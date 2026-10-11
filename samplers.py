"""Uniform-t samplers. History always contains raw guided velocities."""
import torch

METHODS = ('euler', 'heun', 'heun_flip', 'ab2', 'cab2', 'sg', 'sg_prev')


class Sampler:
    def __init__(self, net, config, cfg_scale, diagnostics=None):
        self.net = net
        self.config = config
        self.cfg_scale = cfg_scale
        self.trace = []
        self.diagnostics = diagnostics
        self.flip_phase = False

    def guided_velocity(self, x, t, labels):
        c = self.config
        if not bool(((t >= 0) & (t < 1)).all()):
            raise ValueError('Model calls must satisfy 0 <= t < 1')
        conditional_only = self.cfg_scale == 1 and c['cfg1_conditional_only']
        model_x = x.flip(-1) if self.flip_phase else x
        if conditional_only:
            clean_c = self.net(model_x, t.flatten(), labels)
            clean_u = None
        else:
            clean = self.net(torch.cat((model_x, model_x)), torch.cat((t.flatten(), t.flatten())),
                             torch.cat((labels, torch.full_like(labels, c['class_num']))))
            clean_c, clean_u = clean.chunk(2)
        if self.flip_phase:
            clean_c = clean_c.flip(-1)
            if clean_u is not None:
                clean_u = clean_u.flip(-1)
        clean_c = clean_c.float()
        if clean_u is not None:
            clean_u = clean_u.float()
        denominator = (1 - t).clamp_min(c['t_eps'])
        v_c = (clean_c - x) / denominator
        v_u = (clean_u - x) / denominator if clean_u is not None else None
        active = (t > c['interval_min']) & (t < c['interval_max'])
        scale = torch.where(active, self.cfg_scale, 1.0)
        if self.cfg_scale == 1:
            velocity, clean_cfg = v_c, clean_c
        else:
            # Preserve upstream velocity-CFG arithmetic for exact Heun regression.
            velocity = v_u + scale * (v_c - v_u)
            clean_cfg = clean_u + scale * (clean_c - clean_u)
        self.trace.append(dict(t=float(t.flatten()[0]), denominator=float(denominator.flatten()[0]),
                               cfg_active=bool(active.flatten()[0]),
                               batch_size=x.shape[0], conditional_only=conditional_only,
                               flipped=self.flip_phase))
        if self.diagnostics is not None:
            self.diagnostics.evaluation(t, denominator, scale, v_c, v_u, velocity)
        return velocity, clean_cfg

    @torch.no_grad()
    def generate(self, noise, labels, method, capture=None):
        if method not in METHODS:
            raise ValueError(method)
        c = self.config
        x = noise * c['noise_scale']
        grid = torch.linspace(0, 1, c['steps'] + 1, device=x.device, dtype=x.dtype)
        times = grid.view(-1, 1, 1, 1, 1).expand(-1, x.shape[0], -1, -1, -1)
        self.trace = []
        v_prev = None
        e_history = []
        if self.diagnostics is not None:
            self.diagnostics.begin_batch()
        for i in range(c['steps']):
            t, t_next = times[i], times[i + 1]
            dt = t_next - t
            self.flip_phase = method == 'heun_flip' and i % 2 == 1
            if self.diagnostics is not None:
                self.diagnostics.begin_step(i, t, dt, self.flip_phase)
            v, clean = self.guided_velocity(x, t, labels)
            e = clean - v
            details = {}
            if method in ('heun', 'heun_flip') and i < c['steps'] - 1:
                v_next, _ = self.guided_velocity(x + dt * v, t_next, labels)
                used = 0.5 * (v + v_next)
                x_next = x + dt * used
            elif method == 'sg':
                v_shifted, _ = self.guided_velocity(x, (t - c['sg_shift']).clamp_min(0), labels)
                used = v + c['sg_scale'] * (v - v_shifted)
                x_next = x + dt * used
            elif method == 'sg_prev':
                used = v if v_prev is None else v + c['sg_prev_scale'] * (v - v_prev)
                x_next = x + dt * used
            elif method == 'cab2' and i >= c['cab_start_step']:
                # At i=3, k_i,k_{i-1},k_{i-2} first all have finite endpoints.
                lam = (1 - grid[i-2:i+2]) / grid[i-2:i+2]
                k_prevprev, k_prev, k = lam[1:] - lam[:-1]
                r, r_prev = k / k_prev, k_prev / k_prevprev
                e_prev, e_prevprev = e_history[-1], e_history[-2]
                extrapolated = (1 + r_prev) * e_prev - r_prev * e_prevprev
                predictor = x / t + k * ((1 + r / 2) * e - r / 2 * e_prev)
                x_next = t_next * (predictor + c['cab_theta'] * k * (e - extrapolated))
                used = (x_next - x) / dt
                if self.diagnostics is not None:
                    details = dict(k=k, r=r, r_prev=r_prev, predictor=predictor,
                                   extrapolated=extrapolated, theta=c['cab_theta'])
            elif method in ('ab2', 'cab2') and v_prev is not None:
                used = 1.5 * v - 0.5 * v_prev
                x_next = x + dt * used
            else:
                used = v
                x_next = x + dt * used
            if self.diagnostics is not None:
                self.diagnostics.end_step(x, x_next, used, e, e_history, details)
            if capture is not None:
                capture(i, x, x_next, v, v_prev)
            v_prev = v
            e_history = (e_history + [e])[-2:]
            x = x_next
        self.flip_phase = False
        return x


def initial_noise(sample_ids, config):
    """CPU per-ID generators make input independent of batching and sharding."""
    if config.get('noise'):
        import numpy as np
        from common import BASE, bounded
        bank=np.load(bounded(BASE/config['noise']),mmap_mode='r',allow_pickle=False)
        if bank.shape!=(config['noise_num_images'],3,config['img_size'],config['img_size']) or bank.dtype!=np.float32:
            raise ValueError('Noise bank shape/dtype mismatch')
        indices=[int(i)*config.get('noise_index_stride',1) for i in sample_ids]
        return torch.from_numpy(np.array(bank[indices],copy=True))
    return torch.stack([torch.randn((3, config['img_size'], config['img_size']),
                        generator=torch.Generator(device='cpu').manual_seed(config['eval_seed'] + int(i)),
                        dtype=torch.float32) for i in sample_ids])
