"""Per-image velocity geometry; observers never run the model or modify tensors."""
import json
from pathlib import Path
import numpy as np
import torch


class VelocityGeometry:
    def __init__(self, method):
        self.method = method
        self.begin_batch()

    def begin_batch(self):
        self.rows = []
        self.previous = None
        self.calls = []

    def begin_step(self, index, t, dt, flipped):
        self.index, self.t, self.dt, self.flipped = index, t, dt, flipped
        self.calls = []

    def evaluation(self, t, denominator, scale, vc, vu, v):
        if vu is None:
            raise ValueError('Geometry requires the unconditional branch even at CFG=1')
        self.calls.append(dict(cond=vc, uncond=vu, cfg=v, t=t,
                               denominator=denominator, scale=scale))

    def end_step(self, x, x_next, combined, e, e_history, cab):
        # Disable BF16 autocast for every reduction and keep all vectors FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            row = {}
            norms = {}
            batch = x.shape[0]
            dimension = x[0].numel()
            tiny = 1e-12

            def scalar(name, value):
                row[name] = torch.as_tensor(value, device=x.device, dtype=torch.float32).reshape(-1)[0].expand(batch)

            def norm(name, vector):
                if name not in norms:
                    norms[name] = torch.linalg.vector_norm(vector.float().flatten(1), dim=1)
                    row[name + '.l2'] = norms[name]
                    row[name + '.rms'] = norms[name] / dimension**.5
                return norms[name]

            def pair(name, a_name, a, b_name, b):
                na, nb = norm(a_name, a), norm(b_name, b)
                dot = (a.float() * b.float()).flatten(1).sum(1)
                valid = (na > tiny) & (nb > tiny)
                nan = torch.full_like(na, float('nan'))
                cos = torch.where(valid, (dot / (na * nb).clamp_min(tiny**2)).clamp(-1, 1), nan)
                row[name + '.dot_mean'] = dot / dimension
                row[name + '.cosine'] = cos
                # atan2 resolves small angles more accurately than acos near cosine=1.
                parallel = dot / nb.clamp_min(tiny)
                perpendicular = torch.linalg.vector_norm(
                    (a.float() - (dot / nb.square().clamp_min(tiny**2))[:, None, None, None] * b.float()).flatten(1), dim=1)
                row[name + '.angle_deg'] = torch.where(valid, torch.rad2deg(torch.atan2(perpendicular, parallel)), nan)
                row[name + '.norm_ratio_a_over_b'] = torch.where(nb > tiny, na / nb.clamp_min(tiny), nan)
                # Explicit difference avoids catastrophic cancellation of close vectors.
                difference = torch.linalg.vector_norm((a.float() - b.float()).flatten(1), dim=1)
                row[name + '.difference_rms'] = difference / dimension**.5

            current = self.calls[0]
            scalar('time', self.t)
            scalar('dt', self.dt)
            scalar('flipped', self.flipped)
            scalar('history_available', self.previous is not None)
            for number, call in enumerate(self.calls):
                prefix = 'current' if number == 0 else ('shifted' if self.method == 'sg' else 'heun_predictor')
                scalar(prefix + '.time', call['t'])
                scalar(prefix + '.denominator', call['denominator'])
                scalar(prefix + '.cfg_scale', call['scale'])
                pair(prefix + '.cond_vs_uncond', prefix + '.cond', call['cond'],
                     prefix + '.uncond', call['uncond'])
                norm(prefix + '.cfg', call['cfg'])
                pair(prefix + '.cfg_vs_cond', prefix + '.cfg', call['cfg'],
                     prefix + '.cond', call['cond'])

            if self.previous is not None:
                scalar('previous.time', self.previous['time'])
                for branch in ('cond', 'uncond', 'cfg'):
                    pair('history.' + branch, 'current.' + branch, current[branch],
                         'previous.' + branch, self.previous[branch])
            if len(self.calls) == 2:
                aux = self.calls[1]
                prefix = 'shifted' if self.method == 'sg' else 'heun_predictor'
                for branch in ('cond', 'uncond', 'cfg'):
                    pair(prefix + '.current_vs_aux.' + branch,
                         'current.' + branch, current[branch], prefix + '.' + branch, aux[branch])

            base = current['cfg']
            update = (x_next - x) / self.dt
            correction = combined - base
            cfg_direction = current['cond'] - current['uncond']
            pair('combined_vs_base', 'combined', combined, 'current.cfg', base)
            pair('update_vs_base', 'update', update, 'current.cfg', base)
            pair('update_vs_combined', 'update', update, 'combined', combined)
            pair('correction_vs_base', 'correction', correction, 'current.cfg', base)
            pair('correction_vs_cfg_direction', 'correction', correction, 'cfg_direction', cfg_direction)
            pair('cfg_direction_vs_base', 'cfg_direction', cfg_direction, 'current.cfg', base)
            pair('cfg_correction_vs_base', 'cfg_correction', base - current['cond'], 'current.cfg', base)
            base_norm = norms['current.cfg']
            dot = (correction * base).flatten(1).sum(1)
            parallel = dot / base_norm.clamp_min(tiny)
            perpendicular = torch.linalg.vector_norm(
                (correction - (dot / base_norm.square().clamp_min(tiny**2))[:, None, None, None] * base).flatten(1), dim=1)
            nan = torch.full_like(base_norm, float('nan'))
            valid = base_norm > tiny
            row['correction.parallel_rms_signed'] = torch.where(valid, parallel / dimension**.5, nan)
            row['correction.perpendicular_rms'] = torch.where(valid, perpendicular / dimension**.5, nan)
            row['correction.parallel_over_base'] = torch.where(valid, parallel / base_norm.clamp_min(tiny), nan)
            row['correction.perpendicular_over_base'] = torch.where(valid, perpendicular / base_norm.clamp_min(tiny), nan)
            scalar('cab_active', bool(cab))
            if cab:
                e_prev, e_prevprev = e_history[-1], e_history[-2]
                for name, other in [('previous', e_prev), ('previous_previous', e_prevprev)]:
                    pair('cab.e_vs_' + name, 'cab.e', e, 'cab.e_' + name, other)
                pair('cab.e_history', 'cab.e_previous', e_prev, 'cab.e_previous_previous', e_prevprev)
                predictor_e = (1 + cab['r']/2) * e - cab['r']/2 * e_prev
                defect = e - cab['extrapolated']
                pair('cab.predictor_vs_defect', 'cab.predictor_e', predictor_e, 'cab.defect_e', defect)
                tn = self.t + self.dt
                drift = x / self.t
                history_velocity = tn * cab['k'] * predictor_e / self.dt
                theta_velocity = tn * cab['theta'] * cab['k'] * defect / self.dt
                predictor_velocity = (tn * cab['predictor'] - x) / self.dt
                pair('cab.drift_vs_history', 'cab.drift_velocity', drift, 'cab.history_velocity', history_velocity)
                pair('cab.theta_vs_predictor', 'cab.theta_velocity', theta_velocity,
                     'cab.predictor_velocity', predictor_velocity)
                pair('cab.theta_vs_base', 'cab.theta_velocity', theta_velocity, 'current.cfg', base)
                pair('cab.predictor_vs_base', 'cab.predictor_velocity', predictor_velocity, 'current.cfg', base)
                pair('cab.theta_vs_cfg_direction', 'cab.theta_velocity', theta_velocity, 'cfg_direction', cfg_direction)
                for key in ('k', 'r', 'r_prev', 'theta'):
                    scalar('cab.' + key, cab[key])
                scalar('cab.e_coefficient', 1 + cab['r']/2 + cab['theta'])
                scalar('cab.e_previous_coefficient', -cab['r']/2 - cab['theta']*(1+cab['r_prev']))
                scalar('cab.e_previous_previous_coefficient', cab['theta']*cab['r_prev'])

            columns = sorted(row)
            values = torch.stack([row[key] for key in columns], dim=1).cpu().numpy()
            if np.isinf(values).any():
                raise FloatingPointError('Infinite geometry statistic')
            # All base norm fields must be finite; NaN is allowed only for undefined comparisons.
            for key in ('current.cond.l2', 'current.uncond.l2', 'current.cfg.l2', 'update.l2'):
                if not np.isfinite(values[:, columns.index(key)]).all():
                    raise FloatingPointError('Nonfinite base velocity statistic: ' + key)
            self.rows.append((columns, values))
            self.previous = {branch: current[branch] for branch in ('cond', 'uncond', 'cfg')}
            self.previous['time'] = float(self.t.flatten()[0])
            self.calls = []

    def arrays(self):
        columns = sorted(set().union(*(keys for keys, _ in self.rows)))
        result = np.full((len(self.rows), self.rows[0][1].shape[0], len(columns)), np.nan, dtype=np.float32)
        mapping = {key: i for i, key in enumerate(columns)}
        for i, (keys, values) in enumerate(self.rows):
            result[i][:, [mapping[key] for key in keys]] = values
        return columns, result

    def save_batch(self, folder, start, sample_ids, class_ids):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        columns, values = self.arrays()
        schema = folder / 'schema.json'
        if not schema.exists():
            with schema.open('x') as handle:
                json.dump(dict(version=1, columns=columns, axes=['step', 'sample', 'metric'],
                    vector_space='FP32 pixel/state coordinates; flipped predictions returned to canonical frame',
                    angle_units='degrees', norm_units='L2 and RMS over all CHW elements per image',
                    undefined='NaN: zero-vector angles/zero denominators or unavailable history/CAB stage',
                    method=self.method, extra_model_evaluations=0), handle, indent=2)
        elif json.loads(schema.read_text())['columns'] != columns:
            raise AssertionError('Geometry schema changed between batches')
        target = folder / f'batch_{start:06d}.npz'
        with target.open('xb') as handle:
            np.savez_compressed(handle, sample_ids=np.asarray(sample_ids, dtype=np.int64),
                class_ids=np.asarray(class_ids, dtype=np.int64), values=values)
        self.begin_batch()
        return dict(path=target.name, steps=values.shape[0], samples=values.shape[1], metrics=values.shape[2])


def summarize_geometry(folder, expected_ids=None):
    folder = Path(folder)
    schema = json.loads((folder / 'schema.json').read_text())
    values, ids = [], []
    for path in sorted(folder.glob('batch_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            ids.extend(data['sample_ids'].tolist())
            values.append(data['values'])
    if expected_ids is not None and ids != list(expected_ids):
        raise AssertionError('Geometry sample IDs differ from generated images')
    all_values = np.concatenate(values, axis=1)
    if np.isinf(all_values).any():
        raise FloatingPointError('Infinite geometry statistics')
    rows = []
    for step, block in enumerate(all_values):
        stats = {}
        for col, key in enumerate(schema['columns']):
            column = block[:, col].astype(np.float64)
            finite = column[np.isfinite(column)]
            stats[key] = dict(valid_count=len(finite), undefined_count=len(column)-len(finite),
                mean=float(finite.mean()) if len(finite) else None,
                std=float(finite.std()) if len(finite) else None,
                p10=float(np.quantile(finite, .1)) if len(finite) else None,
                p50=float(np.quantile(finite, .5)) if len(finite) else None,
                p90=float(np.quantile(finite, .9)) if len(finite) else None)
        rows.append(dict(step=step, time=stats['time']['mean'], metrics=stats))
    result = dict(num_images=len(ids), steps=len(rows), method=schema['method'],
                  extra_model_evaluations=0, rows=rows)
    with (folder / 'summary.json').open('x') as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write('\n')
    return dict(num_images=len(ids), steps=len(rows), metric_columns=len(schema['columns']),
                extra_model_evaluations=0, summary='velocity_geometry/summary.json')
