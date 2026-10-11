"""Conditional/unconditional predictions and canonical-coordinate CFG velocity."""

import torch


class VelocityField:
    def __init__(self, net, config, cfg_scale, diagnostics=None):
        self.net = net
        self.config = config
        self.cfg_scale = cfg_scale
        self.diagnostics = diagnostics
        self.trace = []
        self.flip_phase = False

    def evaluate(self, state, time, labels):
        config = self.config
        if not bool(((time >= 0) & (time < 1)).all()):
            raise ValueError("Model calls must satisfy 0 <= t < 1")
        conditional_only = self.cfg_scale == 1 and config["cfg1_conditional_only"]
        model_state = state.flip(-1) if self.flip_phase else state
        if conditional_only:
            conditional_clean = self.net(model_state, time.flatten(), labels)
            unconditional_clean = None
        else:
            clean = self.net(
                torch.cat((model_state, model_state)),
                torch.cat((time.flatten(), time.flatten())),
                torch.cat((labels, torch.full_like(labels, config["class_num"]))),
            )
            (conditional_clean, unconditional_clean) = clean.chunk(2)
        if self.flip_phase:
            conditional_clean = conditional_clean.flip(-1)
            if unconditional_clean is not None:
                unconditional_clean = unconditional_clean.flip(-1)
        conditional_clean = conditional_clean.float()
        if unconditional_clean is not None:
            unconditional_clean = unconditional_clean.float()
        denominator = (1 - time).clamp_min(config["t_eps"])
        conditional_velocity = (conditional_clean - state) / denominator
        unconditional_velocity = (
            (unconditional_clean - state) / denominator if unconditional_clean is not None else None
        )
        active = (time > config["interval_min"]) & (time < config["interval_max"])
        scale = torch.where(active, self.cfg_scale, 1.0)
        if self.cfg_scale == 1:
            (velocity, guided_clean) = (conditional_velocity, conditional_clean)
        else:
            velocity = unconditional_velocity + scale * (
                conditional_velocity - unconditional_velocity
            )
            guided_clean = unconditional_clean + scale * (conditional_clean - unconditional_clean)
        self.trace.append(
            dict(
                t=float(time.flatten()[0]),
                denominator=float(denominator.flatten()[0]),
                cfg_active=bool(active.flatten()[0]),
                batch_size=state.shape[0],
                conditional_only=conditional_only,
                flipped=self.flip_phase,
            )
        )
        if self.diagnostics is not None:
            self.diagnostics.evaluation(
                time, denominator, scale, conditional_velocity, unconditional_velocity, velocity
            )
        return (velocity, guided_clean)
