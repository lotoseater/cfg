"""Conditional/reference predictions and canonical-coordinate CFG velocity."""

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
        reference = config.get("reference_class")
        if reference is not None and self.diagnostics is not None:
            raise ValueError("Legacy geometry field names assume an unconditional reference")
        if (
            reference is not None
            and reference != "target"
            and (not isinstance(reference, int) or not 0 <= reference < config["class_num"])
        ):
            raise ValueError("Invalid replacement reference class")
        reference_labels = (
            labels
            if reference == "target"
            else torch.full_like(labels, config["class_num"] if reference is None else reference)
        )
        conditional_only = self.cfg_scale == 1 and config["cfg1_conditional_only"]
        model_state = state.flip(-1) if self.flip_phase else state
        if conditional_only:
            conditional_clean = self.net(model_state, time.flatten(), labels)
            reference_clean = None
        else:
            clean = self.net(
                torch.cat((model_state, model_state)),
                torch.cat((time.flatten(), time.flatten())),
                torch.cat((labels, reference_labels)),
            )
            (conditional_clean, reference_clean) = clean.chunk(2)
        if self.flip_phase:
            conditional_clean = conditional_clean.flip(-1)
            if reference_clean is not None:
                reference_clean = reference_clean.flip(-1)
        conditional_clean = conditional_clean.float()
        if reference_clean is not None:
            reference_clean = reference_clean.float()
        denominator = (1 - time).clamp_min(config["t_eps"])
        conditional_velocity = (conditional_clean - state) / denominator
        reference_velocity = (
            (reference_clean - state) / denominator if reference_clean is not None else None
        )
        active = (time > config["interval_min"]) & (time < config["interval_max"])
        scale = torch.where(active, self.cfg_scale, 1.0)
        if self.cfg_scale == 1:
            (velocity, guided_clean) = (conditional_velocity, conditional_clean)
        else:
            velocity = reference_velocity + scale * (conditional_velocity - reference_velocity)
            guided_clean = reference_clean + scale * (conditional_clean - reference_clean)
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
                time, denominator, scale, conditional_velocity, reference_velocity, velocity
            )
        return (velocity, guided_clean)
