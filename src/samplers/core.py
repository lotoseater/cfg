"""Uniform-time sampling with raw velocity history and optional observers."""

import torch

from .guidance import VelocityField
from .steps import ab2_step, cab2_step, euler_step, heun_step, sg_step

METHODS = ("euler", "heun", "heun_flip", "ab2", "cab2", "sg", "sg_prev")


class Sampler:
    def __init__(self, net, config, cfg_scale, diagnostics=None):
        self.config = config
        self.diagnostics = diagnostics
        self.field = VelocityField(net, config, cfg_scale, diagnostics)

    @property
    def trace(self):
        return self.field.trace

    def guided_velocity(self, state, time, labels):
        return self.field.evaluate(state, time, labels)

    @torch.no_grad()
    def generate(self, noise, labels, method, capture=None):
        if method not in METHODS:
            raise ValueError(f"Unknown sampling method: {method}")
        config = self.config
        state = noise * config["noise_scale"]
        grid = torch.linspace(0, 1, config["steps"] + 1, device=state.device, dtype=state.dtype)
        times = grid.view(-1, 1, 1, 1, 1).expand(-1, state.shape[0], -1, -1, -1)
        self.field.trace = []
        previous_velocity = None
        derivative_history = []
        if self.diagnostics is not None:
            self.diagnostics.begin_batch()

        for index in range(config["steps"]):
            time, next_time = times[index], times[index + 1]
            dt = next_time - time
            self.field.flip_phase = method == "heun_flip" and index % 2 == 1
            if self.diagnostics is not None:
                self.diagnostics.begin_step(index, time, dt, self.field.flip_phase)

            velocity, clean = self.guided_velocity(state, time, labels)
            derivative = clean - velocity
            details = {}
            if method in ("heun", "heun_flip") and index < config["steps"] - 1:
                predictor_velocity, _ = self.guided_velocity(
                    state + dt * velocity, next_time, labels
                )
                next_state, combined = heun_step(state, dt, velocity, predictor_velocity)
            elif method == "sg":
                shifted_time = (time - config["sg_shift"]).clamp_min(0)
                shifted_velocity, _ = self.guided_velocity(state, shifted_time, labels)
                next_state, combined = sg_step(
                    state, dt, velocity, shifted_velocity, config["sg_scale"]
                )
            elif method == "sg_prev" and previous_velocity is not None:
                next_state, combined = sg_step(
                    state, dt, velocity, previous_velocity, config["sg_prev_scale"]
                )
            elif method == "cab2" and index >= config["cab_start_step"]:
                next_state, combined, details = cab2_step(
                    state,
                    time,
                    next_time,
                    grid,
                    index,
                    derivative,
                    derivative_history,
                    config["cab_theta"],
                )
            elif method in ("ab2", "cab2") and previous_velocity is not None:
                next_state, combined = ab2_step(state, dt, velocity, previous_velocity)
            else:
                next_state, combined = euler_step(state, dt, velocity)

            if self.diagnostics is not None:
                self.diagnostics.end_step(
                    state,
                    next_state,
                    combined,
                    derivative,
                    derivative_history,
                    details,
                )
            if capture is not None:
                capture(index, state, next_state, velocity, previous_velocity)
            previous_velocity = velocity
            derivative_history = (derivative_history + [derivative])[-2:]
            state = next_state

        self.field.flip_phase = False
        return state
