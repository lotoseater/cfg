"""Per-image velocity geometry; observers never run the model or modify tensors."""

import json
from pathlib import Path

import numpy as np
import torch

from .vectors import VectorMetrics


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
            raise ValueError("Geometry requires the unconditional branch even at CFG=1")
        self.calls.append(
            dict(cond=vc, uncond=vu, cfg=v, t=t, denominator=denominator, scale=scale)
        )

    def end_step(self, x, x_next, combined, e, e_history, cab):
        # Disable BF16 autocast for every reduction and keep all vectors FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            metrics = VectorMetrics(x)

            current = self.calls[0]
            metrics.scalar("time", self.t)
            metrics.scalar("dt", self.dt)
            metrics.scalar("flipped", self.flipped)
            metrics.scalar("history_available", self.previous is not None)
            for number, call in enumerate(self.calls):
                prefix = (
                    "current"
                    if number == 0
                    else ("shifted" if self.method == "sg" else "heun_predictor")
                )
                metrics.scalar(prefix + ".time", call["t"])
                metrics.scalar(prefix + ".denominator", call["denominator"])
                metrics.scalar(prefix + ".cfg_scale", call["scale"])
                metrics.pair(
                    prefix + ".cond_vs_uncond",
                    prefix + ".cond",
                    call["cond"],
                    prefix + ".uncond",
                    call["uncond"],
                )
                metrics.norm(prefix + ".cfg", call["cfg"])
                metrics.pair(
                    prefix + ".cfg_vs_cond",
                    prefix + ".cfg",
                    call["cfg"],
                    prefix + ".cond",
                    call["cond"],
                )

            if self.previous is not None:
                metrics.scalar("previous.time", self.previous["time"])
                for branch in ("cond", "uncond", "cfg"):
                    metrics.pair(
                        "history." + branch,
                        "current." + branch,
                        current[branch],
                        "previous." + branch,
                        self.previous[branch],
                    )
            if len(self.calls) == 2:
                aux = self.calls[1]
                prefix = "shifted" if self.method == "sg" else "heun_predictor"
                for branch in ("cond", "uncond", "cfg"):
                    metrics.pair(
                        prefix + ".current_vs_aux." + branch,
                        "current." + branch,
                        current[branch],
                        prefix + "." + branch,
                        aux[branch],
                    )

            self._record_update(metrics, current, x, x_next, combined)
            metrics.scalar("cab_active", bool(cab))
            if cab:
                self._record_cab(metrics, x, e, e_history, cab)

            columns = sorted(metrics.values)
            values = torch.stack([metrics.values[key] for key in columns], dim=1).cpu().numpy()
            if np.isinf(values).any():
                raise FloatingPointError("Infinite geometry statistic")
            # All base norm fields must be finite; NaN is allowed only for undefined comparisons.
            for key in ("current.cond.l2", "current.uncond.l2", "current.cfg.l2", "update.l2"):
                if not np.isfinite(values[:, columns.index(key)]).all():
                    raise FloatingPointError("Nonfinite base velocity statistic: " + key)
            self.rows.append((columns, values))
            self.previous = {branch: current[branch] for branch in ("cond", "uncond", "cfg")}
            self.previous["time"] = float(self.t.flatten()[0])
            self.calls = []

    def _record_update(self, metrics, current, x, x_next, combined):
        dimension = metrics.dimension
        tiny = metrics.tiny
        base = current["cfg"]
        update = (x_next - x) / self.dt
        correction = combined - base
        cfg_direction = current["cond"] - current["uncond"]
        metrics.pair("combined_vs_base", "combined", combined, "current.cfg", base)
        metrics.pair("update_vs_base", "update", update, "current.cfg", base)
        metrics.pair("update_vs_combined", "update", update, "combined", combined)
        metrics.pair("correction_vs_base", "correction", correction, "current.cfg", base)
        metrics.pair(
            "correction_vs_cfg_direction", "correction", correction, "cfg_direction", cfg_direction
        )
        metrics.pair("cfg_direction_vs_base", "cfg_direction", cfg_direction, "current.cfg", base)
        metrics.pair(
            "cfg_correction_vs_base", "cfg_correction", base - current["cond"], "current.cfg", base
        )
        base_norm = metrics.norms["current.cfg"]
        dot = (correction * base).flatten(1).sum(1)
        parallel = dot / base_norm.clamp_min(tiny)
        perpendicular = torch.linalg.vector_norm(
            (
                correction
                - (dot / base_norm.square().clamp_min(tiny**2))[:, None, None, None] * base
            ).flatten(1),
            dim=1,
        )
        nan = torch.full_like(base_norm, float("nan"))
        valid = base_norm > tiny
        metrics.values["correction.parallel_rms_signed"] = torch.where(
            valid, parallel / dimension**0.5, nan
        )
        metrics.values["correction.perpendicular_rms"] = torch.where(
            valid, perpendicular / dimension**0.5, nan
        )
        metrics.values["correction.parallel_over_base"] = torch.where(
            valid, parallel / base_norm.clamp_min(tiny), nan
        )
        metrics.values["correction.perpendicular_over_base"] = torch.where(
            valid, perpendicular / base_norm.clamp_min(tiny), nan
        )

    def _record_cab(self, metrics, x, e, e_history, cab):
        current = self.calls[0]
        base = current["cfg"]
        cfg_direction = current["cond"] - current["uncond"]
        e_prev, e_prevprev = e_history[-1], e_history[-2]
        for name, other in [("previous", e_prev), ("previous_previous", e_prevprev)]:
            metrics.pair("cab.e_vs_" + name, "cab.e", e, "cab.e_" + name, other)
        metrics.pair(
            "cab.e_history", "cab.e_previous", e_prev, "cab.e_previous_previous", e_prevprev
        )
        predictor_e = (1 + cab["r"] / 2) * e - cab["r"] / 2 * e_prev
        defect = e - cab["extrapolated"]
        metrics.pair(
            "cab.predictor_vs_defect", "cab.predictor_e", predictor_e, "cab.defect_e", defect
        )
        tn = self.t + self.dt
        drift = x / self.t
        history_velocity = tn * cab["k"] * predictor_e / self.dt
        theta_velocity = tn * cab["theta"] * cab["k"] * defect / self.dt
        predictor_velocity = (tn * cab["predictor"] - x) / self.dt
        metrics.pair(
            "cab.drift_vs_history",
            "cab.drift_velocity",
            drift,
            "cab.history_velocity",
            history_velocity,
        )
        metrics.pair(
            "cab.theta_vs_predictor",
            "cab.theta_velocity",
            theta_velocity,
            "cab.predictor_velocity",
            predictor_velocity,
        )
        metrics.pair("cab.theta_vs_base", "cab.theta_velocity", theta_velocity, "current.cfg", base)
        metrics.pair(
            "cab.predictor_vs_base",
            "cab.predictor_velocity",
            predictor_velocity,
            "current.cfg",
            base,
        )
        metrics.pair(
            "cab.theta_vs_cfg_direction",
            "cab.theta_velocity",
            theta_velocity,
            "cfg_direction",
            cfg_direction,
        )
        for key in ("k", "r", "r_prev", "theta"):
            metrics.scalar("cab." + key, cab[key])
        metrics.scalar("cab.e_coefficient", 1 + cab["r"] / 2 + cab["theta"])
        metrics.scalar(
            "cab.e_previous_coefficient", -cab["r"] / 2 - cab["theta"] * (1 + cab["r_prev"])
        )
        metrics.scalar("cab.e_previous_previous_coefficient", cab["theta"] * cab["r_prev"])

    def arrays(self):
        columns = sorted(set().union(*(keys for keys, _ in self.rows)))
        result = np.full(
            (len(self.rows), self.rows[0][1].shape[0], len(columns)), np.nan, dtype=np.float32
        )
        mapping = {key: i for i, key in enumerate(columns)}
        for i, (keys, values) in enumerate(self.rows):
            result[i][:, [mapping[key] for key in keys]] = values
        return columns, result

    def save_batch(self, folder, start, sample_ids, class_ids):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        columns, values = self.arrays()
        schema = folder / "schema.json"
        if not schema.exists():
            with schema.open("x") as handle:
                json.dump(
                    dict(
                        version=1,
                        columns=columns,
                        axes=["step", "sample", "metric"],
                        vector_space="FP32 pixel/state coordinates; flipped predictions returned to canonical frame",
                        angle_units="degrees",
                        norm_units="L2 and RMS over all CHW elements per image",
                        undefined="NaN: zero-vector angles/zero denominators or unavailable history/CAB stage",
                        method=self.method,
                        extra_model_evaluations=0,
                    ),
                    handle,
                    indent=2,
                )
        elif json.loads(schema.read_text())["columns"] != columns:
            raise AssertionError("Geometry schema changed between batches")
        target = folder / f"batch_{start:06d}.npz"
        with target.open("xb") as handle:
            np.savez_compressed(
                handle,
                sample_ids=np.asarray(sample_ids, dtype=np.int64),
                class_ids=np.asarray(class_ids, dtype=np.int64),
                values=values,
            )
        self.begin_batch()
        return dict(
            path=target.name,
            steps=values.shape[0],
            samples=values.shape[1],
            metrics=values.shape[2],
        )
