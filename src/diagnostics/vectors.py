"""FP32 per-image vector lengths, pair angles and differences."""

import torch


class VectorMetrics:
    def __init__(self, state):
        self.values = {}
        self.norms = {}
        self.batch = state.shape[0]
        self.dimension = state[0].numel()
        self.device = state.device
        self.tiny = 1e-12

    def scalar(self, name, value):
        self.values[name] = (
            torch.as_tensor(value, device=self.device, dtype=torch.float32)
            .reshape(-1)[0]
            .expand(self.batch)
        )

    def norm(self, name, vector):
        if name not in self.norms:
            self.norms[name] = torch.linalg.vector_norm(vector.float().flatten(1), dim=1)
            self.values[name + ".l2"] = self.norms[name]
            self.values[name + ".rms"] = self.norms[name] / self.dimension**0.5
        return self.norms[name]

    def pair(self, name, a_name, a, b_name, b):
        na, nb = self.norm(a_name, a), self.norm(b_name, b)
        dot = (a.float() * b.float()).flatten(1).sum(1)
        valid = (na > self.tiny) & (nb > self.tiny)
        nan = torch.full_like(na, float("nan"))
        cos = torch.where(valid, (dot / (na * nb).clamp_min(self.tiny**2)).clamp(-1, 1), nan)
        self.values[name + ".dot_mean"] = dot / self.dimension
        self.values[name + ".cosine"] = cos
        # atan2 resolves small angles more accurately than acos near cosine=1.
        parallel = dot / nb.clamp_min(self.tiny)
        perpendicular = torch.linalg.vector_norm(
            (
                a.float()
                - (dot / nb.square().clamp_min(self.tiny**2))[:, None, None, None] * b.float()
            ).flatten(1),
            dim=1,
        )
        self.values[name + ".angle_deg"] = torch.where(
            valid, torch.rad2deg(torch.atan2(perpendicular, parallel)), nan
        )
        self.values[name + ".norm_ratio_a_over_b"] = torch.where(
            nb > self.tiny, na / nb.clamp_min(self.tiny), nan
        )
        # Explicit difference avoids catastrophic cancellation of close vectors.
        difference = torch.linalg.vector_norm((a.float() - b.float()).flatten(1), dim=1)
        self.values[name + ".difference_rms"] = difference / self.dimension**0.5
