"""Independent geometry values, canonical flip, and nonintrusive observer tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import configured_groups, expected_nfe, group_config, load_config
from src.diagnostics import VelocityGeometry, summarize_geometry
from src.samplers import Sampler
from tests.test_samplers import Toy, upstream

HERE = Path(__file__).resolve().parents[1]
CONFIG = load_config(HERE / "configs/velocity100.json")


class AsymmetricToy(Toy):
    def forward(self, x, t, labels):
        base = super().forward(x, t, labels)
        return base + torch.linspace(-0.2, 0.3, x.shape[-1], device=x.device)[None, None, None, :]


class GeometryTests(unittest.TestCase):
    def test_analytic_angles_norms_and_projection(self):
        vc = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 0.0]]).view(4, 1, 1, 2)
        vu = torch.tensor([[0.0, 4.0], [0.0, -2.0], [1.0, 0.0001], [0.0, 0.0]]).view(4, 1, 1, 2)
        t = torch.full((4, 1, 1, 1), 0.2)
        g = VelocityGeometry("ab2")
        g.begin_step(0, t, torch.tensor(0.01), False)
        g.evaluation(t, 1 - t, torch.ones_like(t), vc, vu, vc)
        combined = vc + 0.5 * (vc - vu)
        x = torch.zeros_like(vc)
        g.end_step(x, x + 0.01 * combined, combined, torch.zeros_like(vc), [], {})
        columns, values = g.arrays()

        def col(key):
            return values[0, :, columns.index(key)]

        np.testing.assert_allclose(col("current.cond.l2"), [3, 2, 1, 0])
        np.testing.assert_allclose(col("current.uncond.l2"), [4, 2, (1 + 0.0001**2) ** 0.5, 0])
        np.testing.assert_allclose(
            col("current.cond_vs_uncond.angle_deg")[:3],
            [90, 180, np.rad2deg(np.arctan(0.0001))],
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertTrue(np.isnan(col("current.cond_vs_uncond.angle_deg")[3]))
        np.testing.assert_allclose(col("correction.parallel_over_base")[:3], [0.5, 1, 0], atol=1e-7)
        np.testing.assert_allclose(
            col("correction.perpendicular_over_base")[:3], [2 / 3, 0, 0.00005], atol=1e-7
        )

    def test_all_groups_observer_does_not_change_outputs_or_nfe(self):
        noise = torch.randn((3, 3, 2, 3), generator=torch.Generator().manual_seed(5))
        labels = torch.tensor([0, 123, 999])
        for group in configured_groups(CONFIG):
            with self.subTest(group=group["name"]):
                c = group_config(CONFIG, group)
                observed = VelocityGeometry(group["method"])
                a = Sampler(AsymmetricToy(), c, group["cfg"], observed)
                b = Sampler(AsymmetricToy(), c, group["cfg"])
                actual = a.generate(noise, labels, group["method"])
                reference = b.generate(noise, labels, group["method"])
                self.assertTrue(torch.equal(actual, reference))
                self.assertEqual(len(a.trace), expected_nfe(group["method"], c["steps"]))
                columns, values = observed.arrays()
                self.assertEqual(values.shape[:2], (c["steps"], 3))
                self.assertTrue(np.isfinite(values[:, :, columns.index("current.uncond.l2")]).all())
                self.assertEqual(
                    np.isfinite(values[:, :, columns.index("history.cfg.angle_deg")]).sum(),
                    (c["steps"] - 1) * 3,
                )
                if group["method"] == "sg":
                    np.testing.assert_allclose(
                        values[:, 0, columns.index("shifted.time")],
                        np.maximum(values[:, 0, columns.index("time")] - 0.05, 0),
                        atol=1e-7,
                    )
                    # At t=.12, CFG3 is on at current time and off at shifted time.
                    if group["cfg"] == 3:
                        self.assertEqual(values[6, 0, columns.index("current.cfg_scale")], 3)
                        self.assertEqual(values[6, 0, columns.index("shifted.cfg_scale")], 1)

    def test_flip_matches_original_frame_steps_and_phase(self):
        c = dict(CONFIG, steps=50)
        noise = torch.randn((2, 3, 2, 3), generator=torch.Generator().manual_seed(99))
        labels = torch.tensor([5, 999])
        for cfg in (1.0, 3.0):
            model = upstream(AsymmetricToy(), c, cfg, "heun")
            grid = torch.linspace(0, 1, 51)
            x = noise.clone()
            for i in range(50):
                flipped = i % 2 == 1
                state = x.flip(-1) if flipped else x
                t = grid[i].expand(2, 1, 1, 1)
                tn = grid[i + 1].expand(2, 1, 1, 1)
                step = model._euler_step if i == 49 else model._heun_step
                x = step(state, t, tn, labels)
                if flipped:
                    x = x.flip(-1)
            sampler = Sampler(AsymmetricToy(), c, cfg)
            actual = sampler.generate(noise, labels, "heun_flip")
            self.assertTrue(torch.equal(actual, x))
            self.assertEqual(
                [r["flipped"] for r in sampler.trace],
                [i % 2 == 1 for i in range(49) for _ in range(2)] + [True],
            )

    def test_batch_files_tail_and_summary_weighting(self):
        c = dict(CONFIG, steps=50)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            folder = Path(directory) / "velocity_geometry"
            for start, count in [(0, 2), (2, 1)]:
                observer = VelocityGeometry("sg")
                noise = torch.randn(
                    (count, 3, 2, 3), generator=torch.Generator().manual_seed(start)
                )
                Sampler(Toy(), c, 1, observer).generate(noise, torch.arange(count), "sg")
                observer.save_batch(
                    folder, start, list(range(start, start + count)), list(range(count))
                )
            result = summarize_geometry(folder, [0, 1, 2])
            self.assertEqual(result["num_images"], 3)
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(
                summary["rows"][0]["metrics"]["current.cond_vs_uncond.angle_deg"]["valid_count"], 3
            )
            self.assertEqual(
                summary["rows"][0]["metrics"]["correction_vs_base.angle_deg"]["undefined_count"], 3
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
