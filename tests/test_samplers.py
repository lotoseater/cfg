"""Analytic numerical checks and exact regression against frozen upstream code."""

import json
import sys
import unittest
from pathlib import Path

import torch

from src.samplers import Sampler, initial_noise

HERE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((HERE / "configs/baseline50.json").read_text())


class Toy(torch.nn.Module):
    def forward(self, x, t, labels):
        t = t.view(-1, 1, 1, 1)
        return 0.17 * x + 0.31 * t + 0.07 * (labels.view(-1, 1, 1, 1) != 1000)


def upstream(net, config, cfg, method):
    sys.path.insert(0, str(HERE / "src/upstream"))
    from denoiser import Denoiser

    obj = Denoiser.__new__(Denoiser)
    torch.nn.Module.__init__(obj)
    obj.net = net
    obj.steps = config["steps"]
    obj.img_size = 2
    obj.num_classes = 1000
    obj.noise_scale = config["noise_scale"]
    obj.t_eps = config["t_eps"]
    obj.cfg_scale = cfg
    obj.cfg_interval = (config["interval_min"], config["interval_max"])
    obj.method = method
    return obj


class Checks(unittest.TestCase):
    def setUp(self):
        self.noise = torch.randn((2, 3, 2, 2), generator=torch.Generator().manual_seed(11))
        self.labels = torch.tensor([12, 998])

    def test_exact_official_euler_heun(self):
        for steps in (25, 50):
            c = dict(CONFIG, steps=steps)
            for cfg in (1.0, 3.0):
                for method in ("euler", "heun"):
                    reference = upstream(Toy(), c, cfg, method)
                    actual = Sampler(Toy(), c, cfg).generate(self.noise, self.labels, method)
                    expected = reference.generate(self.labels, None, noise=self.noise)
                    self.assertTrue(
                        torch.equal(actual, expected),
                        (steps, cfg, method, (actual - expected).abs().max()),
                    )

    def test_followup_group_overrides(self):
        from src.config import configured_groups, expected_nfe, group_config, load_config

        c = load_config(HERE / "configs/followup50.json")
        groups = configured_groups(c)
        self.assertEqual(len(groups), 8)
        for group in groups:
            effective = group_config(c, group)
            sampler = Sampler(Toy(), effective, group["cfg"])
            output = sampler.generate(self.noise, self.labels, group["method"])
            nfe = 49 if group["method"] == "heun" else 50
            self.assertEqual(expected_nfe(group["method"], effective["steps"]), nfe)
            self.assertEqual(len(sampler.trace), nfe)
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertLess(max(row["t"] for row in sampler.trace), 1.0)
            if group["method"] == "cab2":
                self.assertEqual(effective["cab_theta"], 0.0)

    def test_cfg_boundaries_and_clamp(self):
        for cfg in (1.0, 3.0):
            sampler = Sampler(Toy(), CONFIG, cfg)
            for t_value in (0.0, 0.1, 0.1001, 0.94, 0.96, 0.98):
                t = torch.full((2, 1, 1, 1), t_value)
                velocity, clean = sampler.guided_velocity(self.noise, t, self.labels)
                scale = cfg if 0.1 < t_value < 1 else 1
                c = Toy()(self.noise, t.flatten(), self.labels)
                u = Toy()(self.noise, t.flatten(), torch.full_like(self.labels, 1000))
                clean_ref = u + scale * (c - u)
                torch.testing.assert_close(clean, clean_ref)
                torch.testing.assert_close(
                    velocity, (clean_ref - self.noise) / (1 - t).clamp_min(0.05)
                )
            with self.assertRaises(ValueError):
                sampler.guided_velocity(self.noise, torch.ones((2, 1, 1, 1)), self.labels)
        cfg1 = Sampler(Toy(), dict(CONFIG, cfg1_conditional_only=True), 1)
        all_branches = Sampler(Toy(), CONFIG, 1)
        self.assertTrue(
            torch.equal(
                cfg1.generate(self.noise, self.labels, "ab2"),
                all_branches.generate(self.noise, self.labels, "ab2"),
            )
        )

    def test_raw_history_and_startup(self):
        for method, omega in (("ab2", 0.5), ("sg_prev", 0.05)):
            captured = []
            sampler = Sampler(Toy(), CONFIG, 3)
            sampler.generate(self.noise, self.labels, method, lambda *row: captured.append(row))
            grid = torch.linspace(0, 1, 51)
            for i, x, x_next, v, v_prev in captured:
                if i == 0:
                    self.assertIsNone(v_prev)
                    wanted = x + (grid[i + 1] - grid[i]) * v
                else:
                    self.assertTrue(torch.equal(v_prev, captured[i - 1][3]))
                    used = 1.5 * v - 0.5 * v_prev if method == "ab2" else v + omega * (v - v_prev)
                    wanted = x + (grid[i + 1] - grid[i]) * used
                self.assertTrue(torch.equal(x_next, wanted))

    def test_sg_shift_same_state_and_nfe(self):
        captured = []
        sampler = Sampler(Toy(), CONFIG, 3)
        sampler.generate(self.noise, self.labels, "sg", lambda *row: captured.append(row))
        self.assertEqual(len(sampler.trace), 100)
        for i in range(50):
            actual, shifted = sampler.trace[2 * i : 2 * i + 2]
            self.assertAlmostEqual(shifted["t"], max(actual["t"] - 0.001, 0), places=6)
            self.assertAlmostEqual(shifted["denominator"], max(1 - shifted["t"], 0.05), places=6)
        # Independent first-step Euler check: equal-time branches cancel exactly.
        first = captured[0]
        self.assertTrue(torch.equal(first[2], first[1] + 0.02 * first[3]))

    def test_cab_constant_rectified_derivative(self):
        # On untruncated straight paths, clean=a and v=(a-x)/(1-t)
        # give constant e=b and analytic x=t*a+(1-t)*b.
        class Constant(torch.nn.Module):
            def forward(self, x, t, labels):
                return torch.ones_like(x) * 0.3

        c = dict(CONFIG, t_eps=1e-9)
        captured = []
        sampler = Sampler(Constant(), c, 1)
        result = sampler.generate(
            self.noise, self.labels, "cab2", lambda *row: captured.append(row)
        )
        torch.testing.assert_close(result, torch.ones_like(result) * 0.3, atol=2e-6, rtol=2e-6)
        grid = torch.linspace(0, 1, 51)
        for i, x, next_x, v, prev in captured[:3]:
            used = v if i == 0 else 1.5 * v - 0.5 * prev
            self.assertTrue(torch.equal(next_x, x + (grid[i + 1] - grid[i]) * used))
        for i in range(3, 50):
            lam = (1 - grid[i - 2 : i + 2]) / grid[i - 2 : i + 2]
            self.assertTrue(bool(((lam[1:] - lam[:-1]) < 0).all()))

    def test_cab_nonlinear_independent_reference(self):
        for theta in (0.0, 0.9):
            c = dict(CONFIG, cab_theta=theta)
            sampler = Sampler(Toy(), c, 3)
            actual = sampler.generate(self.noise, self.labels, "cab2")
            x = self.noise.clone()
            grid = torch.linspace(0, 1, 51)
            velocities = []
            derivatives = []
            for i in range(50):
                t = grid[i].expand(2, 1, 1, 1)
                cond = Toy()(x, t.flatten(), self.labels)
                uncond = Toy()(x, t.flatten(), torch.full_like(self.labels, 1000))
                w = 3 if bool(t[0] > 0.1) else 1
                d = (1 - t).clamp_min(0.05)
                vc = (cond - x) / d
                vu = (uncond - x) / d
                v = vu + w * (vc - vu)
                clean = uncond + w * (cond - uncond)
                derivatives.append(clean - v)
                if i < 3:
                    used = v if i == 0 else 1.5 * v - 0.5 * velocities[-1]
                    x = x + (grid[i + 1] - grid[i]) * used
                else:
                    lambdas = [(1 - grid[j]) / grid[j] for j in (i - 2, i - 1, i, i + 1)]
                    ka, kb, kc = (
                        lambdas[1] - lambdas[0],
                        lambdas[2] - lambdas[1],
                        lambdas[3] - lambdas[2],
                    )
                    ratio = kc / kb
                    predicted = x / grid[i] + kc * (
                        (1 + ratio / 2) * derivatives[-1] - ratio / 2 * derivatives[-2]
                    )
                    extrap = derivatives[-2] + (kb / ka) * (derivatives[-2] - derivatives[-3])
                    x = grid[i + 1] * (predicted + theta * kc * (derivatives[-1] - extrap))
                velocities.append(v)
            torch.testing.assert_close(actual, x, rtol=3e-6, atol=3e-6)

    def test_noise_independent_of_batch_and_shard(self):
        c = dict(CONFIG, img_size=2, noise=None)
        full = initial_noise([0, 1, 2, 10, 9999], c)
        self.assertTrue(
            torch.equal(
                full, torch.cat([initial_noise([0, 1], c), initial_noise([2, 10, 9999], c)])
            )
        )
        self.assertTrue(torch.equal(full[[4, 1]], initial_noise([9999, 1], c)))
        labels = [i // 10 for i in range(10000)]
        self.assertEqual([labels.count(i) for i in range(1000)], [10] * 1000)

    def test_endpoints_all_methods(self):
        for method, nfe in [
            ("euler", 50),
            ("heun", 99),
            ("ab2", 50),
            ("cab2", 50),
            ("sg", 100),
            ("sg_prev", 50),
        ]:
            sampler = Sampler(Toy(), CONFIG, 3)
            result = sampler.generate(self.noise, self.labels, method)
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertEqual(len(sampler.trace), nfe)
            self.assertLess(max(r["t"] for r in sampler.trace), 1)
            self.assertAlmostEqual(sampler.trace[-1]["denominator"], 0.05, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
