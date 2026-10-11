"""Compare refactored trajectories and every metric against a frozen checkout."""

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import configured_groups, group_config, load_config
from src.diagnostics import VelocityGeometry
from src.paths import CONFIG_DIR
from src.samplers import Sampler
from tests.test_geometry import AsymmetricToy


def legacy_module(folder, filename):
    spec = importlib.util.spec_from_file_location("legacy_" + filename, folder / (filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    os.environ.get("CFG_LEGACY_SOURCE"), "Set CFG_LEGACY_SOURCE for refactor regression"
)
class RefactorRegression(unittest.TestCase):
    def test_all_methods_outputs_traces_and_geometry(self):
        folder = Path(os.environ["CFG_LEGACY_SOURCE"])
        old_sampler = legacy_module(folder, "samplers").Sampler
        old_geometry = legacy_module(folder, "geometry").VelocityGeometry
        generator = torch.Generator().manual_seed(1729)
        noise = torch.randn((2, 3, 2, 3), generator=generator)
        labels = torch.tensor([12, 999])

        for config_name in ("baseline50.json", "followup50.json", "velocity100.json"):
            config = load_config(CONFIG_DIR / config_name)
            for group in configured_groups(config):
                effective = group_config(config, group)
                with self.subTest(config=config_name, group=group["name"]):
                    old_observer = old_geometry(group["method"])
                    new_observer = VelocityGeometry(group["method"])
                    old = old_sampler(AsymmetricToy(), effective, group["cfg"], old_observer)
                    new = Sampler(AsymmetricToy(), effective, group["cfg"], new_observer)
                    old_states, new_states = [], []

                    def capture_old(index, state, next_state, velocity, previous):
                        old_states.append(next_state)

                    def capture_new(index, state, next_state, velocity, previous):
                        new_states.append(next_state)

                    expected = old.generate(noise, labels, group["method"], capture_old)
                    actual = new.generate(noise, labels, group["method"], capture_new)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.stack(new_states), torch.stack(old_states)))
                    self.assertEqual(new.trace, old.trace)
                    old_columns, old_values = old_observer.arrays()
                    new_columns, new_values = new_observer.arrays()
                    self.assertEqual(new_columns, old_columns)
                    np.testing.assert_array_equal(new_values, old_values)


if __name__ == "__main__":
    unittest.main()
