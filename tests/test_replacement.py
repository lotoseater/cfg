"""Analytic class-reference formulas, paired noise and independently checked inference."""

import itertools
import unittest

import numpy as np
import torch

from src.config import load_config
from src.replacement.protocol import generator_config, groups, load_protocol, samples
from src.replacement.statistics import compare, holm, kernel, kid, paired_kid_scores, weighted_u
from src.samplers import Sampler, initial_noise


class LabelToy(torch.nn.Module):
    def forward(self, x, t, labels):
        label = labels.float().view(-1, 1, 1, 1) / 1000
        return 0.17 * x + 0.31 * t.view(-1, 1, 1, 1) + 0.07 * label


class ReplacementTests(unittest.TestCase):
    def test_reference_branch_and_cfg1_invariance(self):
        config = dict(load_config(), img_size=2)
        noise = torch.randn((2, 3, 2, 2), generator=torch.Generator().manual_seed(4))
        labels = torch.tensor([12, 998])
        time = torch.full((2, 1, 1, 1), 0.5)
        model = LabelToy()
        conditional = (model(noise, time.flatten(), labels) - noise) / (1 - time)
        replacement = (model(noise, time.flatten(), torch.full_like(labels, 548)) - noise) / (
            1 - time
        )
        actual, _ = Sampler(model, dict(config, reference_class=548), 3).guided_velocity(
            noise, time, labels
        )
        torch.testing.assert_close(actual, replacement + 3 * (conditional - replacement))
        for reference in (None, 548, "target"):
            actual, _ = Sampler(model, dict(config, reference_class=reference), 1).guided_velocity(
                noise, time, labels
            )
            self.assertTrue(torch.equal(actual, conditional))
        same_target, _ = Sampler(model, dict(config, reference_class="target"), 3).guided_velocity(
            noise, time, labels
        )
        self.assertTrue(torch.equal(same_target, conditional))

    def test_paired_seed_manifests_and_batch_independence(self):
        study = load_protocol()
        pilot, formal = samples(study, "pilot"), samples(study, "formal")
        self.assertEqual(len(pilot), 8 * 128)
        self.assertEqual(len(groups(study)), 6)
        self.assertFalse({r["seed"] for r in pilot} & {r["seed"] for r in formal})
        self.assertEqual(
            [f"{r['sample_id']:06d}" for r in pilot], sorted(f"{r['sample_id']:06d}" for r in pilot)
        )
        config = dict(generator_config(study), img_size=2)
        ids = [row["sample_id"] for row in pilot[:3]]
        whole = initial_noise(ids, config)
        split = torch.cat((initial_noise(ids[:1], config), initial_noise(ids[1:], config)))
        self.assertTrue(torch.equal(whole, split))
        self.assertEqual([r["seed"] for r in pilot[:3]], [config["eval_seed"] + i for i in ids])

    def test_unbiased_kid_and_bootstrap_with_duplicate_observations(self):
        x = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        y = np.asarray([[0.0, 1.0], [2.0, 0.0], [2.0, 2.0]])

        def polynomial(a, b):
            return (1 + np.dot(a, b) / 2) ** 3

        expected = (
            sum(polynomial(a, b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j) / 6
        )
        expected += (
            sum(polynomial(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j) / 6
        )
        expected -= 2 * sum(polynomial(a, b) for a in x for b in y) / 9
        self.assertAlmostEqual(kid(x, y), expected)
        indices = [0, 0, 2]
        counts = np.asarray([[2.0, 0.0, 1.0]])
        expected_u = (
            sum(
                polynomial(x[a], x[b])
                for i, a in enumerate(indices)
                for j, b in enumerate(indices)
                if i != j
            )
            / 6
        )
        self.assertAlmostEqual(weighted_u(kernel(x, x), counts)[0], expected_u)

    def test_paired_kid_permutation_scores_match_every_explicit_swap(self):
        rng = np.random.default_rng(4)
        baseline, treatment, real = (
            rng.normal(size=(3, 4)),
            rng.normal(size=(3, 4)),
            rng.normal(size=(7, 4)),
        )
        scores = paired_kid_scores(baseline, treatment, real)
        for signs in itertools.product((-1, 1), repeat=3):
            mask = np.asarray(signs)[:, None] == 1
            swapped_t = np.where(mask, treatment, baseline)
            swapped_b = np.where(mask, baseline, treatment)
            self.assertAlmostEqual(
                np.dot(signs, scores), kid(swapped_t, real) - kid(swapped_b, real), places=10
            )

    def test_identical_arms_have_zero_paired_differences(self):
        rng = np.random.default_rng(9)
        fake, real = rng.normal(size=(12, 4)), rng.normal(size=(50, 4))
        probabilities = rng.dirichlet(np.ones(3), size=12)
        real_probabilities = rng.dirichlet(np.ones(3), size=50)
        result = compare(
            fake,
            fake,
            real,
            probabilities,
            probabilities,
            real_probabilities,
            1,
            bootstrap=64,
            permutations=128,
        )
        for metric in ("accuracy", "target_probability", "dino_kid", "diversity_ratio"):
            self.assertEqual(result[metric]["delta"], 0)
            np.testing.assert_allclose(result[metric]["delta_ci95"], [0, 0], atol=1e-12)
            self.assertEqual(result[metric]["p_value"], 1)
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
