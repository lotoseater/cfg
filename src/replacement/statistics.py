"""Unbiased DINO polynomial MMD, shared-reference bootstrap and paired tests."""

import numpy as np


def kernel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (a @ b.T / a.shape[1] + 1) ** 3


def off_diagonal_mean(matrix):
    n = len(matrix)
    if n < 2:
        raise ValueError("At least two samples are required")
    return float((matrix.sum() - np.trace(matrix)) / (n * (n - 1)))


def kid(fake, real):
    return (
        off_diagonal_mean(kernel(fake, fake))
        + off_diagonal_mean(kernel(real, real))
        - 2 * kernel(fake, real).mean()
    )


def cosine_distance(features):
    values = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError("Zero DINO feature has undefined cosine distance")
    normalized = values / norm
    return np.clip(1 - normalized @ normalized.T, 0, 2)


def basic_interval(estimate, distribution):
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    return [float(2 * estimate - upper), float(2 * estimate - lower)]


def weighted_u(matrix, counts):
    n = counts.sum(axis=1)
    return ((counts @ matrix * counts).sum(axis=1) - counts @ np.diag(matrix)) / (n * (n - 1))


def paired_kid_scores(baseline, treatment, real):
    """Exact linear scores for independently swapping each paired fake observation."""
    n = len(baseline)
    if n < 2 or treatment.shape != baseline.shape:
        raise ValueError("Paired KID requires equal feature shapes and at least two samples")
    bb, tt, tb = (
        kernel(baseline, baseline),
        kernel(treatment, treatment),
        kernel(treatment, baseline),
    )
    within = tt - bb + tb - tb.T
    np.fill_diagonal(within, 0)
    return (
        within.sum(axis=1) / (n * (n - 1))
        - 2 * (kernel(treatment, real) - kernel(baseline, real)).mean(axis=1) / n
    )


def sign_permutation(scores, rng, repeats):
    observed = abs(float(scores.sum()))
    exceed = 0
    for start in range(0, repeats, 256):
        count = min(256, repeats - start)
        signs = rng.integers(0, 2, size=(count, len(scores))) * 2 - 1
        exceed += int((np.abs(signs @ scores) >= observed - 1e-12).sum())
    return (exceed + 1) / (repeats + 1)


def compare(
    baseline_features,
    treatment_features,
    real_features,
    baseline_probability,
    treatment_probability,
    real_probability,
    target,
    bootstrap=2000,
    permutations=10000,
    seed=0,
):
    """Resample fake seed pairs together and the same real draw for both arms."""
    baseline_features = np.asarray(baseline_features, dtype=np.float64)
    treatment_features = np.asarray(treatment_features, dtype=np.float64)
    real_features = np.asarray(real_features, dtype=np.float64)
    n, m = len(baseline_features), len(real_features)
    if n < 2 or m < 2 or baseline_features.shape != treatment_features.shape:
        raise ValueError("Invalid paired feature coverage")
    if (
        len(baseline_probability) != n
        or len(treatment_probability) != n
        or len(real_probability) != m
    ):
        raise ValueError("Classification and feature counts differ")
    b_hit = (baseline_probability.argmax(1) == target).astype(float)
    t_hit = (treatment_probability.argmax(1) == target).astype(float)
    b_prob, t_prob = baseline_probability[:, target], treatment_probability[:, target]
    bb, tt, rr = (
        kernel(baseline_features, baseline_features),
        kernel(treatment_features, treatment_features),
        kernel(real_features, real_features),
    )
    br, tr = kernel(baseline_features, real_features), kernel(treatment_features, real_features)
    bd, td, rd = (
        cosine_distance(baseline_features),
        cosine_distance(treatment_features),
        cosine_distance(real_features),
    )
    real_diversity = off_diagonal_mean(rd)
    if real_diversity <= 0:
        raise ValueError("Real diversity must be nonzero")
    b_kid, t_kid = (
        off_diagonal_mean(bb) + off_diagonal_mean(rr) - 2 * br.mean(),
        off_diagonal_mean(tt) + off_diagonal_mean(rr) - 2 * tr.mean(),
    )
    point = {
        "accuracy": (b_hit.mean(), t_hit.mean()),
        "target_probability": (b_prob.mean(), t_prob.mean()),
        "dino_kid": (b_kid, t_kid),
        "diversity_ratio": (
            off_diagonal_mean(bd) / real_diversity,
            off_diagonal_mean(td) / real_diversity,
        ),
    }
    distributions = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for start in range(0, bootstrap, 32):
        count = min(32, bootstrap - start)
        fake_counts = rng.multinomial(n, np.full(n, 1 / n), size=count).astype(float)
        real_counts = rng.multinomial(m, np.full(m, 1 / m), size=count).astype(float)
        shared_real = weighted_u(rr, real_counts)
        b_boot = (
            weighted_u(bb, fake_counts)
            + shared_real
            - 2 * ((fake_counts @ br) * real_counts).sum(1) / (n * m)
        )
        t_boot = (
            weighted_u(tt, fake_counts)
            + shared_real
            - 2 * ((fake_counts @ tr) * real_counts).sum(1) / (n * m)
        )
        real_distance = weighted_u(rd, real_counts)
        if (real_distance <= 0).any():
            raise ValueError(
                "Bootstrap real diversity undefined; too few distinct real observations"
            )
        distributions["accuracy"].append(
            np.stack((fake_counts @ b_hit / n, fake_counts @ t_hit / n), 1)
        )
        distributions["target_probability"].append(
            np.stack((fake_counts @ b_prob / n, fake_counts @ t_prob / n), 1)
        )
        distributions["dino_kid"].append(np.stack((b_boot, t_boot), 1))
        distributions["diversity_ratio"].append(
            np.stack(
                (
                    weighted_u(bd, fake_counts) / real_distance,
                    weighted_u(td, fake_counts) / real_distance,
                ),
                1,
            )
        )
    result = {}
    for key, (base, treatment) in point.items():
        draws = np.concatenate(distributions[key])
        result[key] = dict(
            baseline=float(base),
            treatment=float(treatment),
            delta=float(treatment - base),
            baseline_ci95=basic_interval(base, draws[:, 0]),
            treatment_ci95=basic_interval(treatment, draws[:, 1]),
            delta_ci95=basic_interval(treatment - base, draws[:, 1] - draws[:, 0]),
        )
    for key, scores in (
        ("accuracy", (t_hit - b_hit) / n),
        ("target_probability", (t_prob - b_prob) / n),
        ("dino_kid", paired_kid_scores(baseline_features, treatment_features, real_features)),
    ):
        result[key]["p_value"] = sign_permutation(scores, rng, permutations)
    baseline_unit = baseline_features / np.linalg.norm(baseline_features, axis=1, keepdims=True)
    treatment_unit = treatment_features / np.linalg.norm(treatment_features, axis=1, keepdims=True)
    mixed_distance = np.clip(1 - treatment_unit @ baseline_unit.T, 0, 2)
    diversity_within = td - bd + mixed_distance - mixed_distance.T
    np.fill_diagonal(diversity_within, 0)
    scores = diversity_within.sum(1) / (n * (n - 1) * real_diversity)
    result["diversity_ratio"]["p_value"] = sign_permutation(scores, rng, permutations)
    result["real_target_accuracy"] = float((real_probability.argmax(1) == target).mean())
    result["real_target_probability"] = float(real_probability[:, target].mean())
    result["confusion"] = {
        arm: confusion(probability, target)
        for arm, probability in (
            ("baseline", baseline_probability),
            ("treatment", treatment_probability),
        )
    }
    result["inference"] = dict(
        bootstrap="basic; paired fake seeds and shared resampled real observations",
        permutation="paired within-seed label swaps; conditional on held-out real set",
        bootstrap_replicates=bootstrap,
        permutation_replicates=permutations,
    )
    return result


def confusion(probabilities, target, top=5):
    predictions = probabilities.argmax(1)
    labels, counts = np.unique(predictions[predictions != target], return_counts=True)
    order = np.argsort(-counts, kind="stable")[:top]
    return [
        dict(
            class_id=int(labels[index]),
            count=int(counts[index]),
            fraction=float(counts[index] / len(predictions)),
        )
        for index in order
    ]


def holm(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty(len(values))
    result[order] = adjusted
    return result.tolist()
