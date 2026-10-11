"""Classwise inference and blinded matched-seed review, using frozen saved features."""

import json
import random

import numpy as np
from PIL import Image

from src.io import write_json
from src.replacement.statistics import compare, holm


def analyze(root, output, reference):
    plan = json.loads((root / "plan.json").read_text())
    study = plan["protocol"]
    records = json.loads((root / "samples.json").read_text())
    labels = np.asarray([row["class_id"] for row in records])
    results = []
    for target in study["target_classes"]:
        mask = labels == target
        with np.load(reference / f"class_{target}.npz", allow_pickle=False) as real:
            real_features, real_probability = real["features"], real["probabilities"]
        with np.load(output / "unconditional.npz", allow_pickle=False) as baseline:
            b_feature, b_probability = baseline["features"][mask], baseline["probabilities"][mask]
        for group in plan["groups"]:
            if group["name"] == "unconditional":
                continue
            with np.load(output / f"{group['name']}.npz", allow_pickle=False) as treatment:
                result = compare(
                    b_feature,
                    treatment["features"][mask],
                    real_features,
                    b_probability,
                    treatment["probabilities"][mask],
                    real_probability,
                    target,
                    bootstrap=study["bootstrap_replicates"],
                    permutations=study["permutation_replicates"],
                    seed=study["statistics_seed"] + target,
                )
            results.append(
                dict(
                    target=target,
                    reference=group["overrides"]["reference_class"],
                    group=group["name"],
                    confirmatory=plan["mode"] == "formal" and group["name"] != "same_target",
                    metrics=result,
                )
            )
    family = [
        (row, metric)
        for row in results
        if row["group"] != "same_target"
        for metric in ("accuracy", "dino_kid")
    ]
    adjusted = holm([row["metrics"][metric]["p_value"] for row, metric in family])
    for (row, metric), value in zip(family, adjusted):
        row["metrics"][metric]["p_holm"] = value
    write_json(
        output / "classwise_results.json",
        dict(
            stage=plan["mode"],
            results=results,
            holm_family_size=len(family),
            pilot_is_exploratory=plan["mode"] == "pilot",
            equivalence_claim=False,
        ),
    )
    lines = [
        "# 每类替代参考速度比较",
        "",
        "差值为 treatment − unconditional；命中率越高越好，DINO-KID 越低越好。",
        "pilot 为探索性分析。非显著不表示等效；同类替代是单列的机制对照。",
        "",
        "| 目标类 | 替代类 | 命中率 baseline → treatment | 命中率差值及95%CI | KID差值及95%CI | 命中率 Holm p | KID Holm p | 多样性 baseline → treatment |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        metrics = row["metrics"]
        accuracy, kid, diversity = (
            metrics[key] for key in ("accuracy", "dino_kid", "diversity_ratio")
        )

        def difference(metric, scale=1):
            low, high = metric["delta_ci95"]
            return f"{scale * metric['delta']:.5f} [{scale * low:.5f}, {scale * high:.5f}]"

        lines.append(
            f"| {row['target']} | {row['reference']} | {accuracy['baseline']:.3%} → {accuracy['treatment']:.3%} | {difference(accuracy, 100)} pp | {difference(kid)} | {accuracy.get('p_holm', 'control')} | {kid.get('p_holm', 'control')} | {diversity['baseline']:.3f} → {diversity['treatment']:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    real_manifest = json.loads((reference / "real_manifest.json").read_text())
    make_blind_review(root, output / "blind", study, records, plan["groups"], real_manifest)


def make_blind_review(root, output, study, records, groups, real_manifest):
    output.mkdir()
    names = {row["class_id"]: row["name"] for row in real_manifest}
    for row in real_manifest:
        for index, image in enumerate(row["images"][:3]):
            with Image.open(image["path"]) as reference_image:
                reference_image.convert("RGB").save(output / f"real_{row['class_id']}_{index}.png")
    rng = random.Random(study["statistics_seed"])
    chosen = {}
    for target in study["target_classes"]:
        candidates = [row for row in records if row["class_id"] == target]
        chosen[target] = rng.sample(
            candidates, min(study["blind_seeds_per_class"], len(candidates))
        )
    trials = [
        (target, sample, group)
        for target in study["target_classes"]
        for sample in chosen[target]
        for group in groups
        if group["name"] != "unconditional"
    ]
    rng.shuffle(trials)
    key, display = [], []
    for index, (target, sample, group) in enumerate(trials):
        name = f"trial_{index:04d}"
        arms = ["unconditional", group["name"]]
        rng.shuffle(arms)
        for side, arm in zip(("A", "B"), arms):
            with Image.open(root / arm / "images" / f"{sample['sample_id']:06d}.png") as image:
                image.save(output / f"{name}_{side}.png")
        key.append(
            dict(trial=name, target=target, sample_id=sample["sample_id"], A=arms[0], B=arms[1])
        )
        real_images = "".join(
            f"<img src='real_{target}_{index}.png' alt='real reference'>" for index in range(3)
        )
        display.append(
            f"<section><h3>{name} · {names[target]} (class {target})</h3><img src='{name}_A.png' alt='A'><img src='{name}_B.png' alt='B'><p>真实参照：</p>{real_images}</section>"
        )
    write_json(output.parent / "blind_key_private.json", key)
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Blind paired review</title><style>img{width:256px;height:256px;object-fit:contain;margin:8px}section{border-bottom:1px solid #ddd}</style><p>左 A，右 B。哪张更像目标类？可选 A / B / 相当 / 都差。记录类别错误、结构错误和伪影。</p>"
        + "\n".join(display)
    )
    (output / "ratings.csv").write_text(
        "trial,target,choice,artifact_A,artifact_B,notes\n"
        + "\n".join(f"{row['trial']},{row['target']},,,," for row in key)
        + "\n"
    )
