"""Resolve a locked protocol, asset identities, class mapping and paired seeds."""

import json
import subprocess
from collections import Counter
from pathlib import Path

from src.config import load_config
from src.io import digest
from src.paths import BASE, CONFIG_DIR, bounded
from src.provenance import assert_upstream

DEFAULT_CONFIG = CONFIG_DIR / "replacement.json"


def load_protocol(path=DEFAULT_CONFIG):
    study = json.loads(Path(path).read_text())
    targets, references = study["target_classes"], study["reference_classes"]
    for labels in (targets, references):
        if (
            not labels
            or len(labels) != len(set(labels))
            or any(
                isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 1000 for x in labels
            )
        ):
            raise ValueError("Class indices must be unique integers in [0, 1000)")
    if set(targets) & set(references):
        raise ValueError("Random references must not overlap target classes; use the named control")
    if study["method"] not in ("heun", "ab2") or study["steps"] not in (50, 100):
        raise ValueError("Study supports Heun/AB2 with 50/100 steps")
    validation_root(study)
    for key in ("dino_source", "dino_weights", "classifier_weights"):
        bounded(BASE / study[key])
    pilot, formal = study["pilot"], study["formal"]
    if max(pilot["images_per_class"], formal["images_per_class"]) > study["seed_stride"]:
        raise ValueError("Sample IDs must not overlap classes")
    if pilot["seed_offset"] + pilot["images_per_class"] > formal["seed_offset"]:
        raise ValueError("Formal seeds must be independent of pilot seeds")
    if formal["seed_offset"] + formal["images_per_class"] > study["seed_stride"]:
        raise ValueError("Formal sample IDs overlap the next target class")
    return study


def verify_assets(study, include_generator=True):
    for prefix in ("dino", "classifier"):
        if digest(BASE / study[prefix + "_weights"]) != study[prefix + "_sha256"]:
            raise RuntimeError("Weight hash mismatch: " + prefix)
    source = bounded(BASE / study["dino_source"])
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != study["dino_source_commit"]:
        raise RuntimeError("DINO source revision changed")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("DINO source has local changes")
    if include_generator:
        config = load_config(CONFIG_DIR / study["base_config"])
        assert_upstream(config)
        if digest(BASE / config["checkpoint"]) != config["checkpoint_sha256"]:
            raise RuntimeError("Generator checkpoint hash mismatch")


def validation_root(study):
    """Read-only dataset link in the user's HDD; never add its target to write boundaries."""
    relative = Path(study["real_validation"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "HDD":
        raise ValueError("Validation data must be addressed through the user HDD namespace")
    own_hdd = bounded(BASE / "HDD")
    return (own_hdd / Path(*relative.parts[1:])).resolve()


def validation_manifest(study):
    from PIL import Image
    from torchvision.models import ResNet50_Weights

    root = validation_root(study)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) != 1000:
        raise ValueError("Expected the 1,000 sorted ImageNet WNID directories")
    names = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    manifest = []
    for label in study["target_classes"]:
        files = sorted(
            p
            for p in (root / classes[label]).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if len(files) != 50:
            raise ValueError(
                f"Expected 50 held-out validation images for {label}, got {len(files)}"
            )
        images = []
        for path in files:
            with Image.open(path) as image:
                image.verify()
            images.append(dict(path=str(path), sha256=digest(path)))
        manifest.append(dict(class_id=label, wnid=classes[label], name=names[label], images=images))
    return manifest


def groups(study):
    result = [
        dict(
            name="unconditional",
            method=study["method"],
            cfg=study["cfg_scale"],
            overrides=dict(reference_class=None),
        )
    ]
    result += [
        dict(
            name=f"reference_{label}",
            method=study["method"],
            cfg=study["cfg_scale"],
            overrides=dict(reference_class=label),
        )
        for label in study["reference_classes"]
    ]
    if study["include_same_target_control"]:
        result.append(
            dict(
                name="same_target",
                method=study["method"],
                cfg=study["cfg_scale"],
                overrides=dict(reference_class="target"),
            )
        )
    return result


def samples(study, stage):
    setting = study[stage]
    return [
        dict(
            sample_id=ordinal * study["seed_stride"] + index,
            class_id=label,
            seed=study["noise_seed"] + ordinal * study["seed_stride"] + index,
            noise_index=ordinal * study["seed_stride"] + index,
        )
        for ordinal, label in enumerate(study["target_classes"])
        for index in range(
            setting["seed_offset"], setting["seed_offset"] + setting["images_per_class"]
        )
    ]


def generator_config(study):
    base = load_config(CONFIG_DIR / study["base_config"])
    return dict(
        base,
        steps=study["steps"],
        batch_size=study["batch_size"],
        noise=None,
        eval_seed=study["noise_seed"],
        cfg1_conditional_only=False,
        velocity_diagnostics=False,
        external_baselines=[],
        noise_policy="CPU per-sample generators; locked class/seed manifest",
    )


def validate_pairing(root):
    root = Path(root)
    records = json.loads((root / "samples.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    if records != samples(plan["protocol"], plan["mode"]):
        raise ValueError("Sample manifest differs from the locked class/seed protocol")
    if plan["groups"] != groups(plan["protocol"]):
        raise ValueError("Reference groups differ from the locked protocol")
    expected_names = [f"{row['sample_id']:06d}.png" for row in records]
    initial = None
    for group in plan["groups"]:
        folder = root / group["name"]
        if sorted(p.name for p in (folder / "images").glob("*.png")) != sorted(expected_names):
            raise ValueError("Generated sample set mismatch: " + group["name"])
        hashes = json.loads((folder / "noise_hashes.json").read_text())
        if [row["sample_id"] for row in hashes] != [row["sample_id"] for row in records]:
            raise ValueError("Noise hash sample IDs mismatch")
        if initial is not None and hashes != initial:
            raise ValueError("Initial noise differs between reference settings")
        initial = hashes
    return dict(
        paired_initial_noise=True,
        images_per_class=dict(Counter(row["class_id"] for row in records)),
    )
