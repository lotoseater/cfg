"""Extract independent frozen features, then perform paired classwise inference."""

import argparse
import json
from pathlib import Path

from src.io import digest, write_json
from src.paths import bounded
from src.provenance import code_hashes
from src.replacement.protocol import validate_pairing, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root, reference = bounded(args.root), bounded(args.reference)
    if (root / "status.txt").read_text().strip() != "GENERATED":
        parser.error("Generation must be complete")
    if (reference / "status.txt").read_text().strip() != "READY":
        parser.error("Real-reference cache is not ready")
    plan = json.loads((root / "plan.json").read_text())
    study = plan["protocol"]
    if study != json.loads((reference / "protocol.json").read_text()):
        parser.error("Real cache protocol differs from generation protocol")
    verify_assets(study, include_generator=False)
    validate_pairing(root)
    records = json.loads((root / "samples.json").read_text())
    output = bounded(args.output or root / "classwise")
    output.mkdir(parents=True, exist_ok=False)
    (output / "status.txt").write_text("EVALUATING\n")
    try:
        import numpy as np
        import torch

        from src.replacement.analysis import analyze
        from src.replacement.features import Evaluators

        torch.set_num_threads(4)
        evaluator = Evaluators(study, args.device)
        metadata = evaluator.metadata(study)
        if metadata != json.loads((reference / "evaluator.json").read_text()):
            raise ValueError(
                "Real/fake extraction metadata differ; rebuild real cache with the same device/environment"
            )
        write_json(
            output / "manifest.json",
            dict(
                evaluator=metadata,
                source_sha256=code_hashes(),
                real_manifest_sha256=digest(reference / "real_manifest.json"),
            ),
        )
        for group in plan["groups"]:
            paths = [
                root / group["name"] / "images" / f"{row['sample_id']:06d}.png" for row in records
            ]
            features, probabilities = evaluator.extract(paths, study["feature_batch_size"])
            np.savez(
                output / f"{group['name']}.npz",
                features=features,
                probabilities=probabilities,
                sample_ids=[row["sample_id"] for row in records],
            )
            print("FEATURES_COMPLETE", group["name"], len(paths), flush=True)
        analyze(root, output, reference)
        (output / "status.txt").write_text("COMPLETE\n")
        print("CLASSWISE_COMPLETE", output, flush=True)
    except BaseException:
        (output / "status.txt").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
