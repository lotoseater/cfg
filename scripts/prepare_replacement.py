"""Validate shared HDD assets and cache held-out real features without generation."""

import argparse
from datetime import datetime
from pathlib import Path

from src.io import write_json
from src.paths import BASE, bounded
from src.provenance import code_hashes
from src.replacement.protocol import (
    DEFAULT_CONFIG,
    load_protocol,
    validation_manifest,
    verify_assets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = load_protocol(args.config)
    verify_assets(study)
    manifest = validation_manifest(study)
    output = bounded(
        args.output
        or BASE / "HDD/cfg/reference" / datetime.now().strftime("replacement_%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", study)
    write_json(output / "real_manifest.json", manifest)
    write_json(output / "source_sha256.json", code_hashes())
    (output / "status.txt").write_text("PREPARING\n")
    try:
        import numpy as np
        import torch

        from src.replacement.features import Evaluators

        torch.set_num_threads(4)
        evaluator = Evaluators(study, args.device)
        write_json(output / "evaluator.json", evaluator.metadata(study))
        for target in manifest:
            paths = [row["path"] for row in target["images"]]
            features, probabilities = evaluator.extract(paths, study["feature_batch_size"])
            np.savez(
                output / f"class_{target['class_id']}.npz",
                features=features,
                probabilities=probabilities,
            )
            print("REAL_FEATURES", target["class_id"], target["name"], len(paths), flush=True)
        (output / "status.txt").write_text("READY\n")
        print("REFERENCE_READY", output, flush=True)
    except BaseException:
        (output / "status.txt").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
