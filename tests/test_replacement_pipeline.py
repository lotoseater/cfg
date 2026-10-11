"""Exercise classwise evaluation, paired manifests and blind review with frozen fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import evaluate_replacement
from src.paths import PROJECT_ROOT
from src.replacement.protocol import groups, load_protocol, samples


class EvaluationPipelineTests(unittest.TestCase):
    def test_paired_evaluation_and_blind_output(self):
        study = load_protocol()
        study["target_classes"] = study["target_classes"][:2]
        study["pilot"]["images_per_class"] = 2
        study["bootstrap_replicates"] = 32
        study["permutation_replicates"] = 64
        records = samples(study, "pilot")
        labels = {row["sample_id"]: row["class_id"] for row in records}
        selected = groups(study)
        metadata = {"fixture": True}
        rng = np.random.default_rng(92)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory) / "run"
            reference = Path(directory) / "reference"
            root.mkdir()
            reference.mkdir()
            (root / "plan.json").write_text(
                json.dumps(dict(mode="pilot", protocol=study, groups=selected))
            )
            (root / "samples.json").write_text(json.dumps(records))
            (root / "status.txt").write_text("GENERATED\n")
            manifest = []
            for target in study["target_classes"]:
                probabilities = np.zeros((50, 1000))
                probabilities[:, target] = 1
                np.savez(
                    reference / f"class_{target}.npz",
                    features=rng.normal(size=(50, 4)),
                    probabilities=probabilities,
                )
                image_paths = []
                for index in range(3):
                    path = reference / f"real_{target}_{index}.png"
                    Image.new("RGB", (256, 256)).save(path)
                    image_paths.append(dict(path=str(path)))
                manifest.append(dict(class_id=target, name=f"class {target}", images=image_paths))
            (reference / "protocol.json").write_text(json.dumps(study))
            (reference / "real_manifest.json").write_text(json.dumps(manifest))
            (reference / "evaluator.json").write_text(json.dumps(metadata))
            (reference / "status.txt").write_text("READY\n")
            for group in selected:
                folder = root / group["name"]
                (folder / "images").mkdir(parents=True)
                for row in records:
                    Image.new("RGB", (256, 256)).save(
                        folder / "images" / f"{row['sample_id']:06d}.png"
                    )
                (folder / "noise_hashes.json").write_text(
                    json.dumps(
                        [
                            dict(sample_id=row["sample_id"], sha256=f"fixture-{row['sample_id']}")
                            for row in records
                        ]
                    )
                )

            def extract(paths, batch_size):
                ids = [int(path.stem) for path in paths]
                features = np.asarray(
                    [
                        [1.0, 2.0, identifier % 2 + 1.0, labels[identifier] / 1000]
                        for identifier in ids
                    ]
                )
                probabilities = np.zeros((len(ids), 1000))
                for index, identifier in enumerate(ids):
                    probabilities[index, labels[identifier]] = 1
                return features, probabilities

            with (
                patch(
                    "sys.argv",
                    ["evaluate_replacement", "--root", str(root), "--reference", str(reference)],
                ),
                patch.object(evaluate_replacement, "verify_assets"),
                patch("src.replacement.features.Evaluators") as evaluator,
            ):
                evaluator.return_value.metadata.return_value = metadata
                evaluator.return_value.extract.side_effect = extract
                evaluate_replacement.main()
            output = root / "classwise"
            result = json.loads((output / "classwise_results.json").read_text())
            self.assertEqual(result["holm_family_size"], 16)
            self.assertEqual(len(result["results"]), 10)
            self.assertEqual((output / "status.txt").read_text().strip(), "COMPLETE")
            for row in result["results"]:
                self.assertFalse(row["confirmatory"])
                self.assertEqual(row["metrics"]["accuracy"]["delta"], 0)
                self.assertAlmostEqual(row["metrics"]["dino_kid"]["delta"], 0)
            review = (output / "blind/index.html").read_text()
            self.assertIn("class 1", review)
            self.assertNotIn("reference_548", review)
            self.assertFalse((output / "blind/blind_key_private.json").exists())
            self.assertTrue((output / "blind_key_private.json").exists())
