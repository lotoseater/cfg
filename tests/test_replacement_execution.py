"""Verify multi-GPU batch partitioning, merged pairing and real-cache compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.io import write_json
from src.paths import PROJECT_ROOT
from src.replacement.execution import merge_workers, partition_samples
from src.replacement.protocol import (
    generator_config,
    groups,
    load_protocol,
    reference_signature,
    samples,
    validate_pairing,
)


class ReplacementExecutionTests(unittest.TestCase):
    def test_eight_gpu_partition_preserves_original_batches(self):
        study = load_protocol()
        for stage, workers in (("pilot", 7), ("pilot", 8), ("formal", 7), ("formal", 8)):
            records = samples(study, stage)
            parts = partition_samples(records, 128, workers)
            self.assertEqual(len(parts), workers)
            flattened = sorted((r for part in parts for r in part), key=lambda r: r["sample_id"])
            self.assertEqual(flattened, records)
            for lane, part in enumerate(parts):
                blocks = [part[start : start + 128] for start in range(0, len(part), 128)]
                expected = [
                    records[start : start + 128]
                    for index, start in enumerate(range(0, len(records), 128))
                    if index % workers == lane
                ]
                self.assertEqual(blocks, expected)
        self.assertEqual(len(partition_samples(records[:2], 128, 8)), 1)

    def test_generation_batch_change_reuses_real_features_only(self):
        study = load_protocol()
        old = dict(study, batch_size=512)
        self.assertEqual(reference_signature(study), reference_signature(old))
        for key in ("dino_sha256", "target_classes", "feature_batch_size"):
            changed = dict(study, **{key: "changed"})
            self.assertNotEqual(reference_signature(study), reference_signature(changed))

    def test_merge_eight_workers_produces_complete_paired_arms(self):
        study = load_protocol()
        study["pilot"]["images_per_class"] = 1
        study["batch_size"] = 1
        records = samples(study, "pilot")
        plan = dict(
            mode="pilot",
            protocol=study,
            config=generator_config(study),
            generation_batch_size=1,
            groups=groups(study),
            code_sha256={"fixture": "fixed"},
        )
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            write_json(root / "plan.json", plan)
            write_json(root / "samples.json", records)
            shards = []
            for index, part in enumerate(partition_samples(records, 1, 8)):
                shard = root / "shards" / str(index)
                shard.mkdir(parents=True)
                shards.append(shard)
                write_json(shard / "samples.json", part)
                for group in plan["groups"]:
                    folder = shard / group["name"]
                    (folder / "images").mkdir(parents=True)
                    for row in part:
                        Image.new("RGB", (8, 8)).save(
                            folder / "images" / f"{row['sample_id']:06d}.png"
                        )
                    write_json(
                        folder / "noise_hashes.json",
                        [dict(sample_id=row["sample_id"], sha256=str(row["seed"])) for row in part],
                    )
                    write_json(
                        folder / "config.json",
                        dict(
                            common=dict(plan["config"], **group["overrides"]),
                            group=group,
                            batch_size=1,
                            num_images=len(part),
                        ),
                    )
                    write_json(
                        folder / "manifest.json", dict(code_sha256=plan["code_sha256"], started=0)
                    )
                    write_json(folder / "network_audit.json", dict(forward_calls=[{}] * 99))
                    write_json(
                        folder / "generation.json",
                        dict(
                            nfe_per_sample=99,
                            batch_network_calls=99,
                            sampling_seconds=1,
                            max_allocated_gib=1,
                            max_reserved_gib=2,
                            validated_png_count=len(part),
                        ),
                    )
                    (folder / "status.txt").write_text("GENERATED\n")
            merge_workers(root, shards, plan, records)
            self.assertTrue(validate_pairing(root)["paired_initial_noise"])
            for group in plan["groups"]:
                folder = root / group["name"]
                audit = json.loads((folder / "network_audit.json").read_text())
                self.assertEqual(len(audit["forward_calls"]), 8 * 99)
                self.assertEqual(len(list((folder / "images").glob("*.png"))), 8)
            path = root / "reference_548/noise_hashes.json"
            hashes = json.loads(path.read_text())
            hashes[0]["sha256"] = "changed"
            path.write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, "Initial noise differs"):
                validate_pairing(root)
