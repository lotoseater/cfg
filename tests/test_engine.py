"""Verify module-based worker dispatch and paired output validation on CPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src import engine
from src.config import load_config
from src.paths import CONFIG_DIR, PROJECT_ROOT


class EngineTests(unittest.TestCase):
    def test_smoke_dispatches_modules_and_validates_pairing(self):
        config = load_config(CONFIG_DIR / "locpu4.json")
        config["groups"] = config["groups"][:2]
        config["velocity_diagnostics"] = False
        asset_hashes = {
            config[key]: config[key + "_sha256"]
            for key in ("checkpoint", "noise", "fid_stats", "inception_weights")
        }
        calls = []

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            base = Path(directory)

            def fake_worker(command, **kwargs):
                self.assertEqual(command[1:3], ["-m", "scripts.generate"])
                self.assertEqual(kwargs["cwd"], PROJECT_ROOT)
                root = Path(command[command.index("--root") + 1])
                name = command[command.index("--group") + 1]
                folder = root / name
                (folder / "images").mkdir(parents=True)
                samples = json.loads((root / "samples.json").read_text())
                hashes = []
                for sample in samples:
                    sample_id = sample["sample_id"]
                    Image.new("RGB", (256, 256)).save(folder / "images" / f"{sample_id:06d}.png")
                    hashes.append(dict(sample_id=sample_id, sha256=f"fixture-{sample_id}"))
                (folder / "noise_hashes.json").write_text(json.dumps(hashes))
                (folder / "generation.json").write_text(json.dumps(dict(steps=100)))
                calls.append((name, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                return SimpleNamespace(returncode=0)

            args = SimpleNamespace(
                mode="smoke",
                config=CONFIG_DIR / "locpu4.json",
                smoke=None,
                smoke_images=4,
                smoke_batch_size=4,
                gpus="2,5",
            )
            with (
                patch.object(engine, "BASE", base),
                patch.object(engine, "load_config", return_value=config),
                patch.object(engine, "assert_upstream"),
                patch.object(
                    engine,
                    "digest",
                    side_effect=lambda path: asset_hashes[str(path.relative_to(base))],
                ),
                patch.object(engine, "code_hashes", return_value={}),
                patch.object(engine.subprocess, "run", side_effect=fake_worker),
            ):
                engine.launch(args)

            self.assertEqual(
                set(calls), {(config["groups"][0]["name"], "2"), (config["groups"][1]["name"], "5")}
            )
            run_root = next((base / "HDD/cfg/smoketest").iterdir())
            self.assertEqual((run_root / "status.txt").read_text().strip(), "SMOKE_COMPLETE")
            summary = json.loads((run_root / "summary.json").read_text())
            self.assertTrue(summary["identical_initial_noise"])
            self.assertEqual(summary["image_count_per_group"], 4)
