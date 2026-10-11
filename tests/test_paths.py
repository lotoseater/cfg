"""Portable data roots retain bounds across SSD/HDD mount symlinks."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


class PathTests(unittest.TestCase):
    def invoke(self, root, script):
        env = dict(
            os.environ,
            CFG_PROJECT_ROOT=str(root),
            PYTHONPATH=str(HERE),
            PYTHONDONTWRITEBYTECODE="1",
        )
        return subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, cwd=HERE
        )

    def test_mounts_and_escape(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix=".test-common-") as tmp:
            folder = Path(tmp)
            root = folder / "home"
            root.mkdir()
            for name in ("SSD", "HDD"):
                target = folder / (name + "-volume")
                target.mkdir()
                (root / name).symlink_to(target, target_is_directory=True)
            outside = folder / "outside"
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)
            result = self.invoke(
                root,
                """
from src.paths import BASE, bounded
from src.config import load_config
from src.provenance import assert_upstream
for path in (BASE/'regular/file', BASE/'SSD/projects/cfg', BASE/'HDD/cfg/runs'):
    bounded(path)
for path in (BASE/'escape/file', BASE.parent/'outside/file'):
    try:
        bounded(path)
    except ValueError:
        pass
    else:
        raise AssertionError('Boundary escape accepted')
config=load_config('configs/velocity100.json')
assert_upstream(config)
""",
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_relative_root_rejected(self):
        result = self.invoke("relative-root", "import src.paths")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be an absolute directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
