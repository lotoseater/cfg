"""Frozen upstream checks and run source provenance."""

import subprocess

from .io import digest
from .paths import PROJECT_ROOT, UPSTREAM_DIR


def code_hashes():
    sources = []
    for directory in ("src", "scripts", "configs", "tests"):
        sources.extend((PROJECT_ROOT / directory).rglob("*"))
    return {
        str(path.relative_to(PROJECT_ROOT)): digest(path)
        for path in sorted(sources)
        if path.is_file() and path.suffix in (".py", ".json") and not path.name.startswith(".")
    }


def git_revision():
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def assert_upstream(config):
    for relative, expected in config["upstream_sha256"].items():
        if digest(UPSTREAM_DIR / relative) != expected:
            raise RuntimeError(f"Upstream snapshot changed: {relative}")
