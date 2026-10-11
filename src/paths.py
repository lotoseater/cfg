"""Server data boundaries and per-worker runtime environment."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
UPSTREAM_DIR = PROJECT_ROOT / "src/upstream"
_root = Path(os.environ.get("CFG_PROJECT_ROOT", "/mnt/data/LONG/hongxu/zjiaak")).expanduser()
if not _root.is_absolute():
    raise ValueError("CFG_PROJECT_ROOT must be an absolute directory")
BASE = _root.resolve()
BOUNDARIES = (BASE, (BASE / "SSD").resolve(), (BASE / "HDD").resolve())


def bounded(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) for root in BOUNDARIES):
        raise ValueError(f"Path outside project boundary: {path}")
    return path


def environment(gpu):
    env = os.environ.copy()
    cache = BASE / "HDD/cfg/cache"
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        TORCH_COMPILE_DISABLE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(cache),
        XDG_CACHE_HOME=str(cache),
        MPLCONFIGDIR=str(cache / "matplotlib"),
    )
    env.pop("FDS_GD_EAGER", None)
    env.pop("FDS_LOCAL_METRIC", None)
    return env
