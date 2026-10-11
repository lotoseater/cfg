#!/usr/bin/env bash
set -euo pipefail
export CFG_PROJECT_ROOT=/home/zjiaak
export PYTHONDONTWRITEBYTECODE=1
cfg_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$cfg_source_dir"
exec /home/zjiaak/SSD/miniconda3/envs/fds/bin/python run.py "$@"
