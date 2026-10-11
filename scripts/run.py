"""Launch smoke/formal experiments or validate a completed run."""

import argparse
import json
from pathlib import Path

from src.engine import launch
from src.paths import CONFIG_DIR, bounded
from src.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "formal", "validate"))
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "baseline50.json")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--smoke-images", type=int, default=16)
    parser.add_argument("--smoke-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "validate":
        if args.root is None:
            parser.error("validate requires --root")
        print(json.dumps(validate(bounded(args.root)), indent=2))
    else:
        launch(args)


if __name__ == "__main__":
    main()
