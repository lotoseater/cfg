"""Run one experiment group in its assigned worker process."""

import argparse
from pathlib import Path

from src.generation import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    generate(args.root, args.group)


if __name__ == "__main__":
    main()
