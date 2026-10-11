"""Render plots and report from saved velocity summaries."""

import argparse
from pathlib import Path

from src.analysis import create_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Run directory or velocity_curves.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_report(args.input, args.output)
    print(f"Analysis saved to {args.output}")


if __name__ == "__main__":
    main()
