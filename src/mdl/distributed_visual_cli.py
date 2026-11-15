from __future__ import annotations

import argparse

from .distributed_visual_experiment import run


def main():
    parser = argparse.ArgumentParser(description="Run distributed 5x5 visual association controls")
    parser.add_argument("--outdir", default="results/distributed_visual_association")
    parser.add_argument("--development-only", action="store_true", help="omit prospective seeds 101-105")
    parser.add_argument("--validation-only", action="store_true", help="run only locked prospective seeds 101-105")
    args = parser.parse_args()
    if args.validation_only:
        run(args.outdir, seeds=(), include_validation=True, validation_only=True)
    else:
        run(args.outdir, include_validation=not args.development_only)


if __name__ == "__main__":
    main()
