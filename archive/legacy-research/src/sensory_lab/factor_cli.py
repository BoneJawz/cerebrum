"""Command line entry point for the independent acoustic factor benchmark."""
from __future__ import annotations

import argparse

from .factor_experiment import DEFAULT_SIZES, FactorConfig, run_factor_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run independent acoustic cue and factor-learning experiments")
    parser.add_argument("--output", default="results/factor-learning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--test-per-combo", type=int, default=40)
    parser.add_argument("--validation-per-combo", type=int, default=25)
    parser.add_argument("--epochs-small", type=int, default=60)
    parser.add_argument("--epochs-large", type=int, default=80)
    args = parser.parse_args(argv)
    config = FactorConfig(seed=args.seed, seeds=args.seeds,
        train_sizes=tuple(args.sizes), test_per_combo=args.test_per_combo,
        validation_per_combo=args.validation_per_combo,
        epochs_small=args.epochs_small, epochs_large=args.epochs_large)
    output = run_factor_experiment(config, args.output)
    print(f"Factor-learning experiment written to {output}")


if __name__ == "__main__":
    main()
