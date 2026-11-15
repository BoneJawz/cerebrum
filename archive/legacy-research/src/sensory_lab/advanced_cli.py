"""CLI entry point: set BLAS thread limits before NumPy is imported."""
import argparse
import os


def _ints(value):
    return tuple(int(x) for x in value.split(",") if x.strip())


def _pairs(value):
    result = []
    for part in value.split(","):
        direction, velocity = part.split(":", 1)
        result.append((int(direction), int(velocity)))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description="Run compositional generalization and efficiency experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-sizes", default="10,25,50,100,250,500")
    parser.add_argument("--generalization-train", type=int, default=500)
    parser.add_argument("--validation-per-combo", type=int, default=25)
    parser.add_argument("--test-per-combo", type=int, default=40)
    parser.add_argument("--heldout", default="-1:2,1:3",
                        help="held-out direction:velocity pairs")
    parser.add_argument("--velocities", default="1,2,3")
    parser.add_argument("--horizons", default="3,6,9,12")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--timed-repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    # These must be set before NumPy/BLAS loads to make CPU comparisons repeatable.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    from .research_benchmark import ResearchConfig, run_all
    config = ResearchConfig(
        seed=args.seed,
        seeds=args.seeds,
        train_sizes=_ints(args.train_sizes),
        generalization_train_episodes=args.generalization_train,
        validation_per_combo=args.validation_per_combo,
        test_per_combo=args.test_per_combo,
        heldout_combinations=_pairs(args.heldout),
        horizons=_ints(args.horizons),
        velocities=_ints(args.velocities),
        hidden=args.hidden,
        timed_repeats=args.timed_repeats,
    )
    output = run_all(config, args.outdir)
    print(f"Wrote compositional results, sample-efficiency curves, compute metrics, plots, and report to {output['output_dir']}")


if __name__ == "__main__":
    main()
