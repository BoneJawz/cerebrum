"""CLI for compositional failure diagnostics; limit numerical-library threads."""
import argparse
import os

from .advanced_cli import _ints, _pairs


def _floats(value):
    return tuple(float(x) for x in value.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose factor recovery, compositional prediction, readouts, and event costs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-sizes", default="10,25,50,100,250,500")
    parser.add_argument("--generalization-train", type=int, default=500)
    parser.add_argument("--validation-per-combo", type=int, default=25)
    parser.add_argument("--test-per-combo", type=int, default=40)
    parser.add_argument("--heldout", default="-1:2,1:3")
    parser.add_argument("--velocities", default="1,2,3")
    parser.add_argument("--horizons", default="3,6,9,12")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--timed-repeats", type=int, default=5)
    parser.add_argument("--event-thresholds", default="0,0.04,0.1,0.25,0.5,1.0")
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--mlp-max-epochs", type=int, default=250)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--outdir", default="results/diagnostics")
    args = parser.parse_args()

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    from .diagnostics import DiagnosticConfig, run_diagnostics
    from .research_benchmark import ResearchConfig

    research = ResearchConfig(
        seed=args.seed, seeds=args.seeds, train_sizes=_ints(args.train_sizes),
        generalization_train_episodes=args.generalization_train,
        validation_per_combo=args.validation_per_combo,
        test_per_combo=args.test_per_combo,
        heldout_combinations=_pairs(args.heldout),
        velocities=_ints(args.velocities), horizons=_ints(args.horizons),
        hidden=args.hidden, timed_repeats=args.timed_repeats,
    )
    diagnostic = DiagnosticConfig(
        research=research, event_thresholds=_floats(args.event_thresholds),
        mlp_hidden=args.mlp_hidden, mlp_max_epochs=args.mlp_max_epochs,
    )
    result = run_diagnostics(diagnostic, args.outdir)
    print(f"Wrote compositional diagnostics to {result['output_dir']}")


if __name__ == "__main__":
    main()
