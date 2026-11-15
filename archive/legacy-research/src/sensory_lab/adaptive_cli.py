"""CLI for the trainable-dynamics follow-up experiment."""
import argparse
import os


def main():
    parser=argparse.ArgumentParser(description="Train and evaluate adaptive recurrent dynamics F-H")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--seeds",type=int,default=5)
    parser.add_argument("--train-sizes",default="10,25,50,100,250,500")
    parser.add_argument("--generalization-train",type=int,default=500)
    parser.add_argument("--validation-per-combo",type=int,default=25)
    parser.add_argument("--test-per-combo",type=int,default=40)
    parser.add_argument("--timed-repeats",type=int,default=5)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--outdir",default="results/adaptive-dynamics")
    args=parser.parse_args()
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]=str(args.threads)
    from .research_benchmark import ResearchConfig
    from .adaptive_dynamics import run_adaptive
    config=ResearchConfig(seed=args.seed,seeds=args.seeds,
        train_sizes=tuple(int(v) for v in args.train_sizes.split(",") if v.strip()),
        generalization_train_episodes=args.generalization_train,
        validation_per_combo=args.validation_per_combo,test_per_combo=args.test_per_combo,
        timed_repeats=args.timed_repeats)
    output=run_adaptive(config,args.outdir)
    print(f"Wrote adaptive dynamics results, plots, report, and checkpoints to {output}")


if __name__=="__main__":
    main()
