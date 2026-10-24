import argparse
from .benchmark import run, BenchmarkConfig

def main():
    parser = argparse.ArgumentParser(description="Run the cross-modal 1D vision + waveform benchmark")
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--observe-through", type=int, default=4)
    parser.add_argument("--predict-at", type=int, default=9)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    rows = run(BenchmarkConfig(args.train, args.test, args.seed, args.hidden, args.seeds,
                               args.observe_through, args.predict_at), output=args.out)
    for row in rows:
        print(f"seed={row['seed']} {row['case']} {row['modality']}: "
              f"within-1px={row['within_1_pixel_accuracy']:.3f}, "
              f"MAE={row['mean_absolute_error_pixels']:.2f}px, "
              f"active updates={row['active_updates']}, "
              f"median step={row['median_step_latency_us']:.1f} us, "
              f"CPU={row['cpu_seconds_train_and_test']:.3f} s, "
              f"peak traced={row['peak_traced_bytes']} B")
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
