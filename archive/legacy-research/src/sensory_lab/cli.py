import argparse
from collections import defaultdict
from statistics import mean
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
    parser.add_argument("--velocities", default="1,2,3",
                        help="discrete pixels per visual frame, for example 1,2,3")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    velocities = tuple(int(value) for value in args.velocities.split(","))
    rows = run(BenchmarkConfig(args.train, args.test, args.seed, args.hidden, args.seeds,
                               args.observe_through, args.predict_at, velocities), output=args.out)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['task'], row['case'], row['modality'])].append(row)
    for (task, case, modality), group in grouped.items():
        print(f"{task} {case} {modality}: "
              f"within-1px={mean(r['within_1_pixel_accuracy'] for r in group):.3f}, "
              f"MAE={mean(r['mean_absolute_error_pixels'] for r in group):.2f}px, "
              f"active updates={mean(r['active_updates'] for r in group):.0f}, "
              f"median step={mean(r['median_step_latency_us'] for r in group):.1f} us")
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
