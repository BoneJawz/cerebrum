"""Multi-seed benchmark for the expanded static spatial A-I viewer task."""
from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
import statistics
import time

import numpy as np

from .spatial_viewer import MODEL_CATALOG, SpatialViewerSession


SEED_T = {5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def _interval(values):
    values = [float(v) for v in values]
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    critical = SEED_T.get(len(values), 1.96 if len(values) > 30 else 2.0)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def run_benchmark(output: str | Path, seeds=(42, 43, 44, 45, 46),
                  neural_epochs=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict] = []
    all_model_ids = [item["id"] for item in MODEL_CATALOG]
    fit_rows = []

    for seed in seeds:
        session = SpatialViewerSession(seed=seed, train_repetitions=1,
                                       test_per_position=1,
                                       neural_epochs=neural_epochs)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        fit_result = session.fit()
        fit_wall = time.perf_counter() - wall_start
        fit_cpu = time.process_time() - cpu_start
        fit_by_model = {row["id"]: row["learning_seconds"]
                        for row in fit_result["models"]}

        observations = {model_id: {"errors": [], "exact": 0, "within_one": 0,
                                   "latency_ms": [], "cpu_ms": []}
                        for model_id in all_model_ids}
        test_wall_start = time.perf_counter()
        test_cpu_start = time.process_time()
        for test_index, item in enumerate(session.test):
            truth = item["position"]
            for row in session.predict(test_index):
                record = observations[row["id"]]
                error = abs(float(row["predicted_position"]) - truth)
                record["errors"].append(error)
                record["exact"] += int(error <= 1e-9)
                record["within_one"] += int(error <= 1.0)
                record["latency_ms"].append(float(row["latency_ms"]))
                record["cpu_ms"].append(float(row["cpu_ms"]))
        test_wall = time.perf_counter() - test_wall_start
        test_cpu = time.process_time() - test_cpu_start

        for model_id in all_model_ids:
            record = observations[model_id]
            errors = np.asarray(record["errors"], dtype=np.float64)
            seed_rows.append({
                "seed": seed, "model": model_id, "train_positions": len(session.train),
                "test_positions": len(session.test), "exact_accuracy": record["exact"] / len(errors),
                "within_one_accuracy": record["within_one"] / len(errors),
                "mae_positions": float(np.mean(errors)),
                "median_absolute_error": float(np.median(errors)),
                "p95_absolute_error": float(np.percentile(errors, 95)),
                "fit_seconds_model": float(fit_by_model[model_id]),
                "fit_seconds_all_models": fit_wall,
                "fit_cpu_seconds_all_models": fit_cpu,
                "mean_prediction_latency_ms": float(np.mean(record["latency_ms"])),
                "p95_prediction_latency_ms": float(np.percentile(record["latency_ms"], 95)),
                "mean_prediction_cpu_ms": float(np.mean(record["cpu_ms"])),
                "test_seconds_all_models": test_wall,
                "test_cpu_seconds_all_models": test_cpu,
            })
        del session
        gc.collect()
        print(f"seed {seed}: fit {fit_wall:.1f}s wall/{fit_cpu:.1f}s CPU; "
              f"all-model evaluation {test_wall:.1f}s wall/{test_cpu:.1f}s CPU", flush=True)

    seed_csv = output / "seed_results.csv"
    with seed_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary = []
    for model_id in all_model_ids:
        rows = [row for row in seed_rows if row["model"] == model_id]
        summary_row = {"model": model_id}
        for metric in ("exact_accuracy", "within_one_accuracy", "mae_positions",
                       "median_absolute_error", "p95_absolute_error",
                       "fit_seconds_model", "mean_prediction_latency_ms",
                       "p95_prediction_latency_ms", "mean_prediction_cpu_ms"):
            mean, low, high = _interval([row[metric] for row in rows])
            if metric in {"exact_accuracy", "within_one_accuracy"}:
                low, high = max(0.0, low), min(1.0, high)
            summary_row[f"{metric}_mean"] = mean
            summary_row[f"{metric}_ci95_low"] = low
            summary_row[f"{metric}_ci95_high"] = high
        summary.append(summary_row)
    summary_csv = output / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    baseline_mae = float(np.mean(np.abs(np.arange(-99, 100, dtype=float))))
    per_seed_costs = [row for row in seed_rows if row["model"] == "A"]
    mean_fit_wall = statistics.mean(row["fit_seconds_all_models"] for row in per_seed_costs)
    mean_fit_cpu = statistics.mean(row["fit_cpu_seconds_all_models"] for row in per_seed_costs)
    mean_test_wall = statistics.mean(row["test_seconds_all_models"] for row in per_seed_costs)
    mean_test_cpu = statistics.mean(row["test_cpu_seconds_all_models"] for row in per_seed_costs)
    lines = [
        "# Expanded Static Spatial A-I Benchmark",
        "",
        "## Protocol",
        "",
        f"This run evaluates {len(seeds)} deterministic seeds ({', '.join(map(str, seeds))}). "
        "Each seed trains on one paired audiovisual example for every integer position from -99 through +99 "
        "and tests all 199 positions once with the visual input blanked. The sign/digit waveform codebook "
        "is randomized by seed and shared between that seed's training and test data. Model initialization, "
        "test ordering, and digit-token codebook are therefore seed-controlled.",
        "",
        f"The browser-demo training profile is used: {neural_epochs} Adam epochs for FF/F/G/H and a "
        "24-unit liquid for I variants. A-D use fixed 64-unit processors and ridge readouts. Each architecture "
        "uses its current implementation and readout procedure; model-specific tuning and parameter matching "
        "were not performed.",
        "",
        "## Metrics",
        "",
        "Exact accuracy (absolute error <= 1e-9), within-one-position accuracy, MAE, median and p95 absolute "
        "error are computed over 199 test positions per seed. Fit time is reported by model; fit and test CPU/wall "
        "times are available per seed in `seed_results.csv`. Per-prediction wall latency and process CPU time are "
        "reported as mean and p95 across test positions. The summary gives mean and two-sided 95% t intervals "
        "across seeds. Timing is specific to the machine and includes Python/NumPy implementation overhead.",
        "",
        f"Across seeds, fitting all 13 predictors took a mean {mean_fit_wall:.1f} s wall time "
        f"({mean_fit_cpu:.1f} s process CPU time). Scoring all 199 test positions with all 13 predictors "
        f"took a mean {mean_test_wall:.1f} s wall time ({mean_test_cpu:.1f} s process CPU time). "
        "See the seed-level file for the per-run range.",
        "",
        "A constant zero-position prediction has MAE " + f"{baseline_mae:.2f}" +
        " positions, 0.50% exact accuracy, and 1.51% within-one accuracy under the uniform test distribution.",
        "",
        "## Results",
        "",
        "| Model | Exact | Within 1 | MAE | Median error | P95 error | Fit s | Mean latency ms | P95 latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['exact_accuracy_mean']:.3f} "
            f"[{row['exact_accuracy_ci95_low']:.3f}, {row['exact_accuracy_ci95_high']:.3f}] | "
            f"{row['within_one_accuracy_mean']:.3f} "
            f"[{row['within_one_accuracy_ci95_low']:.3f}, {row['within_one_accuracy_ci95_high']:.3f}] | "
            f"{row['mae_positions_mean']:.2f} "
            f"[{row['mae_positions_ci95_low']:.2f}, {row['mae_positions_ci95_high']:.2f}] | "
            f"{row['median_absolute_error_mean']:.2f} | {row['p95_absolute_error_mean']:.2f} | "
            f"{row['fit_seconds_model_mean']:.2f} | {row['mean_prediction_latency_ms_mean']:.3f} | "
            f"{row['p95_prediction_latency_ms_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        "This is a reproducible expanded-range check of the A-I viewer models. It tests whether each fitted "
        "system maps an audio instruction to one of the positions after paired audiovisual training. It does "
        "not establish a need for multimodal integration: the acoustic instruction itself specifies the complete "
        "position, every target position appears in training, and the same seed-specific token waveforms recur "
        "at test. It also does not test compositional generalization to unseen positions, noisy/new waveform "
        "realizations, a changed codebook at test, or few-shot learning.",
        "",
        "The five seeds vary token assignment, model initialization, and test order, but share the same 199 "
        "underlying position labels and full-coverage train/test protocol. Confidence intervals describe this "
        "seed variation; they are not population-level uncertainty over task designs. The 24-unit liquid and "
        "four-epoch neural settings are the CPU-oriented viewer profile, so timing and scores must not be merged "
        "with the prior spatial-grounding benchmark. No memory/RSS comparison is claimed.",
        "",
        "## Files",
        "",
        "- `seed_results.csv`: one row per seed and model, including quality and cost measurements.",
        "- `summary.csv`: seed-averaged metrics and 95% t intervals.",
        "",
    ])
    (output / "BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return seed_csv, summary_csv, output / "BENCHMARK_REPORT.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/spatial-range-benchmark")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--neural-epochs", type=int, default=4)
    args = parser.parse_args(argv)
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("provide at least two unique seeds")
    if args.neural_epochs < 1:
        parser.error("--neural-epochs must be positive")
    paths = run_benchmark(args.output, seeds, args.neural_epochs)
    print("Wrote:", *(str(path) for path in paths))


if __name__ == "__main__":
    main()
