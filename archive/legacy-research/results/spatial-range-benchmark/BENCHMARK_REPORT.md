# Expanded Static Spatial A-I Benchmark

## Protocol

This run evaluates 5 deterministic seeds (42, 43, 44, 45, 46). Each seed trains on one paired audiovisual example for every integer position from -99 through +99 and tests all 199 positions once with the visual input blanked. The sign/digit waveform codebook is randomized by seed and shared between that seed's training and test data. Model initialization, test ordering, and digit-token codebook are therefore seed-controlled.

The browser-demo training profile is used: 4 Adam epochs for FF/F/G/H and a 24-unit liquid for I variants. A-D use fixed 64-unit processors and ridge readouts. Each architecture uses its current implementation and readout procedure; model-specific tuning and parameter matching were not performed.

## Metrics

Exact accuracy (absolute error <= 1e-9), within-one-position accuracy, MAE, median and p95 absolute error are computed over 199 test positions per seed. Fit time is reported by model; fit and test CPU/wall times are available per seed in `seed_results.csv`. Per-prediction wall latency and process CPU time are reported as mean and p95 across test positions. The summary gives mean and two-sided 95% t intervals across seeds. Timing is specific to the machine and includes Python/NumPy implementation overhead.

Across seeds, fitting all 13 predictors took a mean 89.1 s wall time (89.5 s process CPU time). Scoring all 199 test positions with all 13 predictors took a mean 59.4 s wall time (59.4 s process CPU time). See the seed-level file for the per-run range.

A constant zero-position prediction has MAE 49.75 positions, 0.50% exact accuracy, and 1.51% within-one accuracy under the uniform test distribution.

## Results

| Model | Exact | Within 1 | MAE | Median error | P95 error | Fit s | Mean latency ms | P95 latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.005 [0.005, 0.005] | 0.010 [0.010, 0.010] | 49.75 [49.75, 49.75] | 50.00 | 94.10 | 0.35 | 1.855 | 2.017 |
| B | 0.000 [0.000, 0.000] | 0.010 [0.010, 0.010] | 49.75 [49.75, 49.75] | 49.55 | 94.13 | 0.54 | 3.148 | 3.407 |
| C | 0.000 [0.000, 0.000] | 0.010 [0.006, 0.014] | 49.76 [49.73, 49.79] | 49.56 | 94.09 | 0.71 | 3.644 | 3.956 |
| D | 0.000 [0.000, 0.000] | 0.011 [0.006, 0.016] | 49.79 [49.74, 49.85] | 49.41 | 94.02 | 0.55 | 2.777 | 3.037 |
| E1 | 0.000 [0.000, 0.000] | 0.022 [0.015, 0.029] | 25.10 [24.78, 25.42] | 24.81 | 49.56 | 0.20 | 1.080 | 1.210 |
| FF_direct | 0.000 [0.000, 0.000] | 0.026 [0.016, 0.036] | 27.17 [19.18, 35.16] | 23.15 | 64.32 | 0.04 | 1.303 | 1.427 |
| F | 0.000 [0.000, 0.000] | 0.010 [0.002, 0.018] | 56.62 [51.10, 62.15] | 44.27 | 133.60 | 0.11 | 1.365 | 1.585 |
| G | 0.000 [0.000, 0.000] | 0.012 [0.005, 0.019] | 59.17 [54.65, 63.70] | 50.00 | 133.08 | 0.11 | 1.372 | 1.622 |
| H | 0.000 [0.000, 0.000] | 0.008 [0.001, 0.015] | 57.78 [52.69, 62.86] | 47.12 | 130.39 | 0.11 | 1.562 | 1.777 |
| I0 | 0.000 [0.000, 0.000] | 0.013 [0.002, 0.024] | 45.38 [38.36, 52.39] | 40.49 | 101.41 | 8.83 | 44.347 | 50.468 |
| I1 | 0.000 [0.000, 0.000] | 0.015 [0.007, 0.023] | 36.92 [29.34, 44.51] | 28.84 | 93.46 | 12.84 | 35.791 | 38.812 |
| I2 | 0.000 [0.000, 0.000] | 0.012 [0.006, 0.018] | 38.95 [29.58, 48.31] | 32.44 | 90.40 | 48.04 | 120.306 | 146.508 |
| I2_rewire_fixed | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.025] | 45.69 [40.61, 50.77] | 41.56 | 97.82 | 16.25 | 80.025 | 87.800 |

## Interpretation and limits

This is a reproducible expanded-range check of the A-I viewer models. It tests whether each fitted system maps an audio instruction to one of the positions after paired audiovisual training. It does not establish a need for multimodal integration: the acoustic instruction itself specifies the complete position, every target position appears in training, and the same seed-specific token waveforms recur at test. It also does not test compositional generalization to unseen positions, noisy/new waveform realizations, a changed codebook at test, or few-shot learning.

The five seeds vary token assignment, model initialization, and test order, but share the same 199 underlying position labels and full-coverage train/test protocol. Confidence intervals describe this seed variation; they are not population-level uncertainty over task designs. The 24-unit liquid and four-epoch neural settings are the CPU-oriented viewer profile, so timing and scores must not be merged with the prior spatial-grounding benchmark. No memory/RSS comparison is claimed.

## Files

- `seed_results.csv`: one row per seed and model, including quality and cost measurements.
- `summary.csv`: seed-averaged metrics and 95% t intervals.
