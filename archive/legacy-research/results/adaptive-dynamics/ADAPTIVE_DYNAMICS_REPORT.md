# Trainable Dynamical States and Recursive World-Model Learning

## Protocol

This run uses 5 seeds beginning at 42, spaced by 1009, and preserves held-out pairs `(-1, 2)` and `(1, 3)`. Training, validation and test trajectories are independent. The visual field is expanded to permit h=18 evaluation, so its pixel count and initial-position range differ from the earlier 129-pixel experiment. Models receive combined sensory input through tick 4.

F/G/H use 32 hidden variables and 10,017 trainable parameters. The direct MLP has 9,956. Each episode begins at zero state. F uses learned tanh recurrence; G uses explicit Euler with dt=0.05 s and tau=0.1 s; H shares G's continuous training dynamics and uses fixed hard thresholds only during evaluation. H's threshold is not learned. NumPy reverse-mode BPTT with Adam (0.003), batches of 32, gradient clipping at 1.0, and 60 epochs (80 at 500 examples) trains position targets at intervals 1–12 with a shared scalar decoder. Normalization uses training data only. No direction/speed labels or motion equation enter model inputs or targets, and prediction receives no future observations. h=18 is excluded from training loss.

The direct MLP maps the flattened five-tick sequence to one output per trained horizon; it has no autonomous transition and does not report h=18. Historical A–D and E1–E3 results remain unchanged in `results/diagnostics/DIAGNOSTIC_REPORT.md`. E3 previously scored 0 px using known factors and kinematics, making it a favorable nonmatched reference.

## Nine-step MAE on original compositional split

Mean and 95% t interval across seeds, after averaging each seed over command pairs.

| Model | Observed | Held out |
|---|---:|---:|
| F | 1.56 ± 0.54 (SD 0.43, n=5) | 37.82 ± 2.27 (SD 1.83, n=5) |
| FF_direct | 2.87 ± 0.94 (SD 0.75, n=5) | 39.22 ± 3.98 (SD 3.20, n=5) |
| G | 1.27 ± 0.32 (SD 0.26, n=5) | 38.75 ± 1.14 (SD 0.92, n=5) |
| H | 1.27 ± 0.32 (SD 0.26, n=5) | 38.75 ± 1.14 (SD 0.92, n=5) |

## Held-out MAE by rollout horizon

| Model | h=3 | h=6 | h=9 | h=12 | h=18 extrapolation |
|---|---:|---:|---:|---:|---:|
| F | 11.96 ± 1.33 (SD 1.07, n=5) | 24.75 ± 1.70 (SD 1.37, n=5) | 37.82 ± 2.27 (SD 1.83, n=5) | 50.69 ± 3.22 (SD 2.60, n=5) | 74.84 ± 5.67 (SD 4.56, n=5) |
| FF_direct | 13.93 ± 2.31 (SD 1.86, n=5) | 25.19 ± 2.08 (SD 1.68, n=5) | 39.22 ± 3.98 (SD 3.20, n=5) | 52.07 ± 4.73 (SD 3.81, n=5) | n/a |
| G | 11.66 ± 0.43 (SD 0.35, n=5) | 25.12 ± 0.59 (SD 0.48, n=5) | 38.75 ± 1.14 (SD 0.92, n=5) | 52.08 ± 1.62 (SD 1.31, n=5) | 76.34 ± 2.65 (SD 2.14, n=5) |
| H | 11.66 ± 0.43 (SD 0.35, n=5) | 25.12 ± 0.59 (SD 0.48, n=5) | 38.75 ± 1.14 (SD 0.92, n=5) | 52.08 ± 1.62 (SD 1.31, n=5) | 76.34 ± 2.65 (SD 2.14, n=5) |

## Sample efficiency

| Model | 10 | 25 | 50 | 100 | 250 | 500 |
|---|---:|---:|---:|---:|---:|---:|
| F | 39.42 | 38.34 | 35.82 | 36.87 | 36.91 | 37.82 |
| FF_direct | 37.96 | 34.80 | 32.16 | 36.66 | 38.50 | 39.22 |
| G | 41.47 | 36.78 | 36.41 | 36.81 | 36.30 | 38.75 |
| H | 41.47 | 36.78 | 36.41 | 36.81 | 36.30 | 38.75 |

## Balanced-distribution control

This secondary control trains on all six direction/speed pairs and tests independent trajectories from all pairs; it does not replace the original holdout.

| Model | h=9 MAE, mean ± 95% CI |
|---|---:|
| F | 1.75 ± 0.71 (SD 0.58, n=5) |
| FF_direct | 2.91 ± 1.02 (SD 0.82, n=5) |
| G | 1.59 ± 0.12 (SD 0.10, n=5) |
| H | 1.59 ± 0.12 (SD 0.10, n=5) |

## Paired model differences

Differences are held-out h=9 MAE, paired by seed and averaged across held-out command pairs. Negative values favor F.

| Comparison | Difference, mean ± 95% CI (px) |
|---|---:|
| F minus G | -0.94 ± 2.25 (SD 1.81, n=5) |
| F minus H | -0.94 ± 2.25 (SD 1.81, n=5) |
| F minus FF_direct | -1.40 ± 4.97 (SD 4.01, n=5) |

## Frozen-state probes

Linear probes are fit after model training using observed training combinations. Position reports MAE; direction and speed report accuracy. Probe labels do not feed back into model training.

| Model | Split | Position MAE | Direction accuracy | Speed exact accuracy |
|---|---|---:|---:|---:|
| F | observed | 1.093 | 1.000 | 1.000 |
| F | heldout | 1.869 | 0.103 | 0.927 |
| FF_direct | observed | 2.416 | 0.995 | 0.504 |
| FF_direct | heldout | 5.306 | 0.125 | 0.562 |
| G | observed | 1.018 | 1.000 | 1.000 |
| G | heldout | 2.033 | 0.070 | 0.922 |
| H | observed | 1.018 | 1.000 | 1.000 |
| H | heldout | 2.033 | 0.070 | 0.922 |

## Latent command-pair clustering

The descriptive silhouette score compares standardized latent-state distances within a direction/speed pair and to the nearest other pair. It is sensitive to initial-position variation and is not evidence that a latent dimension has a physical meaning.

| Model | Split | Mean within distance | Nearest between distance | Mean silhouette |
|---|---|---:|---:|---:|
| F | observed | 5.960 | 7.073 | 0.151 |
| F | heldout | 6.273 | 8.216 | 0.233 |
| FF_direct | observed | 2.716 | 3.437 | 0.198 |
| FF_direct | heldout | 2.865 | 4.641 | 0.381 |
| G | observed | 5.779 | 6.997 | 0.162 |
| G | heldout | 6.114 | 8.469 | 0.274 |
| H | observed | 5.779 | 6.997 | 0.162 |
| H | heldout | 6.114 | 8.469 | 0.274 |

## H event threshold sweep

Thresholds act on the candidate autonomous state change. Computing that candidate is part of trigger evaluation, so skipped writes need not reduce total compute. CPU process-clock time can be coarse for short runs; latency quantiles and component wall time are also recorded.

The reduced held-out error at high thresholds is consistent with suppressing some trained motion updates on the direction-reversed combinations; it is a prediction change from damping, not evidence that event selection learned a useful computation policy.

| Threshold | h=9 MAE | Skipped opportunities | Median ms/episode | CPU s/pass |
|---:|---:|---:|---:|---:|
| 0 | 38.75 | 0.0% | 0.2303 | 0.00313 |
| 0.05 | 38.74 | 0.4% | 0.2290 | 0.00625 |
| 0.1 | 37.80 | 10.7% | 0.2287 | 0.00313 |
| 0.25 | 30.03 | 49.8% | 0.2280 | 0.00000 |
| 0.5 | 26.49 | 67.8% | 0.2278 | 0.00313 |
| 1 | 24.34 | 79.9% | 0.2273 | 0.00000 |

## G timestep refinement

At dt=0.025 s (two Euler substeps per 0.05 s interval), h=9 MAE was 38.86 px versus 38.75 px at dt=0.05 s; mean absolute prediction change was 0.19 px. This inference-only sensitivity check uses the same learned parameters and is not a retrained convergence study.

## Training and inference compute

Inference timings compare a common h=3,6,9,12 request; CPU time is per complete held-out test pass and latency is per episode. The h=18 extrapolation is measured separately for accuracy. Peak traced allocation excludes native NumPy/BLAS allocations and is not process RSS.

| Model | Train wall s | Train CPU s | Median ms/episode | p95 ms | p99 ms | Inference CPU s/pass | Peak traced KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 1.943 | 1.916 | 0.1775 | 0.2168 | 0.2591 | 0.00125 | 1602.6 |
| FF_direct | 0.352 | 0.366 | 0.0698 | 0.0801 | 0.0876 | 0.00000 | 1602.7 |
| G | 2.022 | 1.991 | 0.2246 | 0.2525 | 0.3014 | 0.00688 | 1602.6 |
| H | 2.022 | 2.000 | 0.2006 | 0.2322 | 0.2680 | 0.00188 | 1602.6 |

## Artifacts and findings

`adaptive_dynamics_results.csv` contains per-seed/per-pair errors and balanced-control results. `adaptive_initial_position_results.csv` stratifies errors by starting position. `adaptive_sample_efficiency.csv` contains train/validation scores by size; `adaptive_learning_curves.csv` contains epoch loss; `adaptive_compute_benchmark.csv` contains latency quantiles, CPU, memory and update counts; `adaptive_latent_probes.csv`, `adaptive_latent_cluster_diagnostics.csv`, `adaptive_timestep_sensitivity.csv` and `adaptive_paired_differences.csv` contain latent, integration and paired-seed diagnostics. Plots cover generalization, learning loss, rollout error and accuracy versus compute. Checkpoints under `checkpoints/` contain weights and normalizers. Reproduce with `python -m sensory_lab.adaptive_cli --outdir results/adaptive-dynamics`.

1. Accuracy and compositional generalization are measured above; historical B was 2.71 px observed and 24.44 px held out at h=9. Paired uncertainty is reported before attributing change to architecture.
2. The original holdout and balanced control separate recombination from the speed/direction training correlation.
3. F/G/H use one shared transition and decoder at h=3, 6, 9 and 12.
4. h=18 measures extrapolation beyond the loss horizon; the direct-head model has no h=18 output.
5. F/G compare discrete and Euler-integrated evolution at matched hidden width; measured accuracy and timing determine whether G helps.
6. H reports skips and costs. Since trigger evaluation computes the candidate transition, skipped assignments alone do not establish efficiency.
7. Probes measure linear recoverability; they do not prove that individual units map to physical factors.
8. E3 is a zero-error prior physics-informed reference with a favorable known factorization.
9. F/G/H share data, objective and width; the direct MLP has similar parameter count but a flattened feedforward encoder and direct-head objective, so it does not isolate the head alone. The balanced control changes the data distribution. No one comparison isolates every cause.
10. The smallest next change is to use probe and balanced-control results to locate information loss in observed encoding or autonomous transition, then alter one component under matched data and parameter budget.

These measurements do not establish biological validity or subjective experience. Negative results are retained.
