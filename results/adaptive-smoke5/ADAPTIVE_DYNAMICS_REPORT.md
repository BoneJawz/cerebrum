# Trainable Dynamical States and Recursive World-Model Learning

## Protocol

This run uses 2 seeds beginning at 42, spaced by 1009, and preserves held-out pairs `(-1, 2)` and `(1, 3)`. Training, validation and test trajectories are independent. The visual field is expanded to permit h=18 evaluation, so its pixel count and initial-position range differ from the earlier 129-pixel experiment. Models receive combined sensory input through tick 4.

F/G/H use 32 hidden variables and 10,017 trainable parameters. The direct MLP has 9,956. Each episode begins at zero state. F uses learned tanh recurrence; G uses explicit Euler with dt=0.05 s and tau=0.1 s; H shares G's continuous training dynamics and uses fixed hard thresholds only during evaluation. H's threshold is not learned. NumPy reverse-mode BPTT with Adam (0.003), batches of 32, gradient clipping at 1.0, and 60 epochs (80 at 500 examples) trains position targets at intervals 1–12 with a shared scalar decoder. Normalization uses training data only. No direction/speed labels or motion equation enter model inputs or targets, and prediction receives no future observations. h=18 is excluded from training loss.

The direct MLP maps the flattened five-tick sequence to one output per trained horizon; it has no autonomous transition and does not report h=18. Historical A–D and E1–E3 results remain unchanged in `results/diagnostics/DIAGNOSTIC_REPORT.md`. E3 previously scored 0 px using known factors and kinematics, making it a favorable nonmatched reference.

## Nine-step MAE on original compositional split

Mean and 95% t interval across seeds, after averaging each seed over command pairs.

| Model | Observed | Held out |
|---|---:|---:|
| F | 17.37 ± 41.67 (SD 4.64, n=2) | 27.74 ± 170.31 (SD 18.96, n=2) |
| FF_direct | 18.51 ± 53.15 (SD 5.92, n=2) | 28.00 ± 106.65 (SD 11.87, n=2) |
| G | 17.06 ± 35.85 (SD 3.99, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) |
| H | 17.06 ± 35.85 (SD 3.99, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) |

## Held-out MAE by rollout horizon

| Model | h=3 | h=6 | h=9 | h=12 | h=18 extrapolation |
|---|---:|---:|---:|---:|---:|
| F | 21.05 ± 124.77 (SD 13.89, n=2) | 22.64 ± 141.27 (SD 15.72, n=2) | 27.74 ± 170.31 (SD 18.96, n=2) | 34.08 ± 217.44 (SD 24.20, n=2) | 48.20 ± 254.17 (SD 28.29, n=2) |
| FF_direct | 22.75 ± 103.75 (SD 11.55, n=2) | 21.96 ± 84.28 (SD 9.38, n=2) | 28.00 ± 106.65 (SD 11.87, n=2) | 38.72 ± 79.86 (SD 8.89, n=2) | n/a |
| G | 18.76 ± 86.10 (SD 9.58, n=2) | 22.44 ± 67.70 (SD 7.53, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) | 44.08 ± 95.83 (SD 10.67, n=2) | 61.92 ± 151.92 (SD 16.91, n=2) |
| H | 18.76 ± 86.10 (SD 9.58, n=2) | 22.44 ± 67.70 (SD 7.53, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) | 44.08 ± 95.83 (SD 10.67, n=2) | 61.92 ± 151.92 (SD 16.91, n=2) |

## Sample efficiency

| Model | 4 |
|---|---:|
| F | 27.74 |
| FF_direct | 28.00 |
| G | 31.39 |
| H | 31.39 |

## Balanced-distribution control

This secondary control trains on all six direction/speed pairs and tests independent trajectories from all pairs; it does not replace the original holdout.

| Model | h=9 MAE, mean ± 95% CI |
|---|---:|
| F | 22.39 ± 55.38 (SD 6.16, n=2) |
| FF_direct | 19.35 ± 84.82 (SD 9.44, n=2) |
| G | 20.90 ± 59.61 (SD 6.63, n=2) |
| H | 20.90 ± 59.61 (SD 6.63, n=2) |

## Paired model differences

Differences are held-out h=9 MAE, paired by seed and averaged across held-out command pairs. Negative values favor F.

| Comparison | Difference, mean ± 95% CI (px) |
|---|---:|
| F minus G | -3.65 ± 95.44 (SD 10.62, n=2) |
| F minus H | -3.65 ± 95.44 (SD 10.62, n=2) |
| F minus FF_direct | -0.26 ± 63.66 (SD 7.09, n=2) |

## Frozen-state probes

Linear probes are fit after model training using observed training combinations. Position reports MAE; direction and speed report accuracy. Probe labels do not feed back into model training.

| Model | Split | Position MAE | Direction accuracy | Speed exact accuracy |
|---|---|---:|---:|---:|
| F | observed | 14.967 | 1.000 | 1.000 |
| F | heldout | 24.430 | 0.125 | 0.875 |
| FF_direct | observed | 15.498 | 0.750 | 0.938 |
| FF_direct | heldout | 26.924 | 0.000 | 0.625 |
| G | observed | 15.301 | 1.000 | 1.000 |
| G | heldout | 23.544 | 0.000 | 1.000 |
| H | observed | 15.301 | 1.000 | 1.000 |
| H | heldout | 23.544 | 0.000 | 1.000 |

## Latent command-pair clustering

The descriptive silhouette score compares standardized latent-state distances within a direction/speed pair and to the nearest other pair. It is sensitive to initial-position variation and is not evidence that a latent dimension has a physical meaning.

| Model | Split | Mean within distance | Nearest between distance | Mean silhouette |
|---|---|---:|---:|---:|
| F | observed | 0.881 | 6.510 | 0.832 |
| F | heldout | 4.998 | 10.697 | 0.531 |
| FF_direct | observed | 1.226 | 3.274 | 0.549 |
| FF_direct | heldout | 1.877 | 5.480 | 0.656 |
| G | observed | 0.826 | 5.704 | 0.790 |
| G | heldout | 3.045 | 11.092 | 0.725 |
| H | observed | 0.826 | 5.704 | 0.790 |
| H | heldout | 3.045 | 11.092 | 0.725 |

## H event threshold sweep

Thresholds act on the candidate autonomous state change. Computing that candidate is part of trigger evaluation, so skipped writes need not reduce total compute. CPU process-clock time can be coarse for short runs; latency quantiles and component wall time are also recorded.

The reduced held-out error at high thresholds is consistent with suppressing some trained motion updates on the direction-reversed combinations; it is a prediction change from damping, not evidence that event selection learned a useful computation policy.

| Threshold | h=9 MAE | Skipped opportunities | Median ms/episode | CPU s/pass |
|---:|---:|---:|---:|---:|
| 0 | 31.39 | 0.0% | 0.3176 | 0.00000 |
| 0.05 | 31.39 | 0.0% | 0.3134 | 0.00781 |
| 0.1 | 31.39 | 0.0% | 0.3124 | 0.00000 |
| 0.25 | 30.18 | 16.7% | 0.3206 | 0.00000 |
| 0.5 | 20.10 | 69.4% | 0.3125 | 0.00000 |
| 1 | 20.06 | 91.7% | 0.4985 | 0.00000 |

## G timestep refinement

At dt=0.025 s (two Euler substeps per 0.05 s interval), h=9 MAE was 31.13 px versus 31.39 px at dt=0.05 s; mean absolute prediction change was 0.38 px. This inference-only sensitivity check uses the same learned parameters and is not a retrained convergence study.

## Training and inference compute

Inference CPU time is per complete held-out test pass; latency is measured per episode. Peak traced allocation excludes native NumPy/BLAS allocations and is not process RSS.

| Model | Train wall s | Train CPU s | Median ms/episode | p95 ms | p99 ms | Inference CPU s/pass | Peak traced KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 0.057 | 0.047 | 0.2593 | 0.2788 | 0.2814 | 0.00000 | 117.5 |
| FF_direct | 0.011 | 0.016 | 0.0810 | 0.0918 | 0.0927 | 0.00000 | 117.1 |
| G | 0.062 | 0.062 | 0.2921 | 0.3034 | 0.3050 | 0.00000 | 117.3 |
| H | 0.056 | 0.062 | 0.4281 | 0.4559 | 0.4588 | 0.00313 | 117.3 |

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
