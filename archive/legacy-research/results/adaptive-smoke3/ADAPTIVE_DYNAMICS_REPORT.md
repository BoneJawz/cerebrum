# Trainable Dynamical States and Recursive World-Model Learning

## Protocol

The run uses 2 seeds beginning at 42, spaced by 1009, and the existing held-out pairs `(-1, 2)` and `(1, 3)`. Train sizes are (4,); validation and test episodes are independent. Dataset generation extends the field and trajectory to allow 18-step evaluation, so it retains the command split and sensory rules but its pixel count/initial positions differ from the earlier 129-pixel run. Primary training uses combined raw sensory inputs through tick 4.

F and G/H use 32 hidden units and about 2,177 trainable parameters; the parameter-matched direct MLP uses 156. Every adaptive model starts from a zero state for each episode. F uses a learned tanh recurrent transition; G uses explicit Euler with dt=0.05s, tau=0.1s, and dt/tau=0.50; H shares G's continuous training dynamics and receives hard, fixed event thresholds only at evaluation. No event threshold is learned. NumPy reverse-mode BPTT trains all position targets for intervals 1–12 with a shared scalar position decoder; training clips the global gradient norm at 1.0. Adam uses learning rate 0.003, batch size 32, and 60 epochs at sizes below 500 or 80 at 500. All input/target normalization uses training data only. There is no teacher forcing during prediction. Horizon 18 is excluded from loss.

The direct MLP predicts each training horizon from a flattened five-tick observation sequence with one hidden layer and a direct output head. It has no autonomous state-transition mechanism and therefore does not report h=18. E3, A–D, and E1/E2 are preserved as historical references in `results/diagnostics/DIAGNOSTIC_REPORT.md`; they were not changed or rerun here. The E3 physics-structured sensory baseline scored 0 px on the prior split; it uses known factors and known kinematics and is not parameter-matched.

## Nine-step position MAE on the compositional split

Cells give the across-seed mean and 95% t interval, computed from each seed's mean across command pairs; SD and seed count are included.

| Model | Observed | Held out |
|---|---:|---:|
| F | 17.37 ± 41.67 (SD 4.64, n=2) | 27.74 ± 170.31 (SD 18.96, n=2) |
| FF_direct | 18.51 ± 53.15 (SD 5.92, n=2) | 28.00 ± 106.65 (SD 11.87, n=2) |
| G | 17.06 ± 35.85 (SD 3.99, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) |
| H | 17.06 ± 35.85 (SD 3.99, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) |

## Held-out MAE by rollout horizon

| Model | h=3 | h=6 | h=9 | h=12 | h=18 (extrapolation) |
|---|---:|---:|---:|---:|---:|
| F | 21.05 ± 124.77 (SD 13.89, n=2) | 22.64 ± 141.27 (SD 15.72, n=2) | 27.74 ± 170.31 (SD 18.96, n=2) | 34.08 ± 217.44 (SD 24.20, n=2) | 48.20 ± 254.17 (SD 28.29, n=2) |
| FF_direct | 22.75 ± 103.75 (SD 11.55, n=2) | 21.96 ± 84.28 (SD 9.38, n=2) | 28.00 ± 106.65 (SD 11.87, n=2) | 38.72 ± 79.86 (SD 8.89, n=2) | n/a |
| G | 18.76 ± 86.10 (SD 9.58, n=2) | 22.44 ± 67.70 (SD 7.53, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) | 44.08 ± 95.83 (SD 10.67, n=2) | 61.92 ± 151.92 (SD 16.91, n=2) |
| H | 18.76 ± 86.10 (SD 9.58, n=2) | 22.44 ± 67.70 (SD 7.53, n=2) | 31.39 ± 74.86 (SD 8.33, n=2) | 44.08 ± 95.83 (SD 10.67, n=2) | 61.92 ± 151.92 (SD 16.91, n=2) |

## Sample efficiency

Mean held-out h=9 MAE by total training episodes (seed-level data is in the CSV).

| Model | " + " | ".join(map(str,config.train_sizes)) + " |
|---|" + "---:|"*len(config.train_sizes) + "
" + size_table + "

"
"## Balanced-distribution control

This secondary control trains on all six direction/speed pairs and evaluates on independent trajectories from every pair. The original held-out benchmark remains untouched.

| Model | h=9 MAE (mean ± 95% CI) |
|---|---:|
" + "
".join(balanced) + "

"
"## Frozen-state probes

Linear probes are trained after the adaptive model is frozen, using only observed training combinations. Position reports MAE; direction and speed report accuracy (speed is snapped to known discrete speed levels only for probe scoring). They do not feed back into model training.

| Model | Split | Position MAE | Direction accuracy | Speed exact accuracy |
|---|---|---:|---:|---:|
" + probe_table + "

"
"## H event threshold sweep

Thresholds apply to the norm of the proposed autonomous state change. The candidate change must be computed to score that trigger, so skipped state writes do not necessarily save total work. CPU timings are local process-clock measurements and may quantize to zero for short runs; latency quantiles and component wall times are also recorded.

| Threshold | h=9 MAE | Skipped opportunities | Median ms/episode | CPU s/pass |
|---:|---:|---:|---:|---:|
" + (event_table or "| No event rows | | | | |") + "

"
"## Artifacts and reproducibility

`adaptive_dynamics_results.csv` stores per-seed/per-pair prediction errors, including h=18 and balanced-control rows. `adaptive_sample_efficiency.csv` stores fitted train/validation metrics by dataset size; `adaptive_learning_curves.csv` stores epoch loss. `adaptive_compute_benchmark.csv` contains per-seed timing, memory, update counts, tail latency and H threshold results. `adaptive_latent_probes.csv` and `adaptive_paired_differences.csv` store analysis records. Plots include generalization, learning curves, rollout error, and accuracy versus compute. Compressed checkpoints under `checkpoints/` include weights, normalization, seed and configuration. Rerun with `python -m sensory_lab.adaptive_cli --outdir results/adaptive-dynamics`; CPU thread count and all seed/config arguments are recorded by the CLI or checkpoint metadata.

"
"## Findings and answers

1. **Does trainable dynamics improve accuracy?** Compare F/G/H against the historical A–D means: B was 2.71 px observed and 24.44 px held out at h=9; the learned models' measured values are above. No improvement is inferred without paired uncertainty and matched information/compute.
2. **Does it improve recombination?** The observed/held-out columns and balanced control separate the original correlation shortcut from a broader training distribution.
3. **Can one transition cover several horizons?** F/G/H share one transition and decoder over h=3,6,9,12. Results are tabulated above.
4. **Can it extrapolate?** h=18 is evaluated without including it in loss; the direct model has no h=18 prediction.
5. **Does continuous time help over discrete recurrence?** F vs G is the matched hidden-size comparison; Euler adds a state matrix operation each interval. Error and latency results decide, not the continuous-time label.
6. **Does event sparsification reduce computation?** H thresholds report accuracy, skips, trigger/update wall time and end-to-end latency. The candidate transition is evaluated for trigger decisions, limiting expected savings.
7. **Are factors recoverable from latent state?** Probe metrics show what a linear decoder recovers on observed and held-out combinations; this is descriptive and does not imply one latent coordinate per factor.
8. **How does it compare with E3?** E3's prior 0 px result is a physics-informed reference with a favorable explicit factorization; it is not an apples-to-apples architecture comparison.
9. **What explains differences?** F/G/H use the same examples, supervision and hidden width; the direct model matches parameter count but has a different objective. The balanced control changes training distribution. This does not isolate every objective/compute interaction; paired seed differences are in the paired CSV.
10. **Smallest next modification:** use the measured probes and balanced control to decide whether the sensory encoder or autonomous transition loses command information, then change one of those interfaces while keeping the same trajectories and parameter budget.

Measured results are not evidence of biological validity or subjective experience. Negative or mixed generalization results are retained as generated.
