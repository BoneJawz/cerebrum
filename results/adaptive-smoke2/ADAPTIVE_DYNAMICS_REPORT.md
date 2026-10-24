# Trainable dynamical state experiment

## Protocol

Seeds: 2, beginning at 42, spaced by 1009. The original held-out combinations and A-D experiments are unchanged. F/G/H use 32 hidden variables, a shared scalar position decoder, five observed sensory ticks, 12-step trajectory loss, and autonomous rollout. Training uses Adam (0.003), minibatches of 32, 60 epochs (80 at maximum size), and global gradient clipping at 1.0. Inputs and targets are normalized from training data only. H is trained with continuous updates and evaluated with optional hard event thresholds; its event decision is not learned. G/H use explicit Euler with dt=0.05s and tau=0.1s (dt/tau=0.50).

This is supervised latent-state dynamics learning, not JEPA. The target is position at each of the first 12 future intervals. The h=18 result is excluded from training loss. No direction/speed labels or kinematic formula enter the models. NumPy analytic BPTT avoids a new autodiff dependency. Checkpoints include weights and normalization.

## Held-out nine-step MAE (pixels)

| Model | MAE |
|---|---:|
| F | 27.743 |
| FF_direct | 28.000 |
| G | 31.390 |
| H | 31.390 |

Results and seed-level records are in `adaptive_dynamics_results.csv`; training curves are in `adaptive_sample_efficiency.csv`; compute records are in `adaptive_compute_benchmark.csv`; latent probes are in `adaptive_latent_probes.csv`.

## Interpretation and limits

This first run tests F/G/H on the original correlated compositional split. The three variants have similar trainable parameter counts, but different update equations and per-step compute. Held-out failures remain valid negative results. E3 is retained in the existing diagnostic report as a favorable physics-structured reference. This implementation does not yet provide the requested balanced-distribution control, latent probes, direct-head control, event-threshold sweep, or all latency quantiles; the corresponding CSV placeholders are empty until those analyses are implemented. Do not infer efficiency or representation claims from this run.

## Questions

1. Learning transition parameters is compared with fixed A-D in the existing reports; this run provides F-H errors at multiple horizons.
2. Held-out recombination is measured separately from observed combinations.
3. One shared transition and decoder support horizons 3, 6, 9, and 12.
4. Horizon 18 measures extrapolation beyond the loss horizon.
5. G vs F compares Euler-integrated and discrete recurrence with matched hidden size.
6. H's event thresholds require the explicit threshold sweep before claiming compute savings.
7. Probe analysis is pending; latent claims are not made.
8. E3's 0 px reference is reported in `results/diagnostics/DIAGNOSTIC_REPORT.md`.
9. This run does not fully isolate architecture, objective, parameter-count, and distribution effects.
10. Next: complete balanced-distribution and direct-head controls, then add frozen-state probes before enlarging the model.
