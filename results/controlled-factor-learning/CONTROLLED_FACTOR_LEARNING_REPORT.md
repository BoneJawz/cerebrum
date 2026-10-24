# Controlled factor-learning and state-dynamics validation

## Protocol audit

Combined acoustic cues are direction ticks 0–1, silence tick 2, and speed ticks 3–4. At 20 Hz these finish at 0.25 s; the tick-4 observation includes the complete speed cue. The visual array at tick 5 is the initial motion origin. A uses observations 0–4 and predicts from frame 5; B uses 0–9 and predicts from frame 9. Horizons 1, 3, 6, 9, 12, and 18 target frames origin+h. The environment has 32 ticks to support B horizon 18.

Inputs contain only the fixed sensory feature vector: 177 visual cells, four waveform summary values, 16 signed waveform samples, and 49 FFT magnitudes. Cue names and simulator labels are kept in experiment metadata and never passed to models. Each acoustic chunk is 100 raw samples at 2 kHz; visual frames arrive at 20 Hz. The same independent vocabulary and waveform generation are used in every condition.

## Measured results

Study configuration: 5 seeds, sizes 10, 25, 50, 100, 250, 500, 40 epochs per fit, six direction-speed combinations, 20 held-out trajectories per combination. Paired seed means and standard deviations at the largest training size are shown below. Detailed per-factor metrics are in `observation_protocol_results.csv`; training/inference costs are in `compute_results.csv`.

| Condition | Model | h1 MAE | h9 MAE | h12 MAE | h18 MAE |
|---|---|---:|---:|---:|---:|
| A_instruction_only | F | 3.88 ± 0.69 | 4.12 ± 0.62 | 4.76 ± 0.81 | 8.80 ± 1.83 |
| A_instruction_only | G | 4.10 ± 0.71 | 4.13 ± 0.47 | 4.48 ± 0.58 | 7.98 ± 1.39 |
| A_instruction_only | H | 4.10 ± 0.71 | 4.13 ± 0.47 | 4.52 ± 0.60 | 8.22 ± 1.39 |
| A_instruction_only | FF_direct | 7.33 ± 0.42 | 9.63 ± 0.67 | 10.88 ± 1.06 | not available |
| B_observed_motion | F | 4.53 ± 0.73 | 6.79 ± 0.96 | 8.28 ± 1.11 | 13.65 ± 1.47 |
| B_observed_motion | G | 4.32 ± 0.50 | 7.14 ± 1.64 | 8.62 ± 2.04 | 13.15 ± 2.05 |
| B_observed_motion | H | 4.32 ± 0.50 | 7.18 ± 1.63 | 8.80 ± 1.97 | 13.83 ± 1.91 |
| B_observed_motion | FF_direct | 7.41 ± 1.19 | 10.04 ± 0.84 | 11.52 ± 0.88 | not available |
| C_Btrain_Aeval | F | 12.81 ± 3.32 | 29.30 ± 9.58 | 36.25 ± 12.38 | 48.60 ± 17.67 |
| C_Btrain_Aeval | G | 13.80 ± 1.54 | 23.31 ± 9.23 | 27.27 ± 11.86 | 34.41 ± 16.31 |
| C_Btrain_Aeval | H | 13.80 ± 1.54 | 23.26 ± 9.20 | 27.09 ± 11.71 | 33.86 ± 15.65 |
| C_Btrain_Aeval | FF_direct | 14.09 ± 3.36 | 18.03 ± 4.75 | 18.86 ± 3.94 | not available |

Acoustic diagnostics: isolated_cue_nearest_centroid: 41.2% (n=80); sequential_pair_nearest_centroid: 100.0% (n=72). Isolated cue classes use nearest-centroid classification over independently generated full two-chunk waveforms; combined instructions use the ordered five-tick waveform sequence.

Results are generated from direct model predictions. A has targets relative to frame 5, B relative to frame 9, and C evaluates a B-trained model from frame 5. Compare matched A with A and B with B; C is explicitly a window-transfer condition. FF_direct has no autonomous hidden-state trace and does not forecast beyond its trained direct horizon.

## Limits and interpretation

This run validates all-six-combination in-distribution prediction under matched windows before interpreting compositional transfer. Its A/B difference combines added visual motion evidence and a shorter forecast distance, so it is not a pure sensory-modality effect. Cue scores are diagnostic nearest-centroid baselines, not the learned models' own concept reports. State trace values are decoded positions from hidden states, not direct evidence that a factor is represented. The current first phase does not execute strict-holdout versus factor-pretraining transfer or curriculum versus shuffled controls; those are deferred until the matched-window baseline is stable. The curriculum CSV contains only training-loss checkpoints from the matched all-six shuffled fit and must not be interpreted as a schedule comparison.

## Next experiment

Use these matched-window results to decide whether the remaining failure is already present on the six-way task. If matched A remains poor, first test normalization, input timing, and supervised optimization on a smaller fixed diagnostic set. If A and B work but C fails, isolate the observation-window transfer. Only then run strict compositional holdout and balanced factor-pretraining transfer, with curriculum and shuffled schedules replaying identical episode IDs and update counts.
