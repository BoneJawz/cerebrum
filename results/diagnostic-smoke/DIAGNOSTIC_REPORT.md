# Compositional Generalization Diagnostic Report

## Scope and protocol

This diagnostic retains A-D processor equations and uses the same deterministic training, validation, observed-test, and held-out-test construction as the existing benchmark. Seeds: 2, starting at 7, spaced by 1009. Training and validation exclude ((-1, 2), (1, 3)). All estimators receive only the permitted observations through tick 4; target labels are stored separately. Audio and vision are synchronized at 20 visual ticks/s, with 100 raw audio samples (2 kHz) per visual tick.

### Pipeline audit

- Every A-D input is an `Observation(t, vision, audio)` for ticks 0 through 4. Vision is a binary one-dimensional pixel vector; each audio chunk is 100 signed waveform samples. Ground-truth direction, speed, initial position, and future frames are stored outside `Observation`.
- `sensory_features` appends four audio statistics (mean, RMS, zero-crossing rate, mean absolute amplitude), 16 signed samples at evenly spaced indices, and 49 absolute rFFT bins. The 100-sample chunk is zero-padded to 128 for the FFT; there is no window. Magnitudes discard phase, while the signed samples retain its polarity. Readouts standardize processor feature matrices using training-set means/scales only.
- A concatenates its most recent four sensory feature vectors. B, C, and D return their current sensory vector concatenated with fixed internal state. B has one recurrent update per input tick; C has two Euler substeps; D checks input-feature change and conditionally applies one state update. All processor weights are seeded and fixed; only the output kernel-ridge readout is fitted.
- The primary readout is a degree-two polynomial kernel ridge model, which already includes implicit pairwise feature interactions. It emits four outputs trained for the four horizons; horizon is not a model input. E1 uses a linear kernel without an explicit target intercept, a limitation for absolute-position predictions. E3 is separately factorized as described below.
- Directions and speeds are cycle-balanced over permitted command pairs; each pair is balanced to within one training episode. Start position is sampled uniformly from the central third of the field, object width from 1/3/5 pixels, and waveform noise independently per episode/tick. Train, validation, observed test, and held-out test use distinct deterministic episode seeds. The only intended distribution difference is the held-out direction-speed pairing; all factor values are represented in training. There is no motion visible in the inputs.
- Visual and audio observations share event timestamp `k/20` seconds. Each audio chunk spans the matching 50 ms interval. Processors receive synchronized chunks, not independently timed events.

No label leakage was found in the model inputs or split construction. The probes below test whether factor cues survive feature extraction and whether the models can compose them.

The currently generated cue pairs phase with direction and carrier frequency with speed. The phase-to-direction map is randomized per seed but shared between train and test for that seed. Signed waveform samples preserve phase polarity; magnitude FFT bins preserve frequency magnitude. These are explicit cue factors, not discovered semantics.

## Prediction results by horizon and pair

At horizon 9, observed / held-out mean MAE (pixels) across seeds:

| Model | Observed MAE | Held-out MAE |
|---|---:|---:|
| A | 6.07 | 34.87 |
| B | 5.79 | 34.25 |
| C | 5.64 | 33.20 |
| D | 5.55 | 34.96 |
| E1 | 59.72 | 74.13 |
| E3_factorized | 0.00 | 0.00 |
| E2_oracle | 0.00 | 0.00 |

`diagnostic_results.csv` includes each direction-speed pair and horizon with MAE, signed bias, within-1-pixel accuracy, and seed-level uncertainty summaries. `prediction_vs_actual.png` shows per-combination scatter; `horizon_error.png` shows horizon scaling. `error_decomposition.csv` fits each trajectory's forecast residual to an intercept plus horizon slope. The intercept is position-like and the slope is displacement-like only as a descriptive decomposition, not a unique causal explanation.

Held-out direction/speed errors at horizon 9:

| Model | Left at speed 2: MAE / bias (px) | Right at speed 3: MAE / bias (px) |
|---|---:|---:|
| A | 20.61 / +20.61 | 49.12 / -49.12 |
| B | 19.50 / +19.50 | 48.99 / -48.99 |
| C | 18.96 / +18.96 | 47.44 / -47.44 |
| D | 19.86 / +19.86 | 50.06 / -50.06 |

Held-out MAE by horizon (pixels):

| Model | 3 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|
| A | 14.26 | 23.91 | 34.87 | 45.82 |
| B | 14.79 | 23.74 | 34.25 | 44.78 |
| C | 13.79 | 22.65 | 33.20 | 43.75 |
| D | 14.91 | 24.30 | 34.96 | 45.57 |

The large positive error bias on left-at-speed-2 and negative bias on right-at-speed-3 indicate systematic direction reversals on both held-out pairs. Error rises close to linearly with horizon there while observed-pair MAE stays nearly flat, which is consistent with a displacement/rate error rather than a large initial-position offset. The error-slope/intercept decomposition remains descriptive and cannot uniquely identify the cause.

## Factor probes and E3

| Split | Initial position | Direction from audio | Speed from audio |
|---|---|---|---|
| observed | 0.00 px MAE | 100.0% accuracy | 0.02 speed MAE / 100.0% exact |
| heldout | 0.00 px MAE | 100.0% accuracy | 0.01 speed MAE / 100.0% exact |

The factor probes use training labels only as supervised targets and the permitted sensory features as inputs. E3 independently estimates position from vision and direction/speed from audio, quantizes speed to the known allowed levels, then applies the known constant-velocity rule. E3 is a deliberately favorable, physics-structured diagnostic baseline, not an architecture-matched model. E2 remains a privileged ground-truth oracle.

The direct linear visual probe is also reported separately; its held-out MAE is 4.11 px. The centroid is exact here because stimuli are unclipped symmetric bars; this makes the position component intentionally easy.

Probes trained directly on A-D outputs:

| Model features | Position MAE | Direction accuracy | Speed MAE / exact accuracy |
|---|---:|---:|---:|
| A | 5.97 px MAE | 0.0% accuracy | 0.47 MAE / 50.0% exact |
| B | 6.46 px MAE | 0.0% accuracy | 0.51 MAE / 50.0% exact |
| C | 6.23 px MAE | 0.0% accuracy | 0.50 MAE / 50.0% exact |
| D | 6.67 px MAE | 0.0% accuracy | 0.48 MAE / 50.0% exact |

The raw sensory suffix is preserved in every processor output. Using only that suffix, the held-out direction probe reached 100.0%, the speed probe reached 100.0% exact accuracy, and the visual centroid had 0.00 px MAE. The all-feature quadratic probes above are weaker; that suggests readout/feature competition in the fixed representations, while the exact sensory probes and E3 establish the factors are not absent.

## Readout capacity

`readout_diagnostics.csv` compares the original quadratic kernel ridge, an explicit compact vision-by-audio pairwise feature expansion with a linear ridge readout, and a 32-unit tanh MLP. The expansion selects six high-variance visual and six high-variance audio coordinates per sensory history block using training features only. The MLP uses L2 weight decay 0.0001, at most 10 epochs, and validation-only early stopping (patience 3). Feature extraction and A-D dynamics are unchanged. B MLP held-out result at horizon 9: 41.25 px MAE. Parameter counts, readout training time, and inference latency are in the CSV.

## Event-driven cost audit

At D threshold 0, mean skip fraction on the held-out inference set was 0.0%; end-to-end process CPU time was 0.0000s for the full timed test pass. `event_threshold_results.csv` reports opportunities, triggered/skipped updates, high-resolution wall time spent in trigger checks and state updates, total process CPU/wall time, latency, and errors for thresholds (0.0, 0.1, 0.5); B and C provide matched inference-cost reference rows. On Windows the process CPU timer is too coarse to resolve individual short checks, so those subcomponent durations use the monotonic wall clock and should be read as timing proxies. `event_accuracy_cost.png` plots held-out error against process CPU time. Fewer calls alone do not establish a speed benefit because trigger detection itself costs compute.

| Model | Threshold | Held-out MAE (px) | Skipped opportunities | CPU seconds | Trigger check (ms, wall) | State update (ms, wall) |
|---|---:|---:|---:|---:|---:|---:|
| B | baseline | 34.25 | — | 0.0000 | 0.00 | 0.00 |
| C | baseline | 33.20 | — | 0.0156 | 0.00 | 0.00 |
| D | 0.0 | 32.68 | 0.0% | 0.0000 | 0.40 | 0.74 |
| D | 0.1 | 32.78 | 37.2% | 0.0078 | 0.39 | 0.47 |
| D | 0.5 | 32.90 | 41.7% | 0.0156 | 0.38 | 0.44 |

## Controls and limits

The same test episodes are reused across architectures, readouts, horizons, and event thresholds within each seed. Alpha selection and MLP early stopping use only observed validation combinations. Paired seed differences and 95% t intervals are recorded separately. Threshold and readout sweeps are diagnostic comparisons; the held-out scores are reported together and were not used for tuning.

The attachment supplied for this task ended mid-list at the deliverable bullet `factor...`; outputs here cover the requested diagnostic tasks 1-8 and the artifact names that can be inferred from the text. No result implies reusable dynamics or biological validity.
