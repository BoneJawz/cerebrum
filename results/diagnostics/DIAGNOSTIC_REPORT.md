# Compositional Generalization Diagnostic Report

## Scope and protocol

This diagnostic retains A-D processor equations and uses the same deterministic training, validation, observed-test, and held-out-test construction as the existing benchmark. Seeds: 5, starting at 42, spaced by 1009. Training and validation exclude ((-1, 2), (1, 3)). All estimators receive only the permitted observations through tick 4; target labels are stored separately. Audio and vision are synchronized at 20 visual ticks/s, with 100 raw audio samples (2 kHz) per visual tick.

### Pipeline audit

- Every A-D input is an `Observation(t, vision, audio)` for ticks 0 through 4. Vision is a binary one-dimensional pixel vector; each audio chunk is 100 signed waveform samples. Ground-truth direction, speed, initial position, and future frames are stored outside `Observation`.
- `sensory_features` appends four audio statistics (mean, RMS, zero-crossing rate, mean absolute amplitude), 16 signed samples at evenly spaced indices, and 49 absolute rFFT bins. The 100-sample chunk is zero-padded to 128 for the FFT; there is no window. Magnitudes discard phase, while the signed samples retain its polarity. Readouts standardize processor feature matrices using training-set means/scales only.
- A concatenates its most recent four sensory feature vectors. B, C, and D return their current sensory vector concatenated with fixed internal state. B has one recurrent update per input tick; C has two Euler substeps; D checks input-feature change and conditionally applies one state update. All processor weights are seeded and fixed; only the output kernel-ridge readout is fitted.
- The primary readout is a degree-two polynomial kernel ridge model, which already includes implicit pairwise feature interactions. It emits four outputs trained for the four horizons; horizon is not a model input. E1 uses a linear kernel without an explicit target intercept, a limitation for absolute-position predictions. E3 is separately factorized as described below.
- Directions and speeds are cycle-balanced over permitted command pairs; each pair is balanced to within one training episode. Start position is sampled uniformly from the central third of the field, object width from 1/3/5 pixels, and waveform noise independently per episode/tick. Train, validation, observed test, and held-out test use distinct deterministic episode seeds. The only intended distribution difference is the held-out direction-speed pairing; all factor values are represented in training. There is no motion visible in the inputs.
- The observed pairs are left/speed-1, left/speed-3, right/speed-1, and right/speed-2. Thus speed 2 occurs only with rightward motion and speed 3 only with leftward motion in training/validation; the held-out pairs reverse those correlations. This is an intentional compositional split, but it creates a strong cue-correlation shortcut that can explain held-out direction reversals.
- Visual and audio observations share event timestamp `k/20` seconds. Each audio chunk spans the matching 50 ms interval. Processors receive synchronized chunks, not independently timed events.

No label leakage was found in the model inputs or split construction. The probes below test whether factor cues survive feature extraction and whether the models can compose them.

The currently generated cue pairs phase with direction and carrier frequency with speed. The phase-to-direction map is randomized per seed but shared between train and test for that seed. Signed waveform samples preserve phase polarity; magnitude FFT bins preserve frequency magnitude. These are explicit cue factors, not discovered semantics.

## Prediction results by horizon and pair

At horizon 9, observed / held-out mean MAE (pixels) across seeds:

| Model | Observed MAE | Held-out MAE |
|---|---:|---:|
| A | 3.91 | 32.15 |
| B | 2.71 | 24.44 |
| C | 2.62 | 25.86 |
| D | 2.63 | 25.03 |
| E1 | 61.76 | 85.00 |
| E3_factorized | 0.00 | 0.00 |
| E2_oracle | 0.00 | 0.00 |

`diagnostic_results.csv` includes each direction-speed pair and horizon with MAE, signed bias, within-1-pixel accuracy, and seed-level uncertainty summaries. `prediction_vs_actual.png` shows per-combination scatter; `horizon_error.png` shows horizon scaling. `error_decomposition.csv` fits each trajectory's forecast residual to an intercept plus horizon slope. The intercept is position-like and the slope is displacement-like only as a descriptive decomposition, not a unique causal explanation.

Held-out direction/speed errors at horizon 9:

| Model | Left at speed 2: MAE / bias (px) | Right at speed 3: MAE / bias (px) |
|---|---:|---:|
| A | 19.85 / +19.84 | 44.46 / -44.46 |
| B | 12.44 / +12.27 | 36.44 / -36.44 |
| C | 13.02 / +12.93 | 38.70 / -38.70 |
| D | 13.62 / +13.62 | 36.44 / -36.44 |

Held-out MAE by horizon (pixels):

| Model | 3 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|
| A | 11.06 | 21.37 | 32.15 | 43.04 |
| B | 9.41 | 16.45 | 24.44 | 32.46 |
| C | 10.40 | 17.86 | 25.86 | 34.00 |
| D | 8.93 | 16.76 | 25.03 | 33.30 |

The large positive error bias on left-at-speed-2 and negative bias on right-at-speed-3 indicate systematic direction reversals on both held-out pairs. Error rises close to linearly with horizon there while observed-pair MAE stays nearly flat, which is consistent with a displacement/rate error rather than a large initial-position offset. The error-slope/intercept decomposition remains descriptive and cannot uniquely identify the cause.

## Factor probes and E3

| Split | Initial position | Direction from audio | Speed from audio |
|---|---|---|---|
| observed | 0.00 px MAE | 100.0% accuracy | 0.00 speed MAE / 100.0% exact |
| heldout | 0.00 px MAE | 100.0% accuracy | 0.00 speed MAE / 100.0% exact |

The factor probes use training labels only as supervised targets and the permitted sensory features as inputs. E3 independently estimates position from vision and direction/speed from audio, quantizes speed to the known allowed levels, then applies the known constant-velocity rule. E3 is a deliberately favorable, physics-structured diagnostic baseline, not an architecture-matched model. E2 remains a privileged ground-truth oracle.

The direct linear visual probe is also reported separately; its held-out MAE is 4.70 px. The centroid is exact here because stimuli are unclipped symmetric bars; this makes the position component intentionally easy.

Probes trained directly on A-D outputs:

| Model features | Position MAE | Direction accuracy | Speed MAE / exact accuracy |
|---|---:|---:|---:|
| A | 4.70 px MAE | 0.0% accuracy | 0.52 MAE / 50.0% exact |
| B | 5.79 px MAE | 50.0% accuracy | 0.75 MAE / 40.5% exact |
| C | 5.58 px MAE | 58.0% accuracy | 0.78 MAE / 40.0% exact |
| D | 4.34 px MAE | 63.7% accuracy | 0.81 MAE / 17.2% exact |

The raw sensory suffix is preserved in every processor output. Using only that suffix, the held-out direction probe reached 100.0%, the speed probe reached 100.0% exact accuracy, and the visual centroid had 0.00 px MAE. The all-feature quadratic probes above are weaker; that suggests readout/feature competition in the fixed representations, while the exact sensory probes and E3 establish the factors are not absent.

## Readout capacity

`readout_diagnostics.csv` compares the original quadratic kernel ridge, an explicit compact vision-by-audio pairwise feature expansion with a linear ridge readout, and a 32-unit tanh MLP. The expansion selects six high-variance visual and six high-variance audio coordinates per sensory history block using training features only. The MLP uses L2 weight decay 0.0001, at most 250 epochs, and validation-only early stopping (patience 30). Feature extraction and A-D dynamics are unchanged. B MLP held-out result at horizon 9: 28.71 px MAE. Parameter counts, readout training time, and inference latency are in the CSV.

| Processor | Readout | Held-out MAE (px) | Fitted coefficients/weights | Fit (ms) | Inference (ms/episode) |
|---|---|---:|---:|---:|---:|
| A | quadratic_ridge | 32.15 | 2000 | 42.2 | 0.1755 |
| A | explicit_pairwise_ridge | 33.39 | 2000 | 47.0 | 0.1968 |
| A | mlp32 | 42.80 | 25508 | 68.3 | 0.0097 |
| B | quadratic_ridge | 24.44 | 2000 | 32.5 | 0.0920 |
| B | explicit_pairwise_ridge | 27.46 | 2000 | 30.4 | 0.0905 |
| B | mlp32 | 28.71 | 8548 | 35.2 | 0.0069 |
| C | quadratic_ridge | 25.86 | 2000 | 33.5 | 0.0917 |
| C | explicit_pairwise_ridge | 46.32 | 2000 | 29.8 | 0.0903 |
| C | mlp32 | 25.60 | 8548 | 33.4 | 0.0072 |
| D | quadratic_ridge | 25.03 | 2000 | 32.5 | 0.0915 |
| D | explicit_pairwise_ridge | 45.50 | 2000 | 29.2 | 0.0899 |
| D | mlp32 | 26.64 | 8548 | 45.4 | 0.0072 |

## Event-driven cost audit

At D threshold 0.04, mean skip fraction on the held-out set was 0.0%; uninstrumented end-to-end process CPU time averaged 0.0781s per pass over 5 repeated passes. `event_threshold_results.csv` reports opportunities, triggered/skipped updates, separate high-resolution wall timings for trigger checks and state updates, total uninstrumented process CPU/wall time per pass, latency, and errors for thresholds (0.0, 0.04, 0.1, 0.25, 0.5, 1.0); B and C provide matched inference-cost references on the same held-out episodes. On Windows the process CPU timer is too coarse to resolve individual short checks, so those subcomponent durations use a separate instrumented monotonic-wall-clock pass and should be read as timing proxies. `event_accuracy_cost.png` plots held-out error against median end-to-end latency. Fewer calls alone do not establish a speed benefit because trigger detection itself costs compute.

| Model | Threshold | Held-out MAE (px) | Skipped opportunities | CPU s/pass (5 passes) | Trigger check (ms, wall) | State update (ms, wall) |
|---|---:|---:|---:|---:|---:|---:|
| B | baseline | 24.44 | — | 0.0725 | 0.00 | 0.00 |
| C | baseline | 25.86 | — | 0.0775 | 0.00 | 0.00 |
| D | 0.0 | 25.76 | 0.0% | 0.0775 | 1.76 | 4.49 |
| D | 0.04 | 25.77 | 0.0% | 0.0781 | 1.75 | 4.49 |
| D | 0.1 | 27.40 | 33.6% | 0.0750 | 1.75 | 3.07 |
| D | 0.25 | 25.94 | 40.0% | 0.0750 | 1.75 | 2.79 |
| D | 0.5 | 27.97 | 40.6% | 0.0750 | 1.76 | 2.77 |
| D | 1.0 | 29.78 | 80.0% | 0.0731 | 1.73 | 1.01 |

## Controls and limits

The same test episodes are reused across architectures, readouts, horizons, and event thresholds within each seed. Alpha selection and MLP early stopping use only observed validation combinations. Paired seed differences and 95% t intervals are recorded separately. Threshold and readout sweeps are diagnostic comparisons; the held-out scores are reported together and were not used for tuning.

The attachment supplied for this task ended mid-list at the deliverable bullet `factor...`; outputs here cover the requested diagnostic tasks 1-8 and the artifact names that can be inferred from the text. No result implies reusable dynamics or biological validity.
