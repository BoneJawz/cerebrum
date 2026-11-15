# Experiential Visual Recognition Report

## Protocol

Seeds: 42. The retina has 199 fixed cells spanning -99..99; only -1, 0, +1 are assessed. One cell is on at a time. Each sign/digit waveform is seed-fixed across contexts. Independent noisy waveform variants are used for comparator calibration and evaluation. Every candidate runs for 0.20 simulation seconds from an identical frozen post-cue snapshot. Candidate order uses seeded rotations; the three-choice chance level is 1/3.

The fixed matcher sums elementwise products of the causal mean acoustic state and candidate-induced visual response. The calibrated matcher fits a diagonal ridge compatibility function on labeled calibration pairs and is frozen during later experiences. Its supervised fit can learn the three associations externally, so only changes with the fixed matcher and paired-versus-control contrasts are relevant to internal-learning claims. Plasticity is frozen in all assessment clones. Online responses are recorded before the corresponding feedback interaction; every teaching/feedback presentation increments the exposure count once.

## Measured three-choice recognition

| Variant | Pairing | Fixed before | Fixed after | Calibrated after |
|---|---|---:|---:|---:|
| I0 | correct | 0.000 | 0.000 | 0.000 |

## Checkpoints

- **I0 three-choice gate:** FAIL/INCOMPLETE; mean 0.000, minimum seed 0.000, 1 seed(s).

## Production reference

The separately supervised fixed-I0 scalar readout achieved mean exact-within-half-unit accuracy 0.667 and MAE 0.426. Its labels are outside the liquid and its result is not evidence of internal association learning.

## Dynamics and compute

- I0: mean propagated events during learning 6812; weight updates 0; formed/pruned connections 0/0; assessment CPU 141.49 ms/trial.

## Interpretation and next experiment

A rise in fixed-matcher accuracy after correctly paired exposure, exceeding mismatched and single-modality controls on paired seeds, would support an effect of internal adaptation under this specific scoring rule. The calibrated comparator has label supervision and cannot independently establish that claim. Independent-episode assessment clears transient state and queue while retaining graph and weights; continuous assessments are labeled separately. Delay values are simulation time, while CPU and wall times are machine measurements. Event counts are scheduler events, not FLOPs. Observed queue size at the end of candidate branches is a lower bound on transient queue peaks.

The three-position experiment cannot test unseen sign-digit composition, two-digit grammar, broad distractor difficulty, or four/five-choice recognition. The composition CSV is intentionally empty until the three-position gate and paired-exposure advantage are established. Candidate waveforms and visual arrays retain the historical stimulus grammar. No new neuron or plasticity equation was added.

The smallest justified next experiment is determined by the measured fixed-matcher paired-versus-control contrast: if it fails, test whether the liquid's audio and visual response subspaces can be aligned by a fixed unsupervised transform before expanding position range. If it passes reproducibly, repeat with -9..+9 and counterbalanced unseen sign-digit pairs.
