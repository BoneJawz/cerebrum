# Experiential Visual Recognition Report

## Protocol

Seeds: 42, 43, 44, 45, 46. The retina has 199 fixed cells spanning -99..99; only -1, 0, +1 are assessed. One cell is on at a time. Each sign/digit waveform is seed-fixed across contexts. Independent noisy waveform variants are used for comparator calibration and evaluation. Every candidate runs for 0.20 simulation seconds from an identical frozen post-cue snapshot. Candidate order uses seeded rotations; the three-choice chance level is 1/3.

The fixed matcher sums elementwise products of the causal mean acoustic state and candidate-induced visual response. The calibrated matcher fits a diagonal ridge compatibility function on labeled calibration pairs and is frozen during later experiences. Its supervised fit can learn the three associations externally, so only changes with the fixed matcher and paired-versus-control contrasts are relevant to internal-learning claims. Plasticity is frozen in all assessment clones. Online responses are recorded before the corresponding feedback interaction; every teaching/feedback presentation increments the exposure count once.

## Sensory projection validity

Projection profile: P0. P0 is the original fixed map. P1 applies a fixed absolute gain floor of 0.05 to every visual cell; it leaves audio gains and liquid equations unchanged and raises total visual input energy. It is a separate sensory-interface intervention, so P0 and P1 results must not be attributed solely to plasticity.

| Seed | Visual positions below sensory threshold |
|---:|---|
| 42 | none |
| 43 | 0 |
| 44 | none |
| 45 | -1 |
| 46 | none |

## Measured three-choice recognition

| Variant | Pairing | Fixed before | Fixed after | Calibrated after |
|---|---|---:|---:|---:|
| I0 | correct | 0.289 | 0.289 | 0.267 |

## Checkpoints

- **I0 three-choice gate:** FAIL/INCOMPLETE; mean 0.289, minimum seed 0.111, 5 seed(s).

## Scoring and control diagnostics

- I0: fixed scorer 0.289 before exposure, 0.289 after 50 correct experiences; cue-absent 0.333.

Stochastic sampled accuracy and deterministic accuracy are separate columns in the learning-curve CSV. Blank-audio trials keep the same balanced candidate roster and test for cue-free shortcuts.

## Retention and interference

- I0: independent reset 0.289; continuous delay scores (0s=0.222, 0s=0.333, 0s=0.556, 0s=0.333, 0s=0.222); interference 0.267. Seed-level rows are in the retention CSV.

## Production reference

The separately supervised fixed-I0 scalar readout achieved mean exact-within-half-unit accuracy 0.689 and MAE 0.462. Its labels are outside the liquid and its result is not evidence of internal association learning.

## Dynamics and compute

- I0: mean propagated events during learning 159675; weight updates 0; formed/pruned connections 0/0; assessment CPU 137.04 ms/trial.

## Interpretation and next experiment

A rise in fixed-matcher accuracy after correctly paired exposure, exceeding mismatched and single-modality controls on paired seeds, would support an effect of internal adaptation under this specific scoring rule. The calibrated comparator has label supervision and cannot independently establish that claim. Independent-episode assessment clears transient state and queue while retaining graph and weights; continuous assessments are labeled separately. Delay values are simulation time, while CPU and wall times are machine measurements. Event counts are scheduler events, not FLOPs. Observed queue size at the end of candidate branches is a lower bound on transient queue peaks. The 24-neuron CPU profile differs from the historical 64-neuron experiential report, so cross-report score differences cannot be attributed to assessment format alone. Control pairings are assessed at exposure 0 and the final count; the full intermediate learning curve is measured for correctly paired exposure.

The three-position experiment cannot test unseen sign-digit composition, two-digit grammar, broad distractor difficulty, or four/five-choice recognition. The composition CSV is intentionally empty until the three-position gate and paired-exposure advantage are established. FF_direct/F/G controls remain behind the same gate; comparing architectures using an invalid matcher would be difficult to interpret. Candidate waveforms and visual arrays retain the historical stimulus grammar. No new neuron or plasticity equation was added.

The smallest justified next experiment is determined by the measured fixed-matcher paired-versus-control contrast: if it fails, test whether the liquid's audio and visual response subspaces can be aligned by a fixed unsupervised transform before expanding position range. If it passes reproducibly, repeat with -9..+9 and counterbalanced unseen sign-digit pairs.
