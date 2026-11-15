# Experiential Visual Recognition Report

## Protocol

Seeds: 42, 43, 44, 45, 46. The retina has 199 fixed cells spanning -99..99; only -1, 0, +1 are assessed. One cell is on at a time. Each sign/digit waveform is seed-fixed across contexts. Independent noisy waveform variants are used for comparator calibration and evaluation. Every candidate runs for 0.20 simulation seconds from an identical frozen post-cue snapshot. Candidate order uses seeded rotations; the three-choice chance level is 1/3.

The fixed matcher sums elementwise products of the causal mean acoustic state and candidate-induced visual response. The calibrated matcher fits a 24-weight diagonal ridge compatibility function (alpha 1.0) on 6 independent noisy audio variants per position times three visual candidates, or 54 labeled pairs per seed. It is frozen during later experiences. Its supervised fit can learn the three associations externally, so only changes with the fixed matcher and paired-versus-control contrasts are relevant to internal-learning claims. Plasticity is frozen in all assessment clones. Online responses are recorded before the corresponding feedback interaction; every teaching/feedback presentation increments the exposure count once.

## Sensory projection validity

Projection profile: P1. P0 is the original fixed map. P1 applies a fixed absolute gain floor of 0.05 to every visual cell and rescales all visual gains to match P0's total projection energy; audio gains and liquid equations remain unchanged. It is a separate sensory-interface intervention, so P0 and P1 results must not be attributed solely to plasticity.

| Seed | Visual positions below sensory threshold |
|---:|---|
| 42 | none |
| 43 | none |
| 44 | none |
| 45 | none |
| 46 | none |

## Measured three-choice recognition

| Variant | Pairing | Fixed before | Fixed after | Calibrated after |
|---|---|---:|---:|---:|
| I0 | correct | 0.311 | 0.311 | 0.244 |

## Checkpoints

- **I0 three-choice gate:** FAIL/INCOMPLETE; mean 0.311, minimum seed 0.111, 5 seed(s).
- **Progression decision:** the predeclared three-position accuracy threshold was not met. Larger position sets, four/five-choice tasks, and unseen sign-digit composition are not interpreted from this run.

## Scoring and control diagnostics

- I0: fixed scorer 0.311 before exposure, 0.311 after 50 correct experiences; cue-absent 0.333.

Stochastic sampled accuracy and deterministic accuracy are separate columns in the learning-curve CSV. Softmax choice weights use fixed temperature 1.0 and have not been calibrated as probabilities. Blank-audio trials keep the same balanced candidate roster and test for cue-free shortcuts.

## Retention and interference

- I0: independent reset 0.311; continuous delay scores (0s=0.378, 0.1s=0.356, 0.5s=0.378, 1s=0.289, 5s=0.333); interference 0.400. Seed-level rows are in the retention CSV.

## Production reference

The separately supervised fixed-I0 scalar readout achieved mean exact-within-half-unit accuracy 0.689 and MAE 0.462. Its labels are outside the liquid and its result is not evidence of internal association learning. This production diagnostic has different supervision from the fixed recognition matcher, so their scores do not by themselves establish which task is intrinsically easier.

## Online pre-feedback choices

- I0: first 10 choices per seed (50 total) 0.280 accuracy; last 10 per seed (50 total) 0.300. These pre-feedback choices include transient state and must be compared with independent-episode retention before calling them learned associations.

## Dynamics and compute

- I0: mean propagated events during learning 158906; weight updates 0; formed/pruned connections 0/0; assessment CPU 171.01 ms/trial.
  First-trial traced Python allocation peak: mean 193.7 KiB across seeds (native NumPy allocation and process RSS excluded). These traced trials are included in the CPU mean and carry instrumentation overhead.

## Research questions

1. **Visual recognition from sound:** the three-choice table and seed-level CSV give the measured answer. No tested variant passed the predeclared gate.
2. **Recognition versus coordinate production:** the separate scalar production reference has different supervision, so this run does not identify which task is intrinsically easier.
3. **Repeated experience:** the fixed-scorer curves and paired controls show whether accuracy rises with counted presentations. A rise without a correct-pairing advantage is not association evidence.
4. **Location of any improvement:** the external calibrated comparator is labeled and frozen; fixed-comparator changes can be attributed to liquid-state or graph changes, but require pairing controls to show useful association.
5. **Weight adaptation:** compare I1 with I0 and their correct-versus-mismatched final scores; weight updates alone are reported separately from recognition.
6. **Structural adaptation:** compare I2 with I1 and I2_rewire_fixed using accuracy, formed/pruned edges, and event cost. Rewiring alone is not a benefit without a pairing-specific gain.
7. **Choice difficulty:** two- and three-choice scores are measured separately. Four/five-choice and broad distractor difficulty are gated off in the three-position subset.
8. **Retention:** independent reset, continuous simulated delays, and interference results are separate. Scores near chance cannot establish retained association.
9. **Continuous learning:** main liquid plasticity operates during the counted feedback experiences with no global optimizer epochs; online choices precede feedback. This is continuous updating, not proof of useful learning.
10. **Efficiency and conventional controls:** liquid CPU, events, and traced memory are measured. FF_direct/F/G are not compared after the candidate-assessment gate fails, so no architecture superiority claim is possible.
11. **Component composition:** unseen sign-digit combinations were not assessed because earlier gates failed; familiar three-position performance cannot establish composition.
12. **Next capability:** audit audio-state information, candidate visual separation, and plasticized graph paths at the same three positions before expanding the spatial range.


## Interpretation and next experiment

A rise in fixed-matcher accuracy after correctly paired exposure, exceeding mismatched and single-modality controls on paired seeds, would support an effect of internal adaptation under this specific scoring rule. The calibrated comparator has label supervision and cannot independently establish that claim. Independent-episode assessment clears transient state and queue while retaining graph and weights; continuous assessments are labeled separately. Delay values are simulation time, while CPU and wall times are machine measurements. Event counts are scheduler events, not FLOPs. Observed queue size at the end of candidate branches is a lower bound on transient queue peaks. The 24-neuron CPU profile differs from the historical 64-neuron experiential report, so cross-report score differences cannot be attributed to assessment format alone. Control pairings are assessed at exposure 0 and the final count; the full intermediate learning curve is measured for correctly paired exposure.

The three-position experiment cannot test unseen sign-digit composition, two-digit grammar, broad distractor difficulty, or four/five-choice recognition. The composition CSV is intentionally empty until the three-position gate and paired-exposure advantage are established. FF_direct/F/G controls remain behind the same gate; comparing architectures using an invalid matcher would be difficult to interpret. Candidate waveforms and visual arrays retain the historical stimulus grammar. No new neuron or plasticity equation was added.

The smallest justified next experiment is an information-path audit in the same three-position task: measure separate held-out decodability of the audio trajectory and each visual branch, then check whether plasticized edges lie on paths that affect the frozen candidate score. Predeclare one fixed, label-free temporal summary or cross-modal normalization and repeat the five-seed paired-versus-mismatched test. Expand to -9..+9 only after that competence and pairing-advantage checkpoint passes.
