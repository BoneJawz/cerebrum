# Spatial grounding experiment report

## Experimental protocol

The environment uses a static 21-cell retina spanning -10 through +10, with one occupied cell. The scored positions are -5 through +5. A shared, seed-fixed vocabulary provides equal-duration, equal-RMS arbitrary synthetic tokens. Nonzero instructions are sign, silence, magnitude; zero uses one canonical ZERO token. Audio and vision timestamps are deterministic. No ground-truth coordinate or token identity is passed in `Observation`.

This run used 5 seed(s), 1 audiovisual exposure(s) per trained position, and 1 audio-only test episode(s) per position. The five-seed checkpoint fits its supervised readout on labeled audiovisual liquid states after unsupervised exposure; the raw ridge control uses the same labeled examples and generic waveform/retina features. Visual-only probes and frozen-readout transfer were not run in checkpoint-only mode.

This is a minimum-exposure competence checkpoint, not a training-size sweep. Inputs are noise-free and token waveforms are fixed; nuisance-phase and sample-efficiency curves remain future work after the association bottleneck is diagnosed.

The predeclared all-combinations gate is exact accuracy >= 80%. Composition is not interpreted when the gate fails.

## Measured all-combinations association

| Model/readout | Exact audio-only accuracy | MAE (position units) |
|---|---:|---:|
| I0 | 0.255 | 2.240 |
| raw_ridge | 0.273 | 1.142 |
| I1 | 0.127 | 3.142 |
| I2 | 0.055 | 4.075 |
| I2_rewire_fixed | 0.145 | 2.825 |

The accuracy values are seed means where multiple seeds are present. Each output CSV retains seed-level rows. ZERO is included in this all-combinations control but excluded from sign/magnitude composition scoring.

This is a checkpoint-only run: held-out composition was not executed. The held-out partition is saved for a later dedicated run.

## Compute observations

| Variant | Adaptation wall seconds | Median audio episode wall seconds | Propagated events | Final active edges |
|---|---:|---:|---:|---:|
| I0 | 8.89 | 0.781 | 427073 | 384.0 |
| I1 | 13.75 | 0.701 | 368038 | 384.0 |
| I2 | 20.15 | 1.593 | 532322 | 509.2 |
| I2_rewire_fixed | 10.74 | 1.098 | 499763 | 509.2 |

Event time is distinct from wall time. Event counts are not FLOPs. Wall latency and CPU time are local Python measurements; traced allocation excludes native NumPy/BLAS memory. Structural rewiring may increase event traffic, so edge sparsity alone is not a compute advantage.

## Scientific assumptions and failure modes

The expected baseline is clear token separability and visual-position recoverability, followed by audio-only position prediction that passes the all-combinations gate. This is a falsifiable expectation, not an assumed property of liquid dynamics. The main confounds are (a) accidental correlations between token identity and simple spectrum or amplitude summaries, (b) audiovisual readout fitting that may rely on the visual channel during training, (c) a readout that cannot decode a useful representation despite information being present, (d) plasticity changing activity or event traffic without improving task information, and (e) event cascades making sparse topology computationally expensive. Equal-duration/RMS checks and randomized token assignment address only some acoustic shortcuts; this noise-free benchmark does not yet test robustness to nuisance variation.

Scope distinction: in this attached task, the complete sound sequence carries the target position at audio-only test. It tests audiovisual grounding during learning and recombination of acoustic components; it is not an inference-time task whose target depends on combining both current sensory streams. Do not use it alone as evidence of genuine multimodal dependency. A complementary static task for a later milestone is a balanced match/mismatch judgment between visual side-of-center and acoustic sign; neither modality alone predicts the target when the factors are independently balanced.

Acceptance for proceeding to held-out composition is at least 80% exact audio-only position accuracy on the all-combinations control, reported across at least five seeds. Below that threshold, diagnose token probes, readout source, cross-modal alignment, and event cost before increasing task complexity. A pass would justify a separate held-out training run; it would not itself demonstrate composition.

Direct observations are limited to the recorded seeds and training schedule. Token feature distance establishes only that this fixed token set is separable under the current feature transform; it is not held-out-token generalization. Visual probe accuracy is supervised and does not show that position labels entered the liquid. A trained AV-state readout predicting from audio-only test states is readout-supported transfer, not proof that the liquid formed explicit symbolic concepts.

## Requested scientific questions

1. **Individual acoustic tokens:** all token feature vectors were distinct (mean minimum pairwise distance 2.258); this is a fixed-vocabulary feature check, not held-out token recognition.
2. **Visual positions:** supervised visual-position probe was not measured in the checkpoint-only run.
3. **AV association:** audio-only exact accuracy is shown above. The liquid variants did not reach the competence gate in this run.
4. **I1/I2 versus I0:** neither adaptive variant improved on I0 across the five-seed means; all liquid variants remained well below the 80% gate.
5. **Unseen combinations:** not run in checkpoint-only mode; the all-combinations gate failed, so composition remains deferred.
6-7. **Sign and magnitude interventions:** not run because the all-combinations checkpoint failed.
8. **Cross-modal state alignment:** mean within-minus-between cosine was not measured in the checkpoint-only run. No frozen transfer probe was run in checkpoint-only mode.
9. **Structural adaptation and retention:** no improvement is established. Retention probes were not run in checkpoint-only mode; event traffic and topology costs are in the compute CSV.
10. **Compute cost:** per-variant adaptation time, inference latency, propagated events, and edge counts are tabulated above. I2's changed topology did not imply a lower event count.
11. **Instantaneous transitions:** not implemented or tested; this remains behind the association gate.
12. **Smallest next step:** compare audio-only supervised readout fitting against AV-state readout fitting, add token-level held-out probe examples with nuisance phase variation, and vary repeated AV exposure. This isolates whether failure comes from feature separability, cross-modal state alignment, or readout source before adding motion.

The five-seed all-combinations gate failed for every liquid variant and the raw-feature ridge baseline. This run does not support a compositional generalization claim. Do not add motion or transitions yet; diagnose audio-only association with token-level probes, an audio-only supervised readout control, and a matched readout-source comparison.

This milestone currently includes I0/I1/I2 and I2_rewire_fixed plus a linear raw-feature baseline. It does not yet include trainable F/G networks or instantaneous transitions; those are deferred until the all-combinations gate passes, as required by the staged protocol.

## Artifacts and reproduction

The directory contains the vocabulary CSV, configuration, exact seed partitions, model snapshots, token, association, and compute CSVs. Composition, counterfactual, cross-modal, and memory CSVs are empty because those studies were correctly deferred after the competence gate failed. The PNGs include explicit no-measurement placeholders for those stages. Reproduce with the command recorded in `spatial_grounding_config.json` and `python -m sensory_lab.spatial_grounding`.
