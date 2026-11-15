# Distributed visual association experiment

This separately selectable Brian2/NumPy experiment used 18 shared LIF neurons, 25 binary visual input channels, three auditory spike channels, and the existing pair-based STDP rule. Development seeds were 42, 43, 44, 45, 46; prospective validation seeds were 101, 102, 103, 104, 105. Total wall time: 3540.3 s.

The three trained arrays are 5×5 vertical, horizontal, and main-diagonal lines. Every active visual cell produces one spike on its own input channel. A fixed Bernoulli visual projection is drawn from a seed-only rule before any training; it does not depend on the pattern or mapping. The auditory-to-associative pathway uses the unchanged STDP update. The network receives only active cell indices and auditory spike indices; pattern names and mapping labels remain in the experiment controller and analysis.

For live inspection, `python -m mdl.live_dashboard --distributed` selects the persistent 5×5 dashboard. It displays the actual visual array, the shared population voltage traces and sensory/association raster, and auditory synaptic weights. Simulation advances in fixed 10 ms chunks independently of plot refresh.

## 1. Cause of the earlier silent recalls

See `../recall_diagnostics/SILENT_RECALL_REPORT.md`. Replaying the preserved three-cell baseline produced four silent seed–cue cases out of 15; the peak-winning neuron remained below the −50 mV threshold, and Brian2 recorded no spikes. The result is subthreshold drive, not a missed monitor event or refractory block. The original five replicates reuse each seed–cue jitter and are duplicates for this question.

## 2–3. Distributed visual representation and distinguishability

Visual input is no longer wired to an identity output. Each of the 25 input cells projects randomly and sparsely to the shared population. `visual_pattern_distinguishability.csv` gives all pairwise pretraining cosine similarities per seed; the initial references are fixed before association. Mean pairwise voltage cosine was 0.798 on development seeds and 0.796 on validation seeds, so the patterns were distinguishable but their population codes were correlated. `visual_projection_connectivity.npz` records exact edges.

## 4–8. Cross-modal association, mappings, controls, and spikes

The primary analysis compares each audio-only associative response with all three visual-only reference vectors using cosine similarity. Recognition is the highest finite cosine; zero vectors are explicitly invalid and count as incorrect. Peak-depolarization and spike-count results are reported independently. The `identity` schedule pairs cue indices with the corresponding visual arrays; `remapped` teaches A→C, B→A, C→B. Conditions are plasticity off, paired-gated STDP, continuous STDP, and an unpaired control: audio-only event, 250 ms silence, visual-only event, then the same silence before the next auditory event; gaps exceed 12 STDP time constants. No output teacher is used.

Mean voltage-based recognition across seeds by mapping and exposure:

| Seed role | Mapping | Condition | Exposures | Voltage cosine | Spike cosine | Spike silent fraction |
|---|---|---|---:|---:|---:|---:|
| development | identity | plasticity_off | 0 | 40.0% | 0.0% | 100.0% |
| development | identity | plasticity_off | 5 | 40.0% | 0.0% | 100.0% |
| development | identity | plasticity_off | 20 | 40.0% | 0.0% | 100.0% |
| development | identity | plasticity_off | 50 | 40.0% | 0.0% | 100.0% |
| development | identity | paired_gated | 0 | 40.0% | 0.0% | 100.0% |
| development | identity | paired_gated | 5 | 66.7% | 0.0% | 100.0% |
| development | identity | paired_gated | 20 | 100.0% | 0.0% | 100.0% |
| development | identity | paired_gated | 50 | 100.0% | 100.0% | 0.0% |
| development | identity | continuous | 0 | 40.0% | 0.0% | 100.0% |
| development | identity | continuous | 5 | 66.7% | 0.0% | 100.0% |
| development | identity | continuous | 20 | 100.0% | 0.0% | 100.0% |
| development | identity | continuous | 50 | 100.0% | 100.0% | 0.0% |
| development | identity | unpaired | 0 | 40.0% | 0.0% | 100.0% |
| development | identity | unpaired | 5 | 40.0% | 0.0% | 100.0% |
| development | identity | unpaired | 20 | 40.0% | 0.0% | 100.0% |
| development | identity | unpaired | 50 | 40.0% | 0.0% | 100.0% |
| development | remapped | plasticity_off | 0 | 20.0% | 0.0% | 100.0% |
| development | remapped | plasticity_off | 5 | 20.0% | 0.0% | 100.0% |
| development | remapped | plasticity_off | 20 | 20.0% | 0.0% | 100.0% |
| development | remapped | plasticity_off | 50 | 20.0% | 0.0% | 100.0% |
| development | remapped | paired_gated | 0 | 20.0% | 0.0% | 100.0% |
| development | remapped | paired_gated | 5 | 40.0% | 0.0% | 100.0% |
| development | remapped | paired_gated | 20 | 100.0% | 0.0% | 100.0% |
| development | remapped | paired_gated | 50 | 100.0% | 80.0% | 6.7% |
| development | remapped | continuous | 0 | 20.0% | 0.0% | 100.0% |
| development | remapped | continuous | 5 | 40.0% | 0.0% | 100.0% |
| development | remapped | continuous | 20 | 100.0% | 0.0% | 100.0% |
| development | remapped | continuous | 50 | 100.0% | 80.0% | 6.7% |
| development | remapped | unpaired | 0 | 20.0% | 0.0% | 100.0% |
| development | remapped | unpaired | 5 | 20.0% | 0.0% | 100.0% |
| development | remapped | unpaired | 20 | 20.0% | 0.0% | 100.0% |
| development | remapped | unpaired | 50 | 20.0% | 0.0% | 100.0% |
| validation | identity | plasticity_off | 0 | 33.3% | 0.0% | 100.0% |
| validation | identity | plasticity_off | 5 | 33.3% | 0.0% | 100.0% |
| validation | identity | plasticity_off | 20 | 33.3% | 0.0% | 100.0% |
| validation | identity | plasticity_off | 50 | 33.3% | 0.0% | 100.0% |
| validation | identity | paired_gated | 0 | 33.3% | 0.0% | 100.0% |
| validation | identity | paired_gated | 5 | 60.0% | 0.0% | 100.0% |
| validation | identity | paired_gated | 20 | 86.7% | 0.0% | 100.0% |
| validation | identity | paired_gated | 50 | 100.0% | 100.0% | 0.0% |
| validation | identity | continuous | 0 | 33.3% | 0.0% | 100.0% |
| validation | identity | continuous | 5 | 60.0% | 0.0% | 100.0% |
| validation | identity | continuous | 20 | 86.7% | 0.0% | 100.0% |
| validation | identity | continuous | 50 | 100.0% | 100.0% | 0.0% |
| validation | identity | unpaired | 0 | 33.3% | 0.0% | 100.0% |
| validation | identity | unpaired | 5 | 33.3% | 0.0% | 100.0% |
| validation | identity | unpaired | 20 | 33.3% | 0.0% | 100.0% |
| validation | identity | unpaired | 50 | 33.3% | 0.0% | 100.0% |
| validation | remapped | plasticity_off | 0 | 46.7% | 0.0% | 100.0% |
| validation | remapped | plasticity_off | 5 | 46.7% | 0.0% | 100.0% |
| validation | remapped | plasticity_off | 20 | 46.7% | 0.0% | 100.0% |
| validation | remapped | plasticity_off | 50 | 46.7% | 0.0% | 100.0% |
| validation | remapped | paired_gated | 0 | 46.7% | 0.0% | 100.0% |
| validation | remapped | paired_gated | 5 | 46.7% | 0.0% | 100.0% |
| validation | remapped | paired_gated | 20 | 93.3% | 0.0% | 100.0% |
| validation | remapped | paired_gated | 50 | 100.0% | 100.0% | 0.0% |
| validation | remapped | continuous | 0 | 46.7% | 0.0% | 100.0% |
| validation | remapped | continuous | 5 | 46.7% | 0.0% | 100.0% |
| validation | remapped | continuous | 20 | 93.3% | 0.0% | 100.0% |
| validation | remapped | continuous | 50 | 100.0% | 100.0% | 0.0% |
| validation | remapped | unpaired | 0 | 46.7% | 0.0% | 100.0% |
| validation | remapped | unpaired | 5 | 46.7% | 0.0% | 100.0% |
| validation | remapped | unpaired | 20 | 46.7% | 0.0% | 100.0% |
| validation | remapped | unpaired | 50 | 46.7% | 0.0% | 100.0% |

Per-seed spike-based outcomes at 50 experiences:

| Cohort | Seed | Identity paired | Remapped paired | Identity no-learning | Identity unpaired |
|---|---:|---:|---:|---:|---:|
| development | 42 | 100% | 67% | 0% | 0% |
| development | 43 | 100% | 100% | 0% | 0% |
| development | 44 | 100% | 67% | 0% | 0% |
| development | 45 | 100% | 100% | 0% | 0% |
| development | 46 | 100% | 67% | 0% | 0% |
| validation | 101 | 100% | 100% | 0% | 0% |
| validation | 102 | 100% | 100% | 0% | 0% |
| validation | 103 | 100% | 100% | 0% | 0% |
| validation | 104 | 100% | 100% | 0% | 0% |
| validation | 105 | 100% | 100% | 0% | 0% |

Per-seed results are in `seed_level_summary.csv`; per cue/reference comparisons and response margins are in `population_similarity.csv` and `recognition_trials.csv`. The saved NPZ files contain raw population vectors. `synaptic_weight_history.csv` reports actual weight changes; a recognition increase without corresponding learned weight changes should be interpreted accordingly.

## 9–10. Retention and held-out visual variants

Every event runs in one persistent network with no membrane or synapse reset between experiences. Paired experiences are separated by 50 ms quiet periods, and evaluation presentations by 200 ms. There is no spontaneous input or weight-decay term, so quiet periods do not change learned synaptic weights; responses at later exposure checkpoints remained accurate. A dedicated repeated same-cue probe after a long post-training quiet interval was not included and is a useful next check. Four held-out arrays (shifted lines and a removed pixel) were never used for association training. On prospective identity-pair trials, auditory-to-base cosine averaged 0.947, compared with 0.806 to the related held-out visual arrays; held-out visual arrays themselves averaged 0.805 cosine to their base pattern. Similarity is representational compatibility, not proof of an abstract visual concept.

## 11–12. Engineered assumptions, compute, and next step

The visual representation depends on an engineered fixed random projection and a one-spike-per-active-pixel encoding. Only auditory afferent weights change through STDP. This is not a learned visual feature extractor. The associative population has no recurrent connections, no new neuron equations, and no target output pathway. If distinct visual references or spike responses fail, that failure is retained as evidence; no parameter search is performed on the prospective seeds.

Mean per-network runtime was 44.22 wall seconds and 44.03 CPU seconds; simulated-time/wall-time averaged 0.853×.

At 50 experiences, paired STDP changed total auditory weight L1 by 7.591 on development seeds and 8.201 on validation seeds; the corrected unpaired control changed weights by 0.00001 and 0.00001. Paired-gated and continuous conditions had the same outcomes. Since the validation cohort reproduced association learning but held-out similarity was lower, the smallest next experiment is a preregistered translation-tolerance study with the current network fixed, plus repeated same-cue recall after a 500 ms post-training quiet interval. Do not enlarge the network or change STDP until that targeted test clarifies the current limitation. No curiosity, language, or structural plasticity is included.

## Important interpretation note

This milestone tests an association between independently generated sensory activity patterns. The learned mapping is assessed by a frozen population-similarity readout, not a trained classifier. The fixed random projection and pattern construction are explicit sources of inductive bias. Do not compare its spike percentages directly to the prior three-output network's forced winner readout.
