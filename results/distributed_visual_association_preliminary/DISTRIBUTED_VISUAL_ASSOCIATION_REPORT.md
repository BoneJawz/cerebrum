# Distributed visual association experiment

This separately selectable Brian2/NumPy experiment used 18 shared LIF neurons, 25 binary visual input channels, three auditory spike channels, and the existing pair-based STDP rule. Development seeds were 42, 43, 44, 45, 46; prospective validation seeds were . Total wall time: 1794.7 s.

The three trained arrays are 5×5 vertical, horizontal, and main-diagonal lines. Every active visual cell produces one spike on its own input channel. A fixed Bernoulli visual projection is drawn from a seed-only rule before any training; it does not depend on the pattern or mapping. The auditory-to-associative pathway uses the unchanged STDP update. The network receives only active cell indices and auditory spike indices; pattern names and mapping labels remain in the experiment controller and analysis.

## 1. Cause of the earlier silent recalls

See `../recall_diagnostics/SILENT_RECALL_REPORT.md`. Replaying the preserved three-cell baseline produced four silent seed–cue cases out of 15; the peak-winning neuron remained below the −50 mV threshold, and Brian2 recorded no spikes. The result is subthreshold drive, not a missed monitor event or refractory block. The original five replicates reuse each seed–cue jitter and are duplicates for this question.

## 2–3. Distributed visual representation and distinguishability

Visual input is no longer wired to an identity output. Each of the 25 input cells projects randomly and sparsely to the shared population. `visual_pattern_distinguishability.csv` gives all pairwise pretraining cosine similarities per seed; the initial references are fixed before association. Lower similarity indicates more separable patterns. `visual_projection_connectivity.npz` records exact edges.

## 4–8. Cross-modal association, mappings, controls, and spikes

The primary analysis compares each audio-only associative response with all three visual-only reference vectors using cosine similarity. Recognition is the highest finite cosine; zero vectors are explicitly invalid and count as incorrect. Peak-depolarization and spike-count results are reported independently. The `identity` schedule pairs cue indices with the corresponding visual arrays; `remapped` teaches A→C, B→A, C→B. Conditions are plasticity off, paired-gated STDP, continuous STDP, and an unpaired control with a 500 ms separation. No output teacher is used.

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

Per-seed results are in `seed_level_summary.csv`; per cue/reference comparisons and response margins are in `population_similarity.csv` and `recognition_trials.csv`. The saved NPZ files contain raw population vectors. `synaptic_weight_history.csv` reports actual weight changes; a recognition increase without corresponding learned weight changes should be interpreted accordingly.

## 9–10. Retention and held-out visual variants

Every event runs in one persistent network with no membrane or synapse reset between experiences; 200 ms quiet intervals are used only between evaluation stimuli. Weight history across the 50-exposure run therefore also documents retention through intervening sensory presentations. Four held-out arrays (shifted lines and a removed pixel) were never used for association training. `generalization_results.csv` compares their visual representation to the relevant base reference and compares the trained auditory response to both. Similarity is representational compatibility, not proof of an abstract visual concept.

## 11–12. Engineered assumptions, compute, and next step

The visual representation depends on an engineered fixed random projection and a one-spike-per-active-pixel encoding. Only auditory afferent weights change through STDP. This is not a learned visual feature extractor. The associative population has no recurrent connections, no new neuron equations, and no target output pathway. If distinct visual references or spike responses fail, that failure is retained as evidence; no parameter search is performed on the prospective seeds.

Mean per-network runtime was 44.84 wall seconds and 44.80 CPU seconds; simulated-time/wall-time averaged 0.871×.

The smallest justified next step depends on results: if paired STDP changes the auditory representation toward the experienced visual reference across both mappings and controls, replicate only that locked configuration with a fresh prospective cohort and modestly varied encodings; if not, inspect auditory-to-association timing and the sparse visual reference geometry before considering any architecture change. No curiosity, language, or structural plasticity is included.

## Important interpretation note

This milestone tests an association between independently generated sensory activity patterns. The learned mapping is assessed by a frozen population-similarity readout, not a trained classifier. The fixed random projection and pattern construction are explicit sources of inductive bias. Do not compare its spike percentages directly to the prior three-output network's forced winner readout.
