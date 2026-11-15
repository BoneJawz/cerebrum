# Internal output failure diagnosis

## Summary

The original silent output result had an upstream cause: the 18-cell associative network began with fresh random auditory weights and its auditory STDP was frozen throughout output training. Auditory-only recall therefore produced no associative spikes. Audiovisual teaching still drove associative spikes through vision, so the output STDP weights changed without learning from activity that later occurred during auditory-only recall.

Training the upstream auditory pathway first fixed that part of the problem. With 50 paired experiences per cue, both seeds produced cue-specific associative spikes and 3/3 correct external cosine matches. However, the output population still made **0/6 correct spike-count selections** for either seed. Five of six paired-STDP trials were ties; one was silent. The learned output layer also produced no better spike-count accuracy than its plasticity-disabled control.

**Conclusion: D. Multiple interacting failures.** The fresh upstream condition lacks auditory-driven associative spikes. Once that is corrected, output cells receive broad, nearly simultaneous excitation and all three tend to emit the same one spike. Output weights show some target-specific change, but ordinary sensory-driven output spikes occur in all three cells during teaching, diluting the separate teacher pulse and leaving spike-count responses tied.

## Protocol and code audit

The active `DistributedVisualAssociator` initializes each auditory-to-associative weight independently from `Uniform(0.05, 0.15)` using a seed-derived RNG. It does not load a learned checkpoint. The original pilot calls `set_plasticity(False)` before output teaching, so those auditory weights remain fresh and frozen. Visual input can still drive the associative population during audiovisual teaching.

The diagnostic compared that fresh/frozen condition with the established paired-gated protocol recovered from Git history: 50 audiovisual experiences per cue, auditory STDP enabled during each 60 ms paired event, followed by 50 ms rest with plasticity disabled. It then froze the learned upstream weights before output training. Both conditions used two seeds, identity pairings, five output-teaching experiences per cue, and identical output parameters. Plasticity-disabled output controls received the same sensory events and target teacher pulse. Output-control order was alternated by seed; only output weights, traces, and output voltages were reset between those control phases. The associative pathway was not reset.

The teacher was a direct 55 mV depolarizing pulse to the designated output neuron's voltage at 40 ms. It remained a separate supervised operation, not an auditory or visual input. The output projection used the existing pair-STDP equations, gain 60 mV, and the original `[0.05, 0.15]` initial weight range.

## Upstream associative activity

| Seed | Upstream condition | Auditory-only associative spikes, cues A/B/C | Active cells, A/B/C | Auditory weight L1 change | Frozen cosine matches |
|---:|---|---|---|---:|---:|
| 42 | Fresh, frozen | 0 / 0 / 0 | 0 / 0 / 0 | 0.000 | 1/3 |
| 43 | Fresh, frozen | 0 / 0 / 0 | 0 / 0 / 0 | 0.000 | 2/3 |
| 42 | 50 paired experiences/cue | 4 / 6 / 1 | 4 / 6 / 1 | 8.576 | 3/3 |
| 43 | 50 paired experiences/cue | 3 / 5 / 4 | 3 / 5 / 4 | 7.917 | 3/3 |

Fresh auditory-only maximum associative depolarization was only 8.38–8.97 mV above rest, and no cells spiked. After paired upstream learning, the maximum was 21.31–28.39 mV above rest. The auditory-only spike patterns were distinct: cosine similarities between cue spike-count vectors were A–B `0.000` for both seeds, A–C `0.000`/`0.289`, and B–C `0.408`/`0.447` for seeds 42/43. The learned upstream condition passed the predeclared requirement of three unique, correct frozen cosine matches for each seed before the output population was attached.

In the five audiovisual output-teaching experiences per cue, every associative neuron that later spiked during auditory-only recall had also spiked during teaching. The teaching-to-recall Jaccard overlaps by cue were `0.333, 0.429, 0.111` for seed 42 and `0.250, 0.455, 0.444` for seed 43. The small cue-2 response for seed 42 is especially fragile: only one associative neuron fired at recall.

## Output synapses and teaching timing

Paired output STDP changed the full 18×3 matrix by L1 `6.025` (fresh upstream) and `6.188` (learned upstream) for seed 42, and `5.509` and `5.661` for seed 43. The disabled controls changed the matrix by exactly zero. These totals alone would overstate the result: cue-specific weight increments were only moderately larger for the intended output than for competitors.

Sum of weight increments across the 18 presynaptic cells, listed by output A/B/C:

| Seed | Taught cue | Weight delta to outputs A / B / C | Intended minus strongest competitor |
|---:|---|---:|---:|
| 42 | A | 0.883 / 0.544 / 0.541 | +0.339 |
| 42 | B | 0.732 / 1.134 / 0.728 | +0.401 |
| 42 | C | 0.462 / 0.468 / 0.697 | +0.229 |
| 43 | A | 0.834 / 0.504 / 0.507 | +0.327 |
| 43 | B | 0.593 / 0.919 / 0.593 | +0.326 |
| 43 | C | 0.480 / 0.483 / 0.749 | +0.267 |

The target column gains more weight, but every competing output also gains substantial weight. During learned-upstream teaching, associative presynaptic events occurred at 10.1 ms and 25.1 ms; output postsynaptic events occurred at 10.2 ms and 25.2 ms, with the target teacher spike at 40.0 ms. Thus local STDP saw both pre-before-post and post-before-pre pairs (observed intervals −14.9 to +29.9 ms). Crucially, sensory-driven postsynaptic spikes occurred in all three outputs before the target teacher pulse. Across the 15 teaching events per seed, each output spiked 30 times for seed 42 and 34 times for seed 43. The target-specific teacher contributed only five of those spikes to that output; the other output cells also fired from sensory drive. Those untargeted postsynaptic events participate in the same local STDP rule.

In the fresh/frozen condition, the associative cells spiked at 25.1 ms during audiovisual teaching, and all three output cells spiked at 25.2 ms. None of those associative cells fired during auditory-only recall. This explains why that condition accumulated about six units of output weight L1 change yet still delivered no associative spikes to the output layer at recall.

## Auditory-only output responses

Counts below are for output neurons A/B/C. Expected outputs are A/B/C for cues A/B/C.

| Seed | Upstream | Output STDP | Cue A | Cue B | Cue C | Correct selections |
|---:|---|---|---|---|---|---:|
| 42 | Fresh/frozen | Paired | 0/0/0 silent | 0/0/0 silent | 0/0/0 silent | 0/3 |
| 42 | Fresh/frozen | Disabled | 0/0/0 silent | 0/0/0 silent | 0/0/0 silent | 0/3 |
| 42 | Paired 50/cue | Paired | 1/1/1 tie | 1/1/1 tie | 0/0/0 silent | 0/3 |
| 42 | Paired 50/cue | Disabled | 1/1/1 tie | 1/1/1 tie | 0/0/0 silent | 0/3 |
| 43 | Fresh/frozen | Paired | 0/0/0 silent | 0/0/0 silent | 0/0/0 silent | 0/3 |
| 43 | Fresh/frozen | Disabled | 0/0/0 silent | 0/0/0 silent | 0/0/0 silent | 0/3 |
| 43 | Paired 50/cue | Paired | 1/1/1 tie | 1/1/1 tie | 1/1/1 tie | 0/3 |
| 43 | Paired 50/cue | Disabled | 0/0/0 silent | 1/1/1 tie | 1/1/1 tie | 0/3 |

Output-voltage peaks did carry a graded preference: the intended neuron had the highest sampled peak in five of six learned-upstream paired trials. Intended-minus-best-competitor margins were `+5.40, +8.61, +2.07 mV` for seed 42 and `+3.80, −0.04, +4.88 mV` for seed 43. But the spike-count readout remained tied whenever the population fired: each cell emitted one spike. In the single silent learned-upstream paired trial (seed 42, cue C), peaks were `−52.08, −53.70, −50.002 mV`, respectively `2.08, 3.70, and 0.002 mV` below the strict −50 mV threshold.

For learned-upstream paired trials, the estimated sum of actual incoming synaptic voltage jumps ranged from 16.30 to 100.80 mV across outputs and cues. This is the sum of `weight × 60 mV` for each recorded associative presynaptic event; it does not account for leak between events. Most trials therefore had enough drive to make every output fire, not too little drive to make any output fire. Output STDP did not improve correct spike selection over the frozen control.

The voltage monitor records every 0.1 ms with Brian2's default `when="start"` scheduling. Its maximum is the largest sampled voltage, not a continuous-time peak. Brian2's SpikeMonitor is the authoritative record of threshold crossings; output spike IDs/times, sampled traces, and threshold margins are retained separately in the CSV files.

## Diagnosis and next experiment

The original pilot combined two problems. First, fresh frozen auditory weights produced no upstream auditory spikes, so visual-driven teaching activity could not be replayed by audio. Second, after upstream learning was verified, associative activity did reach the output, but all-to-all 60 mV output synapses drove all output neurons together. The teacher pulse added a modest target-specific weight bias, while sensory-driven postsynaptic firing in every output also drove STDP. Counts saturated at one spike per output, obscuring the graded voltage preference. The single near-threshold silent trial shows a residual excitation boundary, but insufficient excitation is not the dominant learned-upstream failure.

The smallest justified next experiment is a prespecified control that gates output STDP to the teacher-designated output and teacher time window while leaving forward sensory drive, weights, neuron equations, gain, and threshold fixed. Compare it with the current ungated local-STDP condition on the same verified upstream representations. This isolates whether untargeted sensory-driven output spikes are causing the broad potentiation. No such change was made here.

## Artifacts and reproduction

The diagnostic script is `experiments/internal_output_diagnostic.py`. It writes:

- `activity_summary.csv`: per-event associative/output spikes and voltage profiles, cue similarities, teaching/recall overlap, and spike timing.
- `weight_analysis.csv`: complete initial/final synaptic matrices and cue-specific weight deltas.
- `output_voltage_summary.csv`: output counts/status, sampled peak voltage, threshold distance, presynaptic counts, delivered excitation estimate, and response margin.

The final run used seeds 42 and 43 and took 90.73 seconds. A gate-scheduling issue in an earlier diagnostic attempt was caught and fixed before the reported run; the final script enables output STDP for each of the five paired experiences per cue and disables it during rest and recall.
