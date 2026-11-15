# Model I — Structurally Adaptive Liquid (SAL)

## Scope of this run

This is the Phase 1/2 prototype and Experiment A only. It evaluates event-queue dynamics and fading memory for five independent acoustic cue classes. Audio reaches the liquid as waveform-derived features; vision is blanked in this isolated memory test. Direction, speed, cue IDs and physical state remain experiment-side labels. The output does not claim motion prediction, audio-visual association, structural advantage, or biological validity.

## Architecture

The liquid uses a directed sparse adjacency map over 64 unlayered units. Every ordered pair is eligible except disabled self-edges; actual edges start with mean out-degree 6. Fixed seed-controlled sensory projections target up to eight units. Each unit uses a bounded tanh response with exponential leaky decay, local adaptation and a recent incoming-activity trace. A deterministic min-heap schedules transmission and internal decay events by (simulation time, insertion sequence). Neurons decay mathematically between events; the decay is computed on the next event or a readout snapshot. Snapshot reads scan the 64 states but do not mutate or propagate them.

I0 freezes weights/topology and fits only a post-hoc ridge readout. I1 adds bounded local correlation-based weight plasticity using pre/post activation at transmission. I2 adds uniformly sampled candidate-pair utility from decayed local coactivity, plus conservative weak-and-inactive pruning. Every potential directed pair can be sampled; candidate search does not scan a dense weight matrix. Topology changes are interval-limited and obey degree/edge caps. This initial rule is a simple testable heuristic, not a claim of biological synaptic plasticity.

## Measured results

Mean cue accuracy by delay across the requested seeds:

| Variant | Delay 0 s | 0.1 s | 0.25 s | 0.5 s | 1.0 s |
|---|---:|---:|---:|---:|---:|
| I0 | 0.943 ± 0.060 | 0.630 ± 0.321 | 0.473 ± 0.268 | 0.497 ± 0.252 | 0.380 ± 0.200 |
| I1 | 0.947 ± 0.055 | 0.633 ± 0.322 | 0.470 ± 0.271 | 0.497 ± 0.252 | 0.377 ± 0.197 |
| I2 | 0.937 ± 0.069 | 0.697 ± 0.302 | 0.533 ± 0.275 | 0.507 ± 0.225 | 0.377 ± 0.159 |
| I2_rewire_fixed | 0.940 ± 0.069 | 0.630 ± 0.313 | 0.500 ± 0.276 | 0.510 ± 0.229 | 0.373 ± 0.179 |

Topology and compute means across seeds:

| Variant | Final active edges | Formed | Pruned | Propagated events | Training/exposure + readout-state wall s | Median inference ms at delay 0 | p95 inference ms at delay 1 | Structural-learning wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0 | 384.0 | 0.0 | 0.0 | 102129 | 0.635 | 3.047 | 398.7 | 0.0000 |
| I1 | 384.0 | 0.0 | 0.0 | 101837 | 0.696 | 3.070 | 387.7 | 0.0000 |
| I2 | 386.2 | 29.4 | 27.2 | 433098 | 0.915 | 3.528 | 748.4 | 0.0455 |
| I2_rewire_fixed | 383.6 | 26.8 | 27.2 | 254039 | 0.702 | 3.140 | 556.0 | 0.0450 |

The table reports measured post-hoc readout accuracy; it is not a measurement of semantic understanding. I1/I2 change only during the adaptation exposure phase. The ridge readout is fitted on new, separate episodes with liquid plasticity frozen. Training wall time includes adaptation and readout-state collection; the separate readout solver time is in `liquid_compute_results.csv`.

`liquid_dynamics_results.csv` also includes a paired 0.002-SD waveform perturbation probe. It changes only the raw audio chunk and reports hidden-state displacement; this is one pair per seed, not a robustness estimate.

## Observations and interpretation

I0 and I1 use the same graph and remain very close on this cue task; the current local weight rule produces little improvement in retention. I2 forms and prunes edges, but increases event propagation and long-delay latency without a clear accuracy gain. Across seeds, all variants lose cue-class accuracy toward chance by the 1-second delay, while some retain high activity norms and many scheduled events. Persistent activity therefore does not by itself show persistent cue information. The I2 event cascade also demonstrates that sparse adjacency does not guarantee lower computation; mean degree remains near six, but recurrent event traffic can dominate cost.

Seed-level variability is substantial, particularly at 0.1–0.5 s. This is a small diagnostic set with five seeds, not a tuned classifier benchmark. No claim of adaptation benefit is supported. Before Experiment B, the next justified step is to stabilize and characterize long-delay event rates, then test whether the fixed reservoir has recoverable cue information under a more controlled readout split and matched compute budget.

## Numerical and compute safeguards

Configured limits: 512 total edges, 8 outgoing edges per unit, queue cap 100000, and 250000 processed events per advance. Activation and adaptation remain bounded; edge weights are clipped to ±1.0. A queue overflow raises a clear runtime error; event insertions beyond queue capacity are counted as drops. `peak_traced_bytes_model_run` includes Python objects and snapshot serialization but omits native NumPy allocations and is not process RSS. CPU measurements are machine-specific and include Python heap scheduling.

## Interpretation and unresolved questions

I0 is the fixed-reservoir baseline; I1 tests whether this local Hebbian-like update changes cue retention; I2 tests whether sampled coactivity rewiring adds anything beyond weight changes. The study must be read together with topology and compute CSVs: fewer state updates do not imply lower CPU cost, and topology statistics do not imply useful computation. Five seeds and a single cue-memory task are insufficient to establish robust advantage. The next phase should verify cue association under a fixed readout budget and compare adaptation exposure separately from readout fitting. Motion prediction and strict/familiar composition remain unrun pending stable baseline dynamics.

## Reproduction

Configuration: `liquid_experiment_config.json`; snapshots include graph edges, current states, event queue, fixed input projections and RNG state. Run with `python -m sensory_lab.liquid_experiment --output results/liquid`.
