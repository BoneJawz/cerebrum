# MDL codebase and neural architecture audit

## Scope and method

Read-only static audit of the active `src/mdl/` package, `tests/`, root configuration and documentation, and the archived legacy package. Usage classifications are based on import edges, declared entry points, direct `__main__` blocks, and test references. No simulation, benchmark, or full test suite was run. The six audit documents are new documentation only; no source was changed.

## Executive summary

MDL currently contains **two separate Brian2 networks**. The original `STDPAssociator` is a three-input, three-output LIF circuit with a dense auditory STDP projection and an optional fixed visual relay. The newer `DistributedVisualAssociator` accepts a binary 5×5 visual event array and three auditory event channels, projecting both into 18 shared LIF neurons. Neither has recurrent connections. The distributed network has no output/action group; Python compares its activity against stored references. Full component details are in [NEURAL_ARCHITECTURE.md](NEURAL_ARCHITECTURE.md).

The network's actual computations are event-driven input delivery, membrane decay/threshold/reset, synaptic current jumps, and (when enabled) bounded pair-based STDP on auditory synapses. Most interpretations are external: `argmax` in `assessment.py`, peak/spike comparisons in experiment scripts, cosine similarity in `distributed_visual_experiment.py`, nearest-centroid direction decoders in the two pilots, and the action-inspired objective in `action_pilot.py`. Text/speech input, learned visual feature extraction, learned action selection, and autonomous prediction are not implemented. See [INFORMATION_FLOW.md](INFORMATION_FLOW.md).

The active three-cell baseline randomizes the cue-to-target permutation by seed (`stimuli.py`, `experiment.py`). Later live/continuous and visual-sensory streams use identity mappings by default; the visual-sensory experiment has a fixed identity condition and a cyclic mismatch, not a seed-randomized mapping. Its fixed one-to-one visual relay directly excites the output cell associated with that visual index. These conditions are useful controls, but they are not equivalent evidence of arbitrary association unless no-learning and mismatch controls are interpreted alongside the correct-mapping result.

## Main findings

1. **“The brain” is not one canonical implementation.** `STDPAssociator` and `DistributedVisualAssociator` have different channel counts, output semantics, connectivity, and checkpoint support. A live-dashboard selection switches between these distinct objects.
2. **The network does not itself emit semantic choices.** The three-cell network emits spikes and voltages; Python selects an index. The distributed network emits population responses; Python computes similarity/centroid scores. Action prediction and action-inspired cost are pilot readouts outside Brian2.
3. **State handling differs by use.** The live `SimulationController` retains one network across events and resets only when requested. Most headless recall routines reuse a trained network but set output voltages to rest before each recall, freeze plasticity, and preserve weights/traces/clock. The pilots instantiate a fresh seeded network for each trajectory. These are different experimental protocols, not interchangeable evaluation conditions.
4. **Memory readout needs a measurement floor.** The temporal-memory pilot reports 66.7% held-out decoding through 100 ms while its absolute voltage separation falls to about 5.65×10⁻⁵ mV. It has no observation or trial noise, and train-only standardization can decode tiny deterministic residuals. The report correctly flags this; it is not evidence of robust 100 ms memory.
5. **Response metrics are inconsistent.** Some experiments count `argmax` ties as predictions; others mark ties unresolved. Peak voltage and spike count can disagree. The `graded.py` softmax is a fixed-temperature normalization of integrated depolarization, not calibrated confidence.
6. **Checkpoint compatibility is implicit.** The compact `.npz` checkpoint is for the fixed 3×3 `STDPAssociator`; the Brian2 full checkpoint requires reconstructing a compatible object graph. Neither format provides architecture migration across the three-cell and distributed configurations.
7. **The live distributed dashboard has an unbounded monitor/event history path.** Its voltage monitor is cleared after each capture, but spike-monitor arrays are retained and accessed by increasing cursors; the event log and input queue also grow with use. The standard controller clears its monitors per capture, but its event log is also an unbounded list.

## Scientific status

The strongest currently controlled learning design is the randomized three-item baseline (`experiment.py`, extended by `fixed_dose.py`): seeded permutation, no-learning and mismatched-teacher controls, and frozen teacher-free recall. The continuous and visual-sensory studies add detailed plasticity gates, event logs, gain controls, and sensory relays, but their mapping defaults and readout variants must remain visible in interpretation. The distributed experiment includes identity/remapped, no-learning, paired-gated, continuous, and unpaired controls; its “recognition” winner is a Python cosine-similarity comparison, not a decision made by a dedicated neural output layer.

The action and temporal pilots use explicit Python classifiers/objectives and very small numbers of network seeds. They are diagnostic demonstrations, not evidence for a learned world model, action minimization, motion understanding, or a need for recurrence.

## Highest-priority observations

- Keep future claims attached to the exact architecture/configuration and external readout used.
- Before comparing dashboard results with headless experiments, state whether membrane state, STDP traces, learned weights, simulator time, and controller/environment state are reset or retained.
- Define tie handling and one primary readout before training/evaluation; report voltage and spike results separately.
- Treat current 100 ms temporal decoding as numerically detectable residual state, not noise-robust memory.

See [MODULE_INVENTORY.md](MODULE_INVENTORY.md), [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md), and [DEVELOPMENT_PRIORITIES.md](DEVELOPMENT_PRIORITIES.md) for evidence and proposed sequencing. Recommendations are not implemented.
