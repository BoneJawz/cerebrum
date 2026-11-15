# Development priorities from the audit

These are recommendations only. This audit did not implement any change.

## Immediate — before further research

1. **Name the exact model and readout in every result.** Separate “3-output STDP”, “3-cell visual sensory relay”, and “25+3→18 distributed association” in report headers and comparisons. Their metrics are not directly comparable.
2. **Make state protocol explicit.** For every experiment record whether each trial starts with a fresh network, reset voltage only, reset traces, reset weights, or continuous state. Dashboard demonstrations are persistent and must not be described as equivalent to reset-state headless recall.
3. **Standardize tie and silence scoring.** Define whether ties and all-silent trials are incorrect/unresolved; report raw spike counts and voltage peaks separately. Avoid presenting `argmax` on ties as a confident response.
4. **Bound or stream long-running dashboard histories.** Preserve exact event logs, but avoid retaining duplicate Brian2 spike-monitor histories indefinitely. Add a long-session validation before relying on unattended operation.
5. **Treat the temporal-memory result as unresolved under noise.** The present 66.7% at 100 ms is accompanied by ~5.65×10⁻⁵ mV between-class voltage separation and no noise. Replicate with a prespecified readout-noise condition and a few additional topology seeds before claiming usable long-term memory.
6. **Keep live identity mapping separate from randomized association evidence.** Baseline `make_task` permutes mappings; live sensory environments and some later experiments default to identity. State which mapping applies and keep no-learning/mismatch controls in every learning claim.

## Near-term — before adding modalities or sizes

1. **Add a small explicit architecture/config manifest.** Include schema version, neuron counts, input adapter dimensions, graph connectivity rules, Brian2 version/codegen target, timestep, gains, threshold/reset, STDP parameters, mapping seed and state-reset protocol. Validate checkpoints against it.
2. **Separate neural core from experiment-specific pattern helpers.** `distributed_live_dashboard.py` currently imports patterns from the 630-line experiment module. A tiny shared pattern/adapter module would remove this coupling without generalizing the network prematurely.
3. **Define replaceable interfaces at existing seams:** raw sensory adapter → spike schedule; Brian network lifecycle; learning gate; activity recorder; external readout; metrics. The current code already partially separates these in `stimuli.py`, `network.py`, `assessment.py`, and `simulation_controller.py`.
4. **Version checkpoint schemas.** Keep compact learned weights distinct from full Brian2 continuation state. Record architecture identifier and shape validation; document that a weights restore is not exact scheduler/RNG continuation. Define explicit migration/adapter behavior before changing counts.
5. **Audit and document calibration data.** The visual-sensory experiment records a separate seed-901 gain pilot and later gain sweep. Keep calibration selection separate from prospective evaluation; do not select the best gain on test rows.
6. **Add targeted tests before refactors:** distributed network dimensions/weights; cue/label exclusion at network boundary; transient versus trace/weight resets; no test-trial references in fit; tie handling; distributed dashboard monitor bounds; and mismatch control direction. Do not substitute one broad regression suite for these invariants.
7. **Clarify package commands and dependency lock.** Register only stable runners intended for users; pin the build backend or use a lockfile if reproducible installation across machines is a project goal.

## Long-term — only when research questions justify it

| Capability | Observed limitation | Research prerequisite |
|---|---|---|
| New sensory modalities / RGB / higher resolution | Current channel counts, adapters, patterns, event-log offsets and checkpoint dimensions are fixed. | Define raw modality event contracts and per-modality encoding controls; preserve comparable compute/readout baselines. |
| Persistent online learning without forgetting | Live model is persistent, but controlled experiments mostly use short fixed tasks and STDP on only auditory synapses. | Evaluate retention/interference with randomized mappings, no-learning/mismatch controls, session continuation and frozen checkpoints. Do not infer broad continual learning from one task. |
| Recurrent neural dynamics | Neither active network has recurrence; current state is transient LIF voltage and decaying STDP traces. | First demonstrate replicated, noise-robust history limitation and state what recurrence is expected to add. Then compare matched nonrecurrent control and recurrent model. |
| Learned output/action population | Current decisions are Python argmax/cosine/centroid, with no environment action loop. | Define action targets, closed-loop environment, action readout and held-out counterfactual metrics; separate controller from neural pathway. |
| Structural plasticity | No active structural plasticity; archived SAL is not current architecture. | Specify local structural objective and matched fixed-topology controls only after learning gains are independently replicated. |
| Predictive latent representations | Motion direction/action pilots use fitted external nearest-centroid rules; action-cost term reuses the same predictor. | First compare explicit history controls, predictive baselines and held-out trajectories. A “latent” claim requires a defined neural training objective and ablation. |
| Curiosity / information-seeking | No intrinsic reward, uncertainty mechanism, or self-initiated action loop exists. | Requires a validated action interface and measurable value-of-information outcome; defer until perception/prediction is tested. |

## Minimal conceptual architecture

The simplest maintainable separation consistent with the actual code is:

- **A. Neural core (partly exists):** Brian2 populations, equations, connection builders, plasticity gates and live neural state. `network.py` and `distributed_network.py` currently implement separate cores; keep them separate until dimensions and output contracts are explicit.
- **B. Sensory adapters (partly exists):** binary-array→spike indices and cue event channel/time. `stimuli.py`/`sensory_environment.py` are beginnings; raw waveform-to-neural encoding does not exist.
- **C. Environment/controller (exists for live use):** `SimulationController`, `StimulusScheduler`, seeded environments. It owns labels, mapping, event order, phase, reset, and feedback; these are external control logic.
- **D. Scientific instrumentation (exists, unevenly bounded):** monitors, `ActivityBuffer`, dashboard, experiment runners, external readouts and reports. Preserve readout and metric definitions as experiment config, not implicit network behavior.
- **E. Persistence (partial):** compact weights and Brian2 full-state checkpoint plus JSON sidecar. Add version/architecture validation and migration only when the first architecture upgrade is planned.

This architecture is a proposed boundary map, not a reorganization performed by this audit.
