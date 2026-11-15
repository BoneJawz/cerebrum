# Repository and module inventory

## Active package and dependency graph

The active installable package is `src/mdl/` (38 Python modules). Runtime dependencies pinned in `pyproject.toml:5-7` are NumPy 1.23.5, Brian2 2.6.0 and Matplotlib 3.9.0; pytest 9.1.1 is the optional `dev` dependency. Brian2 is explicitly configured to NumPy code generation in both network constructors. No external model or pretrained dependency exists.

### Core neural and input modules

| File | Responsibility / primary symbols | Called by / status |
|---|---|---|
| `src/mdl/__init__.py` | Re-exports `AssociationTask`, `make_task`. | Public package import; imports `stimuli`. |
| `src/mdl/stimuli.py` | `one_cell_visual_array`, `encode_visual_array`, `Episode`, `AssociationTask`, `make_task`; seeded three-way permutation and episode order. | `network`, `experiment`, `fixed_dose`, `diagnostic`, `efficacy`, `recall_diagnostics`, tests. Core and active. |
| `src/mdl/network.py` | `STDPAssociator`, scheduling/presentation, weight/gain/gate methods. | Base experiment family, controller, validation; tests. Canonical only for the three-output branch. |
| `src/mdl/distributed_network.py` | `DistributedVisualAssociator`: 25+3→18 LIF circuit, fixed visual projection, auditory STDP. | Distributed experiment/dashboard and action/temporal pilots. Separate architecture. |
| `src/mdl/assessment.py` | `choose`, `recall`, `evaluate`; argmax spike readout. | Baseline `experiment.py`; narrow utility. |
| `src/mdl/sensory_environment.py` | `AudibleCue`, `audible_cue`, teacher pattern, seeded teacher and visual-array streams. | Controller, dashboards, continuous and visual-sensory experiments. |
| `src/mdl/stimulus_scheduler.py` | `StimulusCommand`, FIFO queue/snapshot/restore. | `SimulationController` and sensory environments. |

### Persistent simulation and UI

| File | Responsibility / primary symbols | Called by / status |
|---|---|---|
| `src/mdl/simulation_controller.py` | One persistent sensory `STDPAssociator`, plasticity modes, command queue, event log, capture, checkpoint API. | `live_dashboard.py`, `live_validation.py`, controller tests. Active. |
| `src/mdl/activity_buffer.py` | Bounded deque of times, voltages and spikes. | Controller and distributed dashboard. Active. |
| `src/mdl/checkpoint_manager.py` | Compact weights/voltage/traces `.npz` and Brian2 full restore wrapper. | Controller; checkpoint tests. Active but architecture-specific. |
| `src/mdl/live_dashboard.py` | Default three-output dashboard; `--distributed` dispatches to other UI. | `mdl-live` console entry and module command. Active. |
| `src/mdl/distributed_live_dashboard.py` | 5×5/18-neuron dashboard, topology plot, event controls. | Dynamically imported by `live_dashboard.main` when `--distributed`; active. |
| `src/mdl/audio_player.py` | Best-effort Windows playback of deterministic cue tones. | Both dashboards. Human-output utility, not model input. |

### Experiment runners and wrappers

| File | Responsibility / primary symbols | Called by / status |
|---|---|---|
| `src/mdl/experiment.py` | Original randomized-permutation baseline, 5 seeds, exposure curve and controls. | `cli.py`, direct module main; active baseline. |
| `src/mdl/cli.py` | Argparse wrapper for `experiment.run`. | `mdl-association` console script. |
| `src/mdl/fixed_dose.py` | No-learning/correct/mismatched training, fixed-dose frozen recall, jitter trials, tie-aware spike metric. | `fixed_dose_cli.py`, direct runner; active validation. |
| `src/mdl/fixed_dose_cli.py` | Wrapper. | Module command; not a project console script. |
| `src/mdl/continuous_experiment.py` | Plasticity-off/teaching-gated/continuous conditions, event/weight histories, recall and unpaired audits, telemetry/plots. | `continuous_learning_cli.py`; active continuous study. |
| `src/mdl/continuous_learning_cli.py` | Wrapper. | `mdl-continuous-learning` console script. |
| `src/mdl/visual_sensory_experiment.py` | Three-cell sensory relay association, correct/mismatched mapping, gain and retention comparisons. | `visual_sensory_cli.py`; active. |
| `src/mdl/visual_sensory_cli.py` | Wrapper. | `mdl-visual-sensory` console script. |
| `src/mdl/distributed_visual_experiment.py` | Pattern/reference generation, distributed training controls, cosine readout, held-out variants, artifact merge/replay and reports. | `distributed_visual_cli.py` and `distributed_live_dashboard.patterns`; active but tightly experiment-coupled. |
| `src/mdl/distributed_visual_cli.py` | Options for development/validation cohorts and output paths. | `mdl-distributed-visual` console script. |
| `src/mdl/diagnostic.py` | STDP timing/manual-weight diagnostic. | `diagnostic_cli.py`, direct main; active targeted diagnostic. |
| `src/mdl/diagnostic_cli.py` | Wrapper. | Module command. |
| `src/mdl/efficacy.py` | Shared `_train`, `_recall`, exposure/gain/control experiment. | `efficacy_cli.py`, plus `graded`, `path_analysis`, `recorder`. Active shared helper/runner. |
| `src/mdl/efficacy_cli.py` | Wrapper. | Module command. |
| `src/mdl/graded.py` | Peak/spike/integrated response and fixed-temperature softmax summaries. | `graded_cli.py`; active diagnostic. |
| `src/mdl/graded_cli.py` | Wrapper. | Module command. |
| `src/mdl/path_analysis.py` | Traces learned row weights to drive/voltage/spike/argmax failures. | `path_analysis_cli.py`; active diagnostic. |
| `src/mdl/path_analysis_cli.py` | Wrapper. | Module command. |
| `src/mdl/recall_diagnostics.py` | Replays silent-recall cases and voltage timing. | Direct module main; active one-off diagnostic. |
| `src/mdl/recorder.py` | Saves trial voltage/spikes and summary plots; imports `_train` from efficacy. | `recorder_cli.py`; active recording tool. |
| `src/mdl/recorder_cli.py` | Wrapper. | Module command. |
| `src/mdl/live_validation.py` | Small controller/checkpoint/recording verification artifact runner. | `live_validation_cli.py`; active validation helper. |
| `src/mdl/live_validation_cli.py` | Direct runner import; no argparse. | Module command, not console script. |
| `src/mdl/action_pilot.py` | Standalone two-seed motion-state and external action-cost pilot. | Direct `python -m` only; no pyproject console entry. Research pilot. |
| `src/mdl/temporal_memory_pilot.py` | Standalone balanced direction decoding and quiet-decay pilot. | Direct `python -m` only; no pyproject console entry. Research pilot. |

### Entrypoints and configuration

`pyproject.toml:10-16` exposes five console scripts: `mdl-association`, `mdl-live`, `mdl-continuous-learning`, `mdl-visual-sensory`, and `mdl-distributed-visual`. The fixed-dose, diagnostic, efficacy, graded, path, recorder, and live-validation wrappers can be run as `python -m mdl.<module>` but are not installed console entry points. The action and temporal pilots have direct `__main__` blocks but no wrapper or console entry.

## Tests

Nine active test files cover: seed permutation (`test_stimuli.py`); network weight/gate, teacher-free call, and gain properties (`test_network.py`); deterministic argmax (`test_assessment.py`); sensory environment/scheduler (`test_continuous_environment.py`); persistent controller, gates, bounded activity buffer (`test_live_simulation.py`); full checkpoint clock/weights/controller restoration (`test_continuous_modes.py`); softmax (`test_graded.py`); efficacy-to-path CSV row ordering/frozen recall (`test_path_analysis.py`); and aligned recorder data/frozen weights (`test_recorder.py`). They do not directly test the distributed network, distributed dashboard monitor growth, the visual-sensory gain calibration, the action/temporal pilots, or cross-architecture checkpoint rejection. They were inspected, not run, per the read-only/no-suite constraint.

## Root documentation and artifacts

| Path | Status |
|---|---|
| `AGENTS.md` | Active project rules: deterministic seeds, modular boundaries, labels excluded from processor input, pair STDP, CPU/NumPy Brian2. |
| `README.md` | Active usage and study overview; several recent pilots are absent and it mixes baseline framing with newer branches. |
| `ARCHITECTURE.md` | Active architecture overview for baseline, 3-cell sensory, and distributed configuration; does not inventory all recent diagnostics. |
| `liquid_architecture.md` | Historical SAL design description; implementation is archived, not active. |
| `pyproject.toml` | Active package/dependencies/scripts; runtime versions pinned, build requirement `setuptools>=61` is not. |
| `example-results.csv` | Root-level A–D sensory-lab benchmark sample; no active `mdl` runner imports or writes it. Treat as historical/context data, not current Brian2 output. |
| `results/*` | Generated active research outputs for baseline, diagnostics, continuous learning, distributed association, dashboards, and pilots. Not source modules. |

## Archived legacy implementation

`archive/legacy-research/README.md` says this is a recoverable pre-foundation snapshot and not imported by the active package. The archived `src/sensory_lab/` inventory is: `__init__.py`, `adaptive_cli.py`, `adaptive_dynamics.py`, `adaptive_report.py`, `advanced_cli.py`, `benchmark.py`, `cli.py`, `contracts.py`, `controlled_factor.py`, `diagnostic_cli.py`, `diagnostics.py`, `environment.py`, `experiential_tokens.py`, `factor_cli.py`, `factor_diagnostics.py`, `factor_environment.py`, `factor_experiment.py`, `factor_learning.py`, `liquid.py`, `liquid_experiment.py`, `liquid_readout.py`, `nengo_associative_baseline.py`, `processors.py`, `recognition_experiment.py`, `research_benchmark.py`, `spatial_environment.py`, `spatial_grounding.py`, `spatial_range_benchmark.py`, `spatial_viewer.py`, `temporal_diagnostics.py`, and `viewer.py`. Archived tests are the corresponding legacy A–I, liquid, recognition, spatial, and token tests under `archive/legacy-research/tests/`. Its reports/checkpoints/results remain historical artifacts; active modules do not import them.
