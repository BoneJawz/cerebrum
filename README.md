# Multisensory Dynamics Lab — v0.3

A CPU-first benchmark for combining binary one-dimensional vision with raw acoustic waveforms. It is a synthetic computational experiment, not a biological brain model or a test of subjective experience.

## Run

Requires Python 3.10+, NumPy, and pytest for development.

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m sensory_lab.cli --train 160 --test 80 --seeds 5 --seed 42 --velocities 1,2,3 --out results/v0.3-velocity.csv
python -m sensory_lab.viewer --train 160 --test 60 --seed 42 --velocities 1,2,3
```

The viewer runs at `http://127.0.0.1:8765`. It plays full training demonstrations, fits 12 readouts when requested, then shows initial held-out scenes and audio. Ask for predictions before revealing the future visual outcome.

## V0.3 experiment

Velocity is measured in pixels per visual frame. The default levels are 1, 2, and 3. Each trial begins from a randomly selected central position; motion starts at frame 5. The environment clamps objects at the retinal edges.

Each demonstration pairs the raw waveform with the actual movement direction and velocity. Tone frequencies encode the six direction–velocity pairs through a codebook randomized per seed and shared across that seed's train/test split. Processor inputs through frame 4 contain only the initial scene and audio cue. The target is the object's center at frame 9, after four visual movement steps.

Three task conditions isolate the factors:

- `direction_only`: vary direction at the lowest configured speed.
- `velocity_only`: vary speed while direction is fixed rightward.
- `direction_and_velocity`: vary both direction and speed.

Each task runs vision-only, audio-only, and combined-input ablations. The omitted sensor stream is zeroed. Neither stream alone determines the future position: audio gives the commanded movement but not the start position; vision gives the start position but not the command. A degree-two kernel ridge readout is trained on fixed features; A–D processor dynamics do not learn.

The processors compute a fixed-bin waveform magnitude spectrum and 16 signed waveform samples from each raw audio chunk, alongside earlier waveform summaries. The samples preserve phase polarity, which magnitudes discard; all features are generic signal transforms and do not receive command labels.

## Compositional generalization and efficiency study

Run the expanded controlled benchmark with:

```powershell
python -m sensory_lab.advanced_cli --outdir results
```

It evaluates the four fixed processors A-D, E1 (a direct linear kernel-ridge baseline), and E2 (a privileged analytic oracle, reported only as a ceiling). The held-out direction/speed pairs are `(-1, 2)` and `(1, 3)`. All individual directions and speeds remain represented in supervised training, so the held-out set tests recombination. Vision-only, audio-only, and combined ablations are evaluated on observed and held-out pairs. Sample-efficiency curves use 10, 25, 50, 100, 250, and 500 training episodes. Five seeds use separate trajectory sets, and validation, normalization, and alpha selection exclude held-out pairs.

The run writes `generalization_results.csv`, `sample_efficiency_results.csv`, `compute_benchmark.csv`, three PNG plots, and `EXPERIMENT_REPORT.md` under the selected output directory. The CSVs include per-seed records plus cross-seed means, standard deviations, and 95% t intervals.

Diagnose the held-out-pair failure with:

```powershell
python -m sensory_lab.diagnostic_cli --outdir results/diagnostics
```

This preserves the original benchmark outputs and A-D equations. It writes per-combination and per-horizon errors, factor probes, the factorized E3 diagnostic, readout-capacity comparisons, paired-seed differences, an error decomposition, and an event-threshold cost sweep under `results/diagnostics/`. `DIAGNOSTIC_REPORT.md` documents the protocol and findings. The default run uses the same five seeds and held-out pairs as the main study.

The diagnostic identifies a strong split-induced shortcut: training contains left/speed-3 and right/speed-2, while the held-out pairs are left/speed-2 and right/speed-3. Standalone phase-only and frequency-only probes recover held-out direction and speed at 100%, and the visual centroid is exact; E3 therefore predicts held-out positions exactly using only sensory inputs plus the known physics rule. A-D, however, systematically reverse direction on both held-out pairs, with errors growing roughly linearly with horizon. The tested pairwise expansion and MLP do not reliably fix this. At D threshold 0.25, about 40% of held-out update opportunities are skipped with a small error change, but CPU time remains slightly above B; at threshold 1.0, 80% are skipped while held-out MAE rises to 29.78 px. The event check costs about as much wall time as the skipped state updates save at moderate thresholds. See the report and seed-level CSVs for uncertainty and full comparisons.

## Trainable state-transition experiment

Run the next experiment with:

```powershell
python -m sensory_lab.adaptive_cli --outdir results/adaptive-dynamics
```

This adds F (trainable discrete recurrence), G (trainable Euler-integrated dynamics), H (G with fixed event thresholds at inference), and a parameter-matched direct multi-horizon MLP. It preserves A-D/E1-E3 and the original held-out command pairs. F/G/H learn position trajectories for intervals 1–12 from observations through tick 4, then predict autonomously with one repeated transition and shared position decoder. Horizon 18 is excluded from the training loss. No direction/speed labels or analytical motion rule are model inputs. NumPy BPTT keeps the CPU prototype dependency-light; gradients are clipped and nonfinite/divergent states are checked.

The run includes five seeds, the six existing training sizes, a separate all-pairs training-distribution control, frozen-state probes and clustering diagnostics, an H event-threshold sweep, compute/latency measurements, plots and checkpoints by seed and training size. The report also records an inference-only G timestep refinement check. It writes `ADAPTIVE_DYNAMICS_REPORT.md` and adaptive CSV/PNG artifacts under `results/adaptive-dynamics/`. Longer-horizon evaluation uses a wider field than the historical 129-pixel run, so its scores are not a clean cross-run architecture comparison. See the generated report for results and uncertainty.

## Independent acoustic-cue factor-learning benchmark

Run the separate symbolic-cue experiment with:

```powershell
python -m sensory_lab.factor_cli --output results/factor-learning
```

It compares the historical combined tone/phase cue with sequential raw-waveform direction, silence, and speed cues. F/G/H and a parameter-matched direct baseline are trained under combined-only, curriculum, and shuffled schedules. The strict arm excludes the two held-out direction/speed pairs from every training phase. A transfer arm balances isolated speed experiences across directions and explicitly includes those withheld physical motions during factor pretraining; its scores are not strict compositional generalization.

The run uses five seeds and 10/25/50/100/250/500 training episodes. At 500 episodes it also crosses the historical A cue with curriculum and shuffled schedules, alongside B, to estimate a cue-by-schedule interaction. It writes the vocabulary, per-horizon results, cue comparison, frozen-state probes, audio counterfactuals, sample-efficiency and compute tables, four plots, checkpoints, full config, and `FACTOR_LEARNING_REPORT.md`. The report documents the strict-arm cue/direction correlation, which follows from forbidding those held-out motions while also trying to balance directions at every speed. Do not interpret the transfer arm as an unseen-motion result.

### Measured compositional benchmark run

On the 9-interval forecast, observed-pair MAE was 3.91 px (A), 2.71 (B), 2.62 (C), and 2.63 (D). Held-out-pair MAE was 32.15, 24.44, 25.86, and 25.03 px, respectively. With combined input, each processor substantially improved over either ablation on observed pairs; held-out recombination remained poor. None reached a 1 px held-out MAE threshold at the tested sample sizes. These results show that this experiment has not demonstrated compositional generalization or an efficiency advantage for event-driven computation. See the generated report for modality-specific scores, confidence intervals, sample-efficiency curves, and machine-specific timing.

The timed combined-input inference results over 240 episodes were:

| Model | Median / p95 episode latency (ms) | CPU seconds | Active updates |
|---|---:|---:|---:|
| A | 0.774 / 0.913 | 0.2031 | 1,200 |
| B | 0.782 / 0.896 | 0.2031 | 1,200 |
| C | 0.885 / 0.946 | 0.2344 | 2,400 |
| D | 0.852 / 0.969 | 0.2188 | 1,200 |

D did not reduce active updates or CPU time versus B in this run. RSS is recorded as a process-level high-water mark, so it is not attributable to an individual model in these sequential timings. Measurements include Python instrumentation and should be compared only within this run.

## Interchangeable processors

- **A:** feedforward recent-history features.
- **B:** recurrent fusion reservoir.
- **C:** coupled leaky dynamics integrated with Euler substeps.
- **D:** event-triggered coupled state updates.

They share the `Processor` contract. Environment, readout, and metrics remain independently replaceable. All use the same hidden-unit count, examples, and readout algorithm; operation counts and feature dimensions differ, so this is not a strict compute-matched architecture comparison.

## Initial measured run

Command: `--train 160 --test 80 --seeds 5 --seed 42 --velocities 1,2,3`. Each CSV row contains one seed/task/processor/modality combination; the artifact has 180 rows, five seeds, and 400 held-out predictions per row.

| Task | Vision-only MAE | Audio-only MAE | Combined MAE | Combined within 1 px |
|---|---:|---:|---:|---:|
| Direction only | 3.92 px | 3.23 px | 0.61 px | 81.8% |
| Velocity only | 2.88 px | 3.01 px | 0.59 px | 83.6% |
| Direction + velocity | 8.12 px | 2.85 px | 0.62 px | 81.1% |

These rows average over A–D and five seeds. The combined task's within-one-pixel score ranged from 78.8% to 83.8% across seeds; its mean absolute error ranged from 0.56 to 0.68 px. All three task conditions improve substantially with both modalities. See [results/v0.3-velocity.csv](results/v0.3-velocity.csv) for seed-level performance and compute measurements.

## Interpretation and limits

- The larger velocity sweep exposes more difficult tone identification and future-position prediction than the two-command v0.2 task. Performance varies by architecture and task; the feedforward case is weaker on some conditions.
- Boundary clamping affects high-speed outcomes near the ends of the visual array. Initial positions are sampled independently of command and speed; edge encounters are part of the task and are present in both splits.
- The readout performs supervised fitting. These results do not show learning in the processor dynamics or an advantage over a strong conventional baseline.
- Update counts are not FLOPs. Latency and CPU measurements are local and instrumentation-sensitive; peak `tracemalloc` allocation is not process RSS.
- The sensor streams remain synchronized at visual ticks. Independently timestamped asynchronous events are a later milestone.

See `ARCHITECTURE.md` and `AGENTS.md` for contracts, safeguards, and the research roadmap.

## Controlled observation-window validation

`python -m sensory_lab.controlled_factor --output results/controlled-factor-learning` runs a matched-window validation in a new results directory, preserving the historical factor-learning outputs. It compares instruction-only input through tick 4, observed motion through tick 9, and the separately labeled B-train/A-evaluate transfer. Direction, silence, and speed audio occupies ticks 0–1, 2, and 3–4 respectively; both cues are complete at the tick-4 cutoff. A/C rollouts start from visual frame 5, while B starts from frame 9.

The default study uses all six direction-speed combinations, five reproducible seeds, the existing 10/25/50/100/250/500 episode schedule, and F/G/H plus `FF_direct`. It records per-combination errors, cue diagnostics, hidden-state decoding, lightweight probes, compute costs, counterfactual cue changes, and figures. F/G/H transition equations remain unchanged. The first phase validates matched-window prediction before interpreting compositional generalization. See `results/controlled-factor-learning/CONTROLLED_FACTOR_LEARNING_REPORT.md` for measured outcomes and limits.

## Model I liquid prototype

`python -m sensory_lab.liquid_experiment --output results/liquid` runs the staged event-driven liquid Experiment A. The first milestone builds a sparse directed graph with asynchronous neuron updates, bounded local Hebbian-style weight adaptation, sampled structural plasticity, deterministic replay snapshots, and independent I0/I1/I2 controls. It measures cue retention using a post-hoc linear readout trained on separate episodes. This does not yet run the audiovisual association or motion-composition experiments; the staged rationale and assumptions are in `liquid_architecture.md`.
