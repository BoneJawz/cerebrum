# Multisensory Dynamics Lab — v0.2

A CPU-first benchmark for combining a binary one-dimensional visual stream with raw acoustic waveforms. It is a synthetic computational experiment, not a biological brain model or a test of subjective experience.

## Run

Requires Python 3.10+, NumPy, and pytest for development.

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m sensory_lab.cli --train 160 --test 80 --seeds 5 --seed 42 --out results/v0.2.csv
python -m sensory_lab.viewer --train 160 --test 60 --seed 42
```

The viewer is at `http://127.0.0.1:8765` while its command is running. It plays and animates full training demonstrations, fits 12 readouts when requested, then shows initial held-out scenes and audio cues. Ask for predictions before revealing the outcome frame and target.

## Task and information flow

During training, one of two raw tone frequencies consistently accompanies rightward movement, and the other accompanies leftward movement. The frequency-to-direction mapping is randomized for each experiment seed and shared between that seed's train and test split. Starting visual position, apparent object width, audio phase, and noise vary. The object begins moving after a stationary interval.

For prediction, the processor sees only the initial visual scene and audio through frame 4, before visible movement. It predicts the object's center position at frame 9, after four frames of movement. The supervised target and future visual frame are kept out of processor inputs. Every processor is evaluated with vision only, audio only, or both; in ablations the omitted stream is zeroed.

This makes the outcome depend on both modalities: audio identifies the movement direction but not the initial position; vision identifies the initial position but not the command. Predicting the later position requires combining the two. The benchmark uses a degree-two kernel ridge readout, fitted on each processor's fixed features. A–D do not learn their internal dynamics.

## Interchangeable processors

- **A:** feedforward recent-history features.
- **B:** recurrent fusion reservoir.
- **C:** coupled leaky dynamics integrated with Euler substeps.
- **D:** event-triggered coupled state updates.

They share the `Processor` contract. The environment, readout, and metrics remain independent. A–D use the same hidden-unit count, train/test trials, and readout algorithm; their feature dimensions and computation differ, so this is not a strict compute-matched comparison.

## Measurements

The CSV includes reproducible seed and task settings, held-out exact-position and within-one-pixel accuracy, mean absolute position error in pixels, median/p95 processor-step latency, train/test CPU time, active update calls, and peak traced Python allocation. Stimulus generation is excluded from timing. Update calls are not FLOPs; traced allocation is not process RSS. Local timing is sensitive to OS scheduling and instrumentation overhead.

## Initial measured run

Command: `--train 160 --test 80 --seeds 5 --seed 42`, five randomized seeds and 80 held-out trials per seed and ablation (Windows, Python 3.10, NumPy 1.23.5). Values below are seed means. The source CSV contains all 60 rows.

| Processor | Vision only MAE (px) | Audio only MAE (px) | Combined MAE (px) | Combined within 1 px | Combined median latency | Combined CPU time | Active updates / seed | Peak traced allocation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 4.00 | 2.85 | 0.45 | 0.882 | 61.0 µs | 0.784 s | 400 | 1.00 MB |
| B | 3.99 | 2.77 | 0.20 | 0.990 | 67.1 µs | 0.794 s | 400 | 0.88 MB |
| C | 3.99 | 2.76 | 0.20 | 0.990 | 86.2 µs | 0.659 s | 800 | 0.88 MB |
| D | 3.99 | 2.83 | 0.20 | 0.990 | 76.1 µs | 0.631 s | 240 | 0.88 MB |

Across seeds, combined within-one-pixel accuracy ranged from 0.762–0.963 for A, 0.963–1.000 for B and C, and 0.963–1.000 for D. Both ablations had much higher position error than combined input. This supports the intended multimodal dependency on this synthetic task. It does not show that the dynamical substrates outperform conventional models: the nonlinear readout performs the supervised fitting, and there is not yet a strict operation-matched baseline.

## Scientific limits

- Tone-to-direction mapping is stable within each train/test seed split. This tests learning a randomized cue mapping from supervised examples, not adaptation to a mapping swap.
- Kernel ridge may perform much of the nonlinear integration. Compare it against matched raw temporal features before attributing performance to A–D.
- The task is synchronized; independent visual and audio timestamps, event queues, and late-event handling belong to a future asynchronous milestone.
- Fixed random substrates do not test learning laws, few-shot transfer, or reusable structure discovery. No architecture superiority claim is supported by these results.

See `ARCHITECTURE.md` and `AGENTS.md` for contracts, scientific safeguards, and the research roadmap.
