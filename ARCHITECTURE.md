# Architecture and scientific roadmap

## Contracts

- `Observation`: event time, raw 1D vision, raw signed audio chunk. Direction, velocity, tone codebook labels, target position, and future frames are not processor inputs.
- `Environment`: generates deterministic raw sensory experience and future ground truth. Targets are used by readout fitting/evaluation and the viewer truth panel only.
- `Processor`: `reset`, `step(observation) -> features`, `updates`; replace without changing environment or benchmark.
- `Readout`: `fit`, `predict`; independent of processing substrate.
- `WorldConfig`: sensor sizes, rates, noise, and time horizon.
- `BenchmarkConfig`: seeds, trial counts, replicate count, hidden dimension, discrete velocity set, observation horizon, and outcome frame.

## A–D processors

A. Feedforward recent-history features. B. Recurrent fusion reservoir. C. Euler-integrated coupled leaky dynamics. D. Event-triggered coupled dynamics. All receive the same raw observation sequence and use the same supervised degree-two kernel ridge readout. Processor dynamics stay fixed.

The audio feature transform includes waveform statistics, 16 signed waveform samples, and a fixed-bin magnitude spectrum computed from the raw chunk. Signed samples retain phase polarity information that a magnitude spectrum loses. These generic signal features do not receive direction/velocity labels or the cue mapping.

## V0.3 direction and velocity forecast

Velocity is an integer number of visual pixels per visual frame, defaulting to 1, 2, and 3. The visual object begins from a randomly selected central position, remains still for five frames, then moves by `direction × velocity` each frame. The bounded retina clamps object position at its edges.

The raw tone cue is paired with the movement's direction and velocity. Six tone frequencies encode the six command pairs under a codebook randomized for each experiment seed and shared by its train/test split. Initial position, apparent width, waveform phase, and noise vary independently.

Processor input ends at frame 4, before visible motion. The target is object center at frame 9, after four movement steps. The model must combine initial visual position with its audio-conditioned displacement to forecast the future position. Three experiment tasks isolate factors: `direction_only` varies direction at the lowest configured velocity; `velocity_only` varies velocity at fixed rightward direction; `direction_and_velocity` varies both. Every task has vision-only, audio-only, and combined ablations. Paired-input tests verify that changing command at a fixed scene changes the target and changing the scene at a fixed command changes the target.

The viewer displays full labeled training demonstrations, while readout features stop before movement. Held-out responses contain only initial sensory inputs and predictions; future target information is returned only by the separate reveal endpoint.

## Metrics and limits

Record exact-position accuracy, within-one-pixel accuracy, mean absolute position error, median/p95 per-step wall latency, CPU seconds, active update calls, and peak `tracemalloc` allocation. Stimulus generation is excluded from timing. Update calls are not FLOPs; traced allocation is not process RSS. Equal hidden-unit counts do not equal equal compute. Report seed-level metrics and ranges.

V0.3 demonstrates task performance only if combined input outperforms both ablations. It does not establish architecture superiority or adaptive learning in the processor substrate. Tone mapping remains fixed within each train/test seed; task evaluation does not yet test mapping swaps. Boundary clamping can affect high-speed outcomes and must remain visible in analyses.

## Compositional generalization and efficiency benchmark

`python -m sensory_lab.advanced_cli --outdir results` runs a second experiment over direction/speed command pairs. It holds out `(-1, 2)` and `(1, 3)` from training and validation while retaining each direction and speed level in the observed pairs. Initial visual position is available through frame 4, before motion begins at frame 5. Targets are future visual centers at 3, 6, 9, and 12 movement intervals. The field expands with maximum displacement to avoid boundary clipping.

The generated cue has carrier frequency `120 + 180 * velocity` Hz and phase `±π/2` for direction; the sign mapping is randomized per seed. This explicit code makes the task factorized by construction and should not be interpreted as cue discovery. A-D processor equations remain fixed and use the same kernel-ridge supervised readout. E1 is a linear-kernel readout; E2 is a privileged oracle using ground truth and is not comparable as a learned system. Each method receives vision-only, audio-only, and combined inputs. Training and held-out trajectories use separate seeds. Normalization and ridge selection use only observed training and validation pairs.

The five-seed run reports per-seed scores, 95% t intervals, train-size curves, latency, CPU time, process RSS, traced allocations, update counts, and approximate operation counts. Current results show strong combined-modality gains for observed pairs, but poor errors on the held-out recombinations for all A-D processors. No tested sample size reached 1 px held-out MAE. This falsifies the current implementation's strong compositional-generalization expectation; it does not establish that a different dynamical substrate cannot generalize. Hardware timings are local, and operation counts are estimates. RSS is a process high-water mark gathered during sequential model runs, not a model-specific measurement; traced allocations omit native NumPy/BLAS memory. In particular, the fixed processors have no trainable internal parameters: learning occurs in the supervised readout.

## Development phases

1. **v0.1:** raw sensory streams and A–D processing with an auditory-only solvable cue task. Wiring baseline.
2. **v0.2:** audiovisual direction-cue forecast at one fixed speed, with modality ablations and multiple seeds.
3. **v0.3:** discrete velocities, direction/velocity/joint tasks, randomized audio codebook, and initial-scene-to-future-position prediction. Implemented.
4. **v0.4:** independently timestamp visual and auditory events, bounded queues, event-time versus wall-time, dropped/late events, and backpressure.
5. **v0.5:** episode-specific cue remapping and few-shot adaptation; compare 0/1/2/5/10 examples with learner/memory as replaceable components.
6. **v0.6:** learned local update rules, compact governing dynamics, and module reuse against capable recurrent and state-space baselines.

## Benchmark discipline

- Match examples, input information, prediction horizon, and readout supervision across A–D.
- Report results across at least five deterministic seeds and include all modality/task controls.
- Separate stimulus generation from timed inference. Warm timing runs before making runtime claims.
- Report process RSS and operation estimates before making strong resource-efficiency claims.
- Keep fixed evolving state distinct from trainable plasticity; state updates are not training.
- For numerical ODE claims, compare timestep refinements and measure discretization error.
