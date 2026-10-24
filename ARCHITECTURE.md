# Architecture and scientific roadmap

## Contracts

- `Observation`: event time, raw 1D vision, raw signed audio chunk. No target position, future frame, direction label, or semantic audio feature is part of model input.
- `Environment`: generates deterministic raw sensory experience and future ground truth. Targets are used by the readout fit/evaluation and viewer truth panel only.
- `Processor`: `reset`, `step(observation) -> features`, `updates`; replace without changing environment or benchmark.
- `Readout`: `fit`, `predict`; independent of processing substrate.
- `WorldConfig`: sensor sizes, rates, noise, and time horizon.
- `BenchmarkConfig`: seed, trial counts, replicate count, hidden dimension, observation horizon, and outcome frame.

## A–D processors

A. Temporal feedforward history features. B. Reservoir recurrent fusion. C. Euler-integrated leaky coupled dynamics. D. Event-triggered coupled dynamics. All receive the same sensory stream and use the same supervised readout algorithm. Processor dynamics stay fixed in v0.2.

## V0.2 directional-cue forecast

Training demonstrations pair a raw tone cue with visible object movement in its direction. The tone-to-direction mapping is randomized per experiment seed and remains fixed between that seed's training and held-out trials. Starting position, apparent width, waveform phase, and noise vary.

For each readout example, processor input ends at frame 4, before visible movement. The target is the object's visual center at frame 9, after four movement frames. Thus audio identifies movement direction but not starting position; vision identifies starting position but not the command. Neither modality alone determines the target position. Three ablations zero vision, audio, or neither at the processor boundary. Paired-stimulus tests verify that a fixed initial scene with opposite commands produces distinct outcomes, and a fixed command at different initial positions produces distinct outcomes.

The readout is degree-two kernel ridge regression. It can represent pairwise feature interactions without explicit polynomial expansion. Readout parameters are fit separately for every A–D/modality pair. Train and test datasets use separate deterministic generation seeds; tone mapping is randomized and recorded per experiment seed. The viewer shows complete labeled demonstrations to the human, fits readouts only from pre-movement features, and serves held-out targets only through the reveal endpoint after predictions.

## Metrics and limits

Record held-out exact-position accuracy, within-one-pixel accuracy, mean absolute position error, median/p95 per-step wall latency, train/test CPU seconds, active update calls, and peak `tracemalloc` allocation. Stimulus generation is excluded from timing. Update counts are not FLOPs; traced allocation is not process RSS. A–D share hidden-unit count and task/readout but have different feature sizes and per-update work, so results are not strict compute-matched comparisons.

V0.2 demonstrates task performance only when combined input outperforms both modality ablations. It does not establish architecture superiority, adaptive processor learning, or transfer beyond this synthetic task. The readout may carry much of the multimodal computation; a matched raw-history baseline remains an important control.

## Development phases

1. **v0.1:** raw sensory streams and A–D processing with an auditory-only solvable cue task. Implemented as a wiring baseline.
2. **v0.2:** directional sound cue paired with visual demonstrations; forecast future visual position from initial scene plus audio cue; three modality ablations and multiple deterministic seeds. Implemented.
3. **v0.3:** independently timestamp visual and auditory events, bounded queues, event-time versus wall-time, dropped/late events, and backpressure. Compare asynchronous and synchronized implementations under equal information.
4. **v0.4:** episode-specific cue remapping and few-shot adaptation; compare 0/1/2/5/10 examples with learner/memory as replaceable components.
5. **v0.5:** test learned local update rules, compact governing dynamics, and module reuse against capable recurrent and state-space baselines.

## Benchmark discipline

- Match samples, input information, prediction horizon, and readout supervision across A–D.
- Report seed-level measurements and uncertainty ranges across at least five seeds.
- Separate stimulus generation from timed inference. Warm timing runs before making runtime claims.
- Report process RSS and operation estimates before making strong resource-efficiency claims.
- Keep parameter-free evolving state distinct from trainable plasticity; state updates are not training.
- For numerical ODE claims, compare timestep refinements and measure discretization error.
