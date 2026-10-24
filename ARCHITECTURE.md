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

`python -m sensory_lab.diagnostic_cli --outdir results/diagnostics` performs the follow-up audit without changing A-D equations. It checks phase-only and frequency-only factor probes, probes A-D feature outputs, evaluates E3 (a physics-structured nonprivileged reference), compares the original quadratic readout with explicit pairwise features and a compact MLP, and sweeps D's event threshold. Fit/early-stop choices use validation combinations only; all tests retain the same seed-specific episodes. Event check/update component timings are collected separately from uninstrumented end-to-end CPU and latency measurements.

## Trainable dynamical-state follow-up

`python -m sensory_lab.adaptive_cli --outdir results/adaptive-dynamics` adds independent experimental models; A-D/E1-E3 and their prior data are left unchanged.

- F is a 32-unit tanh recurrent state model. It learns sensory-to-state, observed-state recurrence, autonomous transition, and a scalar position decoder.
- G uses the same dimensions and shared decoder with Euler-integrated leaky dynamics. The default simulation step is 0.05 s and time constant 0.1 s (`dt/tau = 0.5`); the constructor validates the scalar decay ratio, and rollout/BPTT checks for nonfinite or divergent states. Learned coupling can still destabilize it, so results must be checked for finite states and timestep sensitivity before interpreting it as a continuous-time advantage.
- H uses G's continuous training rule and tests fixed event thresholds only during inference. It does not learn a gate. Its event trigger computes a proposed state change, so update skips may not save compute.
- `FF_direct` is a roughly parameter-matched tanh MLP with one direct output for each trained horizon. It has no recursive transition and is not evaluated beyond h=12.

F/G/H are trained by supervised position trajectory loss for intervals 1–12. Inputs contain only observations through tick 4; direction/speed labels and the analytic motion rule are not used in training. Prediction repeatedly applies one shared transition without future sensory input. Horizon 18 is excluded from training. Training uses deterministic NumPy reverse-mode BPTT and Adam; gradients are globally clipped at 1.0, and minimum/maximum unclipped norms are recorded. No autodiff package is added. The all-pairs balanced-distribution control is separate from the original held-out split. Frozen-state linear probes are diagnostics only. CSVs include per-seed and per-command errors, parameter counts, train/inference timing, latency quantiles, active updates, event skips, traced allocation and probe results. Traced allocation excludes native NumPy/BLAS memory. CPU timing is machine-specific; process CPU clock resolution can be too coarse for very short model calls.

The longer-horizon configuration increases the world width to avoid clipping through h=18, so its pixel count and start-position distribution differ from the prior 129-pixel benchmark. An inference-only refinement check compares dt=0.05 s with two dt=0.025 s Euler substeps using the same learned G weights; it checks sensitivity but is not a convergence study. Compare within this new run and retain the original report as historical baseline; do not attribute cross-run score changes only to architecture.

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
