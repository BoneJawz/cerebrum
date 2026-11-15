# Fast Neural Temporal Memory Experiment

## Design

Reused the existing 5×5 binary visual input, fixed sparse projection, and 18-neuron LIF population, with STDP disabled. No neuron, projection, learning-rule, or timestep changes were made. Each trial started from a fresh network with equivalent initial conditions.

To remove the previous opposite-edge-start shortcut, trajectories use a **circular five-column track**: positions wrap at the boundary. Evaluation columns were 1, 2, and 3; both directions occur at each one. Each trial has 1–4 one-column steps before evaluation, balanced exactly across direction, evaluated position, and split. Thus, conditional on the evaluated column, starting columns are uniformly represented in both directions. For each paired direction and replicate, inter-frame intervals were identical, independently drawn from 8–14 ms; stimulus counts are balanced. This explicit wrap-around is a control assumption and is not a claim about a physical environment.

There were 192 trajectories total: 48 train and 48 test per seed, balanced by direction. Repeats 0–1 were assigned to training and repeats 2–3 to test before network runs. Whole trajectories were kept within one split. Nearest-centroid scaling and prototypes used training trajectories only.

### Direction decoders at the evaluated frame

- **A:** current 5×5 visual array only.
- **B:** current 18-neuron voltage vector.
- **C:** current and preceding two neural voltage snapshots.
- **D:** current and preceding two visual arrays.

Missing history at the first step was zero-padded; the number of preceding frames is balanced across direction and does not enter as a separate feature.

## Experiment A results

| Readout | Seed 42 | Seed 43 | Chance |
|---|---:|---:|---:|
| A. Current visual frame | 50.0% | 50.0% | 50% |
| B. Current neural state | 66.7% | 66.7% | 50% |
| C. Neural state + two prior snapshots | 85.4% | 89.6% | 50% |
| D. Visual frame + two prior frames | 95.8% | 95.8% | 50% |

The opposite-direction centroid distance for current voltage was 1.246 mV in the 18-dimensional population feature space; mean within-direction distance was 7.026 mV. For neural-history features, those distances were 5.668 and 110.414, respectively. These distances describe feature geometry; they are not calibrated probabilities or evidence of semantic understanding.

## Experiment B results

At each delay, a fresh nearest-centroid decoder was fit using **training trajectories at that same delay only**. It was then applied once to held-out trials; no held-out fitting or calibration occurred. Quiet intervals are cumulative from the final visual event. 

| Quiet interval (ms) | Seed 42 accuracy | Seed 43 accuracy | Mean accuracy | Mean between-direction distance (mV) | Mean within-direction distance (mV) |
|---:|---:|---:|---:|---:|---:|
| 0 | 66.7% | 66.7% | 66.7% | 1.25 | 7.03 |
| 20 | 66.7% | 66.7% | 66.7% | 0.169 | 0.951 |
| 50 | 66.7% | 66.7% | 66.7% | 0.00839 | 0.0473 |
| 100 | 66.7% | 66.7% | 66.7% | 5.65e-05 | 0.000319 |

The decoder stayed at 66.7% at every tested delay; it did **not** return to chance within 100 ms. At the same time, the absolute between-direction voltage distance fell from about 1.25 mV at 0 ms to about 5.65×10⁻⁵ mV at 100 ms (roughly a 22,000-fold reduction). The train-only feature scaling makes the decoder sensitive to this tiny deterministic residual. With no injected measurement noise or trial-to-trial neural noise, accuracy alone cannot establish that the 100 ms trace is robustly usable. The separation trajectory is the clearer evidence that voltages relax toward rest.

Spikes and voltage snapshots during movement and quiet periods are retained in the CSV files. A near-chance result at longer delays would have been meaningful; this pilot instead finds chance-exceeding classification alongside a very large collapse in absolute response separation. Read both measures together and seed by seed.

## Interpretation and limitations

This pilot tests whether a simple readout can recover direction from the network's present voltage after balanced recent sensory histories, and how recoverability changes during silence. Model D is the explicit visual-history control: it receives recent observations directly. Differences between C and D indicate how well the network state/history preserves information relative to that simple control in this task.

Only two fixed sparse projection seeds, three evaluated columns, one-column steps, and a small number of trajectories were tested. The circular boundary avoids start-location leakage but introduces wrap-around. The classifier is deliberately simple, and accuracy is not a measure of motion understanding. No causal ablation of individual neurons or synapses was performed. This experiment cannot establish autonomous prediction, generalization to new speeds or visual transformations, or a need for recurrence.

## Smallest justified next experiment

Because decoding stayed at 66.7% while absolute state separation collapsed, the smallest next step is to repeat the balanced protocol with a few additional fixed projection seeds and one preregistered, fixed readout-noise condition alongside the noise-free baseline. This will test whether the residual is robust to small measurement variation. Do not add recurrence unless that replicated, noise-aware test shows current state is insufficient while explicit history remains informative.

## Runtime and artifacts

Wall-clock generation, decoding, and plotting took 65.14 seconds. Simulated biological time across trials was 26.657 seconds; the former is machine-dependent and the latter is the sum of simulated trajectory and quiet intervals. No hyperparameter search or external dependency was added.

- `direction_results.csv`: held-out per-trial predictions for A–D, trajectory metadata, voltage history, and spike events.
- `memory_decay.csv`: held-out direction predictions and neural voltage response by delay.
- `temporal_memory.png`: per-seed direction accuracy and quiet-decay curves.
