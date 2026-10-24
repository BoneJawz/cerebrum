# Model I: Structurally Adaptive Liquid (SAL)

## Status and scope

The current implementation completes the graph and neuron-dynamics foundation (Phase 1), adds fixed-liquid and local weight-adaptive controls (I0/I1), and enables bounded sampled structural adaptation (I2; Phase 2/4 mechanisms). The current experiment exercises only isolated acoustic dynamics and fading memory. Audio-visual association (Experiment B), trajectory composition (Experiment C), F/G/FF comparisons, and parameter sweeps are intentionally staged after stability and memory behavior are reviewed.

## Graph and unit state

`LiquidNetwork` stores directed active connections in `list[dict[destination, Connection]]`; it does not allocate an `n × n` weight matrix. Defaults use 64 units, initial out-degree 6, maximum out-degree 8, and 512 active edges. Every ordered pair is a candidate except a self-pair when `allow_self_connections=False`. Candidate sampling is uniform over all eligible pair indices, with a bounded sparse utility cache. A degree limit restricts concurrent edges only; it does not exclude a pair from future sampling.

Each unit has a bounded activation `x ∈ [-1,1]`, recent incoming magnitude, adaptation, last update timestamp, and a decaying activity trace. Between events, activation obeys `x(t+Δ) = x(t) exp(-Δ/τ)`. On input `u`, the response is `x' = tanh(x_decayed + u - adaptation_decayed)`. Local adaptation rises with `|x'|` and decays with its own time constant. No detailed ion-channel or spiking model is implied.

The min-heap key is `(event_time_seconds, insertion_sequence, ...)`, giving deterministic tie order. Sensory events, delayed connection transmissions, and scheduled internal decay events update individual units. The simulator does not globally step all 64 units on every observation. A readout snapshot does calculate the analytic decay for every unit, which is a small `O(N)` observation cost and is separately counted; it does not mutate state or emit messages. Wall time and simulated time are distinct.

## Sensory interface

The initial projection is a seed-controlled sparse signed map from the existing generic `sensory_features` vector to graph units. Vision and audio remain separate index ranges for future modality ablations; only the strongest bounded number of nonzero projected units are injected per call. `Observation` contains only timestamp, binary 1D vision and raw signed waveform chunk. Experiment A blanks the visual array and uses only the audio waveform-derived features. Vocabulary IDs and labels remain outside the liquid.

## Plasticity controls

- **I0 fixed liquid:** no connection-weight learning and no topology changes.
- **I1 weight-adaptive liquid:** when a connection transmits, update its signed weight by a bounded local pre/post correlation rule, `w ← clip(w + η(pre·post − λw), ±w_max)`. No graph rewiring.
- **I2 structurally adaptive liquid:** I1 plus interval-limited connection formation and pruning. Candidate utility is an EMA of the absolute product of source and destination decayed activity traces; candidate pairs are sampled across the unrestricted directed-pair set. Pruning requires all of weak weight, low utility, sufficient inactivity, and minimum age. This conservative heuristic is falsifiable and may fail to identify useful edges.
- **I2_rewire_fixed diagnostic:** structural changes with fixed-magnitude newly formed weights and no weight adaptation; available for later ablation.

Weight plasticity and structural plasticity are independent flags on the network. Activity is always enabled. Structural work is invoked at most once per configured simulation-time interval while events are processed. Queue and event budgets bound cascades; saturated queues count dropped events, and exceeding the per-advance event-work limit raises an error instead of hanging silently. Deleting an edge invalidates queued messages by connection-generation ID.

## Learning boundary and experiment design

The liquid is not trained with global backpropagation. The initial experiment adapts I1/I2 on a labeled sequence of cue exposures, then freezes the liquid and collects separate labeled state snapshots to fit a ridge readout. Independent test episodes evaluate the same readout at retention delays. The labels exist only in the readout dataset and evaluation code.

The initial local structural signal is coactivity, not prediction improvement; it cannot yet claim predictive structure. Structural controls must match the initial graph, sensory map, exposures, and readout training examples. Candidate sampling cost and topology statistics must be reported alongside accuracy. If I0 cannot produce stable distinguishable states, adaptation results are not interpretable.

## Scientific assumptions and tradeoffs

Exponential decay provides a low-cost continuous-time state definition but is only computed when an event or snapshot requires it. Event propagation reduces state-update calls while adding heap operations and candidate transition checks; CPU savings are empirical, not implied by sparse degree. Sparse adjacency saves memory, while topology snapshots and readouts scan units. Coactivity-based rewiring is simple and local but can reinforce common input correlations or noise; the fixed-structure and fixed-weight controls are needed to isolate its effect. Signed weights allow excitation-like and inhibition-like contributions but do not represent biological synapses.

No graph metric is treated as intelligence. A topology with more edges or higher clustering has no value absent a measured retention or prediction improvement under a matched compute budget. The experiment makes no claims about consciousness, biological fidelity, semantic understanding, or learned physical laws.

## Commands and artifacts

Run the staged Experiment A with `python -m sensory_lab.liquid_experiment --output results/liquid`. It writes the complete configuration, five-seed dynamics and memory results, topology and compute telemetry, JSON snapshots (including input projection and RNG state), plots, and `LIQUID_ARCHITECTURE_REPORT.md`. The default run does not execute the larger motion benchmark.

Next stages: validate the full default five-seed memory run and inspect queue/activity stability; then test I0/I1/I2 on matched audiovisual association with fixed readout budgets; only after those checks, add the matched familiar/strict-heldout motion experiment and compare against F/G/FF_direct.
