# Implemented neural architectures

The active neural implementations are `src/mdl/network.py:17-128` and `src/mdl/distributed_network.py:17-146`. They share a LIF equation and auditory STDP form but are distinct Brian2 networks. There is no active SAL/liquid implementation; that project is under `archive/legacy-research/src/sensory_lab/liquid.py` and described by the legacy `liquid_architecture.md`.

## Population and connection table

| Configuration / component | Neuron/channel count | Inputs | Outputs | Plasticity | Function |
|---|---:|---|---|---|---|
| `STDPAssociator.input` | 3 spike channels | Experiment/controller cue index | Dense auditory synapses | None in group | Arbitrary auditory event channels A/B/C |
| `STDPAssociator.visual_input` | 3 spike channels, only if `visual_sensory=True` | Binary 3-cell visual array → active indices | Fixed one-to-one visual relay | None in group or relay | Direct sensory drive for the three-cell task |
| `STDPAssociator.teacher` | 3 spike channels, always constructed | Target index during teaching/legacy demo | Fixed one-to-one teacher synapses | None | External teaching signal; absent in recall calls |
| `STDPAssociator.output` | 3 LIF neurons | Auditory STDP, optional visual relay, teacher jump | `SpikeMonitor`, `StateMonitor`; external Python readout | Membrane state transient | Three candidate response channels |
| `STDPAssociator.plastic` | 3×3 = 9 synapses | Each auditory channel | Each output neuron | Bounded pair STDP, shared gain and gate | Store cue-to-output association |
| `STDPAssociator.visual_syn` | 3 fixed one-to-one synapses, conditional | Visual sensor `i` | Output `i` | Fixed; gain only | Hand-designed visual pathway; not a learned encoder |
| `STDPAssociator.teacher_syn` | 3 fixed one-to-one synapses | Teacher channel `i` | Output `i` | Fixed +55 mV voltage jump | Provides teaching target drive |
| `DistributedVisualAssociator.visual_input` | 25 channels | Raw binary 5×5 array converted to active spike indices | Sparse random visual projection | None in group | Binary grid sensor channels |
| `DistributedVisualAssociator.auditory_input` | 3 channels | Cue index | Dense auditory STDP | None in group | Synthetic auditory event channels |
| `DistributedVisualAssociator.associative` | 18 by default; constructor allows 12–24 | Visual and auditory projections | Voltage/spike monitors; external similarity readout | Membrane state transient | Shared population response; no dedicated output group |
| `DistributedVisualAssociator.visual_projection` | Bernoulli sparse subset of 25×18 possible edges; p=.35 by default | Each visual channel | Associative population | Fixed weights, default 12 mV | Seeded label-independent visual projection |
| `DistributedVisualAssociator.auditory_stdp` | 3×18 = 54 synapses | Each auditory channel | Associative population | Bounded pair STDP, shared gain and gate | Modifiable auditory projection |

No active network has recurrent synapses. No lateral, inhibitory, normalization, adaptation, homeostatic, or structural-plasticity population/pathway is constructed.

## Neuron and synapse equations

Both associative/output populations use Brian2 `NeuronGroup` with

```text
dv/dt = (-70*mV - v)/(10*ms)
threshold: v > -50*mV
reset: v = -70*mV
refractory: 5*ms
method: exact
```

These are current-based voltage jumps followed by exponential leak. The input groups are `SpikeGeneratorGroup`s; input representation is spike-channel identity and event time, not raw waveform samples.

For the auditory synapses, each connection has weight `w`, a shared voltage gain and plasticity gate, and event-driven traces `dapre/dt=-apre/(20*ms)` and `dapost/dt=-apost/(20*ms)`. On a presynaptic event: `v_post += w*gain`, `apre += gate*0.01`, and `w=clip(w+gate*apost,0,1)`. On a postsynaptic spike: `apost += gate*(-0.012)` and `w=clip(w+gate*apre,0,1)`. Thus the negative post trace supports depression for post-before-pre order and the positive pre trace supports potentiation for pre-before-post order. Initial auditory weights are uniform in [0.05, 0.15] using a seed-derived RNG. Learning is controlled by a shared binary gate.

The teacher relay adds `55*mV`; the optional three-cell sensory relay uses configurable `visual_gain_mv` (default 55 mV). Distributed visual edges add a fixed `w` in volts (default 12 mV). The distributed auditory synapses use configurable `auditory_gain_mv` (default 60 mV), initialized with a separate deterministic seed stream.

## State and dimensions

Configurable dimensions are limited. `STDPAssociator` hardcodes three auditory channels, three teacher channels, three output neurons, and a 3×3 weight reshape/validation. Its optional visual sensor is also three channels. The distributed model hardcodes 25 visual channels and three auditory channels, but exposes associative size only within 12–24; its auditory weights reshape as `(3, n_associative)`. Patterns and the dashboard assume 5×5.

The state variables that can persist in a live network are output/associative voltage `v`, auditory synaptic `w`, `apre`, `apost`, simulation clock, scheduled spike events, and monitor history. Group sizes and equation objects are fixed when each constructor runs.

## Configuration distinction

1. **Three-output baseline:** `STDPAssociator(..., visual_sensory=False)` still constructs auditory input, teacher input, and three outputs. Used by baseline learning and most controlled association diagnostics.
2. **Three-cell sensory variant:** `STDPAssociator(..., visual_sensory=True)` adds a three-cell visual `SpikeGeneratorGroup` and one-to-one fixed relay. Used by the persistent live controller and visual-sensory experiment. Teacher group remains present but separate.
3. **Distributed 5×5 variant:** `DistributedVisualAssociator` is an independent 25+3→18 circuit, with sparse fixed visual projection and plastic dense auditory projection. Used by distributed experiments/dashboard and temporal/action pilots.

There is no adapter that makes these network constructors interchangeable. `live_dashboard.py --distributed` selects configuration 3; default dashboard selects configuration 2. The current CLI is not a parameterized choice of sizes within one canonical architecture.

## Neural computation versus output decisions

Brian2 outputs voltage samples and threshold spikes only. The three-neuron baseline has no explicit winner-take-all or tie-resolution circuit; `assessment.choose` uses `numpy.argmax`. The distributed network has no output neurons: `distributed_visual_experiment` compares peak-voltage/spike vectors against frozen reference vectors via external cosine similarity. The action/temporal pilots use external nearest-centroid classifiers, and the action-inspired score is Python arithmetic. These are evaluation/readout procedures, not neural populations.
