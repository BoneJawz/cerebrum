# Minimal neural competition experiment

## Result

Adding fixed mutual inhibition did **not** improve output selection. Both baseline and competition produced **0 correct, 0 incorrect, 5 tied, and 1 silent** auditory-only recall responses across the six cue trials (three cues × two seeds). No condition produced a unique output-neuron decision.

The inhibition was delivered, but it did not suppress competitors before their first spikes. In every active competition trial, output neurons that spiked did so simultaneously at 10.2 ms. Inhibitory arrivals were also timestamped at 10.2 ms. Under Brian2's schedule, threshold checks occur before synaptic delivery, so same-timestep inhibition arrives too late to prevent those threshold crossings. There were no later output spikes for inhibition to separate.

## Protocol

For seeds 42 and 43, each condition independently reproduced the verified upstream procedure: 50 audiovisual paired experiences per cue, frozen associative weights, then five output-teaching experiences per cue using the same target-gated STDP procedure. The upstream recall spike counts matched across conditions (seed 42: 4/6/1; seed 43: 3/5/4), with 3/3 correct frozen cosine matches for both seeds. Seed-derived output initialization and cue order were identical across conditions; initial and final output weight sums matched between conditions for each seed.

The competition condition added only fixed all-to-all mutual inhibition between distinct output cells, with no self-connections. Each output spike delivered a 10 mV inhibitory voltage jump to the other two cells. This single strength was fixed before the run at half the existing 20 mV rest-to-threshold gap. The output LIF equation, excitatory gain, threshold, refractory period, teacher pulse, and target-gated plasticity were unchanged. The neural core was not modified.

## Baseline versus competition

Output counts are A/B/C; first-spike times are in ms. A/B/C are the intended outputs for cues A/B/C. A `—` means that output did not spike.

| Seed | Cue | Baseline counts | Competition counts | First spikes, competition A/B/C | Inhibition arrivals | Response in both |
|---:|:---:|:---:|:---:|---|---|---|
| 42 | A | 1/1/1 | 1/1/1 | 10.2 / 10.2 / 10.2 | all at 10.2 | Tie |
| 42 | B | 1/1/1 | 1/1/1 | 10.2 / 10.2 / 10.2 | all at 10.2 | Tie |
| 42 | C | 0/0/0 | 0/0/0 | — / — / — | none | Silent |
| 43 | A | 1/0/1 | 1/0/1 | 10.2 / — / 10.2 | all cells at 10.2 | Tie |
| 43 | B | 1/1/1 | 1/1/1 | 10.2 / 10.2 / 10.2 | all at 10.2 | Tie |
| 43 | C | 1/1/1 | 1/1/1 | 10.2 / 10.2 / 10.2 | all at 10.2 | Tie |

| Condition | Correct | Incorrect | Tied | Silent | Unique correct selections |
|---|---:|---:|---:|---:|---:|
| Baseline | 0 | 0 | 5 | 1 | 0/6 |
| Competition | 0 | 0 | 5 | 1 | 0/6 |

Across the five active trials, every output that spiked did so in the same timestep as at least one other output. Competition did not change counts, first-spike times, or response categories. In seed 43 cue A, output B was already silent in baseline; inhibition did not produce a unique response from A or C. The seed 42 cue C silence also remained unchanged.

The target-gated output weights remained matched between baseline and competition (L1 totals: 6.625 for seed 42 and 6.109 for seed 43 after teaching). Non-target weight changes were zero for every cue in both conditions. Thus response differences cannot be attributed to different trained output weights.

## Interpretation

The most likely failure mechanism is timing: the shared feedforward associative spikes drive several output neurons across threshold on the same 0.1 ms simulation step. Event-triggered lateral inhibition follows those threshold checks and cannot veto spikes already emitted in that step. With at most one spike per active output during recall, there is no later spike volley for this fixed inhibition to suppress. This experiment therefore shows that adding mutual inhibition alone, at the prespecified strength, did not resolve the output-selection failure. It does not establish that all forms or timings of competition would fail.

## Artifacts and reproducibility

- Experiment script: `experiments/output_competition_pilot.py`
- Trial-level counts, first-spike times, simultaneous groups, inhibition arrivals, and peaks: `results/output_competition/output_competition_trials.csv`
- Seeds: 42 and 43
- Fixed inhibition: 10 mV
- Runtime: 116.25 seconds

This is a small diagnostic pilot, not a broad estimate across seeds. No strength sweep, tuning, additional neurons, or core changes were performed.
