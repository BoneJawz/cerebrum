# Quick output plasticity gating pilot

## Result

**Decision B: gating made weight changes target-specific, but actual output spike decisions remained tied or silent.** Across six auditory-only recall trials per condition, ungated and target-gated STDP each produced **0 correct, 0 incorrect, 5 ties, and 1 silent**. No trial yielded a unique output-neuron selection.

The target-gated manipulation did what it was designed to do: for each taught cue, the two non-target output columns changed by exactly zero. This is a gate compliance check, not independent evidence that the network learned selective associations. The target output's weight increased, but gating did not produce more selective spike counts. Peak voltages retained a target preference in five of six trials in both conditions.

## Protocol

For seeds 42 and 43, the experiment recreated the verified upstream auditory training: 50 paired audiovisual experiences per cue, then frozen auditory-to-associative weights. The independently recreated networks matched the prior diagnostic's auditory-only associative spike counts (seed 42: 4/6/1; seed 43: 3/5/4) and had 3/3 correct frozen cosine matches. Since Model I has no output-to-associative feedback, the output conditions use the same trained parameters and deterministic stimulus schedule. Both conditions initialize their output weights from the same seed-derived matrix.

- **Ungated baseline:** existing pair-STDP runs throughout the 40 ms audiovisual teaching event and teacher pulse.
- **Target-gated:** sensory transmission stays active and pair-STDP traces continue updating, while weight changes are disabled during the sensory event. For the one simulator step from 40.0 to 40.1 ms, weight updates are allowed only on synapses ending at the teacher-designated output. The gate is then closed. This preserves the existing LIF equations, STDP increments, gains, and teaching pulse.

Each condition used five teaching experiences per cue. Auditory-only recall had no teacher and no plasticity. The target-gated STDP expression lives only in this experiment script; the neural core was not modified.

## Results

Counts are output A/B/C. A/B/C are also the intended target for cues A/B/C. Peak margins are intended-output peak voltage minus the strongest competitor peak, in mV.

| Seed | Condition | Output counts for cues A / B / C | Correct / incorrect / tied / silent | Peak margins A / B / C (mV) | Total output weight L1 change |
|---:|---|---|---|---|---:|
| 42 | Ungated | 1/1/1, 1/1/1, 0/0/0 | 0 / 0 / 2 / 1 | +5.395 / +8.606 / +2.075 | 6.188 |
| 42 | Target-gated | 1/1/1, 1/1/1, 0/0/0 | 0 / 0 / 2 / 1 | +5.057 / +8.446 / +1.980 | 0.960 |
| 43 | Ungated | 1/1/1, 1/1/1, 1/1/1 | 0 / 0 / 3 / 0 | +3.800 / −0.045 / +4.881 | 5.661 |
| 43 | Target-gated | 1/0/1, 1/1/1, 1/1/1 | 0 / 0 / 3 / 0 | +3.705 / −0.165 / +4.732 | 0.900 |

Cue-specific summed weight changes to outputs A/B/C were:

| Seed | Condition | Cue A deltas | Cue B deltas | Cue C deltas |
|---:|---|---|---|---|
| 42 | Ungated | 0.883 / 0.544 / 0.541 | 0.732 / 1.134 / 0.728 | 0.462 / 0.468 / 0.697 |
| 42 | Target-gated | 0.332 / 0 / 0 | 0 / 0.401 / 0 | 0 / 0 / 0.227 |
| 43 | Ungated | 0.834 / 0.504 / 0.507 | 0.593 / 0.919 / 0.593 | 0.480 / 0.483 / 0.749 |
| 43 | Target-gated | 0.320 / 0 / 0 | 0 / 0.319 / 0 | 0 / 0 / 0.261 |

Gating reduced total weight change by limiting updates to the target column and pulse window. It did **not** improve recall selection. Spike counts still tie because sensory-driven transmission remains strong enough to make competing output neurons spike during recall; the one near-threshold cue remains silent for seed 42. Target peak margins are similar to the ungated baseline, so restricting plasticity alone did not materially separate the neural responses at threshold.

## Interpretation and next step

This result supports the specific diagnosis that untargeted postsynaptic events contribute to broad STDP potentiation: restricting updates eliminates competitor-column changes. But that broad potentiation is not the only obstacle to output decisions. With forward transmission unchanged, competing output cells continue to spike together. The experiment does not establish that target-gated STDP improves recognition.

The next justified experiment is a controlled local output-neuron competition mechanism, compared against this target-gated condition while holding upstream weights, sensory drive, teacher schedule, and output readout fixed. Do not infer a benefit until unique correct spike selections improve across additional seeds.

## Artifacts and reproducibility

- Script: `experiments/output_gating_pilot.py`
- Trial-level spikes and voltage peaks: `results/output_gating_pilot/output_gating_trials.csv`
- Per-cue synaptic changes: `results/output_gating_pilot/output_gating_weights.csv`
- Seeds: 42, 43
- Runtime: 113.06 seconds on the current CPU environment

The small two-seed pilot is diagnostic rather than a population-level estimate. No parameter search or neural-core changes were made.
