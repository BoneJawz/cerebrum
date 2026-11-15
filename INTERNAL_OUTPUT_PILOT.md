# Minimal internal neural output pilot

## Question and design

Can the existing 18-neuron associative population drive a learned three-cell spiking response after audiovisual teaching, then select the taught output from auditory-only input?

The pilot added three LIF output neurons and a dense 18×3 associative-to-output synapse group to the same Brian2 network. Output neurons use the existing LIF equation. Output synapses use the existing bounded pair-STDP equations and the existing initial weight range. No recurrence, new sensory pathways, extra neuronal populations, or new learning equations were added. The existing auditory-to-associative STDP was frozen in every condition to isolate changes in the new output projection.

Each association experience presented one of three existing binary 5×5 patterns together with one auditory channel. After 40 ms, a separate supervised teaching operation depolarized the designated output neuron by 55 mV, evoking its postsynaptic spike. This teaching signal was delivered directly to the output voltage state; it was not encoded as an ordinary auditory or visual event. The network then continued for 60 ms without resetting neural state or weights. Each cue-pattern pair was experienced five times. Recall presented auditory-only cues for 60 ms, with output plasticity disabled and no teacher pulse.

Conditions were paired output STDP with identity mapping, plasticity disabled with identity mapping, and paired output STDP with remapping `(1, 2, 0)`. All conditions ran with seeds 42 and 43. The diagnostic external cosine readout compared each auditory-only associative response with visual-only reference responses recorded before teaching. Python scored output spikes but did not inject a prediction.

## Results

The experiment took **32.99 seconds** according to its internal wall-clock timer. Each output recall was silent: all three output spike counts were `[0, 0, 0]`. Thus there were no selections, no correct output associations, and no ties in any of the 18 auditory-only recall trials.

| Seed | Condition | Output weight L1 change | Output recall accuracy | Silent / tied trials | External cosine accuracy |
|---:|---|---:|---:|---:|---:|
| 42 | Paired STDP, identity | 6.0254 | 0/3 | 3 / 0 | 1/3 |
| 42 | Plasticity off, identity | 0.0000 | 0/3 | 3 / 0 | 1/3 |
| 42 | Paired STDP, remapped | 6.0245 | 0/3 | 3 / 0 | 1/3 |
| 43 | Paired STDP, identity | 5.5091 | 0/3 | 3 / 0 | 2/3 |
| 43 | Plasticity off, identity | 0.0000 | 0/3 | 3 / 0 | 2/3 |
| 43 | Paired STDP, remapped | 5.5076 | 0/3 | 3 / 0 | 1/3 |

Paired teaching changed output weights in both seeds, while the plasticity-disabled controls changed none. However, these local synaptic changes did not produce output spikes during recall. The neural output pathway therefore **failed to demonstrate learned auditory-only output selection** in this pilot. The external cosine diagnostic produced selections, but its identity-condition accuracy was the same with and without output plasticity for each seed; it did not provide evidence that the internal output pathway worked. Under remapping, its fixed pretraining references were generally misaligned with the remapped targets.

## Interpretation and limits

The teacher pulse guarantees a postsynaptic output spike during teaching, allowing the local STDP rule to update weights near active associative presynaptic cells. The resulting weight change is evidence that the update mechanism ran, not evidence that the system learned a usable mapping. At recall, the auditory-only drive did not bring any output neuron to threshold. This experiment does not establish that the associative population's auditory responses contain enough separable information, or that the selected output synaptic efficacy is sufficient to express it.

The outcome is limited to two seeds, five teaching experiences per cue, one fixed gain and timing, and three synthetic patterns. The cosine readout is an external diagnostic and is not a neural decision. No parameter was changed after seeing the results. A next experiment would first diagnose, without tuning, whether the associative presynaptic spike patterns during auditory-only recall overlap the cells potentiated by audiovisual teaching; any later gain or timing study should be separately prespecified with matched controls.

## Reproduction

```powershell
python experiments/internal_output_pilot.py
```

The script uses seeds 42 and 43, five experiences per cue, the fixed timing and teacher amplitude above, and prints the per-seed results as JSON.
