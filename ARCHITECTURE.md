# Active architecture

The active foundation separates deterministic stimulus generation (`mdl.stimuli`), Brian2 neuron/synapse dynamics (`mdl.network`), candidate readout (`mdl.assessment`), experiment scheduling/controls (`mdl.experiment`), and telemetry/artifacts. The network receives only spike events. Teacher events are present only in teaching episodes and are absent during recall.

Archived A–I and liquid experiments remain under `archive/legacy-research/`; this milestone does not modify their equations or reinterpret their results.

## Binary visual sensory pathway

`STDPAssociator(visual_sensory=True)` adds a three-cell binary visual event group, with a fixed one-to-one sensory relay into the three output cells. `visual_array` is encoded only as active visual sensor spike indices; the separate `teacher` spike group remains a distinct legacy control. This deliberately simple relay lets visual input evoke postsynaptic spikes while auditory cue synapses adapt under the existing pair-based STDP rule. It is not a learned visual encoder. The live dashboard uses the sensory pathway for Alt+A/B/C audiovisual pairings and the seeded `N` stream; Shift+A/B/C and `E` preserve the direct-teacher demonstration.

## Distributed visual association configuration

`mdl.distributed_network.DistributedVisualAssociator` is a separately selectable Brian2 configuration. It accepts a raw 5×5 binary array as 25 visual spike channels and three auditory spike channels. Both project to a shared 18-cell LIF population. The visual projection is fixed, sparse, random, and determined only by the network seed; it has no pattern-label or mapping input. Auditory afferents use the existing pair STDP equations. There is no output teacher, recurrent population, learned visual encoder, or structural plasticity. `mdl.distributed_visual_experiment` compares frozen cosine similarity of audio-only population responses against pretraining visual references under no-learning, paired-gated, continuous, unpaired, identity, and remapped controls. The readout, event encoding, connectivity generator, and metrics remain separate from the baseline three-cell network.

The distributed live dashboard renders that network's actual auditory, visual, and associative populations and its fixed and plastic projections. It reads current Brian2 weights, voltages, and recorded spike events for visualization; these displays do not feed back into simulation. `T` and `R` select teaching and assessment phases, with assessment explicitly disabling plasticity.
