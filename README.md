# Multisensory Dynamics Lab — literature-based foundation

The active Version 0.2 milestone is a minimal, reproducible associative-learning experiment. Three arbitrary auditory spike patterns must be paired with three one-cell visual outputs. A seed-specific permutation prevents index conventions from carrying the answer. During teaching, a delayed visual teacher spike drives the selected output; during recall the teacher is removed and output spike counts select the candidate.

The learning mechanism is bounded pair-based spike-timing-dependent plasticity (STDP), implemented with Brian2. Controls are no learning and mismatched teaching. Five seeds and 0/1/5/10/20/50 exposures are run by:

```powershell
python -m pip install -e ".[dev]"
python -m mdl.cli --outdir results/association_baseline
```

Artifacts include `configuration.json`, `trial_results.csv`, `learning_curve.csv`, `learning_curve.png`, and `ASSOCIATION_BASELINE_REPORT.md`. Prior work is archived in [archive/legacy-research](archive/legacy-research/README.md).

To evaluate fixed teaching doses without corrective feedback, run:

```powershell
python -m mdl.fixed_dose_cli --outdir results/fixed_dose_validation
```

This compares no-learning, correct, and fixed mismatched teaching across five seeds and 0/5/20/50 presentations per association, then evaluates 20 jittered recall trials per cue with plasticity disabled. Peak membrane depolarization is the primary response measure; seed-level records and the report are saved with the results.

## Live continuous experience

Run `python -m mdl.live_dashboard` for the persistent keyboard-driven dashboard. `A/B/C` present auditory cues; `Alt+A/B/C` pairs a cue with a binary visual array; `Shift+A/B/C` preserves the legacy direct-teacher pairing. Press `1/2/3` to choose a visual cell for either pairing. `J` presents a visual array alone and `N` toggles the seeded visual sensory stream. `V` presents a teacher-only event and `E` toggles the legacy teacher stream. `0/G/P` select plasticity off / paired-gated / continuous, and `M` mutes only the human-audible tone representation. `+/-` changes its volume. Press Space to pause/resume, `S` saves learned parameters plus the event log, and `F` saves a Brian2 full-state checkpoint.

Use `python -m mdl.live_dashboard --distributed` to select the separate 5×5 dashboard. `A/B/C` play auditory-only cues; `Alt+A/B/C` pair each cue with its corresponding 5×5 pattern; `Shift+A/B/C` pairs the selected cue with the currently selected pattern. `1/2/3` selects vertical, horizontal, or diagonal; `J` presents that visual pattern alone; `N` toggles seeded automatic audiovisual experiences. `T` enables the selected learning mode for teaching; `R` enters assessment mode and forces plasticity off. `0/G/P`, Space, `M`, `S`, and `F` control learning mode, pause, audio, session recording, and Brian2 checkpoints. The dashboard shows all 18 associative voltages, a compact 46-channel sensory/association spike raster, and the seeded network topology. Auditory STDP synapses are colored and weighted by current efficacy; brighter neurons indicate recent spikes, and briefly highlighted connections mark recent presynaptic-to-postsynaptic spike pairs. Simulation advances in fixed 10 ms biological chunks independently of the plot refresh interval.

In the legacy teacher controls, the three visual cells depict the externally supplied teacher target. In the sensory controls, the binary visual array is encoded as an event for the separate visual sensory group. Audible tones are reproducible human-audible representations of the neural cue IDs; the network receives spike events, not audio waveforms. The dashboard labels these pathways separately.

Run the headless plasticity-mode comparison with:

```powershell
python -m mdl.continuous_learning_cli --outdir results/continuous_learning
```

It compares plasticity-off, teaching-gated, and continuously enabled STDP across five seeds and correct/mismatched teacher mappings, with paired-exposure checkpoints and frozen recall evaluation.

## Binary visual sensory pathway

Run `python -m mdl.visual_sensory_cli --outdir results/visual-sensory-pathway` to evaluate an actual three-cell binary visual array input. The array is encoded as a spike in a separate visual sensory group and reaches the output through a fixed one-to-one relay; the legacy direct teacher pathway is not used in this experiment. Audio-only recall is assessed by peak voltage and by unique, nonzero output spikes at 60, 90, and 120 mV readout gains. It compares plasticity-off, paired-gated STDP, and continuous STDP across five seeds and correct/mismatched audiovisual pairings. The fixed visual relay remains an explicit modeling assumption, not a learned visual encoder.

## Distributed visual association

Run `python -m mdl.distributed_visual_cli` for the separate 5×5 distributed-visual experiment, including development seeds 42–46 and prospective seeds 101–105. Use `--development-only` to omit the prospective cohort, or `--validation-only --outdir results/distributed_visual_association_validation` to run the locked prospective cohort separately. It compares a fixed seed-controlled sparse random visual projection and auditory STDP into a shared 18-neuron population. Audio-only population responses are matched to pretraining visual references by frozen cosine similarity; voltage and spike-count evidence are reported separately. Controls include plasticity off, paired-gated STDP, continuous STDP, temporally unpaired presentations, and the remapped A→C, B→A, C→B association. Results are written under `results/distributed_visual_association/`. The preceding three-cell silent-recall diagnosis is in `results/recall_diagnostics/`.
