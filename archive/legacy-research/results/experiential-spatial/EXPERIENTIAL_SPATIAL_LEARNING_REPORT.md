# Experiential Spatial Learning — Checkpoint 1

## Acoustic codebook audit

The decimal vocabulary contains sign tokens `+` and `-` and digit tokens `0` through `9`. Zero is one `0` token. Nonzero positions use sign followed by magnitude digits from most significant to least significant: -37 is `-`, `3`, `7`; +3 is `+`, `3`; +30 is `+`, `3`, `0`. Each token lasts 4 ticks (0.2 s), with 2 ticks (0.1 s) of silence between adjacent tokens. The full instruction is 16 ticks (0.8 s) at 2,000 waveform samples/s, divided into 100-sample observations at 20 Hz. Two-digit instructions exactly fill the duration; shorter instructions end in silence padding. Identical digit tokens use the same underlying waveform in every context.

Each model observation contains timestamp, binary vision, and raw audio chunk. The existing liquid feature transform receives those streams. Probe labels and vocabulary metadata are diagnostic targets only. Current `make_digit_episode` presents vision and audio simultaneously; the phase-separated teaching sequence has not yet been added.

## Checkpoint 1 protocol

The fixed I0 reservoir was tested across 5 seeds. Each of 12 tokens had 8 training and 4 held-out waveform variants with modest gain, circular shift, and additive-noise nuisance. One-hot ridge probes used the frozen final state, the concatenated four-tick state trajectory, or the concatenated 69-D-per-tick audio features that feed the liquid. The liquid never received class labels. The token gate uses the time-resolved state trajectory (mean >= 80%; every seed >= 70%); terminal-state performance is reported separately.

## Results

| Probe | Mean accuracy | Per-seed range | Chance |
|---|---:|---:|---:|
| all_tokens_12way | 0.300 | 0.146–0.542 | 0.083 |
| liquid_trajectory_12way | 0.950 | 0.854–1.000 | 0.083 |
| raw_audio_features_12way | 1.000 | 1.000–1.000 | 0.083 |
| sign_2way | 0.375 | 0.125–0.875 | 0.500 |
| digit_10way_conditional | 0.310 | 0.150–0.600 | 0.100 |
| digit_in_12way | 0.285 | 0.125–0.600 | 0.083 |

**Token-recognition checkpoint: PASS.** Trajectory-probe mean: 0.950; minimum seed: 0.854.

The raw features and time-resolved liquid trajectory retain token identity, but a single terminal liquid state performs much worse. This points to temporal aggregation or state retention as the immediate bottleneck rather than inadequate acoustic feature separation. The probe results do not establish that the liquid learned token semantics. Per-token confusions are in `experiential_token_confusion.csv`.

## Ordered-instruction diagnostics

These supervised probes measure information recoverable from complete audio instructions; they do not show that the liquid learned numerical composition through local plasticity.

| Diagnostic | Readout | Mean accuracy | Mean MAE | Mean within 1 |
|---|---|---:|---:|---:|
| complete_instruction_position | terminal | 0.000 | 56.58 | 0.014 |
| complete_instruction_position | trajectory | 0.000 | 48.99 | 0.021 |
| digit_place_role | token_end_state | 0.801 | — | — |
| instruction_length | terminal | 0.922 | — | — |
| instruction_length | trajectory | 0.951 | — | — |

### Counterfactual component changes

Stimuli intervene on one component relative to +37. Predictions are from supervised full-position probes; targets are analysis labels only.

| Intervention | Target | Terminal prediction | Trajectory prediction |
|---|---:|---:|---:|
| change ones digit only | 38 | -5.13 | 44.77 |
| change sign only | -37 | -14.66 | -59.02 |
| change tens digit only | 47 | 30.86 | 27.06 |
| reference +37 | 37 | -38.14 | 66.56 |
| swap digit order | 73 | -17.92 | 22.20 |

## Three-position readout-source checkpoint

I0 is a fixed liquid here. Readouts are fit on labels from their indicated training modality. The vision-only readout is a pre-exposure calibration baseline; this experiment includes no audiovisual plasticity. Terminal-state and four-ordered-bin temporal readouts are compared. Exact accuracy counts a prediction within 0.5 position units of its integer target.

| Representation | Readout source | Test modality | Exact accuracy | Within 1 | MAE |
|---|---|---|---:|---:|---:|
| terminal | audio | audio | 0.283 | 0.567 | 0.921 |
| terminal | audio | both | 0.383 | 0.617 | 0.845 |
| terminal | audio | vision | 0.333 | 0.400 | 4.172 |
| terminal | both | audio | 0.333 | 0.550 | 0.931 |
| terminal | both | both | 0.317 | 0.533 | 1.000 |
| terminal | both | vision | 0.200 | 0.533 | 5.730 |
| terminal | vision | audio | 0.367 | 0.683 | 0.799 |
| terminal | vision | both | 0.350 | 0.633 | 0.825 |
| terminal | vision | vision | 0.867 | 1.000 | 0.188 |
| trajectory | audio | audio | 0.467 | 0.750 | 0.656 |
| trajectory | audio | both | 0.450 | 0.733 | 0.646 |
| trajectory | audio | vision | 0.200 | 0.267 | 11.684 |
| trajectory | both | audio | 0.400 | 0.817 | 0.693 |
| trajectory | both | both | 0.583 | 0.850 | 0.510 |
| trajectory | both | vision | 0.133 | 0.267 | 16.043 |
| trajectory | vision | audio | 0.133 | 0.200 | 4.723 |
| trajectory | vision | both | 0.083 | 0.117 | 5.343 |
| trajectory | vision | vision | 1.000 | 1.000 | 0.001 |
| raw_audio_features | audio | audio | 0.883 | 1.000 | 0.265 |

The AV-trained readout evaluated on audio-only states measures modality-induced state/readout shift. The audio-supervised row is a labeled diagnostic control. Vision calibration evaluated on audio is a pre-exposure transfer baseline; transfer after plastic adaptation remains gated.

## Checkpoint decision

The token-recognition checkpoint passes on time-resolved states, but fails for a terminal state. Complete-instruction position decoding remains near the constant-zero baseline (trajectory MAE 48.99 vs 49.75 baseline; exact accuracy 0). On the simpler three-position task, mean exact accuracy is 0.467 for an audio-trained trajectory readout, 0.400 for an AV-trained trajectory readout evaluated on audio, and 0.583 for that AV readout evaluated on AV. The supervised raw-audio-feature control reaches 0.883. Each seed contributes 12 held-out noisy trials to each three-position condition.

**The simple position-association checkpoint is not established for the liquid.** The audio-feature control indicates the cues and labels are recoverable by a direct supervised readout; the liquid-state readouts do not reliably recover them, and the AV-to-audio shift remains. Do not interpret this as evidence for or against adaptive plasticity: I0 is fixed and no repeated audiovisual teaching or target feedback occurred.

Next controlled experiment: hold the three-position data fixed and compare terminal state, ordered liquid trajectory, and raw acoustic features with an equally declared readout and training budget. Diagnose projection/temporal summarization before enabling I1/I2 plasticity or repeated feedback.

No repeated-association, feedback, retention, curriculum, or adaptive-advantage result is claimed here.
