# Nengo Associative Baseline Report

## Reference reproduction

Nengo 4.0.0 was used with the documented Voja/PES architecture: a 200-neuron key ensemble, Voja encoder adaptation, PES decoder learning, sequential five-key presentation, and a negative learning signal during recall. The reference run used seed 42 and dt=0.001 s. Recall RMSE was 0.7386; encoder change L2 was 418.2747. This reproduces the mechanism and produces finite recall; the full time-series plots are saved separately.

## Spatial adaptation

The adapted task uses only -1, 0, and +1. Each 4-tick acoustic instruction is transformed through the existing 69-D waveform-derived audio feature interface and a fixed 69x16 random projection. The PES target is a fixed 199x16 projection of the actual one-cell visual array; candidate images are still the raw 199-cell arrays and are passed through the same frozen projection before cosine scoring. This intermediate control keeps the image provenance explicit while making the decoder CPU-sized. Coordinates, token IDs, candidate indices, and answer labels remain controller-side. Training presents the projected visual target to the PES error pathway; recognition presents only the audio cue, runs the neural dynamics, and scores cosine similarity between recalled output and each projected candidate image. Learning is disabled during recall.

| Condition | Exposure 0 accuracy | Exposure 50 accuracy |
|---|---:|---:|
| no_learning | 0.333 | 0.333 |
| correct | 0.333 | 0.333 |
| mismatched | 0.333 | 0.333 |

Chance for the three-choice assessment is 0.333. Results are five-seed means; seed-level rows remain in `nengo_learning_curves.csv`. The no-learning and mismatched conditions use the same number and duration of presentations as correct pairing. Independent nuisance waveform realizations are used for recognition.

## Interpretation

Voja/PES is supervised error-modulated associative learning. A successful spatial result would establish a conventional reference mechanism, not autonomous discovery of meaning. The engineered reference vectors are unlike the raw waveform interface; the spatial adaptation therefore tests a materially harder sensor-to-pattern mapping. Input projection and Nengo neuron counts are fixed before evaluation; no test score selects them.

Model I was not imported or modified. Future comparison should match output representation, candidate scorer, exposure schedule, CPU budget, and learning supervision. Only after this baseline passes its own recognition check should the liquid be compared against it.
