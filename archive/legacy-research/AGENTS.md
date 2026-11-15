# Codex project instructions

## Mission
Investigate whether a modular, distributed dynamical processing substrate can integrate raw visual and acoustic streams with higher **sample efficiency and/or compute efficiency** than strong conventional baselines. Ultimately examine compact transferable governing dynamics learned across experiences; do not assume they emerge.

## Engineering rules
1. Keep environment, sensor interfaces, processors A-D, learner, readout, scheduler and telemetry independent and easily swapped.
2. CPU-first, Python 3.10+, NumPy first. Introduce larger frameworks only with a measured reason.
3. Include deterministic seed control; ground truth and sound codebooks must never leak into processor input.
4. Maintain tests before and after replacing modules. Never report invented benchmark numbers.
5. Preserve explicit distinction between continuous mathematical dynamics and continuous real-time computation.
6. Instrument wall-clock latency separately from model event time and update counts.
7. Prioritize falsifiable experiments, ablations, and honest comparisons over biomimetic storytelling.
8. Retain raw stream semantics; current audio chunk summary is a placeholder local interface, not an audio recognizer.

## Immediate Codex assignment
Read README.md and ARCHITECTURE.md; run the tests and CLI. First implement **v0.2 truly cross-modal dependency**, not a bigger model. Propose two complementary sensory tasks whose targets cannot be derived from vision or audio alone; implement one with unit tests for no single-modality leakage. Add vision-only/audio-only/both ablations, multiple seeds, and compute-matched baselines. Before editing explain the experiment, the failure modes, and acceptance criteria. Do not claim the current v0.1 demonstrates multimodal integration or learning laws.
