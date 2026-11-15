# Preliminary distributed association run

This folder preserves the first development run for auditability. Its unpaired control used a 500 ms audio-to-visual gap but only a 50 ms post-visual gap before the next audio event. That allowed residual STDP traces to span adjacent trials, so it was not a clean temporally unpaired negative control.

Use `results/distributed_visual_association/` for the corrected benchmark. The final unpaired schedule uses 250 ms gaps on both sides of the visual event, each longer than 12 STDP trace time constants. The network, pairing schedules, readout, and other settings were unchanged.
