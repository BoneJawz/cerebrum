# Neural learning foundation instructions

The active stack is a small Brian2/NumPy association baseline. Historical sensory-lab models, experiments, reports, and results are preserved under `archive/legacy-research/` and must not be deleted or silently rewritten.

- Use deterministic seeds and report seed-level results; never invent benchmark values.
- Keep stimulus generation, neural dynamics, learning rule, readout, and metrics replaceable.
- Inputs are raw spike events/arrays; simulator labels and target indices must not enter processor inputs.
- The first learning rule is established pair-based STDP. Do not add structural plasticity, liquid-state mechanisms, neuromodulation, or Model I changes in this milestone.
- Distinguish biological-time simulation from wall-clock performance and instrument both when benchmarking.
- Compare no-learning, correct-teacher, and mismatched-teacher controls. Treat failure as a useful result.
- CPU-first; keep the Brian2 runtime in NumPy mode and pin dependencies.
