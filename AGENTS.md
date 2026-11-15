# MDL neural-core instructions

- The canonical implementation is the CPU/NumPy Brian2 network in `src/mdl/`.
- Preserve the 25-channel binary visual input, 3 auditory spike channels, sparse fixed visual projection, and 18-cell default LIF population.
- Keep the established membrane and pair-based STDP equations unchanged unless a separately reviewed experiment explicitly authorizes a change.
- Keep labels and targets outside the neural input API. Sensory inputs are binary arrays and spike-channel events.
- Use deterministic seeds, report seed-level results, and keep biological simulation time distinct from wall-clock time.
- Keep configuration, sensory encoding, neural dynamics, plasticity, persistence, and tests as small replaceable modules.
- Do not add dashboards, pretrained encoders, recurrence, structural plasticity, new populations, or a general plugin framework to the neural core.
