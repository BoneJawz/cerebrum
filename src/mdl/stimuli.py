from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def one_cell_visual_array(position_index: int, n_cells: int = 3) -> np.ndarray:
    """Construct a binary one-dimensional visual field with one occupied cell."""
    if n_cells < 1 or position_index not in range(n_cells):
        raise ValueError("position_index must identify a cell in the visual field")
    pattern = np.zeros(n_cells, dtype=np.uint8)
    pattern[position_index] = 1
    return pattern


def encode_visual_array(pattern: np.ndarray) -> np.ndarray:
    """Convert a raw binary visual array to active sensor-cell spike indices."""
    values = np.asarray(pattern)
    if values.ndim != 1 or values.size == 0 or not np.isin(values, (0, 1)).all():
        raise ValueError("visual input must be a non-empty one-dimensional binary array")
    return np.flatnonzero(values).astype(int)


@dataclass(frozen=True)
class Episode:
    cue: int
    target: int
    audio_times: tuple[float, ...] = (0.010,)
    teacher_times: tuple[float, ...] = (0.025,)


@dataclass(frozen=True)
class AssociationTask:
    """Three arbitrary spike-pattern to one-cell visual associations.

    ``cue`` is an input-neuron index and ``target`` is an output-neuron index;
    neither is supplied as a semantic label to the network.  A seed-specific
    permutation prevents a fixed index convention from being learned.
    """

    permutation: tuple[int, ...]
    episodes: tuple[Episode, ...]

    @property
    def n_items(self) -> int:
        return len(self.permutation)

    def episode(self, cue: int) -> Episode:
        return Episode(cue=cue, target=self.permutation[cue])


def make_task(seed: int = 0, exposures: int = 1) -> AssociationTask:
    if exposures < 0:
        raise ValueError("exposures must be non-negative")
    rng = np.random.default_rng(seed)
    permutation = tuple(int(x) for x in rng.permutation(3))
    episodes = tuple(
        Episode(cue=cue, target=permutation[cue])
        for _ in range(exposures)
        for cue in rng.permutation(3)
    )
    return AssociationTask(permutation, episodes)
