"""Convert raw numerical/binary stimuli to spike-channel events."""
from __future__ import annotations

import numpy as np


def encode_visual_array(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape or not np.isin(array, (0, 1, False, True)).all():
        raise ValueError(f"visual input must be a binary array with shape {shape}")
    return np.flatnonzero(array.reshape(-1)).astype(int)


def encode_auditory_event(channel: int, onset_s: float, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= int(channel) < channels:
        raise ValueError(f"auditory channel must be in [0, {channels})")
    if onset_s < 0:
        raise ValueError("event onset must be nonnegative")
    return np.array([int(channel)], dtype=int), np.array([float(onset_s)])
