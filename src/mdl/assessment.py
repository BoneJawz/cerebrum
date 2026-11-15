from __future__ import annotations

import numpy as np
from .network import STDPAssociator


def choose(scores: np.ndarray) -> int:
    scores = np.asarray(scores, dtype=float)
    return int(np.argmax(scores))


def recall(network: STDPAssociator, cue: int) -> tuple[int, np.ndarray]:
    spikes = network.present(cue, target=None)
    return choose(spikes), spikes


def evaluate(network: STDPAssociator, permutation: tuple[int, ...]) -> float:
    hits = []
    for cue, target in enumerate(permutation):
        pred, _ = recall(network, cue)
        hits.append(pred == target)
    return float(np.mean(hits))
