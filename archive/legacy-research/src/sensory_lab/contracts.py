from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class Observation:
    t: float
    vision: np.ndarray  # binary 1D retinal pixels
    audio: np.ndarray  # raw signed waveform chunk, NOT acoustic labels

class Processor(Protocol):
    name: str
    def reset(self) -> None: ...
    def step(self, observation: Observation) -> np.ndarray: ...
    @property
    def updates(self) -> int: ...

class Environment(Protocol):
    def episode(self, seed: int, direction: int, tone_hz: float) -> list[Observation]: ...

class Readout(Protocol):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None: ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...
