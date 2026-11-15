"""Explicit configuration for the canonical distributed association network."""
from __future__ import annotations

from dataclasses import asdict, dataclass


ARCHITECTURE_VERSION = "mdl-distributed-lif-stdp-v1"


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    visual_shape: tuple[int, int] = (5, 5)
    auditory_inputs: int = 3
    associative_neurons: int = 18
    visual_probability: float = 0.35
    visual_weight_mv: float = 12.0
    auditory_gain_mv: float = 60.0
    rest_mv: float = -70.0
    threshold_mv: float = -50.0
    membrane_tau_ms: float = 10.0
    refractory_ms: float = 5.0
    timestep_ms: float = 0.1
    stdp_tau_ms: float = 20.0
    stdp_pre_increment: float = 0.01
    stdp_post_increment: float = -0.012
    weight_min: float = 0.0
    weight_max: float = 1.0
    auditory_weight_min: float = 0.05
    auditory_weight_max: float = 0.15
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        if len(self.visual_shape) != 2 or any(int(x) < 1 for x in self.visual_shape):
            raise ValueError("visual_shape must contain two positive dimensions")
        if self.auditory_inputs < 1 or self.associative_neurons < 1:
            raise ValueError("population sizes must be positive")
        if not 0.0 <= self.visual_probability <= 1.0:
            raise ValueError("visual_probability must be between 0 and 1")
        if self.timestep_ms <= 0 or self.membrane_tau_ms <= 0 or self.stdp_tau_ms <= 0:
            raise ValueError("time constants and timestep must be positive")
        if self.weight_min >= self.weight_max:
            raise ValueError("weight bounds must be increasing")
        if not self.weight_min <= self.auditory_weight_min <= self.auditory_weight_max <= self.weight_max:
            raise ValueError("initial auditory weights must fit within permitted bounds")

    @property
    def visual_inputs(self) -> int:
        return int(self.visual_shape[0] * self.visual_shape[1])

    def manifest(self) -> dict:
        data = asdict(self)
        data["visual_shape"] = list(self.visual_shape)
        return data
