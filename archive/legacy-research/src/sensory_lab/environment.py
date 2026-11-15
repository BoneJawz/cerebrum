"""Ground-truth generator. The label/mapping must NEVER enter Observation."""
from dataclasses import dataclass
import numpy as np
from .contracts import Observation

MOTION_START_TICK = 5

@dataclass(frozen=True)
class WorldConfig:
    pixels: int = 33
    visual_hz: int = 20
    audio_hz: int = 2000
    ticks: int = 24
    tone_amplitude: float = 0.5
    noise_std: float = 0.015

    def __post_init__(self):
        if self.pixels < 5 or self.visual_hz < 1 or self.audio_hz < 2 * 400:
            raise ValueError("Invalid sensor dimensions/rates")
        if self.audio_hz % self.visual_hz:
            raise ValueError("audio_hz must be divisible by visual_hz for v0.1")
        if self.ticks < 6:
            raise ValueError("ticks must be >= 6")

class LineWorld:
    def __init__(self, config: WorldConfig):
        self.config = config

    def sample_initial_position(self, rng: np.random.Generator) -> int:
        """Keep the original randomized central start distribution at every speed."""
        return int(rng.integers(self.config.pixels // 3,
                                2 * self.config.pixels // 3))

    def episode(self, seed: int, direction: int, tone_hz: float,
                initial_position: int | None = None,
                velocity: int = 1,
                phase_offset: float | None = None) -> list[Observation]:
        """Pair a raw tone cue with movement direction and visual speed.

        Velocity is measured in visual pixels per visual frame. The benchmark
        predicts a future visual position from the initial scene and raw audio.
        Neither direction, velocity, nor target is attached to observations.
        Motion begins after a stationary interval.
        """
        if direction not in (-1, 1) or int(velocity) != velocity or velocity < 1:
            raise ValueError("direction must be +/-1 and velocity a positive integer")
        cfg = self.config
        rng = np.random.default_rng(seed)
        chunk = cfg.audio_hz // cfg.visual_hz
        if initial_position is not None and not 0 <= initial_position < cfg.pixels:
            raise ValueError("initial_position is outside the visual array")
        initial = (int(rng.integers(cfg.pixels // 3, 2 * cfg.pixels // 3))
                   if initial_position is None else int(initial_position))
        size = int(rng.choice([1, 3, 5]))
        start = MOTION_START_TICK
        out = []
        phase = (float(rng.uniform(0, 2*np.pi)) if phase_offset is None
                 else float(phase_offset))
        for k in range(cfg.ticks):
            pos = int(np.clip(initial + direction * int(velocity) * max(0, k - start),
                              0, cfg.pixels - 1))
            vision = np.zeros(cfg.pixels, dtype=np.float64)
            left = max(0, pos - size//2)
            right = min(cfg.pixels, pos + size//2 + 1)
            vision[left:right] = 1.0
            idx = np.arange(k*chunk, (k+1)*chunk, dtype=np.float64)
            # The experiment benchmark maps speed to carrier frequency and
            # direction to a phase offset. Both remain raw waveform properties.
            envelope = min(1., (k+1)/3.)
            audio = cfg.tone_amplitude * envelope * np.sin(
                2*np.pi*tone_hz*idx/cfg.audio_hz + phase
            ) + rng.normal(0, cfg.noise_std, chunk)
            out.append(Observation(k/cfg.visual_hz, vision, audio.astype(np.float64)))
        return out
