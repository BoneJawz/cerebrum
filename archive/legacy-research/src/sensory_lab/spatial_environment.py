"""Synthetic compositional acoustic vocabulary and static spatial scene.

Vocabulary meanings remain simulator metadata. Model observations contain only
binary retina samples, raw waveform chunks, and timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path

import numpy as np

from .contracts import Observation


FIELD_MIN = -10
FIELD_MAX = 10
POSITIONS = tuple(range(-5, 6))
SAMPLE_RATE = 2000
VISUAL_RATE = 20
CHUNK_SIZE = SAMPLE_RATE // VISUAL_RATE
TOKEN_TICKS = 4
SILENCE_TICKS = 2
EPISODE_TICKS = 10
SIGNS = (-1, 1)
MAGNITUDES = tuple(range(1, 6))


@dataclass(frozen=True)
class Token:
    name: str
    category: str
    waveform: np.ndarray


class SpatialVocabulary:
    """Stable arbitrary word-like waveforms, balanced for duration and RMS."""

    def __init__(self, seed: int = 731):
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # Scrambled patterns avoid a monotonic number-to-carrier assignment.
        names = ["POSITIVE", "NEGATIVE", "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        categories = ["sign", "sign", "magnitude", *(["magnitude"] * 5)]
        frequency_code = rng.permutation(np.array([170., 230., 310., 390., 470., 550., 630., 710.]))
        self.tokens: dict[str, Token] = {}
        t = np.arange(TOKEN_TICKS * CHUNK_SIZE, dtype=np.float64) / SAMPLE_RATE
        # Each token has four pieces with smooth edges. All have the same RMS.
        window = np.ones_like(t)
        ramp = 8
        window[:ramp] = np.linspace(0, 1, ramp, endpoint=False)
        window[-ramp:] = np.linspace(1, 0, ramp, endpoint=False)
        for i, (name, category) in enumerate(zip(names, categories)):
            base = frequency_code[i]
            # Distinct temporal pattern: the same multiset of offsets is used
            # for every token, while their order is seed-scrambled.
            offsets = rng.permutation(np.array([-37., -11., 13., 41.]))
            signal = np.zeros_like(t)
            for part in range(TOKEN_TICKS):
                mask = (np.arange(len(t)) // CHUNK_SIZE) == part
                phase = rng.uniform(0, 2 * np.pi)
                signal[mask] = np.sin(2 * np.pi * (base + offsets[part]) * t[mask] + phase)
            signal *= window
            signal /= np.sqrt(np.mean(signal ** 2))
            self.tokens[name] = Token(name, category, signal.astype(np.float64))

    def token(self, name: str) -> np.ndarray:
        if name not in self.tokens:
            raise KeyError(name)
        return self.tokens[name].waveform.copy()

    def instruction(self, position: int) -> np.ndarray:
        """Return fixed 0.5 s grammar; ZERO is one canonical magnitude token."""
        position = int(position)
        if position == 0:
            chunks = [self.token("ZERO")]
            chunks += [np.zeros(TOKEN_TICKS * CHUNK_SIZE)]
            chunks += [np.zeros(SILENCE_TICKS * CHUNK_SIZE)]
        else:
            sign = "POSITIVE" if position > 0 else "NEGATIVE"
            magnitude = ("ONE", "TWO", "THREE", "FOUR", "FIVE")[abs(position) - 1]
            chunks = [self.token(sign), np.zeros(SILENCE_TICKS * CHUNK_SIZE), self.token(magnitude)]
        waveform = np.concatenate(chunks)
        expected = EPISODE_TICKS * CHUNK_SIZE
        if len(waveform) != expected:
            raise AssertionError(f"instruction has {len(waveform)} samples; expected {expected}")
        return waveform

    def vocabulary_rows(self) -> list[dict[str, object]]:
        rows = []
        for name, token in self.tokens.items():
            wave = token.waveform
            spectrum = np.abs(np.fft.rfft(wave))
            freqs = np.fft.rfftfreq(len(wave), 1 / SAMPLE_RATE)
            rows.append({"token": name, "category": token.category,
                         "duration_seconds": len(wave) / SAMPLE_RATE,
                         "rms": float(np.sqrt(np.mean(wave ** 2))),
                         "peak": float(np.max(np.abs(wave))),
                         "mean_absolute_amplitude": float(np.mean(np.abs(wave))),
                         "zero_crossing_rate": float(np.mean(np.diff(np.signbit(wave)) != 0)),
                         "spectral_centroid_hz": float(np.sum(freqs * spectrum) / np.sum(spectrum)),
                         "waveform_seed": self.seed})
        return rows

    def write_csv(self, path: str | Path) -> None:
        rows = self.vocabulary_rows()
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class DigitSpatialVocabulary:
    """Stable sign and digit waveforms for decimal positions through 99.

    Instructions are always 16 visual ticks long. Sign and digit tokens are
    separated by fixed silence; unused tail time is silence rather than an
    extra cue, so one- and two-digit positions have the same duration.
    """

    TOKEN_TICKS = 4
    SILENCE_TICKS = 2
    INSTRUCTION_TICKS = 16

    def __init__(self, seed: int = 731):
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        names = ["+", "-", *(str(i) for i in range(10))]
        categories = ["sign", "sign", *("digit" for _ in range(10))]
        carrier_bank = np.arange(130.0, 850.0, 60.0)[:len(names)]
        carriers = rng.permutation(carrier_bank)
        t = np.arange(self.TOKEN_TICKS * CHUNK_SIZE, dtype=np.float64) / SAMPLE_RATE
        window = np.ones_like(t)
        ramp = 8
        window[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
        window[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False)
        self.tokens: dict[str, Token] = {}
        sample_ticks = np.arange(len(t)) // CHUNK_SIZE
        for name, category, carrier in zip(names, categories, carriers):
            offsets = rng.permutation(np.array([-37., -11., 13., 41.]))
            wave = np.zeros_like(t)
            for part in range(self.TOKEN_TICKS):
                mask = sample_ticks == part
                phase = rng.uniform(0.0, 2.0 * np.pi)
                wave[mask] = np.sin(2.0 * np.pi * (carrier + offsets[part]) * t[mask] + phase)
            wave *= window
            wave /= np.sqrt(np.mean(wave ** 2))
            self.tokens[name] = Token(name, category, wave.astype(np.float64))

    def token(self, name: str) -> np.ndarray:
        if name not in self.tokens:
            raise KeyError(name)
        return self.tokens[name].waveform.copy()

    def token_sequence(self, position: int) -> list[str]:
        if int(position) != position or not -99 <= int(position) <= 99:
            raise ValueError("position must be an integer from -99 through +99")
        value = int(position)
        if value == 0:
            return ["0"]
        sign = "+" if value > 0 else "-"
        return [sign, *(str(digit) for digit in str(abs(value)))]

    def instruction(self, position: int) -> np.ndarray:
        sequence = self.token_sequence(position)
        chunks: list[np.ndarray] = []
        for index, name in enumerate(sequence):
            if index:
                chunks.append(np.zeros(self.SILENCE_TICKS * CHUNK_SIZE, dtype=np.float64))
            chunks.append(self.token(name))
        waveform = np.concatenate(chunks)
        remaining = self.INSTRUCTION_TICKS * CHUNK_SIZE - len(waveform)
        if remaining < 0:
            raise AssertionError("instruction exceeds its fixed duration")
        if remaining:
            waveform = np.concatenate([waveform, np.zeros(remaining, dtype=np.float64)])
        return waveform

    def vocabulary_rows(self) -> list[dict[str, object]]:
        """Return auditable waveform statistics without changing token data."""
        rows = []
        for name, token in self.tokens.items():
            wave = token.waveform
            spectrum = np.abs(np.fft.rfft(wave))
            freqs = np.fft.rfftfreq(len(wave), 1 / SAMPLE_RATE)
            rows.append({"token": name, "category": token.category,
                "duration_seconds": len(wave) / SAMPLE_RATE,
                "rms": float(np.sqrt(np.mean(wave ** 2))),
                "peak": float(np.max(np.abs(wave))),
                "mean_absolute_amplitude": float(np.mean(np.abs(wave))),
                "zero_crossing_rate": float(np.mean(np.diff(np.signbit(wave)) != 0)),
                "spectral_centroid_hz": float(np.sum(freqs * spectrum) / np.sum(spectrum)),
                "waveform_seed": self.seed})
        return rows


def visual_array(position: int, field_min: int = FIELD_MIN,
                 field_max: int = FIELD_MAX) -> np.ndarray:
    if field_min >= field_max or int(position) != position or not field_min <= position <= field_max:
        raise ValueError(f"position must be an integer from {field_min} through {field_max}")
    vision = np.zeros(field_max - field_min + 1, dtype=np.float64)
    vision[int(position) - field_min] = 1.0
    return vision


def make_digit_episode(position: int, vocabulary: DigitSpatialVocabulary | None = None,
                       modality: str = "both", field_min: int = -99,
                       field_max: int = 99) -> list[Observation]:
    """Make fixed-duration raw audiovisual events for a decimal position."""
    if modality not in {"both", "audio", "vision"}:
        raise ValueError("modality must be both, audio, or vision")
    if not field_min <= position <= field_max:
        raise ValueError("position is outside the configured visual field")
    vocab = vocabulary or DigitSpatialVocabulary()
    waveform = vocab.instruction(position)
    retina = visual_array(position, field_min, field_max)
    tick_count = len(waveform) // CHUNK_SIZE
    observations = []
    for tick in range(tick_count):
        audio = waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
        vision = retina.copy()
        if modality == "audio":
            vision.fill(0.0)
        elif modality == "vision":
            audio.fill(0.0)
        observations.append(Observation(t=tick / VISUAL_RATE, vision=vision, audio=audio))
    return observations


def make_episode(position: int, vocabulary: SpatialVocabulary | None = None,
                 mode: str = "simultaneous", delay_seconds: float = 0.2,
                 modality: str = "both") -> list[Observation]:
    """Create a deterministic static audiovisual experience.

    In sequential mode, the complete instruction precedes a fixed-delay visual
    display. Audio-only evaluation contains a zeroed retina and no target image.
    """
    if position not in POSITIONS:
        raise ValueError("evaluated position must be -5 through +5")
    if mode not in {"simultaneous", "sequential"}:
        raise ValueError("mode must be simultaneous or sequential")
    if modality not in {"both", "audio", "vision"}:
        raise ValueError("modality must be both, audio, or vision")
    vocab = vocabulary or SpatialVocabulary()
    waveform = vocab.instruction(position)
    retina = visual_array(position)
    observations: list[Observation] = []
    delay_ticks = int(math.ceil(delay_seconds * VISUAL_RATE)) if mode == "sequential" else 0
    total_ticks = EPISODE_TICKS + delay_ticks + (TOKEN_TICKS if mode == "sequential" else 0)
    for tick in range(total_ticks):
        t = tick / VISUAL_RATE
        if tick < EPISODE_TICKS:
            audio = waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
        else:
            audio = np.zeros(CHUNK_SIZE, dtype=np.float64)
        vision = retina.copy()
        if modality == "audio":
            vision.fill(0.0)
        elif modality == "vision":
            audio.fill(0.0)
        if mode == "sequential" and modality != "audio":
            # Audio ends at 0.5 s; vision starts after a position-independent delay.
            if tick < EPISODE_TICKS + delay_ticks:
                vision.fill(0.0)
        observations.append(Observation(t=t, vision=vision, audio=audio))
    return observations


def rotate_composition_split(split_id: int) -> tuple[tuple[int, int], ...]:
    """Two held-out sign/magnitude pairs; every factor remains in training."""
    pairs = [(sign, magnitude) for magnitude in MAGNITUDES for sign in SIGNS]
    index = int(split_id) % len(pairs)
    # Opposite signs of distinct magnitudes; cycling covers each pair over four splits.
    first = pairs[index]
    second = (-first[0], ((first[1]) % len(MAGNITUDES)) + 1)
    return tuple(sorted({first, second}))

