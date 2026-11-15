from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import wave
import numpy as np

from .stimuli import one_cell_visual_array
from .stimulus_scheduler import StimulusCommand


@dataclass(frozen=True)
class AudibleCue:
    cue: int
    frequency_hz: float
    sample_rate_hz: int
    samples: np.ndarray
    wav_bytes: bytes
    label: str


def audible_cue(cue: int, duration_s: float = 0.18, sample_rate_hz: int = 22050, amplitude: float = 0.22) -> AudibleCue:
    """Human-audible tone representing the synthetic spike cue; not model input."""
    if cue not in range(3):
        raise ValueError("cue must be 0, 1, or 2")
    frequency = (440.0, 554.37, 659.25)[cue]
    t = np.arange(round(duration_s * sample_rate_hz), dtype=np.float64) / sample_rate_hz
    envelope = np.minimum(1.0, np.minimum(t / 0.008, (duration_s - t) / 0.012)).clip(0, 1)
    samples = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
    pcm = np.asarray(np.clip(samples, -1, 1) * 32767, dtype="<i2").tobytes()
    stream = BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate_hz); wav.writeframes(pcm)
    return AudibleCue(cue, frequency, sample_rate_hz, samples.astype(np.float32), stream.getvalue(),
                      f"Audible representation of synthetic neural cue {'ABC'[cue]}")


def teacher_pattern(target: int | None) -> np.ndarray:
    """One-hot visualization of which teacher neuron is driven (not a retinal image)."""
    pattern = np.zeros(3, dtype=np.uint8)
    if target is not None:
        pattern[target] = 1
    return pattern


class ContinuousSensoryEnvironment:
    """Seeded repeated sequence of paired, unpaired, and quiet experiences."""

    def __init__(self, seed: int = 42, mapping: tuple[int, int, int] = (0, 1, 2), quiet_s: float = 0.20):
        if sorted(mapping) != [0, 1, 2]:
            raise ValueError("mapping must be a permutation of A/B/C")
        self.seed = seed
        self.mapping = tuple(mapping)
        self.quiet_s = float(quiet_s)
        self.rng = np.random.default_rng(seed)
        self.cycle = 0
        self._events: list[tuple[int | None, int | None, str, float]] = []
        self._cursor = 0

    def _build_cycle(self) -> None:
        paired = [int(x) for x in self.rng.permutation(3)]
        auditory_only = [int(x) for x in self.rng.permutation(3)]
        visual_only = [int(x) for x in self.rng.permutation(3)]
        events: list[tuple[int | None, int | None, str, float]] = []
        for cue in paired:
            events.extend([(cue, self.mapping[cue], "paired_teaching", 0.060), (None, None, "quiet", self.quiet_s)])
        for cue in auditory_only:
            events.extend([(cue, None, "auditory_only", 0.060), (None, None, "quiet", self.quiet_s)])
        for target in visual_only:
            events.extend([(None, target, "visual_teacher_only", 0.060), (None, None, "quiet", self.quiet_s)])
        self._events = events
        self._cursor = 0
        self.cycle += 1

    def next_event(self) -> tuple[int | None, int | None, str, float]:
        if self._cursor >= len(self._events):
            self._build_cycle()
        event = self._events[self._cursor]
        self._cursor += 1
        return event

    def next_command(self, scheduler) -> StimulusCommand:
        cue, target, event_type, duration = self.next_event()
        return scheduler.submit_event(cue, target, event_type, duration)

    def snapshot(self) -> dict:
        return {"seed": self.seed, "mapping": self.mapping, "quiet_s": self.quiet_s,
                "cycle": self.cycle, "events": self._events, "cursor": self._cursor,
                "rng_state": self.rng.bit_generator.state}

    def restore(self, state: dict) -> None:
        self.seed = int(state["seed"])
        self.mapping = tuple(state["mapping"])
        self.quiet_s = float(state["quiet_s"])
        self.cycle = int(state["cycle"])
        self._events = [tuple(item) for item in state["events"]]
        self._cursor = int(state["cursor"])
        self.rng.bit_generator.state = state["rng_state"]


class VisualSensoryEnvironment:
    """Seeded audiovisual stream whose visual events are binary arrays, not teacher targets."""

    def __init__(self, seed: int = 42, mapping: tuple[int, int, int] = (0, 1, 2), quiet_s: float = 0.20):
        if sorted(mapping) != [0, 1, 2]:
            raise ValueError("mapping must be a permutation of the three visible locations")
        self.seed = int(seed)
        self.mapping = tuple(int(x) for x in mapping)
        self.quiet_s = float(quiet_s)
        self.rng = np.random.default_rng(seed)
        self.cycle = 0
        self._events = []
        self._cursor = 0

    def _build_cycle(self):
        cues = [int(x) for x in self.rng.permutation(3)]
        auditory = [int(x) for x in self.rng.permutation(3)]
        visual = [int(x) for x in self.rng.permutation(3)]
        events = []
        for cue in cues:
            events.extend([(cue, tuple(int(v) for v in one_cell_visual_array(self.mapping[cue])), "paired_sensory", .060),
                           (None, None, "quiet", self.quiet_s)])
        for cue in auditory:
            events.extend([(cue, None, "auditory_only", .060), (None, None, "quiet", self.quiet_s)])
        for position in visual:
            events.extend([(None, tuple(int(v) for v in one_cell_visual_array(position)), "visual_sensory_only", .060),
                           (None, None, "quiet", self.quiet_s)])
        self._events = events
        self._cursor = 0
        self.cycle += 1

    def next_event(self):
        if self._cursor >= len(self._events):
            self._build_cycle()
        event = self._events[self._cursor]
        self._cursor += 1
        return event

    def next_command(self, scheduler):
        cue, visual_array, event_type, duration = self.next_event()
        return scheduler.submit_event(cue, None, event_type, duration, visual_array)

    def snapshot(self):
        return {"seed": self.seed, "mapping": self.mapping, "quiet_s": self.quiet_s,
                "cycle": self.cycle, "events": self._events, "cursor": self._cursor,
                "rng_state": self.rng.bit_generator.state}

    def restore(self, state):
        self.seed = int(state["seed"])
        self.mapping = tuple(state["mapping"])
        self.quiet_s = float(state["quiet_s"])
        self.cycle = int(state["cycle"])
        self._events = [tuple(item) for item in state["events"]]
        self._cursor = int(state["cursor"])
        self.rng.bit_generator.state = state["rng_state"]
