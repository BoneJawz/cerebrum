from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import time
import numpy as np


@dataclass(frozen=True)
class StimulusCommand:
    cue: int | None
    target: int | None
    event_type: str
    duration_s: float
    submitted_wall_time: float
    sequence: int
    visual_array: tuple[int, ...] | None = None

    @property
    def teaching(self) -> bool:
        return self.event_type == "paired_teaching"


class StimulusScheduler:
    """FIFO scheduler; commands are resolved against simulation time at dispatch."""

    def __init__(self):
        self._queue: deque[StimulusCommand] = deque()
        self._sequence = 0

    def submit_event(self, cue: int | None, target: int | None, event_type: str,
                     duration_s: float = 0.060, visual_array=None) -> StimulusCommand:
        if cue is not None and cue not in range(3):
            raise ValueError("cue must be A/B/C (0/1/2) or None")
        if target is not None and target not in range(3):
            raise ValueError("target must be A/B/C (0/1/2) or None")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if target is not None and visual_array is not None:
            raise ValueError("a scheduled event cannot combine a visual array with direct teacher drive")
        if visual_array is not None:
            values = np.asarray(visual_array)
            if values.ndim != 1 or values.size != 3 or not np.isin(values, (0, 1)).all():
                raise ValueError("visual_array must contain three binary cells")
            visual_array = tuple(int(value) for value in values)
        cmd = StimulusCommand(cue, target, event_type, float(duration_s), time.perf_counter(), self._sequence, visual_array)
        self._sequence += 1
        self._queue.append(cmd)
        return cmd

    def submit(self, cue: int, teaching: bool = False, target: int | None = None) -> StimulusCommand:
        if teaching and target is None:
            target = cue
        kind = "paired_teaching" if teaching else "auditory_only"
        return self.submit_event(cue, target, kind)

    def submit_sensory_pair(self, cue: int, visual_array, duration_s: float = 0.060) -> StimulusCommand:
        return self.submit_event(cue, None, "paired_sensory", duration_s, visual_array)

    def pop(self) -> StimulusCommand | None:
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)

    def snapshot(self) -> dict:
        return {"sequence": self._sequence, "queue": [asdict(command) for command in self._queue]}

    def restore(self, state: dict) -> None:
        self._sequence = int(state["sequence"])
        self._queue = deque(StimulusCommand(**{**item, "visual_array": None if item.get("visual_array") is None else tuple(item["visual_array"])})
                            for item in state["queue"])
