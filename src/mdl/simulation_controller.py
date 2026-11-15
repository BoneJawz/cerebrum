from __future__ import annotations

import csv
from dataclasses import asdict
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from .activity_buffer import ActivityBuffer
from .network import STDPAssociator
from .checkpoint_manager import CheckpointManager
from .sensory_environment import ContinuousSensoryEnvironment, VisualSensoryEnvironment, teacher_pattern
from .stimuli import one_cell_visual_array
from .stimulus_scheduler import StimulusCommand, StimulusScheduler


LEARNING_MODES = ("off", "teaching_gated", "continuous")


class SimulationController:
    """Own one persistent Brian2 network and stream experiences through it."""

    def __init__(self, seed=42, chunk_s=0.020, buffer_samples=20000):
        self.session_id = uuid.uuid4().hex
        self.seed = int(seed)
        self.chunk_s = float(chunk_s)
        self.network = STDPAssociator(seed, True, 60.0, visual_sensory=True)
        self.scheduler = StimulusScheduler()
        self.buffer = ActivityBuffer(buffer_samples)
        self.checkpoints = CheckpointManager()
        self.running = True
        self.learning_mode = "teaching_gated"
        self.plasticity = False
        self.experiences = 0
        self.paired_experiences = 0
        self.last_event = "idle"
        self.last_response = None
        self.current_auditory = None
        self.current_visual = None
        self.current_event_type = "quiet"
        self.current_audio_until_s = 0.0
        self.current_visual_until_s = 0.0
        self.current_visual_onset_s = 0.0
        self.current_visual_pattern = teacher_pattern(None)
        self.current_visual_kind = "none"
        self.active_event: dict | None = None
        self.started_wall = time.perf_counter()
        self.event_log: list[dict] = []
        self.audio_play_callback = None
        self.audio_muted = False
        self.audio_amplitude = 0.22
        self.environment = ContinuousSensoryEnvironment(seed)
        self.sensory_environment = VisualSensoryEnvironment(seed)
        self.environment_enabled = False
        self.sensory_environment_enabled = False
        self.recording_enabled = True
        self._event_id = 0
        self.network.set_plasticity(False)

    @property
    def simulation_time_s(self):
        return float(self.network.net.t)

    @property
    def wall_elapsed_s(self):
        return time.perf_counter() - self.started_wall

    def set_learning_mode(self, mode: str):
        if mode not in LEARNING_MODES:
            raise ValueError(f"learning mode must be one of {LEARNING_MODES}")
        self.learning_mode = mode
        self._apply_plasticity()

    def _apply_plasticity(self):
        enabled = self.learning_mode == "continuous"
        if self.learning_mode == "teaching_gated":
            enabled = bool(self.active_event and self.active_event["command"].event_type in ("paired_teaching", "paired_sensory"))
        self.plasticity = enabled
        self.network.set_plasticity(enabled)

    def submit(self, cue, teaching=False, target=None):
        return self.scheduler.submit(cue, teaching, target)

    def submit_audio(self, cue: int):
        return self.scheduler.submit_event(cue, None, "auditory_only", 0.060)

    def submit_pair(self, cue: int, target: int, event_type="paired_teaching"):
        return self.scheduler.submit_event(cue, target, event_type, 0.060)

    def submit_sensory_pair(self, cue: int, visual_array=None):
        pattern = one_cell_visual_array(cue) if visual_array is None else np.asarray(visual_array)
        return self.scheduler.submit_sensory_pair(cue, pattern)

    def submit_visual(self, target: int):
        return self.scheduler.submit_event(None, target, "visual_teacher_only", 0.060)

    def submit_visual_array(self, visual_array):
        return self.scheduler.submit_event(None, None, "visual_sensory_only", 0.060, visual_array)

    def submit_quiet(self, duration_s=0.20):
        return self.scheduler.submit_event(None, None, "quiet", duration_s)

    def _start_event(self, command: StimulusCommand):
        start_s = self.simulation_time_s
        audio_onset_s, visual_onset_s = self.network.schedule(
            command.cue, command.target, visual_array=command.visual_array)
        self.active_event = {
            "command": command,
            "start_s": start_s,
            "end_s": start_s + command.duration_s,
            "audio_onset_s": audio_onset_s,
            "visual_onset_s": visual_onset_s,
            "peaks_mv": np.zeros(3, dtype=float),
            "spike_counts": np.zeros(3, dtype=int),
            "weights_before": self.network.weights().copy(),
        }
        self.current_auditory = command.cue
        self.current_visual = command.target
        self.current_event_type = command.event_type
        self.current_audio_until_s = (audio_onset_s or start_s) + 0.18
        has_visual = command.target is not None or command.visual_array is not None
        self.current_visual_until_s = start_s + command.duration_s if has_visual else start_s
        self.current_visual_onset_s = visual_onset_s if visual_onset_s is not None else start_s
        self.current_visual_pattern = (np.asarray(command.visual_array, dtype=np.uint8)
                                      if command.visual_array is not None else teacher_pattern(command.target))
        self.current_visual_kind = "sensory_array" if command.visual_array is not None else ("teacher" if command.target is not None else "none")
        self._apply_plasticity()
        if command.cue is not None and self.audio_play_callback is not None and not self.audio_muted:
            self.audio_play_callback(command.cue, self.audio_amplitude)
        self._event_id += 1
        if self.recording_enabled:
            self.event_log.append({
                "event_id": self._event_id, "sequence": command.sequence,
                "event_type": command.event_type, "cue": command.cue, "visual_teacher_target": command.target,
                "visual_pattern": "".join(map(str, self.current_visual_pattern.tolist())) if command.target is not None or command.visual_array is not None else "",
                "visual_kind": self.current_visual_kind,
                "submitted_wall_time": command.submitted_wall_time,
                "dispatch_wall_time": time.perf_counter(), "start_simulation_s": start_s,
                "audio_onset_simulation_s": audio_onset_s, "visual_onset_simulation_s": visual_onset_s,
                "actual_end_simulation_s": "", "duration_s": command.duration_s,
                "learning_mode": self.learning_mode, "plasticity_enabled": self.plasticity,
                "weight_change": "", "peak_prediction": "", "audio_play_request_wall_time": time.perf_counter() if command.cue is not None else "",
            })

    def _finish_event(self):
        active = self.active_event
        if active is None:
            return
        command = active["command"]
        peaks = active["peaks_mv"]
        prediction = int(np.argmax(peaks)) if command.cue is not None else None
        counts = active["spike_counts"]
        weights_after = self.network.weights().copy()
        delta = float(np.abs(weights_after - active["weights_before"]).sum())
        if command.event_type != "quiet":
            self.experiences += 1
        if command.event_type in ("paired_teaching", "paired_sensory"):
            self.paired_experiences += 1
        if prediction is not None:
            self.last_response = {"peak_mv": peaks.copy(), "spike_counts": counts.copy(), "prediction": prediction,
                                  "margin_mv": float(np.max(peaks) - np.partition(peaks, -2)[-2]),
                                  "spike_tie": bool(np.count_nonzero(counts == counts.max()) > 1),
                                  "silent": bool(counts.sum() == 0)}
        visual_description = (f"visual-array {''.join(map(str, command.visual_array))}"
                              if command.visual_array is not None else f"visual-teacher {command.target}")
        self.last_event = f"{command.event_type}: audio {command.cue}, {visual_description}"
        if self.event_log and self.event_log[-1].get("sequence") == command.sequence:
            self.event_log[-1].update({"actual_end_simulation_s": self.simulation_time_s, "plasticity_enabled": self.plasticity,
                                       "weight_change": delta, "peak_prediction": prediction,
                                       "peak_a_mv": float(peaks[0]), "peak_b_mv": float(peaks[1]), "peak_c_mv": float(peaks[2]),
                                       "spikes_a": int(counts[0]), "spikes_b": int(counts[1]), "spikes_c": int(counts[2])})
        self.active_event = None
        self._apply_plasticity()

    def tick(self):
        if not self.running:
            return
        if self.active_event is None:
            if self.sensory_environment_enabled and len(self.scheduler) == 0:
                self.sensory_environment.next_command(self.scheduler)
            elif self.environment_enabled and len(self.scheduler) == 0:
                self.environment.next_command(self.scheduler)
            command = self.scheduler.pop()
            if command is not None:
                self._start_event(command)
            else:
                self._apply_plasticity()
                self.network.advance(self.chunk_s)
                self._capture()
                return
        remaining = self.active_event["end_s"] - self.simulation_time_s
        step_s = min(self.chunk_s, max(remaining, 0.0))
        if step_s > 0:
            self.network.advance(step_s)
            self._capture()
        if self.simulation_time_s >= self.active_event["end_s"] - 1e-12:
            self._finish_event()

    def _capture(self):
        voltage_monitor = self.network.voltage
        times = np.asarray(voltage_monitor.t[:], dtype=float) / 1e-3
        values = np.asarray(voltage_monitor.v[:], dtype=float) / 1e-3
        if len(times):
            for idx, timestamp in enumerate(times):
                self.buffer.append(float(timestamp), values[:, idx], ())
            if self.active_event is not None:
                self.active_event["peaks_mv"] = np.maximum(self.active_event["peaks_mv"], np.max(values + 70.0, axis=1))
            voltage_monitor.resize(0)
        all_spikes = []
        monitors = [(0, self.network.input_monitor)]
        if self.network.visual_monitor is not None:
            monitors.append((3, self.network.visual_monitor))
        monitors.extend(((6, self.network.teacher_monitor), (9 if self.network.visual_sensory else 6, self.network.monitor)))
        for offset, monitor in monitors:
            if monitor is None:
                continue
            st = np.asarray(monitor.t[:], dtype=float) / 1e-3
            si = np.asarray(monitor.i[:], dtype=int)
            all_spikes.extend((float(ts), int(i + offset)) for ts, i in zip(st, si))
            output_offset = 9 if self.network.visual_sensory else 6
            if self.active_event is not None and offset == output_offset and len(si):
                self.active_event["spike_counts"] += np.bincount(si, minlength=3)[:3]
            if len(st):
                monitor.resize(0)
        self.buffer.spikes.extend(all_spikes)

    def pause(self):
        self.running = False

    def resume(self):
        self.running = True

    def reset_transient(self):
        self.network.reset_transient()
        self.event_log.append({"event_id": "manual_reset", "event_type": "transient_reset", "start_simulation_s": self.simulation_time_s})

    def set_environment_enabled(self, enabled: bool):
        self.environment_enabled = bool(enabled)

    def set_sensory_environment_enabled(self, enabled: bool):
        self.sensory_environment_enabled = bool(enabled)

    def save_checkpoint(self, path):
        return self.checkpoints.save_weights(self.network, path, {
            "session_id": self.session_id, "seed": self.seed, "experiences": self.experiences,
            "paired_experiences": self.paired_experiences, "learning_mode": self.learning_mode,
            "environment_seed": self.environment.seed, "environment_cycle": self.environment.cycle,
            "sensory_environment_seed": self.sensory_environment.seed, "sensory_environment_cycle": self.sensory_environment.cycle,
            "model": "STDPAssociator", "checkpoint_type": "learned_parameters_and_voltage",
        })

    def save_full_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.network.net.store(name="continuation", filename=str(path))
        def json_value(value):
            return value.tolist() if isinstance(value, np.ndarray) else value
        active = None
        if self.active_event is not None:
            active = {key: (asdict(value) if isinstance(value, StimulusCommand) else json_value(value))
                      for key, value in self.active_event.items()}
        response = None if self.last_response is None else {key: json_value(value) for key, value in self.last_response.items()}
        state = {
            "version": 1, "session_id": self.session_id, "seed": self.seed, "chunk_s": self.chunk_s,
            "running": self.running, "learning_mode": self.learning_mode,
            "experiences": self.experiences, "paired_experiences": self.paired_experiences,
            "last_event": self.last_event, "last_response": response,
            "current_auditory": self.current_auditory, "current_visual": self.current_visual,
            "current_event_type": self.current_event_type, "current_audio_until_s": self.current_audio_until_s,
            "current_visual_until_s": self.current_visual_until_s, "current_visual_onset_s": self.current_visual_onset_s,
            "current_visual_pattern": self.current_visual_pattern.tolist(), "current_visual_kind": self.current_visual_kind,
            "active_event": active, "event_log": self.event_log, "event_id": self._event_id,
            "environment_enabled": self.environment_enabled, "environment": self.environment.snapshot(),
            "sensory_environment_enabled": self.sensory_environment_enabled,
            "sensory_environment": self.sensory_environment.snapshot(),
            "scheduler": self.scheduler.snapshot(), "recording_enabled": self.recording_enabled,
            "audio_muted": self.audio_muted, "audio_amplitude": self.audio_amplitude,
        }
        sidecar = path.with_suffix(path.suffix + ".json")
        fd, temp_name = tempfile.mkstemp(prefix=sidecar.name, suffix=".tmp", dir=sidecar.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream)
            os.replace(temp_name, sidecar)
        except Exception:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
            raise
        return path

    def restore_full_checkpoint(self, path):
        path = Path(path)
        with path.with_suffix(path.suffix + ".json").open(encoding="utf-8") as stream:
            state = json.load(stream)
        self.checkpoints.restore_full(self.network, path)
        self.session_id = state["session_id"]
        self.seed = int(state["seed"])
        self.chunk_s = float(state["chunk_s"])
        self.running = bool(state["running"])
        self.learning_mode = state["learning_mode"]
        self.experiences = int(state["experiences"])
        self.paired_experiences = int(state["paired_experiences"])
        self.last_event = state["last_event"]
        response = state["last_response"]
        self.last_response = None if response is None else {
            key: np.asarray(value) if isinstance(value, list) else value for key, value in response.items()
        }
        for key in ("current_auditory", "current_visual", "current_event_type", "current_audio_until_s",
                    "current_visual_until_s", "current_visual_onset_s"):
            setattr(self, key, state[key])
        self.current_visual_pattern = np.asarray(state["current_visual_pattern"], dtype=np.uint8)
        self.current_visual_kind = state["current_visual_kind"]
        self.active_event = state["active_event"]
        if self.active_event is not None:
            command_state = self.active_event["command"]
            command_state["visual_array"] = None if command_state.get("visual_array") is None else tuple(command_state["visual_array"])
            self.active_event["command"] = StimulusCommand(**command_state)
            for key in ("peaks_mv", "spike_counts", "weights_before"):
                self.active_event[key] = np.asarray(self.active_event[key])
        self.event_log = state["event_log"]
        self._event_id = int(state["event_id"])
        self.environment_enabled = bool(state["environment_enabled"])
        self.environment.restore(state["environment"])
        self.sensory_environment_enabled = bool(state["sensory_environment_enabled"])
        self.sensory_environment.restore(state["sensory_environment"])
        self.scheduler.restore(state["scheduler"])
        self.recording_enabled = bool(state["recording_enabled"])
        self.audio_muted = bool(state["audio_muted"])
        self.audio_amplitude = float(state["audio_amplitude"])
        self.network.set_plasticity(self.learning_mode == "continuous" or
                                    (self.learning_mode == "teaching_gated" and self.active_event is not None and
                                     self.active_event["command"].event_type == "paired_teaching"))
        self.started_wall = time.perf_counter()
        return self

    def save_event_log(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.event_log:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in self.event_log for k in row)))
                writer.writeheader()
                writer.writerows(self.event_log)
        return path

    def status(self):
        return {"session_id": self.session_id, "running": self.running, "simulation_time_s": self.simulation_time_s,
                "wall_elapsed_s": self.wall_elapsed_s, "speed_ratio": self.simulation_time_s / max(self.wall_elapsed_s, 1e-9),
                "pending": len(self.scheduler), "experiences": self.experiences, "paired_experiences": self.paired_experiences,
                "learning_mode": self.learning_mode, "plasticity": self.plasticity, "last_event": self.last_event,
                "environment_enabled": self.environment_enabled,
                "sensory_environment_enabled": self.sensory_environment_enabled}
