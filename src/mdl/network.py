from __future__ import annotations

import os
from pathlib import Path
import numpy as np
# Brian2 inspects the user's home directory at import time even in NumPy mode.
# Keep generated caches inside this reproducible project workspace.
_cache = str(Path(__file__).resolve().parents[2] / ".brian")
os.makedirs(_cache, exist_ok=True)
os.environ["USERPROFILE"] = _cache
os.environ["HOME"] = _cache
from brian2 import (NeuronGroup, SpikeGeneratorGroup, Synapses, SpikeMonitor, StateMonitor,
                    Network, ms, second, mV, start_scope, prefs, seed as brian_seed)
from .stimuli import encode_visual_array


class STDPAssociator:
    """Small feed-forward LIF network with additive, bounded pair-based STDP."""

    def __init__(self, seed: int = 0, learning: bool = True, synaptic_gain: float = 18.0,
                 visual_sensory: bool = False, visual_gain_mv: float = 55.0):
        start_scope()
        prefs.codegen.target = "numpy"
        brian_seed(seed)
        self.learning = learning
        self.visual_sensory = bool(visual_sensory)
        self.input = SpikeGeneratorGroup(3, [], [] * ms, name="auditory_input")
        self.visual_input = (SpikeGeneratorGroup(3, [], [] * ms, name="visual_sensory_input")
                             if self.visual_sensory else None)
        self.teacher = SpikeGeneratorGroup(3, [], [] * ms, name="visual_teacher")
        self.output = NeuronGroup(
            3, "dv/dt = (-70*mV-v)/(10*ms) : volt", threshold="v > -50*mV",
            reset="v = -70*mV", refractory=5*ms, method="exact", name="visual_output")
        self.output.v = -70 * mV
        gain = float(synaptic_gain)
        on_pre = "v_post += w*gain; apre += plasticity_enabled*0.01; w = clip(w + plasticity_enabled*apost, 0, 1)"
        on_post = "apost += plasticity_enabled*(-0.012); w = clip(w + plasticity_enabled*apre, 0, 1)"
        self.plastic = Synapses(
            self.input, self.output,
            "w : 1\n gain : volt (shared)\n plasticity_enabled : 1 (shared)\n dapre/dt = -apre/(20*ms) : 1 (event-driven)\n"
            " dapost/dt = -apost/(20*ms) : 1 (event-driven)",
            on_pre=on_pre, on_post=on_post,
            name="auditory_visual_stdp")
        self.plastic.connect()
        self.plastic.gain = gain * mV
        self.plastic.plasticity_enabled = 1 if learning else 0
        self.plastic.w = np.random.default_rng(seed).uniform(0.05, 0.15, len(self.plastic.w))
        if self.visual_sensory:
            self.visual_syn = Synapses(
                self.visual_input, self.output, "gain : volt (shared)",
                on_pre="v_post += gain", name="visual_sensory_relay")
            self.visual_syn.connect(j="i")
            self.visual_syn.gain = float(visual_gain_mv) * mV
            self.visual_monitor = SpikeMonitor(self.visual_input, name="visual_sensory_spikes")
        else:
            self.visual_syn = None
            self.visual_monitor = None
        self.teacher_syn = Synapses(self.teacher, self.output, on_pre="v_post += 55*mV", name="teacher")
        self.teacher_syn.connect(j="i")
        # Stable object names let Brian2 restore full-state checkpoints into a
        # newly constructed network (auto-generated monitor names are process
        # counter dependent and therefore do not match across sessions).
        self.monitor = SpikeMonitor(self.output, name="output_spikes")
        self.input_monitor = SpikeMonitor(self.input, name="auditory_spikes")
        self.teacher_monitor = SpikeMonitor(self.teacher, name="teacher_spikes")
        self.voltage = StateMonitor(self.output, "v", record=True, name="output_voltage")
        objects = [self.input, self.teacher, self.output, self.plastic, self.teacher_syn,
                   self.monitor, self.input_monitor, self.teacher_monitor, self.voltage]
        if self.visual_sensory:
            objects.extend((self.visual_input, self.visual_syn, self.visual_monitor))
        self.net = Network(*objects)

    def _set_spikes(self, group, indices, time: float | None):
        if indices is None or time is None:
            group.set_spikes(np.array([], dtype=int), np.array([]) * second, period=0 * second)
        else:
            values = np.asarray(indices, dtype=int).reshape(-1)
            group.set_spikes(values, np.full(len(values), time) * second, period=0 * second)

    def schedule(self, cue: int | None, target: int | None = None, jitter: float = 0.0,
                 visual_array: np.ndarray | None = None) -> tuple[float | None, float | None]:
        """Schedule auditory and either sensory-visual or legacy teacher events."""
        if target is not None and visual_array is not None:
            raise ValueError("a presentation cannot use both visual sensation and direct teacher drive")
        if visual_array is not None and not self.visual_sensory:
            raise ValueError("this network was constructed without a visual sensory pathway")
        base = float(self.net.t / second)
        audio_onset = base + 0.010 + float(jitter) if cue is not None else None
        visual_onset = base + 0.025 if target is not None or visual_array is not None else None
        self._set_spikes(self.input, None if cue is None else [cue], audio_onset)
        active_visual = None if visual_array is None else encode_visual_array(visual_array)
        if self.visual_sensory:
            self._set_spikes(self.visual_input, active_visual, visual_onset if visual_array is not None else None)
        self._set_spikes(self.teacher, None if target is None else [target], visual_onset if target is not None else None)
        return audio_onset, visual_onset

    def advance(self, duration: float) -> None:
        self.net.run(float(duration) * second)

    def present(self, cue: int | None, target: int | None = None, duration: float = 0.060,
                jitter: float = 0.0, visual_array: np.ndarray | None = None) -> np.ndarray:
        """Present a cue and optional separate visual sensation or direct teacher."""
        self.schedule(cue, target, jitter, visual_array)
        before = np.asarray(self.monitor.count[:], dtype=int).copy()
        self.advance(duration)
        after = np.asarray(self.monitor.count[:], dtype=int).copy()
        return after - before

    def weights(self) -> np.ndarray:
        return np.asarray(self.plastic.w[:], dtype=float).reshape(3, 3)

    def reset_transient(self) -> None:
        self.output.v = -70 * mV

    def set_weights(self, matrix: np.ndarray) -> None:
        values = np.asarray(matrix, dtype=float)
        if values.shape != (3, 3):
            raise ValueError("expected a 3x3 auditory-to-visual matrix")
        self.plastic.w = values.reshape(-1)

    def set_gain(self, gain_mv: float) -> None:
        if gain_mv <= 0:
            raise ValueError("gain must be positive")
        self.plastic.gain = float(gain_mv) * mV

    def set_plasticity(self, enabled: bool) -> None:
        self.plastic.plasticity_enabled = int(bool(enabled))

