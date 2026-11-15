"""Small distributed visual–auditory association network for MDL experiments."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
# Brian2 creates its cache during import; keep it under the project workspace.
_cache = str(Path(__file__).resolve().parents[2] / ".brian")
os.makedirs(_cache, exist_ok=True)
os.environ["USERPROFILE"] = _cache
os.environ["HOME"] = _cache
from brian2 import (NeuronGroup, SpikeGeneratorGroup, Synapses, SpikeMonitor,
                    StateMonitor, Network, ms, second, mV, start_scope, prefs, seed as brian_seed)


class DistributedVisualAssociator:
    """25 visual inputs and 3 auditory inputs converge on one LIF population.

    Visual connectivity is a seed-controlled sparse random projection, independent
    of the three stimulus patterns and their audiovisual mapping. Auditory
    afferents use the project's existing bounded pair-based STDP equation.
    """

    def __init__(self, seed: int, n_associative: int = 18, visual_probability: float = 0.35,
                 visual_weight_mv: float = 12.0, auditory_gain_mv: float = 60.0,
                 learning: bool = True):
        if n_associative < 12 or n_associative > 24:
            raise ValueError("n_associative must be between 12 and 24")
        start_scope()
        prefs.codegen.target = "numpy"
        brian_seed(int(seed))
        self.seed = int(seed)
        self.n_associative = int(n_associative)
        self.visual_probability = float(visual_probability)
        self.visual_input = SpikeGeneratorGroup(25, [], [] * ms, name="grid_visual_input")
        self.auditory_input = SpikeGeneratorGroup(3, [], [] * ms, name="association_auditory_input")
        self.associative = NeuronGroup(
            self.n_associative, "dv/dt = (-70*mV-v)/(10*ms) : volt",
            threshold="v > -50*mV", reset="v = -70*mV", refractory=5*ms,
            method="exact", name="associative_population")
        self.associative.v = -70 * mV

        # This projection is sampled once per seed and never uses pattern labels.
        projection_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 517]))
        mask = projection_rng.random((25, self.n_associative)) < self.visual_probability
        src, dst = np.where(mask)
        self.visual_projection = Synapses(
            self.visual_input, self.associative, "w : volt",
            on_pre="v_post += w", name="fixed_random_visual_projection")
        self.visual_projection.connect(i=src, j=dst)
        self.visual_projection.w = float(visual_weight_mv) * mV
        self.visual_edges = np.column_stack((src, dst)).astype(int)

        self.auditory_stdp = Synapses(
            self.auditory_input, self.associative,
            "w : 1\n gain : volt (shared)\n plasticity_enabled : 1 (shared)\n"
            " dapre/dt = -apre/(20*ms) : 1 (event-driven)\n"
            " dapost/dt = -apost/(20*ms) : 1 (event-driven)",
            on_pre="v_post += w*gain; apre += plasticity_enabled*0.01; w = clip(w + plasticity_enabled*apost, 0, 1)",
            on_post="apost += plasticity_enabled*(-0.012); w = clip(w + plasticity_enabled*apre, 0, 1)",
            name="auditory_to_associative_stdp")
        self.auditory_stdp.connect()
        self.auditory_stdp.gain = float(auditory_gain_mv) * mV
        self.auditory_stdp.plasticity_enabled = int(bool(learning))
        weight_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 991]))
        self.auditory_stdp.w = weight_rng.uniform(0.05, 0.15, len(self.auditory_stdp.w))

        self.visual_spikes = SpikeMonitor(self.visual_input, name="grid_visual_spikes")
        self.auditory_spikes = SpikeMonitor(self.auditory_input, name="association_auditory_spikes")
        self.associative_spikes = SpikeMonitor(self.associative, name="associative_spikes")
        self.voltage = StateMonitor(self.associative, "v", record=True, name="associative_voltage")
        self.net = Network(self.visual_input, self.auditory_input, self.associative,
                           self.visual_projection, self.auditory_stdp,
                           self.visual_spikes, self.auditory_spikes,
                           self.associative_spikes, self.voltage)

    def set_plasticity(self, enabled: bool):
        self.auditory_stdp.plasticity_enabled = int(bool(enabled))

    def weights(self) -> np.ndarray:
        return np.asarray(self.auditory_stdp.w[:], dtype=float).reshape(3, self.n_associative).copy()

    def present(self, cue: int | None = None, pattern: np.ndarray | None = None,
                duration_s: float = 0.060, audio_delay_s: float = 0.010,
                visual_delay_s: float = 0.025, capture: bool = True) -> dict:
        if cue is not None and cue not in range(3):
            raise ValueError("cue must be 0, 1, 2, or None")
        if pattern is not None:
            values = np.asarray(pattern, dtype=np.uint8)
            if values.shape != (5, 5) or not np.isin(values, (0, 1)).all():
                raise ValueError("visual input must be a binary 5x5 array")
            visual_indices = np.flatnonzero(values.reshape(-1))
        else:
            visual_indices = np.array([], dtype=int)
        start = float(self.net.t / second)
        audio_time = start + audio_delay_s if cue is not None else None
        visual_time = start + visual_delay_s if len(visual_indices) else None
        self.auditory_input.set_spikes(
            np.array([], dtype=int) if cue is None else np.array([cue], dtype=int),
            np.array([]) * second if cue is None else np.array([audio_time]) * second,
            period=0 * second)
        self.visual_input.set_spikes(visual_indices,
                                     np.full(len(visual_indices), visual_time) * second if len(visual_indices) else np.array([]) * second,
                                     period=0 * second)
        vm0 = len(self.voltage.t)
        as0, vs0, xs0 = (len(self.associative_spikes.t), len(self.visual_spikes.t), len(self.auditory_spikes.t))
        self.net.run(float(duration_s) * second)
        vtimes = np.asarray(self.voltage.t[vm0:] / second, dtype=float)
        volts = np.asarray(self.voltage.v[:, vm0:] / mV, dtype=float)
        spike_ids = np.asarray(self.associative_spikes.i[as0:], dtype=int)
        spike_times = np.asarray(self.associative_spikes.t[as0:] / second, dtype=float)
        visual_ids = np.asarray(self.visual_spikes.i[vs0:], dtype=int)
        visual_times = np.asarray(self.visual_spikes.t[vs0:] / second, dtype=float)
        audio_ids = np.asarray(self.auditory_spikes.i[xs0:], dtype=int)
        audio_times = np.asarray(self.auditory_spikes.t[xs0:] / second, dtype=float)
        if capture:
            if volts.size:
                peak_depol = np.maximum(np.max(volts, axis=1) - (-70.0), 0.0)
                area_depol = np.maximum(volts - (-70.0), 0.0).mean(axis=1)
            else:
                peak_depol = np.zeros(self.n_associative)
                area_depol = np.zeros(self.n_associative)
            spike_counts = np.bincount(spike_ids, minlength=self.n_associative)
            result = {
                "start_s": start, "audio_onset_s": audio_time, "visual_onset_s": visual_time,
                "voltage_times_s": vtimes, "voltage_mv": volts,
                "peak_depolarization_mv": peak_depol, "mean_depolarization_mv": area_depol,
                "spike_counts": spike_counts, "spike_ids": spike_ids, "spike_times_s": spike_times,
                "visual_ids": visual_ids, "visual_times_s": visual_times,
                "auditory_ids": audio_ids, "auditory_times_s": audio_times,
            }
        else:
            result = {"start_s": start, "audio_onset_s": audio_time, "visual_onset_s": visual_time,
                      "spike_counts": np.bincount(spike_ids, minlength=self.n_associative),
                      "visual_ids": visual_ids, "visual_times_s": visual_times,
                      "auditory_ids": audio_ids, "auditory_times_s": audio_times}
            self.voltage.resize(0)
        return result

    def rest(self, duration_s: float = 0.20):
        self.auditory_input.set_spikes(np.array([], dtype=int), np.array([]) * second, period=0 * second)
        self.visual_input.set_spikes(np.array([], dtype=int), np.array([]) * second, period=0 * second)
        self.net.run(float(duration_s) * second)
        self.voltage.resize(0)

