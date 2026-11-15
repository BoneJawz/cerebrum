"""Canonical Brian2 visual-auditory LIF association network."""
from __future__ import annotations

import numpy as np
from brian2 import (
    Clock, Network, NeuronGroup, SpikeGeneratorGroup, SpikeMonitor, StateMonitor,
    Synapses, ms, mV, prefs, second, start_scope, seed as brian_seed,
)

from .config import ModelConfig
from .plasticity import AUDITORY_ON_POST, AUDITORY_ON_PRE, AUDITORY_SYNAPSE_MODEL
from .sensory import encode_auditory_event, encode_visual_array


class DistributedVisualAssociator:
    """Fixed sparse visual and plastic auditory inputs converge on one LIF group."""

    def __init__(self, config: ModelConfig | None = None, *, seed: int | None = None,
                 learning: bool = True):
        if config is not None and seed is not None:
            raise ValueError("pass either config or seed, not both")
        self.config = config or ModelConfig(seed=42 if seed is None else int(seed))
        self.seed = self.config.seed
        start_scope()
        prefs.codegen.target = "numpy"
        brian_seed(self.seed)
        self.clock = Clock(dt=self.config.timestep_ms * ms, name="mdl_clock")

        c = self.config
        self.visual_input = SpikeGeneratorGroup(
            c.visual_inputs, [], [] * ms, clock=self.clock, name="grid_visual_input")
        self.auditory_input = SpikeGeneratorGroup(
            c.auditory_inputs, [], [] * ms, clock=self.clock, name="association_auditory_input")
        self.associative = NeuronGroup(
            c.associative_neurons,
            f"dv/dt = ({c.rest_mv}*mV-v)/({c.membrane_tau_ms}*ms) : volt",
            threshold=f"v > {c.threshold_mv}*mV",
            reset=f"v = {c.rest_mv}*mV", refractory=c.refractory_ms * ms,
            method="exact", clock=self.clock, name="associative_population")
        self.associative.v = c.rest_mv * mV

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 517]))
        mask = rng.random((c.visual_inputs, c.associative_neurons)) < c.visual_probability
        src, dst = np.where(mask)
        self.visual_projection = Synapses(
            self.visual_input, self.associative, "w : volt", on_pre="v_post += w",
            clock=self.clock, name="fixed_random_visual_projection")
        self.visual_projection.connect(i=src, j=dst)
        self.visual_projection.w = c.visual_weight_mv * mV
        self.visual_edges = np.column_stack((src, dst)).astype(int)

        self.auditory_stdp = Synapses(
            self.auditory_input, self.associative, model=AUDITORY_SYNAPSE_MODEL,
            on_pre=AUDITORY_ON_PRE, on_post=AUDITORY_ON_POST,
            clock=self.clock, namespace={
                "pre_increment": c.stdp_pre_increment,
                "post_increment": c.stdp_post_increment,
                "weight_min": c.weight_min,
                "weight_max": c.weight_max,
            }, name="auditory_to_associative_stdp")
        self.auditory_stdp.connect()
        self.auditory_stdp.gain = c.auditory_gain_mv * mV
        self.auditory_stdp.tau = c.stdp_tau_ms
        self.auditory_stdp.plasticity_enabled = int(bool(learning))
        wrng = np.random.default_rng(np.random.SeedSequence([self.seed, 991]))
        self.auditory_stdp.w = wrng.uniform(
            c.auditory_weight_min, c.auditory_weight_max,
            len(self.auditory_stdp.w))

        self.visual_spikes = SpikeMonitor(self.visual_input, name="grid_visual_spikes")
        self.auditory_spikes = SpikeMonitor(self.auditory_input, name="association_auditory_spikes")
        self.associative_spikes = SpikeMonitor(self.associative, name="associative_spikes")
        self.voltage = StateMonitor(
            self.associative, "v", record=True, name="associative_voltage")
        self.net = Network(
            self.visual_input, self.auditory_input, self.associative,
            self.visual_projection, self.auditory_stdp, self.visual_spikes,
            self.auditory_spikes, self.associative_spikes, self.voltage)

    def set_plasticity(self, enabled: bool) -> None:
        self.auditory_stdp.plasticity_enabled = int(bool(enabled))

    def weights(self) -> np.ndarray:
        return np.asarray(self.auditory_stdp.w[:], dtype=float).reshape(
            self.config.auditory_inputs, self.config.associative_neurons).copy()

    def run_event(self, *, cue: int | None = None, visual: np.ndarray | None = None,
                  duration_s: float = 0.060, audio_delay_s: float = 0.010,
                  visual_delay_s: float = 0.025) -> dict:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if cue is not None and not 0 <= cue < self.config.auditory_inputs:
            raise ValueError("cue is outside the configured auditory channels")
        visual_ids = (np.array([], dtype=int) if visual is None else
                      encode_visual_array(visual, self.config.visual_shape))
        start_s = float(self.net.t / second)
        audio_onset = start_s + audio_delay_s if cue is not None else None
        visual_onset = start_s + visual_delay_s if len(visual_ids) else None
        ai, at = (encode_auditory_event(cue, audio_delay_s, self.config.auditory_inputs)
                  if cue is not None else (np.array([], dtype=int), np.array([])))
        self.auditory_input.set_spikes(ai, (start_s + at) * second, period=0 * second)
        self.visual_input.set_spikes(
            visual_ids, np.full(len(visual_ids), visual_delay_s + start_s) * second,
            period=0 * second)

        vi0, ai0 = len(self.voltage.t), len(self.associative_spikes.t)
        vs0, as0 = len(self.visual_spikes.t), len(self.auditory_spikes.t)
        self.net.run(duration_s * second)
        times = np.asarray(self.voltage.t[vi0:] / second, dtype=float)
        volts = np.asarray(self.voltage.v[:, vi0:] / mV, dtype=float)
        spike_ids = np.asarray(self.associative_spikes.i[ai0:], dtype=int)
        spike_times = np.asarray(self.associative_spikes.t[ai0:] / second, dtype=float)
        visual_event_ids = np.asarray(self.visual_spikes.i[vs0:], dtype=int)
        visual_event_times = np.asarray(self.visual_spikes.t[vs0:] / second, dtype=float)
        audio_event_ids = np.asarray(self.auditory_spikes.i[as0:], dtype=int)
        audio_event_times = np.asarray(self.auditory_spikes.t[as0:] / second, dtype=float)
        peak = (np.maximum(np.max(volts, axis=1) - self.config.rest_mv, 0.0)
                if volts.size else np.zeros(self.config.associative_neurons))
        mean = (np.maximum(volts - self.config.rest_mv, 0.0).mean(axis=1)
                if volts.size else np.zeros(self.config.associative_neurons))
        return {
            "start_s": start_s, "audio_onset_s": audio_onset,
            "visual_onset_s": visual_onset, "voltage_times_s": times,
            "voltage_mv": volts, "peak_depolarization_mv": peak,
            "mean_depolarization_mv": mean,
            "spike_counts": np.bincount(spike_ids, minlength=self.config.associative_neurons),
            "spike_ids": spike_ids, "spike_times_s": spike_times,
            "visual_ids": visual_event_ids, "visual_times_s": visual_event_times,
            "auditory_ids": audio_event_ids, "auditory_times_s": audio_event_times,
        }

    def rest(self, duration_s: float = 0.2) -> None:
        if duration_s < 0:
            raise ValueError("duration_s cannot be negative")
        self.auditory_input.set_spikes(np.array([], dtype=int), np.array([]) * second, period=0 * second)
        self.visual_input.set_spikes(np.array([], dtype=int), np.array([]) * second, period=0 * second)
        if duration_s:
            self.net.run(duration_s * second)

    def run(self, duration_s: float) -> None:
        """Advance biological time without scheduling new sensory input."""
        self.rest(duration_s)
