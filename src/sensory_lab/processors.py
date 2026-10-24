"""Interchangeable processing substrates; readout training is separate."""
from collections import deque
import numpy as np
from .contracts import Observation


def sensory_features(o: Observation) -> np.ndarray:
    """Fixed local summaries, signed waveform samples, and FFT magnitudes.

    The signed samples retain phase/polarity information discarded by spectral
    magnitudes. They are generic samples of the raw waveform, not cue labels.
    """
    audio = o.audio
    zcr = np.mean(np.abs(np.diff(np.signbit(audio).astype(float)))) if len(audio)>1 else 0
    audio_summary = np.array([np.mean(audio), np.sqrt(np.mean(audio**2)), zcr,
                              np.mean(np.abs(audio))])
    spectrum = 2.0 * np.abs(np.fft.rfft(audio, n=128))[1:50] / max(1, len(audio))
    sample_indices = np.linspace(0, max(0, len(audio) - 1), 16, dtype=int)
    waveform_samples = audio[sample_indices]
    return np.concatenate([o.vision.astype(float), audio_summary,
                           waveform_samples, spectrum])


class Base:
    audio_spectrum_bins = 49
    audio_waveform_samples = 16

    def __init__(self, pixels: int = 33, hidden: int = 64, seed: int = 7):
        self.pixels = pixels
        self.hidden = hidden
        self.feature_dim = pixels + 4 + self.audio_waveform_samples + self.audio_spectrum_bins
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.12, (hidden, self.feature_dim))
        a = rng.normal(0, 1., (hidden, hidden))
        self.a = a / max(1., np.linalg.norm(a, ord=2)) * 0.8
        self.reset()

    @property
    def updates(self) -> int:
        return self._updates

    def reset(self):
        self.state = np.zeros(self.hidden)
        self._updates = 0
        self._last_t = None
        self._last_input = None

    def input_features(self, o: Observation):
        return sensory_features(o)

    def _dt(self, t: float):
        dt = 0.05 if self._last_t is None else t - self._last_t
        if dt <= 0:
            raise ValueError("timestamps must increase")
        self._last_t = t
        return dt

class FeedforwardHistory(Base):
    name = "A_feedforward_history"
    def reset(self):
        super().reset()
        self.history = deque(maxlen=4)
    def step(self, o):
        self._dt(o.t)
        self.history.append(self.input_features(o))
        self._updates += 1
        parts = list(self.history)
        padded = [np.zeros(self.feature_dim)] * (4-len(parts)) + parts
        return np.concatenate(padded)

class RecurrentFusion(Base):
    name = "B_recurrent_fusion"
    def step(self, o):
        self._dt(o.t)
        u = self.input_features(o)
        self.state = np.tanh(self.a @ self.state + self.w @ u)
        self._updates += 1
        return np.concatenate([self.state, u])

class CoupledDynamics(Base):
    name = "C_coupled_dynamics"
    def step(self, o):
        dt = self._dt(o.t)
        u = self.input_features(o)
        # Euler integrate a stable leaky nonlinear state, two substeps / frame.
        for _ in range(2):
            self.state += (dt/2) * 12.0 * (-self.state + np.tanh(self.a @ self.state + self.w @ u))
            self._updates += 1
        return np.concatenate([self.state, u])

class EventDrivenDynamics(Base):
    name = "D_event_driven"
    def __init__(self, pixels=33, hidden=64, seed=7, threshold=0.04):
        self.threshold = threshold
        super().__init__(pixels, hidden, seed)
    def step(self, o):
        dt = self._dt(o.t)
        u = self.input_features(o)
        trigger = self._last_input is None or np.linalg.norm(u - self._last_input) > self.threshold
        if trigger:
            # Cheap event-triggered state update. Not a continuous-time solver.
            self.state += min(1.0, 12.0*dt) * (-self.state + np.tanh(self.a @ self.state + self.w @ u))
            self._updates += 1
        self._last_input = u.copy()
        return np.concatenate([self.state, u])

PROCESSORS = {"A":FeedforwardHistory,"B":RecurrentFusion,"C":CoupledDynamics,"D":EventDrivenDynamics}
