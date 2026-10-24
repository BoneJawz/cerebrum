"""Reproducible cross-modal benchmark with independent sensor ablations."""
from dataclasses import dataclass
import csv
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .contracts import Observation
from .environment import LineWorld, WorldConfig
from .processors import PROCESSORS


@dataclass(frozen=True)
class BenchmarkConfig:
    train_episodes: int = 160
    test_episodes: int = 80
    seed: int = 42
    hidden: int = 64
    seeds: int = 5
    prediction_tick: int = 4
    outcome_tick: int = 9

    def __post_init__(self):
        if min(self.train_episodes, self.test_episodes, self.hidden, self.seeds) < 1:
            raise ValueError("episode counts, hidden size and seeds must be positive")
        if self.prediction_tick < 0 or self.outcome_tick <= self.prediction_tick:
            raise ValueError("outcome_tick must follow the nonnegative prediction_tick")


class RidgeReadout:
    """Degree-two kernel ridge readout; processors remain fixed and untrained.

    The quadratic kernel exposes pairwise feature interactions needed to combine
    the initial position and audio-conditioned movement, without materializing
    a very wide polynomial feature matrix.
    """
    def __init__(self, alpha=0.05):
        self.alpha = float(alpha)

    def fit(self, x, y):
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = (x - self.mean) / self.scale
        self.dimension = z.shape[1]
        gram = (1.0 + (z @ z.T) / self.dimension) ** 2
        self.dual = np.linalg.solve(gram + self.alpha*np.eye(len(gram)), y)
        self.train = z

    def predict(self, x):
        z = (x - self.mean) / self.scale
        cross = (1.0 + (z @ self.train.T) / self.dimension) ** 2
        return cross @ self.dual


MODALITIES = ("vision_only", "audio_only", "combined")


def create_data(world, count, seed, frequency_map, outcome_tick=9):
    """Create command-conditioned motion trials and future-position targets."""
    rng = np.random.default_rng(seed)
    data = []
    for i in range(count):
        direction = int(rng.choice([-1, 1]))
        observations = world.episode(seed=seed*10000+i, direction=direction,
                                    tone_hz=frequency_map[direction])
        frame = observations[outcome_tick]
        pixels = np.arange(len(frame.vision), dtype=float)
        target_position = float(np.sum(pixels * frame.vision) / np.sum(frame.vision))
        data.append((observations, target_position))
    return data


def _ablate(observation: Observation, modality: str) -> Observation:
    vision = observation.vision if modality != "audio_only" else np.zeros_like(observation.vision)
    audio = observation.audio if modality != "vision_only" else np.zeros_like(observation.audio)
    return Observation(observation.t, vision, audio)


def _features(processor, dataset, modality, prediction_tick):
    xs, ys, latencies, updates = [], [], [], 0
    for observations, target in dataset:
        processor.reset()
        for observation in observations[:prediction_tick + 1]:
            start = time.perf_counter_ns()
            feature = processor.step(_ablate(observation, modality))
            latencies.append(time.perf_counter_ns() - start)
        xs.append(feature)
        ys.append(target)
        updates += processor.updates
    return np.asarray(xs), np.asarray(ys), np.asarray(latencies), updates


def _run_seed(config, world_config, seed):
    if config.outcome_tick >= world_config.ticks:
        raise ValueError("outcome_tick must be inside the episode")
    world = LineWorld(world_config)
    codebook = np.random.default_rng(seed).permutation([140., 340.])
    frequency_map = {-1: float(codebook[0]), 1: float(codebook[1])}
    train = create_data(world, config.train_episodes, seed + 1, frequency_map, config.outcome_tick)
    test = create_data(world, config.test_episodes, seed + 2, frequency_map, config.outcome_tick)
    rows = []
    for code, kind in PROCESSORS.items():
        for modality in MODALITIES:
            processor = kind(pixels=world_config.pixels, hidden=config.hidden, seed=seed)
            tracemalloc.start()
            cpu_start = time.process_time()
            x_train, y_train, _, train_updates = _features(
                processor, train, modality, config.prediction_tick)
            readout = RidgeReadout()
            readout.fit(x_train, y_train)
            x_test, y_test, latencies, test_updates = _features(
                processor, test, modality, config.prediction_tick)
            scores = readout.predict(x_test)
            cpu_seconds = time.process_time() - cpu_start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            absolute_errors = np.abs(scores - y_test)
            exact_accuracy = float(np.mean(np.rint(scores) == y_test))
            rows.append({
                "seed": seed,
                "train_episodes": config.train_episodes,
                "test_episodes": config.test_episodes,
                "prediction_tick": config.prediction_tick,
                "outcome_tick": config.outcome_tick,
                "pixels": world_config.pixels,
                "visual_hz": world_config.visual_hz,
                "audio_hz": world_config.audio_hz,
                "episode_ticks": world_config.ticks,
                "tone_for_left_direction_hz": frequency_map[-1],
                "tone_for_right_direction_hz": frequency_map[1],
                "noise_std": world_config.noise_std,
                "tone_amplitude": world_config.tone_amplitude,
                "case": code,
                "architecture": processor.name,
                "modality": modality,
                "exact_position_accuracy": exact_accuracy,
                "within_1_pixel_accuracy": float(np.mean(absolute_errors <= 1.0)),
                "mean_absolute_error_pixels": float(np.mean(absolute_errors)),
                "test_episodes": len(test),
                "train_updates": train_updates,
                "active_updates": test_updates,
                "median_step_latency_us": float(np.median(latencies) / 1000),
                "p95_step_latency_us": float(np.percentile(latencies, 95) / 1000),
                "cpu_seconds_train_and_test": cpu_seconds,
                "peak_traced_bytes": peak_bytes,
                "hidden_units": config.hidden,
                "readout_training_examples": int(readout.train.shape[0]),
            })
    return rows


def run(config=BenchmarkConfig(), world_config=WorldConfig(), output="results.csv"):
    if config.outcome_tick >= world_config.ticks:
        raise ValueError("outcome_tick must be inside the episode")
    rows = []
    for replicate in range(config.seeds):
        rows.extend(_run_seed(config, world_config, config.seed + replicate * 1009))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
