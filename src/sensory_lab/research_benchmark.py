"""Compositional generalization, sample-efficiency, and compute experiments."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .benchmark import _ablate
from .contracts import Observation
from .environment import LineWorld, MOTION_START_TICK, WorldConfig
from .processors import PROCESSORS, FeedforwardHistory, sensory_features


DEFAULT_HELDOUT = ((-1, 2), (1, 3))
DEFAULT_SIZES = (10, 25, 50, 100, 250, 500)
DEFAULT_HORIZONS = (3, 6, 9, 12)
DEFAULT_ALPHAS = (0.001, 0.01, 0.05, 0.2, 1.0)
MODEL_IDS = ("A", "B", "C", "D", "E1")
MODALITIES = ("vision_only", "audio_only", "combined")


@dataclass(frozen=True)
class ResearchConfig:
    seed: int = 42
    seeds: int = 5
    train_sizes: tuple[int, ...] = DEFAULT_SIZES
    generalization_train_episodes: int = 500
    validation_per_combo: int = 25
    test_per_combo: int = 40
    heldout_combinations: tuple[tuple[int, int], ...] = DEFAULT_HELDOUT
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    velocities: tuple[int, ...] = (1, 2, 3)
    prediction_tick: int = 4
    hidden: int = 64
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    timed_repeats: int = 5

    def __post_init__(self):
        if self.seeds < 2 or self.generalization_train_episodes < 1:
            raise ValueError("use at least two seeds and a positive training size")
        if tuple(sorted(set(self.train_sizes))) != self.train_sizes or min(self.train_sizes) < 1:
            raise ValueError("train_sizes must be strictly increasing positive integers")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive motion-interval counts")
        if self.prediction_tick >= MOTION_START_TICK:
            raise ValueError("prediction inputs must end before motion begins")
        all_pairs = {(d, v) for d in (-1, 1) for v in self.velocities}
        held = set(self.heldout_combinations)
        if not held or not held < all_pairs:
            raise ValueError("held-out pairs must be a nonempty subset of command pairs")
        observed = all_pairs - held
        if {d for d, _ in observed} != {-1, 1} or {v for _, v in observed} != set(self.velocities):
            raise ValueError("every direction and velocity must occur in observed training pairs")
        if min(self.velocities) < 1 or len(set(self.velocities)) != len(self.velocities):
            raise ValueError("velocities must be unique positive integers")
        if self.timed_repeats < 2 or not self.alphas:
            raise ValueError("use at least two timing repeats and one alpha")


@dataclass
class Split:
    train_pool: list[dict]
    validation: list[dict]
    observed_test: list[dict]
    heldout_test: list[dict]
    phase_by_direction: dict[int, float]
    world: LineWorld
    world_config: WorldConfig


@dataclass
class FittedKernel:
    mean: np.ndarray
    scale: np.ndarray
    train: np.ndarray
    dual: np.ndarray
    kind: str

    def predict_one(self, x):
        z = (np.asarray(x) - self.mean) / self.scale
        return float((_kernel(z[None, :], self.train, self.kind) @ self.dual)[0])


class DirectLinearReadout:
    """Linear kernel ridge baseline, represented by a primal coefficient vector."""
    kernel = "linear"


class QuadraticKernelReadout:
    """Degree-two kernel ridge readout used by A-D."""
    kernel = "quadratic"


def _balanced_pairs(pairs, count, rng):
    order = []
    while len(order) < count:
        cycle = list(pairs)
        rng.shuffle(cycle)
        order.extend(cycle)
    return order[:count]


def _target_center(observation: Observation) -> float:
    pixels = np.arange(len(observation.vision), dtype=float)
    total = float(np.sum(observation.vision))
    return float(np.sum(pixels * observation.vision) / total)


def _generate_examples(world, count, seed, pairs, phase_map, horizons, pixels):
    rng = np.random.default_rng(seed)
    commands = _balanced_pairs(pairs, count, rng)
    examples = []
    for i, (direction, velocity) in enumerate(commands):
        initial = world.sample_initial_position(rng)
        max_tick = MOTION_START_TICK + max(horizons)
        max_target = initial + direction * velocity * max(horizons)
        radius = 2  # largest existing visual object has width five
        if not radius <= max_target < pixels - radius:
            raise ValueError("world field clips a requested target horizon")
        tone_hz = 120.0 + 180.0 * velocity
        observations = world.episode(
            seed=seed * 100_000 + i,
            direction=direction,
            velocity=velocity,
            tone_hz=tone_hz,
            initial_position=initial,
            phase_offset=phase_map[direction],
        )
        targets = {
            horizon: _target_center(observations[MOTION_START_TICK + horizon])
            for horizon in horizons
        }
        # Targets are stored beside, never inside, Observation objects.
        examples.append({
            "observations": observations,
            "trajectory_seed": seed * 100_000 + i,
            "targets": targets,
            "direction": direction,
            "velocity": velocity,
            "initial_position": initial,
            "combination": (direction, velocity),
            "target_frames": {h: MOTION_START_TICK + h for h in horizons},
        })
    return examples


def make_split(config: ResearchConfig, seed: int) -> Split:
    """Build train/validation/test data with no held-out supervised examples."""
    max_displacement = max(config.velocities) * max(config.horizons)
    pixels = max(129, 3 * (max_displacement + 5))
    world_config = WorldConfig(pixels=pixels, ticks=MOTION_START_TICK + max(config.horizons) + 1)
    world = LineWorld(world_config)
    all_pairs = [(d, v) for d in (-1, 1) for v in config.velocities]
    heldout = set(config.heldout_combinations)
    observed = [pair for pair in all_pairs if pair not in heldout]
    phase_sign = int(np.random.default_rng(seed).choice([-1, 1]))
    phase_map = {-1: -phase_sign * np.pi/2, 1: phase_sign * np.pi/2}
    max_train = max(config.train_sizes + (config.generalization_train_episodes,))
    train_pool = _generate_examples(world, max_train, seed + 1, observed,
                                    phase_map, config.horizons, pixels)
    validation = _generate_examples(world, config.validation_per_combo * len(observed),
                                    seed + 2, observed, phase_map,
                                    config.horizons, pixels)
    observed_test = _generate_examples(world, config.test_per_combo * len(observed),
                                       seed + 3, observed, phase_map,
                                       config.horizons, pixels)
    heldout_test = _generate_examples(world, config.test_per_combo * len(heldout),
                                      seed + 4, sorted(heldout), phase_map,
                                      config.horizons, pixels)
    return Split(train_pool, validation, observed_test, heldout_test,
                 phase_map, world, world_config)


def _feature_vector(model_id, observations, modality, hidden, seed, pixels):
    if model_id == "E1":
        history = []
        for observation in observations:
            masked = _ablate(observation, modality)
            history.append(sensory_features(masked))
        width = pixels + 4 + 16 + 49
        history = history[-4:]
        return np.concatenate([np.zeros(width * (4 - len(history)))] + history)
    processor = PROCESSORS[model_id](pixels=pixels, hidden=hidden, seed=seed)
    result = None
    for observation in observations:
        result = processor.step(_ablate(observation, modality))
    return result


def extract_matrix(model_id, examples, modality, config, seed, pixels):
    rows, updates = [], 0
    for item in examples:
        if model_id == "E1":
            rows.append(_feature_vector(model_id,
                                        item["observations"][:config.prediction_tick + 1],
                                        modality, config.hidden, seed, pixels))
        else:
            processor = PROCESSORS[model_id](pixels=pixels, hidden=config.hidden, seed=seed)
            feature = None
            for observation in item["observations"][:config.prediction_tick + 1]:
                feature = processor.step(_ablate(observation, modality))
            rows.append(feature)
            updates += processor.updates
    return np.asarray(rows), updates


def _targets(examples, horizons):
    return np.asarray([[item["targets"][h] for h in horizons] for item in examples], dtype=float)


def _kernel(a, b, kind):
    dot = (a @ b.T) / max(1, a.shape[1])
    return dot if kind == "linear" else (1.0 + dot) ** 2


def fit_validate_predict(x_train, y_train, x_val, y_val, x_eval, alphas, kind):
    """Select alpha only on seen-combination validation; normalize on train only."""
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-8] = 1.0
    z_train, z_val, z_eval = ((x - mean) / scale for x in (x_train, x_val, x_eval))
    gram = _kernel(z_train, z_train, kind)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected_targets = eigenvectors.T @ y_train
    val_kernel = _kernel(z_val, z_train, kind)
    eval_kernel = _kernel(z_eval, z_train, kind)
    best_alpha = np.full(y_train.shape[1], float(alphas[0]))
    best_error = np.full(y_train.shape[1], np.inf)
    for alpha in alphas:
        coeff = eigenvectors @ (projected_targets / (eigenvalues[:, None] + alpha))
        errors = np.mean(np.abs(val_kernel @ coeff - y_val), axis=0)
        take = errors < best_error
        best_error[take] = errors[take]
        best_alpha[take] = alpha
    coeff = eigenvectors @ (projected_targets / (eigenvalues[:, None] + best_alpha[None, :]))
    predictions = eval_kernel @ coeff
    readouts = [FittedKernel(mean, scale, z_train, coeff[:, j], kind)
                for j in range(y_train.shape[1])]
    # Count stored numeric values: normalized support vectors, the shared
    # normalization vectors, and one dual coefficient per support vector and
    # target horizon. This is an implementation storage count, not a count of
    # biological or trainable processor parameters.
    stored_values = x_train.size + mean.size + scale.size + coeff.size
    return predictions, best_alpha, stored_values, readouts


def direct_linear_baseline(x_train, y_train, x_val, y_val, x_eval, alphas):
    predictions, alpha, _, _ = fit_validate_predict(
        x_train, y_train, x_val, y_val, x_eval, alphas, "linear")
    return predictions, alpha, x_train.shape[1] + 1 + 2 * x_train.shape[1]


def _metrics(predictions, examples, horizon):
    truth = np.asarray([x["targets"][horizon] for x in examples])
    error = np.abs(predictions - truth)
    return {
        "mae_pixels": float(np.mean(error)),
        "within_1px_accuracy": float(np.mean(error <= 1.0)),
        "exact_position_accuracy": float(np.mean(np.rint(predictions) == truth)),
    }


def _raw_result_rows(seed, model, modality, horizon, split, examples, predictions):
    rows = []
    by_combo = {}
    for pair in sorted(set(x["combination"] for x in examples)):
        indices = [i for i, x in enumerate(examples) if x["combination"] == pair]
        by_combo[pair] = indices
        metrics = _metrics(predictions[indices], [examples[i] for i in indices], horizon)
        rows.append({"seed": seed, "model": model, "modality": modality,
                     "horizon_intervals": horizon,
                     "target_frame": MOTION_START_TICK + horizon,
                     "evaluation_split": f"{split}_combination",
                     "direction": pair[0], "velocity": pair[1], **metrics})
    all_metrics = _metrics(predictions, examples, horizon)
    rows.append({"seed": seed, "model": model, "modality": modality,
                 "horizon_intervals": horizon,
                 "target_frame": MOTION_START_TICK + horizon,
                 "evaluation_split": split,
                 "direction": "all", "velocity": "all", **all_metrics})
    return rows, by_combo


def _oracle_predictions(examples, horizon):
    return np.asarray([x["initial_position"] + x["direction"] * x["velocity"] * horizon
                       for x in examples], dtype=float)


def _one_seed_generalization(config, seed):
    split = make_split(config, seed)
    train = split.train_pool[:config.generalization_train_episodes]
    observed_commands = {x["combination"] for x in train}
    if not observed_commands.issuperset(
            {pair for pair in [(d, v) for d in (-1, 1) for v in config.velocities]
             if pair not in set(config.heldout_combinations)}):
        raise AssertionError("training data failed to include all observed command pairs")
    test = split.observed_test + split.heldout_test
    result_rows = []
    for model_id in MODEL_IDS:
        for modality in MODALITIES:
            model_seed = seed + (list(MODEL_IDS).index(model_id) * 10007)
            x_train, _ = extract_matrix(model_id, train, modality, config, model_seed,
                                        split.world_config.pixels)
            x_val, _ = extract_matrix(model_id, split.validation, modality, config, model_seed,
                                      split.world_config.pixels)
            x_test, _ = extract_matrix(model_id, test, modality, config, model_seed,
                                       split.world_config.pixels)
            y_train = _targets(train, config.horizons)
            y_val = _targets(split.validation, config.horizons)
            y_test = _targets(test, config.horizons)
            kind = "linear" if model_id == "E1" else "quadratic"
            predictions, alphas, parameter_count, _ = fit_validate_predict(
                x_train, y_train, x_val, y_val, x_test, config.alphas, kind)
            for j, horizon in enumerate(config.horizons):
                pred = predictions[:, j]
                observed_indices = [i for i, item in enumerate(test)
                                    if item["combination"] not in set(config.heldout_combinations)]
                held_indices = [i for i, item in enumerate(test)
                                if item["combination"] in set(config.heldout_combinations)]
                for group, indices in (("observed", observed_indices), ("heldout", held_indices)):
                    rows, _ = _raw_result_rows(seed, model_id, modality, horizon, group,
                                               [test[i] for i in indices], pred[indices])
                    for row in rows:
                        row["selected_alpha"] = float(alphas[j])
                        row["readout_parameters"] = int(parameter_count)
                    result_rows.extend(rows)
                seen_mae = float(np.mean(np.abs(pred[observed_indices] - y_test[observed_indices, j])))
                held_mae = float(np.mean(np.abs(pred[held_indices] - y_test[held_indices, j])))
                result_rows.append({"seed": seed, "model": model_id, "modality": modality,
                                    "horizon_intervals": horizon,
                                    "target_frame": MOTION_START_TICK + horizon,
                                    "evaluation_split": "generalization_gap",
                                    "direction": "all", "velocity": "all",
                                    "mae_pixels": held_mae - seen_mae,
                                    "within_1px_accuracy": float(
                                        np.mean(np.abs(pred[held_indices]-y_test[held_indices,j]) <= 1)
                                        - np.mean(np.abs(pred[observed_indices]-y_test[observed_indices,j]) <= 1)),
                                    "exact_position_accuracy": np.nan,
                                    "selected_alpha": float(alphas[j]),
                                    "readout_parameters": int(parameter_count)})
    for horizon in config.horizons:
        for group, examples in (("observed", split.observed_test), ("heldout", split.heldout_test)):
            pred = _oracle_predictions(examples, horizon)
            rows, _ = _raw_result_rows(seed, "E2_oracle", "privileged", horizon,
                                       group, examples, pred)
            result_rows.extend(rows)
    phase_map_json = json.dumps({str(k): float(v) for k, v in split.phase_by_direction.items()})
    heldout_json = json.dumps(sorted([list(pair) for pair in config.heldout_combinations]))
    for row in result_rows:
        row["train_episodes"] = len(train)
        row["validation_episodes"] = len(split.validation)
        row["observed_test_episodes"] = len(split.observed_test)
        row["heldout_test_episodes"] = len(split.heldout_test)
        row["heldout_combinations"] = heldout_json
        row["velocity_levels"] = ",".join(map(str, config.velocities))
        row["audio_frequency_hz_formula"] = "120 + 180 * velocity"
        row["direction_phase_map_rad"] = phase_map_json
    return result_rows


def _sample_efficiency_seed(config, seed):
    split = make_split(config, seed)
    test = split.observed_test + split.heldout_test
    horizon = 9 if 9 in config.horizons else max(config.horizons)
    rows = []
    for train_size in config.train_sizes:
        train = split.train_pool[:train_size]
        for model_id in MODEL_IDS:
            model_seed = seed + (list(MODEL_IDS).index(model_id) * 10007)
            tracemalloc.start()
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            x_train, updates_train = extract_matrix(model_id, train, "combined", config,
                                                    model_seed, split.world_config.pixels)
            x_val, _ = extract_matrix(model_id, split.validation, "combined", config,
                                      model_seed, split.world_config.pixels)
            x_eval, updates_test = extract_matrix(model_id, test, "combined", config,
                                                  model_seed, split.world_config.pixels)
            feature_wall = time.perf_counter() - wall_start
            fit_wall_start = time.perf_counter()
            y_train = np.asarray([[x["targets"][horizon]] for x in train])
            y_val = np.asarray([[x["targets"][horizon]] for x in split.validation])
            kind = "linear" if model_id == "E1" else "quadratic"
            # The held-out set is never involved in normalization or alpha selection.
            prediction, alpha, parameters, _ = fit_validate_predict(
                x_train, y_train, x_val, y_val, x_eval, config.alphas, kind)
            fit_wall = time.perf_counter() - fit_wall_start
            cpu_seconds = time.process_time() - cpu_start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            for group, examples in (("observed", split.observed_test),
                                    ("heldout", split.heldout_test)):
                start = len(split.observed_test) if group == "heldout" else 0
                end = start + len(examples)
                metrics = _metrics(prediction[start:end, 0], examples, horizon)
                rows.append({"seed": seed, "model": model_id,
                             "train_episodes": train_size,
                             "evaluation_split": group,
                             "horizon_intervals": horizon,
                             "target_frame": MOTION_START_TICK + horizon,
                             "modality": "combined", **metrics,
                             "selected_alpha": float(alpha[0]),
                             "feature_extraction_wall_s": feature_wall,
                             "readout_fit_wall_s": fit_wall,
                             "training_wall_s": feature_wall + fit_wall,
                             "training_cpu_s": cpu_seconds,
                             "process_peak_rss_bytes": _rss_bytes(),
                             "active_updates_train": updates_train,
                             "active_updates_test": updates_test,
                             "readout_parameters": int(parameters),
                             "peak_traced_bytes": peak_bytes})
    return rows


def _t_critical(n):
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
            7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)


def summarize(records, keys, metric_names):
    buckets = {}
    for row in records:
        key = tuple(row.get(k, "") for k in keys)
        buckets.setdefault(key, []).append(row)
    summaries = []
    for key, group in buckets.items():
        out = dict(zip(keys, key))
        out["seed_count"] = len(group)
        out["seed_ids"] = json.dumps(sorted({int(r["seed"]) for r in group if "seed" in r}))
        for metric in metric_names:
            values = np.asarray([float(r[metric]) for r in group if r.get(metric) is not None], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = _t_critical(len(values)) * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = sd
            out[f"{metric}_ci95_low"] = mean - half
            out[f"{metric}_ci95_high"] = mean + half
            out[f"{metric}_by_seed"] = json.dumps([float(v) for v in values])
        summaries.append(out)
    return summaries


def _csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _generalization(config):
    raw = []
    for i in range(config.seeds):
        seed = config.seed + i * 1009
        raw.extend(_one_seed_generalization(config, seed))
    keys = ("model", "modality", "horizon_intervals", "target_frame",
            "evaluation_split", "direction", "velocity")
    return raw, summarize(raw, keys, ("mae_pixels", "within_1px_accuracy",
                                      "exact_position_accuracy"))


def _sample_efficiency(config):
    raw = []
    for i in range(config.seeds):
        raw.extend(_sample_efficiency_seed(config, config.seed + i * 1009))
    keys = ("model", "train_episodes", "evaluation_split", "horizon_intervals",
            "target_frame", "modality")
    return raw, summarize(raw, keys, ("mae_pixels", "within_1px_accuracy",
                                     "exact_position_accuracy", "feature_extraction_wall_s",
                                     "readout_fit_wall_s", "training_cpu_s",
                                     "training_wall_s", "peak_traced_bytes",
                                     "process_peak_rss_bytes"))


def _rss_bytes():
    """Best-effort process working-set/RSS measurement using only stdlib APIs."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else 0
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return 0


def _fixed_parameters(model_id, processor):
    if model_id == "E1" or processor is None:
        return 0, 0
    allocated = int(processor.w.size + processor.a.size)
    active = 0 if model_id == "A" else allocated
    return allocated, active


def _instrumented_inference(model_id, examples, modality, fitted, config, seed, pixels):
    processor = None if model_id == "E1" else PROCESSORS[model_id](
        pixels=pixels, hidden=config.hidden, seed=seed)
    preproc_ns = [0]
    if processor is not None:
        original = processor.input_features

        def timed_input(observation):
            start = time.perf_counter_ns()
            value = original(observation)
            preproc_ns[0] += time.perf_counter_ns() - start
            return value

        processor.input_features = timed_input
    feature_rows = []
    step_ns = []
    readout_ns = []
    episode_ns = []
    updates = 0
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for item in examples:
        observations = item["observations"][:config.prediction_tick + 1]
        if model_id == "E1":
            prep_start = time.perf_counter_ns()
            feature = _feature_vector(model_id, observations, modality, config.hidden, seed, pixels)
            total_step = time.perf_counter_ns() - prep_start
            preproc_ns[0] += total_step
        else:
            processor.reset()
            feature = None
            total_step = 0
            for observation in observations:
                masked = _ablate(observation, modality)
                start = time.perf_counter_ns()
                feature = processor.step(masked)
                elapsed = time.perf_counter_ns() - start
                total_step += elapsed
                step_ns.append(elapsed)
            updates += processor.updates
        start_readout = time.perf_counter_ns()
        fitted.predict_one(feature)
        rtime = time.perf_counter_ns() - start_readout
        if processor is None:
            step_ns.extend([0] * (config.prediction_tick + 1))
        readout_ns.append(rtime)
        episode_ns.append(total_step + rtime)
        feature_rows.append(feature)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    return {
        "processor": processor,
        "preprocessing_ns": preproc_ns[0],
        "step_ns": step_ns,
        "readout_ns": readout_ns,
        "episode_ns": episode_ns,
        "updates": updates,
        "inference_cpu_s": cpu_seconds,
        "inference_wall_s": wall_seconds,
        "features": np.asarray(feature_rows),
    }


def _timing_summary(config, seed):
    split = make_split(config, seed)
    train = split.train_pool[:config.generalization_train_episodes]
    test = split.observed_test + split.heldout_test
    horizon = 9 if 9 in config.horizons else max(config.horizons)
    rows = []
    timed_examples = test[:min(48, len(test))]
    metadata = {
        "seed": seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor_name": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
        "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "warmup_trials": 1,
        "timed_repeats": config.timed_repeats,
        "velocity_levels": ",".join(map(str, config.velocities)),
        "heldout_combinations": json.dumps(sorted([list(p) for p in config.heldout_combinations])),
        "horizons": ",".join(map(str, config.horizons)),
        "audio_frequency_hz_formula": "120 + 180 * velocity",
        "direction_phase_map_rad": json.dumps({str(k): float(v)
                                               for k, v in split.phase_by_direction.items()}),
        "visual_pixels": split.world_config.pixels,
    }
    for model_id in MODEL_IDS:
        modality = "combined"
        model_seed = seed + list(MODEL_IDS).index(model_id) * 10007
        tracemalloc.start()
        train_feature_wall0 = time.perf_counter()
        train_cpu0 = time.process_time()
        x_train, train_updates = extract_matrix(model_id, train, modality, config,
                                                model_seed, split.world_config.pixels)
        x_val, _ = extract_matrix(model_id, split.validation, modality, config,
                                  model_seed, split.world_config.pixels)
        train_feature_wall = time.perf_counter() - train_feature_wall0
        y_train = np.asarray([[x["targets"][horizon]] for x in train])
        y_val = np.asarray([[x["targets"][horizon]] for x in split.validation])
        fit0 = time.perf_counter()
        _, alpha, readout_params, readouts = fit_validate_predict(
            x_train, y_train, x_val, y_val, x_train[:1], config.alphas,
            "linear" if model_id == "E1" else "quadratic")
        fit_wall = time.perf_counter() - fit0
        training_cpu = time.process_time() - train_cpu0
        fitted = readouts[0]
        # One warm-up episode is explicitly excluded from latency summaries.
        _instrumented_inference(model_id, timed_examples[:1], modality, fitted,
                                config, model_seed, split.world_config.pixels)
        repeated = []
        for _ in range(config.timed_repeats):
            repeated.append(_instrumented_inference(
                model_id, timed_examples, modality, fitted, config,
                model_seed, split.world_config.pixels))
        step_lat = np.asarray([v for run in repeated for v in run["step_ns"]], dtype=float)
        episode_lat = np.asarray([v for run in repeated for v in run["episode_ns"]], dtype=float)
        readout_lat = np.asarray([v for run in repeated for v in run["readout_ns"]], dtype=float)
        prep_s = sum(run["preprocessing_ns"] for run in repeated) / 1e9
        updates = sum(run["updates"] for run in repeated)
        processor = repeated[-1]["processor"]
        fixed_alloc, fixed_active = _fixed_parameters(model_id, processor)
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dimension = x_train.shape[1]
        obs_per_trial = config.prediction_tick + 1
        calls = len(timed_examples) * obs_per_trial
        hidden = config.hidden
        if model_id == "A":
            proc_ops_per_step = 0
        elif model_id == "E1":
            proc_ops_per_step = 0
        else:
            base_ops = 2 * hidden * (hidden + dimension)
            rate = updates / max(1, calls * config.timed_repeats)
            proc_ops_per_step = int(base_ops * (2 if model_id == "C" else 1) * rate)
            if model_id == "D":
                proc_ops_per_step += 3 * dimension
        preprocess_ops = int(5 * 128 * math.log2(128)) + 36 * dimension
        readout_ops = int(2 * len(train) * dimension + 5 * len(train))
        fit_ops = int((2/3) * len(train)**3 + len(train)**2 * dimension)
        learned_params = int(readout_params)
        rows.append({
            **metadata,
            "model": model_id,
            "modality": modality,
            "horizon_intervals": horizon,
            "train_episodes": len(train),
            "selected_alpha": float(alpha[0]),
            "feature_extraction_wall_s": train_feature_wall,
            "readout_training_wall_s": fit_wall,
            "training_cpu_s": training_cpu,
            "prediction_episode_median_ms": float(np.median(episode_lat) / 1e6),
            "prediction_episode_p95_ms": float(np.percentile(episode_lat, 95) / 1e6),
            "prediction_episode_p99_ms": float(np.percentile(episode_lat, 99) / 1e6),
            "processor_step_median_us": float(np.median(step_lat) / 1000),
            "processor_step_p95_us": float(np.percentile(step_lat, 95) / 1000),
            "readout_predict_median_us": float(np.median(readout_lat) / 1000),
            "feature_preprocessing_total_s": prep_s,
            "processor_internal_estimated_total_s": max(
                0.0, float(np.sum(step_lat) / 1e9) - prep_s),
            "inference_cpu_total_s": sum(r["inference_cpu_s"] for r in repeated),
            "inference_wall_total_s": sum(r["inference_wall_s"] for r in repeated),
            "timed_prediction_trials": len(episode_lat),
            "trainable_processor_parameters": 0,
            "readout_fitted_parameters": learned_params,
            "fixed_parameters_allocated": fixed_alloc,
            "fixed_parameters_active": fixed_active,
            "active_updates_training": train_updates,
            "active_updates_timed_inference": updates,
            "estimated_preprocessing_ops_per_observation": preprocess_ops,
            "estimated_processor_ops_per_observation": proc_ops_per_step,
            "estimated_readout_ops_per_prediction": readout_ops,
            "estimated_readout_fit_ops": fit_ops,
            "peak_traced_bytes": peak_traced,
            "process_peak_rss_bytes": _rss_bytes(),
        })
    # Privileged analytic reference; timed separately and never included as a fair model.
    oracle_times = []
    oracle_cpu0 = time.process_time()
    oracle_wall0 = time.perf_counter()
    for _ in range(config.timed_repeats):
        for item in timed_examples:
            tick0 = time.perf_counter_ns()
            item["initial_position"] + item["direction"] * item["velocity"] * horizon
            oracle_times.append(time.perf_counter_ns() - tick0)
    oracle_cpu = time.process_time() - oracle_cpu0
    oracle_wall = time.perf_counter() - oracle_wall0
    rows.append({**metadata, "model": "E2_oracle", "modality": "privileged",
                 "horizon_intervals": horizon, "train_episodes": 0,
                 "selected_alpha": "", "feature_extraction_wall_s": 0.0,
                 "readout_training_wall_s": 0.0, "training_cpu_s": 0.0,
                 "prediction_episode_median_ms": float(np.median(oracle_times)/1e6),
                 "prediction_episode_p95_ms": float(np.percentile(oracle_times,95)/1e6),
                 "prediction_episode_p99_ms": float(np.percentile(oracle_times,99)/1e6),
                 "processor_step_median_us": 0.0, "processor_step_p95_us": 0.0,
                 "readout_predict_median_us": 0.0,
                 "feature_preprocessing_total_s": 0.0,
                 "inference_cpu_total_s": oracle_cpu,
                 "inference_wall_total_s": oracle_wall,
                 "timed_prediction_trials": len(oracle_times),
                 "trainable_processor_parameters": 0, "readout_fitted_parameters": 0,
                 "fixed_parameters_allocated": 0, "fixed_parameters_active": 0,
                 "active_updates_training": 0, "active_updates_timed_inference": 0,
                 "estimated_preprocessing_ops_per_observation": 0,
                 "estimated_processor_ops_per_observation": 3,
                 "estimated_readout_ops_per_prediction": 3,
                 "estimated_readout_fit_ops": 0, "peak_traced_bytes": 0,
                 "process_peak_rss_bytes": _rss_bytes(),
                 "note": "privileged oracle; uses ground-truth start, direction, velocity"})
    return rows


def _write_plots(generalization_rows, sample_rows, compute_rows, outdir, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_order = ["A", "B", "C", "D", "E1", "E2_oracle"]
    command_order = [(d, v) for d in (-1, 1) for v in config.velocities]
    heldout = set(config.heldout_combinations)
    h = 9 if 9 in config.horizons else max(config.horizons)
    matrix = np.full((len(model_order), len(command_order)), np.nan)
    lookup = {(r["model"], str(r["evaluation_split"]), int(r["direction"]), int(r["velocity"])): r
              for r in generalization_rows
              if r.get("evaluation_split") in ("observed_combination", "heldout_combination")
              and r.get("modality") in ("combined", "privileged")
              and int(r.get("horizon_intervals", -1)) == h}
    for mi, model in enumerate(model_order):
        for ci, (direction, velocity) in enumerate(command_order):
            split_name = "heldout_combination" if (direction, velocity) in heldout else "observed_combination"
            modality = "privileged" if model == "E2_oracle" else "combined"
            row = lookup.get((model, split_name, direction, velocity))
            if row is not None and row.get("mae_pixels_mean") is not None:
                matrix[mi, ci] = float(row["mae_pixels_mean"])
    labels = [("L" if d == -1 else "R") + str(v) + (" *" if (d, v) in heldout else "")
              for d, v in command_order]
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(model_order)), model_order)
    ax.set_xlabel("Direction and velocity combination (* held out from supervised training)")
    ax.set_title(f"Compositional generalization · combined input · {h} motion intervals")
    fig.colorbar(image, ax=ax, label="Mean absolute position error (pixels)")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.savefig(outdir / "generalization_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, split in zip(axes, ("observed", "heldout")):
        for model in ["A", "B", "C", "D", "E1"]:
            points = sorted((int(r["train_episodes"]), float(r["mae_pixels_mean"]))
                            for r in sample_rows
                            if r.get("model") == model and r.get("evaluation_split") == split
                            and r.get("modality") == "combined")
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_xlabel("Training episodes")
        ax.set_title(f"{split.capitalize()} combinations")
        ax.grid(True, alpha=.25)
    axes[0].set_ylabel("Mean absolute position error (pixels)")
    axes[1].legend(title="Model", fontsize=8)
    fig.suptitle("Sample efficiency at the 9-interval horizon · combined input")
    fig.savefig(outdir / "learning_curves.png", dpi=160)
    plt.close(fig)

    models = ["A", "B", "C", "D", "E1", "E2_oracle"]
    med, p95, p99 = [], [], []
    for model in models:
        rows = [r for r in compute_rows if r.get("model") == model]
        med.append(float(np.mean([r["prediction_episode_median_ms"] for r in rows])))
        p95.append(float(np.mean([r["prediction_episode_p95_ms"] for r in rows])))
        p99.append(float(np.mean([r["prediction_episode_p99_ms"] for r in rows])))
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(models))
    width = .24
    ax.bar(x-width, med, width, label="Median")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x+width, p99, width, label="p99")
    ax.set_xticks(x, models)
    ax.set_ylabel("End-to-end prediction latency (ms)")
    ax.set_title("CPU inference latency · combined input (E2 is privileged)")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.savefig(outdir / "latency_comparison.png", dpi=160)
    plt.close(fig)


def _report(generalization_rows, sample_rows, compute_rows, config, output_path):
    def find_g(model, split, metric="mae_pixels_mean", modality="combined", horizon=9):
        h = horizon if horizon in config.horizons else max(config.horizons)
        row = next((r for r in generalization_rows
                    if r.get("model") == model and r.get("modality") == modality
                    and r.get("evaluation_split") == split and r.get("direction") == "all"
                    and int(r.get("horizon_intervals", -1)) == h), None)
        return float(row[metric]) if row and row.get(metric) not in (None, "") else float("nan")

    general_lines = []
    for model in ("A", "B", "C", "D", "E1", "E2_oracle"):
        mod = "privileged" if model == "E2_oracle" else "combined"
        seen = find_g(model, "observed", modality=mod)
        held = find_g(model, "heldout", modality=mod)
        gap = held - seen
        general_lines.append(f"| {model} | {seen:.3f} | {held:.3f} | {gap:.3f} |")

    ablation_lines = []
    for model in ("A", "B", "C", "D"):
        values = [find_g(model, "observed", modality=modality)
                  for modality in ("vision_only", "audio_only", "combined")]
        held_combined = find_g(model, "heldout", modality="combined")
        ablation_lines.append(
            f"| {model} | {values[0]:.3f} | {values[1]:.3f} | "
            f"{values[2]:.3f} | {held_combined:.3f} |"
        )

    sample_lines = []
    for model in ("A", "B", "C", "D", "E1"):
        vals = [(int(r["train_episodes"]), float(r["mae_pixels_mean"]))
                for r in sample_rows if r.get("model") == model
                and r.get("evaluation_split") == "heldout" and r.get("modality") == "combined"]
        vals.sort()
        threshold = next((n for n, mae in vals if mae <= 1.0), None)
        target = "not reached" if threshold is None else str(threshold)
        sample_lines.append(f"| {model} | {target} | "
                            + ", ".join(f"{n}: {mae:.2f}" for n, mae in vals) + " |")

    compute_lookup = {r["model"]: r for r in compute_rows}
    timing_metadata = compute_rows[0] if compute_rows else {}
    d_cpu = compute_lookup.get("D", {}).get("inference_cpu_total_s", float("nan"))
    b_cpu = compute_lookup.get("B", {}).get("inference_cpu_total_s", float("nan"))
    c_seen, c_held = find_g("C", "observed"), find_g("C", "heldout")
    b_seen, b_held = find_g("B", "observed"), find_g("B", "heldout")
    compute_lines = []
    for model in ("A", "B", "C", "D"):
        row = compute_lookup[model]
        compute_lines.append(
            f"| {model} | {float(row['prediction_episode_median_ms']):.3f} | "
            f"{float(row['prediction_episode_p95_ms']):.3f} | "
            f"{float(row['inference_cpu_total_s']):.4f} | "
            f"{int(row['active_updates_timed_inference'])} |"
        )
    report = f'''# Multimodal Generalization and Efficiency Experiment

## Protocol

- Seeds: {config.seeds}, beginning at {config.seed} and spaced by 1009.
- Training direction/speed pairs: all pairs except {config.heldout_combinations}.
- Training sizes: {config.train_sizes}; validation examples are sampled only from permitted training pairs.
- Horizons: {config.horizons} movement intervals. Motion begins at frame {MOTION_START_TICK}; target frame is `motion_start + horizon`.
- The field is sized to avoid edge clipping for every tested horizon and speed.
- Audio cue relationship: carrier frequency is `120 + 180 × speed` Hz; direction is encoded by phase `±π/2`, with the sign-to-direction assignment randomized per seed. These are explicit factor cues; the test is compositional use of known factors, not discovery of an unknown physics law.
- Candidate ridge alpha is selected on validation combinations only. Feature normalization is fitted on training examples only. Held-out combinations do not enter training, normalization, or alpha selection.
- A–D processor equations remain fixed. E1 is a direct linear ridge baseline over the same fixed temporal sensory history. E2 is a privileged analytic oracle using ground-truth initial position, direction, and speed; it is not a fair competitor.

## Generalization at nine motion intervals

Mean absolute position error (pixels), averaged across seeds:

| Model | Observed combinations | Held-out combinations | Generalization gap (held-out − observed) |
|---|---:|---:|---:|
{chr(10).join(general_lines)}

The machine-readable CSV includes modality ablations, each horizon, individual held-out pairs, within-one-pixel accuracy, exact-position accuracy, seed standard deviations, and 95% t-intervals.

### A-D modality ablations at nine intervals

Mean absolute position error in pixels, averaged over seeds:

| Model | Vision-only, observed | Audio-only, observed | Combined, observed | Combined, held-out |
|---|---:|---:|---:|---:|
{chr(10).join(ablation_lines)}

Combined input sharply improves seen-pair prediction because the initial visual position and audio command provide complementary information. Held-out recombination remains poor for all four processors; that is a failed compositional-generalization result under this readout and feature design, not evidence for reusable dynamics.

## Sample efficiency

Held-out combination MAE by independent training-set size; the threshold column is the first size with mean MAE ≤ 1 pixel (or not reached):

| Model | Episodes to ≤1 px MAE | MAE by training episodes |
|---|---:|---|
{chr(10).join(sample_lines)}

Each size is fit from scratch using a prefix of a shared, balanced training pool; no fitted parameters or normalization are carried over from larger sizes. The test set is identical across models and sizes within a seed.

## Compute results

Repeated inference uses one warm-up trial followed by {config.timed_repeats} timing passes. `compute_benchmark.csv` separates feature/preprocessing, internal processor, readout, and end-to-end timings. It records median/p95/p99 latency, process RSS, traced allocation, parameters, updates, and approximate operation counts. Environment thread variables were: `OMP_NUM_THREADS={timing_metadata.get('omp_num_threads','unset')}`, `OPENBLAS_NUM_THREADS={timing_metadata.get('openblas_num_threads','unset')}`, `MKL_NUM_THREADS={timing_metadata.get('mkl_num_threads','unset')}`.

| Model | Median episode latency (ms) | p95 episode latency (ms) | CPU seconds over 240 timed predictions | Active updates over timed predictions |
|---|---:|---:|---:|---:|
{chr(10).join(compute_lines)}

RSS is the process peak working set, a cumulative high-water mark, and cannot be attributed to one model from these sequential measurements. Per-run `tracemalloc` peaks are also recorded, but omit native NumPy/BLAS allocations.

- Event-driven D inference CPU time: {d_cpu:.4f}s; recurrent B: {b_cpu:.4f}s, over the same timed trials. These local measurements include Python instrumentation and should not be treated as portable speed rankings.
- B vs C compositional errors: observed {b_seen:.3f} vs {c_seen:.3f} pixels and held-out {b_held:.3f} vs {c_held:.3f} pixels. Statistical reliability should be judged from the seed-level confidence intervals in the CSV; this report does not infer a mechanism from low error.

## Answers and limits

1. **Does recurrence improve over feedforward history?** Compare A and B in the observed/held-out columns and learning curves. Any difference is task-, horizon-, and seed-dependent; the report does not treat the fixed random recurrent state as learning.
2. **Do continuous-time dynamics improve over ordinary recurrence?** Compare B with C at identical inputs and readout selection. C's Euler substeps add work; error differences alone do not establish a benefit.
3. **Does event-driven computation reduce actual CPU time?** Compare D's measured CPU and end-to-end latency with B/C. Update counts and FLOP estimates are supporting diagnostics, not timing substitutes.
4. **Do architectures need fewer examples?** The first ≤1px sample-size threshold and held-out learning curves provide the observed answer; `not reached` means the configured sizes did not meet it.
5. **Are differences reliable?** Five seeds and t-based 95% intervals are reported. Overlapping intervals are not evidence of a reliable separation; no multiple-comparison hypothesis test is claimed.
6. **Could simple relationships explain the results?** Yes. Frequency directly encodes speed, phase encodes direction, and the target is analytically `initial + direction × speed × horizon`. E1 and privileged E2 expose how much performance follows from a simple linear readout or the known kinematics. This benchmark tests composition of sensory factors, not discovery of governing equations.

Plots are descriptive views of these tables. All outputs are generated from the CSV records by the benchmark runner.
'''
    output_path.write_text(report, encoding="utf-8")


def run_all(config=ResearchConfig(), output_dir="results"):
    """Run and write all requested benchmark tables, figures, and report."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    general_raw, general_summary = _generalization(config)
    sample_raw, sample_summary = _sample_efficiency(config)
    compute_rows = _timing_summary(config, config.seed)
    _csv_write(outdir / "generalization_results.csv", general_summary)
    _csv_write(outdir / "sample_efficiency_results.csv", sample_summary)
    _csv_write(outdir / "compute_benchmark.csv", compute_rows)
    _write_plots(general_summary, sample_summary, compute_rows, outdir, config)
    _report(general_summary, sample_summary, compute_rows, config,
            outdir / "EXPERIMENT_REPORT.md")
    return {"generalization": general_summary, "sample_efficiency": sample_summary,
            "compute": compute_rows, "output_dir": str(outdir)}
