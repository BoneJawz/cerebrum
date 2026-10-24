"""Diagnostics for compositional generalization, readouts, and event costs.

This module leaves A-D state equations untouched. All target labels remain
outside Observation and are used only to fit diagnostic probes/readouts.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .benchmark import _ablate
from .contracts import Observation
from .environment import MOTION_START_TICK
from .processors import PROCESSORS, EventDrivenDynamics, sensory_features
from .research_benchmark import (
    MODEL_IDS, MODALITIES, ResearchConfig, _feature_vector, _metrics, _oracle_predictions,
    _target_center, _targets, fit_validate_predict, make_split, summarize,
)

DIAGNOSTIC_MODELS = ("A", "B", "C", "D", "E1", "E3_factorized", "E2_oracle")
DEFAULT_EVENT_THRESHOLDS = (0.0, 0.04, 0.10, 0.25, 0.5, 1.0)
READOUTS = ("quadratic_ridge", "explicit_pairwise_ridge", "mlp32")


@dataclass(frozen=True)
class DiagnosticConfig:
    research: ResearchConfig = ResearchConfig()
    event_thresholds: tuple[float, ...] = DEFAULT_EVENT_THRESHOLDS
    mlp_hidden: int = 32
    mlp_max_epochs: int = 250
    mlp_patience: int = 30
    mlp_weight_decay: float = 1e-4
    pairwise_features_per_modality: int = 6

    def __post_init__(self):
        if len(self.event_thresholds) < 2 or any(x < 0 for x in self.event_thresholds):
            raise ValueError("provide at least two nonnegative event thresholds")
        if tuple(sorted(set(self.event_thresholds))) != self.event_thresholds:
            raise ValueError("event thresholds must be strictly increasing")
        if self.mlp_hidden < 1 or self.mlp_max_epochs < 1 or self.mlp_patience < 1:
            raise ValueError("MLP dimensions and training limits must be positive")


def _csv_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _target_position(item):
    return float(item["initial_position"])


def _audio_vector(item, config, pixels, cue="all"):
    """Average generic acoustic features; optionally isolate phase or spectrum."""
    frames = item["observations"][:config.prediction_tick + 1]
    vectors = []
    for obs in frames:
        zero_vision = np.zeros(pixels, dtype=float)
        acoustic = Observation(obs.t, zero_vision, obs.audio)
        feature = sensory_features(acoustic)[pixels:]
        # Existing feature order: four waveform summaries, sixteen signed
        # samples, then 49 magnitude-spectrum bins.
        if cue == "phase":
            feature = feature[4:20]
        elif cue == "frequency":
            feature = feature[20:]
        vectors.append(feature)
    return np.mean(vectors, axis=0)


def _vision_vector(item, config):
    return np.asarray(item["observations"][0].vision, dtype=float)


def _vision_centroid(item):
    vision = np.asarray(item["observations"][0].vision, dtype=float)
    return float(np.dot(np.arange(len(vision), dtype=float), vision) / np.sum(vision))


def _factor_inputs(examples, config, pixels):
    visual = np.asarray([_vision_vector(item, config) for item in examples])
    phase = np.asarray([_audio_vector(item, config, pixels, "phase") for item in examples])
    frequency = np.asarray([_audio_vector(item, config, pixels, "frequency") for item in examples])
    return visual, phase, frequency


def _fit_factor_probes(config, split, train, seed):
    visual_train, phase_train, frequency_train = _factor_inputs(
        train, config, split.world_config.pixels)
    visual_val, phase_val, frequency_val = _factor_inputs(
        split.validation, config, split.world_config.pixels)
    x0_train = np.asarray([[_target_position(x)] for x in train])
    x0_val = np.asarray([[_target_position(x)] for x in split.validation])
    direction_train = np.asarray([[x["direction"]] for x in train], dtype=float)
    direction_val = np.asarray([[x["direction"]] for x in split.validation], dtype=float)
    speed_train = np.asarray([[x["velocity"]] for x in train], dtype=float)
    speed_val = np.asarray([[x["velocity"]] for x in split.validation], dtype=float)
    def affine_ridge(x_train, y_train, x_val, y_val, x_eval):
        target_mean = np.mean(y_train, axis=0, keepdims=True)
        _, _, _, base = fit_validate_predict(
            x_train, y_train - target_mean, x_val, y_val - target_mean,
            x_eval, config.alphas, "linear")
        return [_AffineKernel(readout, float(target_mean[0, 0])) for readout in base]

    x0_probe = affine_ridge(visual_train, x0_train, visual_val, x0_val, visual_val)
    direction_probe = affine_ridge(phase_train, direction_train, phase_val,
                                   direction_val, phase_val)
    speed_probe = affine_ridge(frequency_train, speed_train, frequency_val,
                               speed_val, frequency_val)
    return x0_probe[0], direction_probe[0], speed_probe[0], (visual_train, phase_train,
                                                              frequency_train)


class _AffineKernel:
    """Adds a train-only target intercept to the normalized linear KRR probe."""
    def __init__(self, base, intercept):
        self.base = base
        self.intercept = intercept

    def predict_one(self, x):
        return self.base.predict_one(x) + self.intercept


def _fit_affine_kernel(x_train, y_train, x_val, y_val, x_eval, alphas, kind):
    """Fit ridge with an unpenalized target intercept using train targets only."""
    target_mean = np.mean(y_train, axis=0, keepdims=True)
    prediction, selected, count, readouts = fit_validate_predict(
        x_train, y_train - target_mean, x_val, y_val - target_mean,
        x_eval, alphas, kind)
    prediction = prediction + target_mean
    wrapped = [_AffineKernel(model, float(target_mean[0, j]))
               for j, model in enumerate(readouts)]
    return prediction, selected, count, wrapped


def _probe_rows(seed, split_name, examples, x0_hat, direction_hat, speed_hat,
                x0_linear=None):
    true_x0 = np.asarray([_target_position(x) for x in examples])
    true_direction = np.asarray([x["direction"] for x in examples])
    true_speed = np.asarray([x["velocity"] for x in examples])
    predicted_direction = np.where(direction_hat >= 0, 1, -1)
    predicted_speed = np.rint(speed_hat).astype(int)
    rows = [
        {"seed": seed, "evaluation_split": split_name, "probe": "initial_position",
         "mae": float(np.mean(np.abs(x0_hat - true_x0))),
         "accuracy": "", "exact_level_accuracy": ""},
        {"seed": seed, "evaluation_split": split_name, "probe": "direction_from_audio",
         "mae": "", "accuracy": float(np.mean(predicted_direction == true_direction)),
         "exact_level_accuracy": float(np.mean(predicted_direction == true_direction))},
        {"seed": seed, "evaluation_split": split_name, "probe": "speed_from_audio",
         "mae": float(np.mean(np.abs(speed_hat - true_speed))),
         "accuracy": float(np.mean(predicted_speed == true_speed)),
         "exact_level_accuracy": float(np.mean(predicted_speed == true_speed))},
    ]
    if x0_linear is not None:
        rows.append({"seed": seed, "evaluation_split": split_name,
                     "probe": "initial_position_linear", "mae": float(np.mean(np.abs(
                         x0_linear - true_x0))), "accuracy": "", "exact_level_accuracy": ""})
    return rows


def _internal_factor_rows(seed, model, examples, predictions, config):
    rows = []
    true_x0 = np.asarray([item["initial_position"] for item in examples])
    true_direction = np.asarray([item["direction"] for item in examples])
    true_speed = np.asarray([item["velocity"] for item in examples])
    x0_hat, direction_score, speed_hat = (predictions[:, i] for i in range(3))
    direction_hat = np.where(direction_score >= 0, 1, -1)
    speed_class = np.rint(speed_hat).astype(int)
    for split_name, mask in (
        ("observed", np.asarray([x["combination"] not in set(config.heldout_combinations)
                                 for x in examples])),
        ("heldout", np.asarray([x["combination"] in set(config.heldout_combinations)
                                for x in examples])),
    ):
        if not np.any(mask):
            continue
        rows.extend([
            {"seed": seed, "model": model, "evaluation_split": split_name,
             "factor": "initial_position", "mae": float(np.mean(np.abs(x0_hat[mask] - true_x0[mask]))),
             "accuracy": "", "exact_level_accuracy": ""},
            {"seed": seed, "model": model, "evaluation_split": split_name,
             "factor": "direction", "mae": "",
             "accuracy": float(np.mean(direction_hat[mask] == true_direction[mask])),
             "exact_level_accuracy": float(np.mean(direction_hat[mask] == true_direction[mask]))},
            {"seed": seed, "model": model, "evaluation_split": split_name,
             "factor": "speed", "mae": float(np.mean(np.abs(speed_hat[mask] - true_speed[mask]))),
             "accuracy": float(np.mean(speed_class[mask] == true_speed[mask])),
             "exact_level_accuracy": float(np.mean(speed_class[mask] == true_speed[mask]))},
        ])
    return rows


def _passthrough_factor_rows(seed, model, examples, x, direction_score,
                             speed_score, pixels, config):
    """Probe the raw sensory suffix that each A-D feature output preserves."""
    sensory_width = pixels + 4 + 16 + 49
    raw = x[:, -sensory_width:]
    vision = raw[:, :pixels]
    phase = raw[:, pixels + 4:pixels + 20]
    spectrum = raw[:, pixels + 20:]
    centers = np.arange(pixels, dtype=float)
    x0 = (vision @ centers) / np.sum(vision, axis=1)
    true_direction = np.asarray([item["direction"] for item in examples])
    true_speed = np.asarray([item["velocity"] for item in examples])
    speed_class = np.rint(speed_score).astype(int)
    direction_class = np.where(direction_score >= 0, 1, -1)
    rows = []
    for split_name, mask in (
        ("observed", np.asarray([x["combination"] not in set(config.heldout_combinations)
                                 for x in examples])),
        ("heldout", np.asarray([x["combination"] in set(config.heldout_combinations)
                                for x in examples])),
    ):
        if not np.any(mask):
            continue
        rows.extend([
            {"seed": seed, "model": model, "evaluation_split": split_name,
             "factor": "passthrough_position", "mae": float(np.mean(np.abs(
                 x0[mask] - np.asarray([item["initial_position"] for item in examples])[mask]))),
             "accuracy": "", "exact_level_accuracy": ""},
            {"seed": seed, "model": model, "evaluation_split": split_name,
             "factor": "passthrough_direction", "mae": "",
             "accuracy": float(np.mean(direction_class[mask] == true_direction[mask])),
             "exact_level_accuracy": float(np.mean(direction_class[mask] == true_direction[mask]))},
            {"seed": seed, "model": model, "evaluation_split": split_name,
             "factor": "passthrough_speed", "mae": float(np.mean(np.abs(
                 speed_score[mask] - true_speed[mask]))),
             "accuracy": float(np.mean(speed_class[mask] == true_speed[mask])),
             "exact_level_accuracy": float(np.mean(speed_class[mask] == true_speed[mask]))},
        ])
    return rows


def _predict_factorized(probes, examples, config, split, seed):
    x0_probe, direction_probe, speed_probe, _ = probes
    visual, phase, frequency = _factor_inputs(examples, config, split.world_config.pixels)
    # The visual centroid is a direct, label-free estimate from the allowed
    # binary array. The learned linear probe is retained as a diagnostic.
    x0_hat = np.asarray([_vision_centroid(item) for item in examples])
    direction_score = np.asarray([direction_probe.predict_one(x) for x in phase])
    direction_hat = np.where(direction_score >= 0, 1, -1)
    speed_score = np.asarray([speed_probe.predict_one(x) for x in frequency])
    speeds = np.asarray(config.velocities)
    speed_hat = speeds[np.argmin(np.abs(speed_score[:, None] - speeds[None, :]), axis=1)]
    return x0_hat, direction_hat, speed_hat


def _run_processor(model_id, item, config, seed, pixels, event_threshold=None,
                   instrument_costs=False):
    if model_id == "D":
        processor = EventDrivenDynamics(
            pixels=pixels, hidden=config.hidden, seed=seed,
            threshold=0.04 if event_threshold is None else event_threshold,
            instrument_costs=instrument_costs,
        )
    else:
        processor = PROCESSORS[model_id](pixels=pixels, hidden=config.hidden, seed=seed)
    feature = None
    for obs in item["observations"][:config.prediction_tick + 1]:
        feature = processor.step(_ablate(obs, "combined"))
    return processor, feature


def _matrix_for(model_id, examples, config, seed, pixels, threshold=None):
    rows = []
    for item in examples:
        _, feature = _run_processor(model_id, item, config, seed, pixels, threshold)
        rows.append(feature)
    return np.asarray(rows)


def _groups(split, test):
    held = set()
    # The caller passes heldout and observed examples in split order. A direct
    # pair check avoids coupling the generated test order to this helper.
    return (("observed", split.observed_test), ("heldout", split.heldout_test))


def _add_prediction_rows(rows, seed, model, examples, predictions, config):
    for h_index, horizon in enumerate(config.horizons):
        groups = (
            ("observed", [item for item in examples if item["combination"] not in set(config.heldout_combinations)]),
            ("heldout", [item for item in examples if item["combination"] in set(config.heldout_combinations)]),
        )
        for split_name, group in groups:
            if not group:
                continue
            group_pairs = {item["combination"] for item in group}
            indices = [i for i, item in enumerate(examples)
                       if item["combination"] in group_pairs]
            pair_rows = []
            for pair in sorted(group_pairs):
                pair_indices = [i for i in indices if examples[i]["combination"] == pair]
                truth = np.asarray([examples[i]["targets"][horizon] for i in pair_indices])
                pred = predictions[pair_indices, h_index]
                error = pred - truth
                row = {
                    "seed": seed, "model": model, "evaluation_split": split_name,
                    "direction": pair[0], "velocity": pair[1],
                    "horizon_intervals": horizon,
                    "target_frame": MOTION_START_TICK + horizon,
                    "mae_pixels": float(np.mean(np.abs(error))),
                    "signed_bias_pixels": float(np.mean(error)),
                    "within_1px_accuracy": float(np.mean(np.abs(error) <= 1.0)),
                    "exact_position_accuracy": float(np.mean(np.rint(pred) == truth)),
                    "n": len(pair_indices),
                }
                rows.append(row)
                pair_rows.append(row)
            truth = np.asarray([examples[i]["targets"][horizon] for i in indices])
            pred = predictions[indices, h_index]
            error = pred - truth
            rows.append({
                "seed": seed, "model": model, "evaluation_split": split_name,
                "direction": "all", "velocity": "all",
                "horizon_intervals": horizon,
                "target_frame": MOTION_START_TICK + horizon,
                "mae_pixels": float(np.mean(np.abs(error))),
                "signed_bias_pixels": float(np.mean(error)),
                "within_1px_accuracy": float(np.mean(np.abs(error) <= 1.0)),
                "exact_position_accuracy": float(np.mean(np.rint(pred) == truth)),
                "n": len(indices),
            })


def _summary_with_ci(rows, keys, metrics):
    cleaned = []
    for row in rows:
        item = dict(row)
        for metric in metrics:
            if item.get(metric) == "":
                item[metric] = None
        cleaned.append(item)
    return summarize(cleaned, keys, metrics)


def _explicit_pairwise(x_train, x_other, pixels, model_id, blocks=6):
    """Append train-variance-selected vision x audio products.

    Only original sensory coordinates are paired; all A-D internal features
    and their state equations remain unchanged. A uses each of its four
    history blocks, while B-D use the sensory suffix of their output vector.
    The 6x6 cap keeps the explicit expansion small on a CPU.
    """
    sensory_width = pixels + 4 + 16 + 49
    if model_id == "A":
        block_count = 4
        starts = [b * sensory_width for b in range(block_count)]
    else:
        block_count = 1
        starts = [x_train.shape[1] - sensory_width]
    pair_indices = []
    for start in starts:
        visual = np.arange(start, start + pixels)
        audio = np.arange(start + pixels, start + sensory_width)
        vi = visual[np.argsort(np.var(x_train[:, visual], axis=0))[-blocks:]]
        ai = audio[np.argsort(np.var(x_train[:, audio], axis=0))[-blocks:]]
        pair_indices.extend((int(v), int(a)) for v in vi for a in ai)

    def transform(x):
        interactions = np.column_stack([x[:, i] * x[:, j] for i, j in pair_indices])
        return np.concatenate([x, interactions], axis=1)
    return transform(x_train), transform(x_other), len(pair_indices)


def _mlp_fit_predict(x_train, y_train, x_val, y_val, x_eval, config, seed):
    """Small tanh MLP with Adam and validation-only early stopping."""
    start_time = time.perf_counter()
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-8] = 1.0
    z_train = (x_train - mean) / scale
    z_val = (x_val - mean) / scale
    z_eval = (x_eval - mean) / scale
    y_mean = np.mean(y_train, axis=0)
    y_scale = np.std(y_train, axis=0)
    y_scale[y_scale < 1e-8] = 1.0
    yt, yv = (y_train - y_mean) / y_scale, (y_val - y_mean) / y_scale
    rng = np.random.default_rng(seed)
    n, d = z_train.shape
    hidden = config.mlp_hidden
    params = {
        "w1": rng.normal(0, 1 / math.sqrt(d), (d, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, 1 / math.sqrt(hidden), (hidden, y_train.shape[1])),
        "b2": np.zeros(y_train.shape[1]),
    }
    moments = {k: np.zeros_like(v) for k, v in params.items()}
    velocities = {k: np.zeros_like(v) for k, v in params.items()}
    best = None
    best_loss = np.inf
    best_epoch = 0
    stale = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, config.mlp_max_epochs + 1):
        h = np.tanh(z_train @ params["w1"] + params["b1"])
        pred = h @ params["w2"] + params["b2"]
        delta = (2.0 / n) * (pred - yt)
        grads = {
            "w2": h.T @ delta + 2 * config.mlp_weight_decay * params["w2"],
            "b2": np.sum(delta, axis=0),
        }
        dh = (delta @ params["w2"].T) * (1.0 - h * h)
        grads["w1"] = z_train.T @ dh + 2 * config.mlp_weight_decay * params["w1"]
        grads["b1"] = np.sum(dh, axis=0)
        for key, grad in grads.items():
            moments[key] = beta1 * moments[key] + (1 - beta1) * grad
            velocities[key] = beta2 * velocities[key] + (1 - beta2) * grad * grad
            mhat = moments[key] / (1 - beta1 ** epoch)
            vhat = velocities[key] / (1 - beta2 ** epoch)
            params[key] -= 0.01 * mhat / (np.sqrt(vhat) + eps)
        val_h = np.tanh(z_val @ params["w1"] + params["b1"])
        val_pred = val_h @ params["w2"] + params["b2"]
        loss = float(np.mean(np.abs(val_pred * y_scale + y_mean - y_val * y_scale - y_mean)))
        if loss < best_loss - 1e-8:
            best_loss = loss
            best = {k: value.copy() for k, value in params.items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.mlp_patience:
                break
    train_wall = time.perf_counter() - start_time
    h_eval = np.tanh(z_eval @ best["w1"] + best["b1"])
    predictions = (h_eval @ best["w2"] + best["b2"]) * y_scale + y_mean
    parameter_count = sum(value.size for value in best.values())
    return predictions, parameter_count, train_wall, best_epoch, (mean, scale, best, y_mean, y_scale)


def _mlp_predict_one(model, x):
    mean, scale, params, y_mean, y_scale = model
    z = (x - mean) / scale
    return (np.tanh(z @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]) * y_scale + y_mean


def _readout_rows(rows, seed, model, readout, examples, predictions, horizon_list,
                  train_wall, params, inference_ms, alpha, config,
                  pairwise_count=0, mlp_epoch=""):
    for j, h in enumerate(horizon_list):
        for split_name in ("observed", "heldout"):
            subset = [(i, item) for i, item in enumerate(examples)
                      if (item["combination"] in set(config.heldout_combinations))
                      == (split_name == "heldout")]
            if not subset:
                continue
            ix = [i for i, _ in subset]
            truth = np.asarray([item["targets"][h] for _, item in subset])
            pred = predictions[ix, j]
            error = pred - truth
            rows.append({
                "seed": seed, "model": model, "readout": readout,
                "evaluation_split": split_name, "horizon_intervals": h,
                "mae_pixels": float(np.mean(np.abs(error))),
                "signed_bias_pixels": float(np.mean(error)),
                "within_1px_accuracy": float(np.mean(np.abs(error) <= 1)),
                "trainable_parameter_count": int(params),
                "readout_training_wall_s": float(train_wall),
                "inference_latency_ms_per_episode": float(inference_ms),
                "selected_alpha": alpha,
                "explicit_interaction_count": pairwise_count,
                "mlp_selected_epoch": mlp_epoch,
            })


def _evaluate_event_processor(examples, fitted, config, seed, pixels, threshold,
                              instrument=True):
    cpu0, wall0 = time.process_time(), time.perf_counter()
    per_episode_ms = []
    predictions = []
    opportunities = triggered = skipped = check_wall_ns = update_wall_ns = 0
    for item in examples:
        episode_start = time.perf_counter_ns()
        processor, feature = _run_processor(
            "D", item, config, seed, pixels, threshold, instrument_costs=instrument)
        prediction = fitted.predict_one(feature)
        per_episode_ms.append((time.perf_counter_ns() - episode_start) / 1e6)
        predictions.append(prediction)
        opportunities += processor.update_opportunities
        triggered += processor.updates
        skipped += processor.skipped_updates
        check_wall_ns += processor.trigger_check_wall_ns
        update_wall_ns += processor.state_update_wall_ns
    return (np.asarray(predictions), {
        "available_update_opportunities": opportunities,
        "triggered_updates": triggered,
        "skipped_updates": skipped,
        "fraction_opportunities_skipped": skipped / max(1, opportunities),
        "trigger_check_wall_s": check_wall_ns / 1e9,
        "state_update_wall_s": update_wall_ns / 1e9,
        "inference_cpu_s": time.process_time() - cpu0,
        "inference_wall_s": time.perf_counter() - wall0,
        "median_episode_latency_ms": float(np.median(per_episode_ms)),
        "p95_episode_latency_ms": float(np.percentile(per_episode_ms, 95)),
        "episode_latencies_ms": per_episode_ms,
    })


def _evaluate_fixed_processor(model_id, examples, fitted, config, seed, pixels):
    cpu0, wall0 = time.process_time(), time.perf_counter()
    predictions, updates = [], 0
    latencies = []
    for item in examples:
        episode_start = time.perf_counter_ns()
        processor, feature = _run_processor(model_id, item, config, seed, pixels)
        predictions.append(fitted.predict_one(feature))
        updates += processor.updates
        latencies.append((time.perf_counter_ns() - episode_start) / 1e6)
    return np.asarray(predictions), {
        "available_update_opportunities": "",
        "triggered_updates": updates,
        "skipped_updates": "",
        "fraction_opportunities_skipped": "",
        "trigger_check_wall_s": 0.0, "state_update_wall_s": 0.0,
        "inference_cpu_s": time.process_time() - cpu0,
        "inference_wall_s": time.perf_counter() - wall0,
        "median_episode_latency_ms": float(np.median(latencies)),
        "p95_episode_latency_ms": float(np.percentile(latencies, 95)),
        "episode_latencies_ms": latencies,
    }


def _repeat_inference(evaluator, repeats):
    """Repeat a fixed inference pass to reduce coarse process-CPU timer noise."""
    cpu0, wall0 = time.process_time(), time.perf_counter()
    latencies = []
    prediction = None
    for _ in range(repeats):
        prediction, cost = evaluator()
        latencies.extend(cost["episode_latencies_ms"])
    total_cpu = time.process_time() - cpu0
    total_wall = time.perf_counter() - wall0
    cost["inference_cpu_s"] = total_cpu / repeats
    cost["inference_wall_s"] = total_wall / repeats
    cost["median_episode_latency_ms"] = float(np.median(latencies))
    cost["p95_episode_latency_ms"] = float(np.percentile(latencies, 95))
    cost.pop("episode_latencies_ms", None)
    return prediction, cost


def _summarize_records(rows, keys, metrics):
    result = []
    for row in _summary_with_ci(rows, keys, metrics):
        result.append({"record_type": "seed_summary", **row})
    result.extend({"record_type": "seed", **row} for row in rows)
    return result


def _paired_differences(rows, left, right, split_name, horizon):
    by_seed = {}
    for row in rows:
        if (row.get("model") in (left, right) and row.get("evaluation_split") == split_name
                and row.get("direction") == "all"
                and int(row.get("horizon_intervals", -1)) == horizon):
            by_seed.setdefault(int(row["seed"]), {})[row["model"]] = float(row["mae_pixels"])
    diffs = [{"seed": seed, "comparison": f"{left}_minus_{right}",
              "evaluation_split": split_name, "horizon_intervals": horizon,
              "mae_difference_pixels": values[left] - values[right]}
             for seed, values in by_seed.items() if left in values and right in values]
    summary = summarize(diffs, ("comparison", "evaluation_split", "horizon_intervals"),
                        ("mae_difference_pixels",))
    return diffs + [{"record_type": "paired_summary", **row} for row in summary]


def _plot_diagnostics(outdir, prediction_cache, diagnostic_summary, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = 9 if 9 in config.research.horizons else max(config.research.horizons)
    models = ("A", "B", "C", "D", "E1", "E3_factorized", "E2_oracle")
    pairs = [(d, v) for d in (-1, 1) for v in config.research.velocities]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    palette = dict(zip(models, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                "#8c564b", "#9467bd", "#222222")))
    for ax, pair in zip(axes.flat, pairs):
        matching = [r for r in prediction_cache if r["pair"] == pair and r["horizon"] == h]
        actual_all = [x for row in matching for x in row["truth"]]
        lo, hi = min(actual_all), max(actual_all)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal")
        for model in models:
            selected = [row for row in matching if row["model"] == model]
            if not selected:
                continue
            x = np.concatenate([row["truth"] for row in selected])
            y = np.concatenate([row["prediction"] for row in selected])
            ax.scatter(x, y, s=8, alpha=.35, color=palette[model], label=model)
        d, v = pair
        ax.set_title(f"{'Left' if d < 0 else 'Right'}, speed {v}"
                     f"{' (held out)' if pair in set(config.research.heldout_combinations) else ''}")
        ax.set_xlabel("Actual final position (pixels)")
        ax.set_ylabel("Predicted final position (pixels)")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Predicted versus actual position at horizon {h}")
    fig.savefig(outdir / "prediction_vs_actual.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for ax, split_name in zip(axes, ("observed", "heldout")):
        for model in ("A", "B", "C", "D", "E1", "E3_factorized"):
            values = []
            for horizon in config.research.horizons:
                seed_rows = [r for r in diagnostic_summary
                             if r.get("record_type") == "seed_summary"
                             and r.get("model") == model
                             and r.get("evaluation_split") == split_name
                             and r.get("direction") == "all"
                             and int(r.get("horizon_intervals", -1)) == horizon]
                values.append((horizon, float(seed_rows[0]["mae_pixels_mean"])) if seed_rows else None)
            points = [v for v in values if v]
            if points:
                ax.plot([v[0] for v in points], [v[1] for v in points], marker="o", label=model)
        ax.set_title(f"{split_name.capitalize()} pairs")
        ax.set_xlabel("Prediction horizon (movement intervals)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean absolute position error (pixels)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Position error by prediction horizon")
    fig.savefig(outdir / "horizon_error.png", dpi=160)
    plt.close(fig)

    # Paired, identical-test comparison of event threshold and compute cost.
    event_path = outdir / "event_threshold_results.csv"
    with event_path.open(newline="", encoding="utf-8-sig") as f:
        event_rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model, color, marker in (("B", "#ff7f0e", "s"), ("C", "#2ca02c", "^") ):
        selected = [r for r in event_rows if r["model"] == model
                    and r["evaluation_split"] == "heldout" and r.get("median_episode_latency_ms")]
        if selected:
            x = [float(r["median_episode_latency_ms"]) for r in selected]
            y = [float(r["mae_pixels"]) for r in selected]
            ax.scatter(x, y, color=color, marker=marker, label=model, s=55)
    selected = [r for r in event_rows if r["model"] == "D"
                and r["evaluation_split"] == "heldout" and r.get("median_episode_latency_ms")]
    selected.sort(key=lambda r: float(r["event_threshold"]))
    if selected:
        x = [float(r["median_episode_latency_ms"]) for r in selected]
        y = [float(r["mae_pixels"]) for r in selected]
        ax.plot(x, y, color="#d62728", marker="o", label="D threshold sweep")
        for row in selected:
            ax.annotate(row["event_threshold"],
                        (float(row["median_episode_latency_ms"]), float(row["mae_pixels"])),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Median end-to-end inference latency (ms per episode)")
    ax.set_ylabel("Held-out MAE (pixels)")
    ax.set_title("Event threshold: accuracy versus measured compute")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(outdir / "event_accuracy_cost.png", dpi=160)
    plt.close(fig)


def _write_report(path, config, diagnostic_summary, factor_summary,
                  internal_factor_summary, readout_summary, event_rows, paired_rows):
    h = 9 if 9 in config.research.horizons else max(config.research.horizons)

    def diag_line(model, split):
        row = next((r for r in diagnostic_summary
                    if r.get("record_type") == "seed_summary" and r.get("model") == model
                    and r.get("evaluation_split") == split and r.get("direction") == "all"
                    and int(r.get("horizon_intervals", -1)) == h), None)
        return "NA" if row is None else f"{float(row['mae_pixels_mean']):.2f}"

    model_lines = "\n".join(
        f"| {m} | {diag_line(m, 'observed')} | {diag_line(m, 'heldout')} |"
        for m in ("A", "B", "C", "D", "E1", "E3_factorized", "E2_oracle"))
    pair_lines = []
    for model in ("A", "B", "C", "D"):
        values = []
        for direction, velocity in ((-1, 2), (1, 3)):
            row = next((r for r in diagnostic_summary
                        if r.get("record_type") == "seed_summary" and r.get("model") == model
                        and r.get("evaluation_split") == "heldout"
                        and str(r.get("direction")) == str(direction)
                        and str(r.get("velocity")) == str(velocity)
                        and int(r.get("horizon_intervals", -1)) == h), None)
            if row:
                values.extend((f"{float(row['mae_pixels_mean']):.2f}",
                               f"{float(row['signed_bias_pixels_mean']):+.2f}"))
            else:
                values.extend(("NA", "NA"))
        pair_lines.append(f"| {model} | {values[0]} / {values[1]} | {values[2]} / {values[3]} |")
    horizon_lines = []
    for model in ("A", "B", "C", "D"):
        cells = []
        for horizon in config.research.horizons:
            row = next((r for r in diagnostic_summary
                        if r.get("record_type") == "seed_summary" and r.get("model") == model
                        and r.get("evaluation_split") == "heldout" and r.get("direction") == "all"
                        and int(r.get("horizon_intervals", -1)) == horizon), None)
            cells.append("NA" if row is None else f"{float(row['mae_pixels_mean']):.2f}")
        horizon_lines.append(f"| {model} | " + " | ".join(cells) + " |")
    factor_lines = []
    for split in ("observed", "heldout"):
        cells = []
        for name in ("initial_position", "direction_from_audio", "speed_from_audio"):
            row = next((r for r in factor_summary
                        if r.get("record_type") == "seed_summary" and r.get("evaluation_split") == split
                        and r.get("probe") == name), None)
            if not row:
                cells.append("NA")
            elif name == "initial_position":
                cells.append(f"{float(row['mae_mean']):.2f} px MAE")
            elif name == "direction_from_audio":
                cells.append(f"{float(row['accuracy_mean']):.1%} accuracy")
            else:
                cells.append(f"{float(row['mae_mean']):.2f} speed MAE / "
                              f"{float(row['exact_level_accuracy_mean']):.1%} exact")
        factor_lines.append(f"| {split} | " + " | ".join(cells) + " |")
    internal_lines = []
    for model in ("A", "B", "C", "D"):
        cells = []
        for factor in ("initial_position", "direction", "speed"):
            row = next((r for r in internal_factor_summary
                        if r.get("record_type") == "seed_summary" and r.get("model") == model
                        and r.get("evaluation_split") == "heldout" and r.get("factor") == factor), None)
            if not row:
                cells.append("NA")
            elif factor == "initial_position":
                cells.append(f"{float(row['mae_mean']):.2f} px MAE")
            elif factor == "direction":
                cells.append(f"{float(row['accuracy_mean']):.1%} accuracy")
            else:
                cells.append(f"{float(row['mae_mean']):.2f} MAE / "
                              f"{float(row['exact_level_accuracy_mean']):.1%} exact")
        internal_lines.append(f"| {model} | " + " | ".join(cells) + " |")
    best_mlp = next((r for r in readout_summary
                     if r.get("record_type") == "seed_summary" and r.get("model") == "B"
                     and r.get("readout") == "mlp32" and r.get("evaluation_split") == "heldout"
                     and int(r.get("horizon_intervals", -1)) == h), None)
    readout_lines = []
    for model in ("A", "B", "C", "D"):
        for readout in READOUTS:
            row = next((r for r in readout_summary
                        if r.get("record_type") == "seed_summary" and r.get("model") == model
                        and r.get("readout") == readout and r.get("evaluation_split") == "heldout"
                        and int(r.get("horizon_intervals", -1)) == h), None)
            if row:
                readout_lines.append(
                    f"| {model} | {readout} | {float(row['mae_pixels_mean']):.2f} | "
                    f"{float(row['trainable_parameter_count_mean']):.0f} | "
                    f"{float(row['readout_training_wall_s_mean'])*1000:.1f} | "
                    f"{float(row['inference_latency_ms_per_episode_mean']):.4f} |"
                )
    threshold_reference = (0.04 if 0.04 in config.event_thresholds
                           else min(config.event_thresholds, key=lambda x: abs(x - 0.04)))
    event_default = [r for r in event_rows if r["model"] == "D"
                     and abs(float(r["event_threshold"]) - threshold_reference) < 1e-10
                     and r["evaluation_split"] == "heldout"]
    event_default_cpu = (float(np.mean([float(r["inference_cpu_s"]) for r in event_default]))
                         if event_default else float("nan"))
    event_default_skip = (float(np.mean([float(r["fraction_opportunities_skipped"])
                                         for r in event_default]))
                          if event_default else float("nan"))
    event_lines = []
    for model, threshold in ([("B", "baseline"), ("C", "baseline")]
                             + [("D", str(t)) for t in config.event_thresholds]):
        selected = [r for r in event_rows if r["model"] == model
                    and str(r["event_threshold"]) == threshold
                    and r["evaluation_split"] == "heldout"]
        if not selected:
            continue
        mae = float(np.mean([float(r["mae_pixels"]) for r in selected]))
        cpu = float(np.mean([float(r["inference_cpu_s"]) for r in selected]))
        skip = ("—" if model != "D" else f"{np.mean([float(r['fraction_opportunities_skipped']) for r in selected]):.1%}")
        check = float(np.mean([float(r["trigger_check_wall_s"]) for r in selected]))
        update = float(np.mean([float(r["state_update_wall_s"]) for r in selected]))
        event_lines.append(f"| {model} | {threshold} | {mae:.2f} | {skip} | "
                           f"{cpu:.4f} | {check*1000:.2f} | {update*1000:.2f} |")
    passthrough_map = {r.get("factor"): r for r in internal_factor_summary
                       if r.get("record_type") == "seed_summary" and r.get("model") == "A"
                       and r.get("evaluation_split") == "heldout"
                       and str(r.get("factor", "")).startswith("passthrough_")}
    report = f"""# Compositional Generalization Diagnostic Report

## Scope and protocol

This diagnostic retains A-D processor equations and uses the same deterministic training, validation, observed-test, and held-out-test construction as the existing benchmark. Seeds: {config.research.seeds}, starting at {config.research.seed}, spaced by 1009. Training and validation exclude {config.research.heldout_combinations}. All estimators receive only the permitted observations through tick {config.research.prediction_tick}; target labels are stored separately. Audio and vision are synchronized at 20 visual ticks/s, with 100 raw audio samples (2 kHz) per visual tick.

### Pipeline audit

- Every A-D input is an `Observation(t, vision, audio)` for ticks 0 through 4. Vision is a binary one-dimensional pixel vector; each audio chunk is 100 signed waveform samples. Ground-truth direction, speed, initial position, and future frames are stored outside `Observation`.
- `sensory_features` appends four audio statistics (mean, RMS, zero-crossing rate, mean absolute amplitude), 16 signed samples at evenly spaced indices, and 49 absolute rFFT bins. The 100-sample chunk is zero-padded to 128 for the FFT; there is no window. Magnitudes discard phase, while the signed samples retain its polarity. Readouts standardize processor feature matrices using training-set means/scales only.
- A concatenates its most recent four sensory feature vectors. B, C, and D return their current sensory vector concatenated with fixed internal state. B has one recurrent update per input tick; C has two Euler substeps; D checks input-feature change and conditionally applies one state update. All processor weights are seeded and fixed; only the output kernel-ridge readout is fitted.
- The primary readout is a degree-two polynomial kernel ridge model, which already includes implicit pairwise feature interactions. It emits four outputs trained for the four horizons; horizon is not a model input. E1 uses a linear kernel without an explicit target intercept, a limitation for absolute-position predictions. E3 is separately factorized as described below.
- Directions and speeds are cycle-balanced over permitted command pairs; each pair is balanced to within one training episode. Start position is sampled uniformly from the central third of the field, object width from 1/3/5 pixels, and waveform noise independently per episode/tick. Train, validation, observed test, and held-out test use distinct deterministic episode seeds. The only intended distribution difference is the held-out direction-speed pairing; all factor values are represented in training. There is no motion visible in the inputs.
- The observed pairs are left/speed-1, left/speed-3, right/speed-1, and right/speed-2. Thus speed 2 occurs only with rightward motion and speed 3 only with leftward motion in training/validation; the held-out pairs reverse those correlations. This is an intentional compositional split, but it creates a strong cue-correlation shortcut that can explain held-out direction reversals.
- Visual and audio observations share event timestamp `k/20` seconds. Each audio chunk spans the matching 50 ms interval. Processors receive synchronized chunks, not independently timed events.

No label leakage was found in the model inputs or split construction. The probes below test whether factor cues survive feature extraction and whether the models can compose them.

The currently generated cue pairs phase with direction and carrier frequency with speed. The phase-to-direction map is randomized per seed but shared between train and test for that seed. Signed waveform samples preserve phase polarity; magnitude FFT bins preserve frequency magnitude. These are explicit cue factors, not discovered semantics.

## Prediction results by horizon and pair

At horizon {h}, observed / held-out mean MAE (pixels) across seeds:

| Model | Observed MAE | Held-out MAE |
|---|---:|---:|
{model_lines}

`diagnostic_results.csv` includes each direction-speed pair and horizon with MAE, signed bias, within-1-pixel accuracy, and seed-level uncertainty summaries. `prediction_vs_actual.png` shows per-combination scatter; `horizon_error.png` shows horizon scaling. `error_decomposition.csv` fits each trajectory's forecast residual to an intercept plus horizon slope. The intercept is position-like and the slope is displacement-like only as a descriptive decomposition, not a unique causal explanation.

Held-out direction/speed errors at horizon {h}:

| Model | Left at speed 2: MAE / bias (px) | Right at speed 3: MAE / bias (px) |
|---|---:|---:|
{chr(10).join(pair_lines)}

Held-out MAE by horizon (pixels):

| Model | """ + " | ".join(str(v) for v in config.research.horizons) + " |\n|---|" + "---:|" * len(config.research.horizons) + "\n" + "\n".join(horizon_lines) + f"""

The large positive error bias on left-at-speed-2 and negative bias on right-at-speed-3 indicate systematic direction reversals on both held-out pairs. Error rises close to linearly with horizon there while observed-pair MAE stays nearly flat, which is consistent with a displacement/rate error rather than a large initial-position offset. The error-slope/intercept decomposition remains descriptive and cannot uniquely identify the cause.

## Factor probes and E3

| Split | Initial position | Direction from audio | Speed from audio |
|---|---|---|---|
{chr(10).join(factor_lines)}

The factor probes use training labels only as supervised targets and the permitted sensory features as inputs. E3 independently estimates position from vision and direction/speed from audio, quantizes speed to the known allowed levels, then applies the known constant-velocity rule. E3 is a deliberately favorable, physics-structured diagnostic baseline, not an architecture-matched model. E2 remains a privileged ground-truth oracle.

The direct linear visual probe is also reported separately; its held-out MAE is {next((f"{float(r['mae_mean']):.2f} px" for r in factor_summary if r.get('record_type') == 'seed_summary' and r.get('evaluation_split') == 'heldout' and r.get('probe') == 'initial_position_linear'), 'NA')}. The centroid is exact here because stimuli are unclipped symmetric bars; this makes the position component intentionally easy.

Probes trained directly on A-D outputs:

| Model features | Position MAE | Direction accuracy | Speed MAE / exact accuracy |
|---|---:|---:|---:|
{chr(10).join(internal_lines)}

The raw sensory suffix is preserved in every processor output. Using only that suffix, the held-out direction probe reached {float(passthrough_map['passthrough_direction']['accuracy_mean']):.1%}, the speed probe reached {float(passthrough_map['passthrough_speed']['exact_level_accuracy_mean']):.1%} exact accuracy, and the visual centroid had {float(passthrough_map['passthrough_position']['mae_mean']):.2f} px MAE. The all-feature quadratic probes above are weaker; that suggests readout/feature competition in the fixed representations, while the exact sensory probes and E3 establish the factors are not absent.

## Readout capacity

`readout_diagnostics.csv` compares the original quadratic kernel ridge, an explicit compact vision-by-audio pairwise feature expansion with a linear ridge readout, and a 32-unit tanh MLP. The expansion selects six high-variance visual and six high-variance audio coordinates per sensory history block using training features only. The MLP uses L2 weight decay {config.mlp_weight_decay:g}, at most {config.mlp_max_epochs} epochs, and validation-only early stopping (patience {config.mlp_patience}). Feature extraction and A-D dynamics are unchanged. B MLP held-out result at horizon {h}: {('NA' if best_mlp is None else f"{float(best_mlp['mae_pixels_mean']):.2f} px MAE")}. Parameter counts, readout training time, and inference latency are in the CSV.

| Processor | Readout | Held-out MAE (px) | Fitted coefficients/weights | Fit (ms) | Inference (ms/episode) |
|---|---|---:|---:|---:|---:|
{chr(10).join(readout_lines)}

## Event-driven cost audit

At D threshold {threshold_reference:g}, mean skip fraction on the held-out set was {event_default_skip:.1%}; uninstrumented end-to-end process CPU time averaged {event_default_cpu:.4f}s per pass over {config.research.timed_repeats} repeated passes. `event_threshold_results.csv` reports opportunities, triggered/skipped updates, separate high-resolution wall timings for trigger checks and state updates, total uninstrumented process CPU/wall time per pass, latency, and errors for thresholds {config.event_thresholds}; B and C provide matched inference-cost references on the same held-out episodes. On Windows the process CPU timer is too coarse to resolve individual short checks, so those subcomponent durations use a separate instrumented monotonic-wall-clock pass and should be read as timing proxies. `event_accuracy_cost.png` plots held-out error against median end-to-end latency. Fewer calls alone do not establish a speed benefit because trigger detection itself costs compute.

| Model | Threshold | Held-out MAE (px) | Skipped opportunities | CPU s/pass ({config.research.timed_repeats} passes) | Trigger check (ms, wall) | State update (ms, wall) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(event_lines)}

## Controls and limits

The same test episodes are reused across architectures, readouts, horizons, and event thresholds within each seed. Alpha selection and MLP early stopping use only observed validation combinations. Paired seed differences and 95% t intervals are recorded separately. Threshold and readout sweeps are diagnostic comparisons; the held-out scores are reported together and were not used for tuning.

The attachment supplied for this task ended mid-list at the deliverable bullet `factor...`; outputs here cover the requested diagnostic tasks 1-8 and the artifact names that can be inferred from the text. No result implies reusable dynamics or biological validity.
"""
    path.write_text(report, encoding="utf-8")


def run_diagnostics(config=DiagnosticConfig(), output_dir="results/diagnostics"):
    """Run per-combination, factor-probe, readout, and event-cost diagnostics."""
    research = config.research
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    diagnostic_rows, factor_rows = [], []
    internal_factor_rows = []
    passthrough_factor_rows = []
    readout_rows, event_rows, decomposition_rows = [], [], []
    prediction_cache = []
    paired_rows = []
    for seed_index in range(research.seeds):
        seed = research.seed + seed_index * 1009
        split = make_split(research, seed)
        train = split.train_pool[:research.generalization_train_episodes]
        test = split.observed_test + split.heldout_test
        y_train = _targets(train, research.horizons)
        y_val = _targets(split.validation, research.horizons)
        test_predictions_by_model = {}
        fitted_by_model = {}
        features_by_model = {}

        for model_index, model in enumerate(("A", "B", "C", "D", "E1")):
            model_seed = seed + model_index * 10007
            def features(examples):
                if model != "E1":
                    return _matrix_for(model, examples, research, model_seed,
                                       split.world_config.pixels)
                return np.asarray([_feature_vector(
                    "E1", item["observations"][:research.prediction_tick + 1],
                    "combined", research.hidden, model_seed,
                    split.world_config.pixels) for item in examples])

            x_train = features(train)
            x_val = features(split.validation)
            x_test = features(test)
            kind = "linear" if model == "E1" else "quadratic"
            pred, alpha, _, readouts = fit_validate_predict(
                x_train, y_train, x_val, y_val, x_test, research.alphas, kind)
            test_predictions_by_model[model] = pred
            fitted_by_model[model] = readouts
            features_by_model[model] = (x_train, x_val, x_test)
            _add_prediction_rows(diagnostic_rows, seed, model, test, pred, research)
            for pair in sorted({item["combination"] for item in test}):
                ix = [i for i, item in enumerate(test) if item["combination"] == pair]
                prediction_cache.append({
                    "model": model, "pair": pair, "horizon": 9 if 9 in research.horizons else max(research.horizons),
                    "truth": np.asarray([test[i]["targets"][9 if 9 in research.horizons else max(research.horizons)] for i in ix]),
                    "prediction": pred[ix, research.horizons.index(9 if 9 in research.horizons else max(research.horizons))],
                })

        # Train factor probes on the actual outputs A-D expose to their readout.
        factor_targets_train = np.column_stack((
            np.asarray([item["initial_position"] for item in train]),
            np.asarray([item["direction"] for item in train]),
            np.asarray([item["velocity"] for item in train]),
        ))
        factor_targets_val = np.column_stack((
            np.asarray([item["initial_position"] for item in split.validation]),
            np.asarray([item["direction"] for item in split.validation]),
            np.asarray([item["velocity"] for item in split.validation]),
        ))
        for model in ("A", "B", "C", "D"):
            x_train, x_val, x_test = features_by_model[model]
            factor_pred, _, _, _ = fit_validate_predict(
                x_train, factor_targets_train, x_val, factor_targets_val,
                x_test, research.alphas, "quadratic")
            internal_factor_rows.extend(_internal_factor_rows(
                seed, model, test, factor_pred, research))
            sensory_width = (split.world_config.pixels + 4 + 16 + 49)
            train_raw = x_train[:, -sensory_width:]
            val_raw = x_val[:, -sensory_width:]
            test_raw = x_test[:, -sensory_width:]
            p0 = split.world_config.pixels
            dir_slice = slice(p0 + 4, p0 + 20)
            speed_slice = slice(p0 + 20, sensory_width)
            dir_prediction, _, _, _ = _fit_affine_kernel(
                train_raw[:, dir_slice], factor_targets_train[:, 1:2],
                val_raw[:, dir_slice], factor_targets_val[:, 1:2],
                test_raw[:, dir_slice], research.alphas, "linear")
            speed_prediction, _, _, _ = _fit_affine_kernel(
                train_raw[:, speed_slice], factor_targets_train[:, 2:3],
                val_raw[:, speed_slice], factor_targets_val[:, 2:3],
                test_raw[:, speed_slice], research.alphas, "linear")
            passthrough_factor_rows.extend(_passthrough_factor_rows(
                seed, model, test, x_test, dir_prediction[:, 0],
                speed_prediction[:, 0], p0, research))

        # Independently trained sensory probes and the factorized E3.
        probes = _fit_factor_probes(research, split, train, seed)
        x0_hat_all, direction_hat_all, speed_hat_all = _predict_factorized(
            probes, test, research, split, seed)
        e3 = np.column_stack([
            x0_hat_all + direction_hat_all * speed_hat_all * horizon
            for horizon in research.horizons
        ])
        test_predictions_by_model["E3_factorized"] = e3
        _add_prediction_rows(diagnostic_rows, seed, "E3_factorized", test, e3, research)
        for pair in sorted({item["combination"] for item in test}):
            ix = [i for i, item in enumerate(test) if item["combination"] == pair]
            h = 9 if 9 in research.horizons else max(research.horizons)
            prediction_cache.append({"model": "E3_factorized", "pair": pair, "horizon": h,
                                     "truth": np.asarray([test[i]["targets"][h] for i in ix]),
                                     "prediction": e3[ix, research.horizons.index(h)]})

        # Privileged analytical oracle, kept separate from sensory models.
        e2 = np.column_stack([_oracle_predictions(test, h) for h in research.horizons])
        test_predictions_by_model["E2_oracle"] = e2
        _add_prediction_rows(diagnostic_rows, seed, "E2_oracle", test, e2, research)
        for pair in sorted({item["combination"] for item in test}):
            ix = [i for i, item in enumerate(test) if item["combination"] == pair]
            h = 9 if 9 in research.horizons else max(research.horizons)
            prediction_cache.append({"model": "E2_oracle", "pair": pair, "horizon": h,
                                     "truth": np.asarray([test[i]["targets"][h] for i in ix]),
                                     "prediction": e2[ix, research.horizons.index(h)]})

        # Factor probe summaries use only acoustics for movement factors and
        # only the initial image for position.
        factor_values = {}
        for split_name, examples in (("observed", split.observed_test),
                                     ("heldout", split.heldout_test)):
            offset = 0 if split_name == "observed" else len(split.observed_test)
            vis, phase, frequency = _factor_inputs(
                examples, research, split.world_config.pixels)
            x0_hat = np.asarray([_vision_centroid(x) for x in examples])
            x0_linear = np.asarray([probes[0].predict_one(x) for x in vis])
            direction_score = np.asarray([probes[1].predict_one(x) for x in phase])
            speed_score = np.asarray([probes[2].predict_one(x) for x in frequency])
            factor_values[split_name] = (x0_hat, direction_score, speed_score)
            factor_rows.extend(_probe_rows(seed, split_name, examples, x0_hat,
                                           direction_score, speed_score, x0_linear))

        # Readout capacity comparisons on identical A-D feature matrices.
        for model in ("A", "B", "C", "D"):
            x_train, x_val, x_test = features_by_model[model]
            model_seed = seed + ("ABCD".index(model) * 10007)
            for readout in READOUTS:
                fit_start = time.perf_counter()
                alpha_used = ""
                pair_count = 0
                mlp_epoch = ""
                inference_features = x_test
                if readout == "quadratic_ridge":
                    prediction, alphas, _, readouts = fit_validate_predict(
                        x_train, y_train, x_val, y_val, x_test,
                        research.alphas, "quadratic")
                    predictor = readouts
                    alpha_used = ";".join(str(float(a)) for a in alphas)
                elif readout == "explicit_pairwise_ridge":
                    p_train, p_val, pair_count = _explicit_pairwise(
                        x_train, x_val, split.world_config.pixels, model,
                        config.pairwise_features_per_modality)
                    _, p_test, _ = _explicit_pairwise(
                        x_train, x_test, split.world_config.pixels, model,
                        config.pairwise_features_per_modality)
                    inference_features = p_test
                    prediction, alphas, _, readouts = _fit_affine_kernel(
                        p_train, y_train, p_val, y_val, p_test,
                        research.alphas, "linear")
                    predictor = readouts
                    alpha_used = ";".join(str(float(a)) for a in alphas)
                else:
                    prediction, parameter_count, _, mlp_epoch, predictor = _mlp_fit_predict(
                        x_train, y_train, x_val, y_val, x_test, config, model_seed)
                    alphas = [""] * len(research.horizons)
                training_wall = time.perf_counter() - fit_start
                # Measure single-example prediction on the same heldout+observed matrix.
                infer_times = []
                for _ in range(2):
                    for x in inference_features:
                        t0 = time.perf_counter_ns()
                        if readout == "mlp32":
                            _mlp_predict_one(predictor, x)
                        else:
                            for fitted in predictor:
                                fitted.predict_one(x)
                        infer_times.append(time.perf_counter_ns() - t0)
                inference_ms = float(np.median(infer_times) / 1e6)
                count = (parameter_count if readout == "mlp32"
                         else len(train) * len(research.horizons))
                _readout_rows(readout_rows, seed, model, readout, test, prediction,
                              research.horizons, training_wall, count, inference_ms,
                              alpha_used, research, pair_count, mlp_epoch)

        # Error intercept/slope across horizons, by trajectory and pair.
        hs = np.asarray(research.horizons, dtype=float)
        for model, pred in test_predictions_by_model.items():
            errors = pred - _targets(test, research.horizons)
            for split_name, examples in (("observed", split.observed_test),
                                         ("heldout", split.heldout_test)):
                offset = 0 if split_name == "observed" else len(split.observed_test)
                for pair in sorted({x["combination"] for x in examples}):
                    indices = [offset + i for i, item in enumerate(examples)
                               if item["combination"] == pair]
                    slopes = []
                    intercepts = []
                    for idx in indices:
                        slope, intercept = np.polyfit(hs, errors[idx], 1)
                        slopes.append(slope)
                        intercepts.append(intercept)
                    decomposition_rows.append({
                        "seed": seed, "model": model, "evaluation_split": split_name,
                        "direction": pair[0], "velocity": pair[1],
                        "error_intercept_pixels": float(np.mean(intercepts)),
                        "error_slope_pixels_per_interval": float(np.mean(slopes)),
                        "n": len(indices),
                    })

        # Event thresholds use the same split and validation-only alpha tuning.
        h_eval = 9 if 9 in research.horizons else max(research.horizons)
        j_eval = research.horizons.index(h_eval)
        for threshold in config.event_thresholds:
            x_train = _matrix_for("D", train, research, seed + 30007,
                                  split.world_config.pixels, threshold)
            x_val = _matrix_for("D", split.validation, research, seed + 30007,
                                split.world_config.pixels, threshold)
            prediction, event_alphas, _, readouts = fit_validate_predict(
                x_train, y_train, x_val, y_val, x_val,
                research.alphas, "quadratic")
            # Warm once, then measure uninstrumented end-to-end cost. A second
            # pass isolates check/update subcomponent timings; that pass is not
            # used for the main latency/CPU comparison.
            _evaluate_event_processor(test[:1], readouts[j_eval], research,
                                      seed + 30007, split.world_config.pixels,
                                      threshold, instrument=False)
            for split_name, examples in (("observed", split.observed_test),
                                         ("heldout", split.heldout_test)):
                test_pred, cost = _repeat_inference(
                    lambda: _evaluate_event_processor(
                        examples, readouts[j_eval], research, seed + 30007,
                        split.world_config.pixels, threshold, instrument=False),
                    research.timed_repeats)
                _, component_cost = _evaluate_event_processor(
                    examples, readouts[j_eval], research, seed + 30007,
                    split.world_config.pixels, threshold, instrument=True)
                for field in ("available_update_opportunities", "triggered_updates",
                              "skipped_updates", "fraction_opportunities_skipped",
                              "trigger_check_wall_s", "state_update_wall_s"):
                    cost[field] = component_cost[field]
                metrics = _metrics(test_pred, examples, h_eval)
                event_rows.append({"seed": seed, "model": "D", "event_threshold": threshold,
                                   "evaluation_split": split_name, "horizon_intervals": h_eval,
                                   "mae_pixels": metrics["mae_pixels"],
                                   "within_1px_accuracy": metrics["within_1px_accuracy"],
                                   "available_update_opportunities": cost["available_update_opportunities"],
                                   "triggered_updates": cost["triggered_updates"],
                                   "skipped_updates": cost["skipped_updates"],
                                   "fraction_opportunities_skipped": cost["fraction_opportunities_skipped"],
                                   "trigger_check_wall_s": cost["trigger_check_wall_s"],
                                   "state_update_wall_s": cost["state_update_wall_s"],
                                   "inference_cpu_s": cost["inference_cpu_s"],
                                   "inference_wall_s": cost["inference_wall_s"],
                                   "median_episode_latency_ms": cost["median_episode_latency_ms"],
                                   "p95_episode_latency_ms": cost["p95_episode_latency_ms"],
                                   "selected_alpha": float(event_alphas[j_eval])})

        # Matched B/C inference timing references on each identical split.
        for model in ("B", "C"):
            fitted = fitted_by_model[model][j_eval]
            _evaluate_fixed_processor(model, test[:1], fitted, research,
                                      seed + "ABCD".index(model) * 10007,
                                      split.world_config.pixels)
            for split_name, examples in (("observed", split.observed_test),
                                         ("heldout", split.heldout_test)):
                _, cost = _repeat_inference(
                    lambda: _evaluate_fixed_processor(
                        model, examples, fitted, research,
                        seed + "ABCD".index(model) * 10007,
                        split.world_config.pixels), research.timed_repeats)
                event_rows.append({"seed": seed, "model": model, "event_threshold": "baseline",
                                   "evaluation_split": split_name, "horizon_intervals": h_eval,
                                   "mae_pixels": next(r["mae_pixels"] for r in diagnostic_rows
                                       if r["seed"] == seed and r["model"] == model
                                       and r["evaluation_split"] == split_name
                                       and r["direction"] == "all"
                                       and r["horizon_intervals"] == h_eval),
                                   "within_1px_accuracy": next(r["within_1px_accuracy"] for r in diagnostic_rows
                                       if r["seed"] == seed and r["model"] == model
                                       and r["evaluation_split"] == split_name
                                       and r["direction"] == "all"
                                       and r["horizon_intervals"] == h_eval),
                                   **cost})

    diagnostic_summary = _summary_with_ci(
        diagnostic_rows, ("model", "evaluation_split", "direction", "velocity", "horizon_intervals"),
        ("mae_pixels", "signed_bias_pixels", "within_1px_accuracy", "exact_position_accuracy"))
    diagnostic_output = _summarize_records(
        diagnostic_rows, ("model", "evaluation_split", "direction", "velocity", "horizon_intervals"),
        ("mae_pixels", "signed_bias_pixels", "within_1px_accuracy", "exact_position_accuracy"))
    factor_summary = _summary_with_ci(
        factor_rows, ("evaluation_split", "probe"), ("mae", "accuracy", "exact_level_accuracy"))
    factor_output = _summarize_records(
        factor_rows, ("evaluation_split", "probe"), ("mae", "accuracy", "exact_level_accuracy"))
    internal_factor_output = _summarize_records(
        internal_factor_rows + passthrough_factor_rows,
        ("model", "evaluation_split", "factor"),
        ("mae", "accuracy", "exact_level_accuracy"))
    readout_summary = _summary_with_ci(
        readout_rows, ("model", "readout", "evaluation_split", "horizon_intervals"),
        ("mae_pixels", "signed_bias_pixels", "within_1px_accuracy",
         "trainable_parameter_count", "readout_training_wall_s",
         "inference_latency_ms_per_episode"))
    readout_output = _summarize_records(
        readout_rows, ("model", "readout", "evaluation_split", "horizon_intervals"),
        ("mae_pixels", "signed_bias_pixels", "within_1px_accuracy",
         "trainable_parameter_count", "readout_training_wall_s",
         "inference_latency_ms_per_episode"))
    paired_rows = []
    h_eval = 9 if 9 in research.horizons else max(research.horizons)
    for split_name in ("observed", "heldout"):
        for model in ("B", "C", "D"):
            paired_rows.extend(_paired_differences(
                diagnostic_rows, model, "A", split_name, h_eval))
        paired_rows.extend(_paired_differences(
            diagnostic_rows, "D", "B", split_name, h_eval))
    decomposition_summary = _summary_with_ci(
        decomposition_rows, ("model", "evaluation_split", "direction", "velocity"),
        ("error_intercept_pixels", "error_slope_pixels_per_interval"))
    event_summary = _summary_with_ci(
        event_rows, ("model", "event_threshold", "evaluation_split", "horizon_intervals"),
        ("mae_pixels", "within_1px_accuracy", "fraction_opportunities_skipped",
         "trigger_check_wall_s", "state_update_wall_s", "inference_cpu_s",
         "inference_wall_s", "median_episode_latency_ms", "p95_episode_latency_ms"))

    _csv_write(outdir / "diagnostic_results.csv", diagnostic_output)
    _csv_write(outdir / "factor_probe_results.csv", factor_output)
    _csv_write(outdir / "internal_factor_probe_results.csv", internal_factor_output)
    _csv_write(outdir / "readout_diagnostics.csv", readout_output)
    _csv_write(outdir / "event_threshold_results.csv", event_rows + event_summary)
    _csv_write(outdir / "error_decomposition.csv", decomposition_rows + decomposition_summary)
    _csv_write(outdir / "paired_architecture_differences.csv", paired_rows)
    _plot_diagnostics(outdir, prediction_cache, diagnostic_output, config)
    _write_report(outdir / "DIAGNOSTIC_REPORT.md", config, diagnostic_output,
                  factor_output, internal_factor_output, readout_output,
                  event_rows, paired_rows)
    return {"output_dir": str(outdir), "diagnostic_rows": diagnostic_rows,
            "factor_rows": factor_rows, "readout_rows": readout_rows,
            "event_rows": event_rows}
