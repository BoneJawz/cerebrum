"""Five-seed temporal retention/readout diagnostics for the fixed I0 liquid.

The runner preserves the existing liquid equations and keeps all targets in
experiment-side records. Acoustic input adapters and readouts are replaceable
components; neither receives position labels at inference.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from .contracts import Observation
from .liquid import LiquidConfig, LiquidNetwork
from .processors import sensory_features
from .spatial_environment import (
    CHUNK_SIZE, SAMPLE_RATE, VISUAL_RATE, DigitSpatialVocabulary,
)
from .experiential_tokens import _variant


POSITIONS = (-1, 0, 1)
SEEDS = (42, 43, 44, 45, 46)
SPLIT_COUNTS = {"train": 12, "validation": 6, "test": 12}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PROJECTIONS = ("P0", "P1", "P2", "P3")
DELAYS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
POST_INSTRUCTION_DELAYS = (0.05, 0.10, 0.25, 0.50, 1.0)
EW_TAU_SECONDS = 0.25


def _dual_ridge(train_x, train_y, test_x, alpha):
    """Standardized multi-output ridge, solved in sample space for wide inputs."""
    x = np.asarray(train_x, dtype=np.float64)
    xt = np.asarray(test_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    z = (x - mean) / scale
    zt = (xt - mean) / scale
    y_mean = y.mean(axis=0)
    dual = np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y - y_mean)
    coef = z.T @ dual
    scores = zt @ coef + y_mean
    return scores, {"mean": mean, "scale": scale, "coef": coef,
                    "intercept": y_mean, "alpha": float(alpha)}


def _targets(positions):
    target = np.zeros((len(positions), 3), dtype=np.float64)
    order = {-1: 0, 0: 1, 1: 2}
    for i, value in enumerate(positions):
        target[i, order[int(value)]] = 1.0
    return target


def _accuracy(scores, labels):
    classes = np.asarray(POSITIONS, dtype=np.int64)
    prediction = classes[np.argmax(scores, axis=1)]
    error = np.abs(prediction - np.asarray(labels, dtype=np.int64))
    return {"exact_accuracy": float(np.mean(error == 0)),
            "within_one_accuracy": float(np.mean(error <= 1)),
            "mae": float(np.mean(error)), "median_absolute_error": float(np.median(error)),
            "p95_absolute_error": float(np.percentile(error, 95)),
            "predicted": prediction, "errors": error}


def _make_trials(seed, counts=SPLIT_COUNTS):
    vocab = DigitSpatialVocabulary(seed=seed + 1_000_003)
    rng = np.random.default_rng(seed + 7_000_019)
    trials = {split: [] for split in counts}
    parts = []
    for split, count in counts.items():
        per_split = []
        for position in POSITIONS:
            base = vocab.instruction(position)
            for replicate in range(count):
                wave = _variant(base, rng, noise_sd=0.01)
                observations = []
                raw_audio_features = []
                for tick in range(DigitSpatialVocabulary.INSTRUCTION_TICKS):
                    chunk = wave[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
                    obs = Observation(t=tick / VISUAL_RATE,
                        vision=np.zeros(199, dtype=np.float64), audio=chunk)
                    observations.append(obs)
                    raw_audio_features.append(sensory_features(obs)[199:])
                token_end = 0.2 if position == 0 else 0.5
                sequence = vocab.token_sequence(position)
                token_spans = ([{"token": "0", "start_tick": 0, "end_tick": 4}]
                               if position == 0 else [
                    {"token": sequence[0], "start_tick": 0, "end_tick": 4},
                    {"token": sequence[1], "start_tick": 6, "end_tick": 10}])
                trial = {"seed": seed, "split": split, "trial_id": f"{split}_{position}_{replicate}",
                    "position": position, "replicate": replicate, "observations": observations,
                    "raw_audio": np.asarray(raw_audio_features), "token_end": token_end,
                    "token_spans": token_spans}
                per_split.append(trial)
                parts.append({"seed": seed, "split": split, "trial_id": trial["trial_id"],
                              "position": position, "replicate": replicate,
                              "vocabulary_seed": vocab.seed, "waveform_noise_sd": 0.01})
        order = rng.permutation(len(per_split))
        trials[split] = [per_split[int(i)] for i in order]
    return trials, parts, vocab


def _projection_matrix(kind, net, seed):
    n = net.config.neurons
    audio_dim = 69
    baseline_energy = float(np.sum(net._input_gains[199:] ** 2))
    rng = np.random.default_rng(seed + {"P1": 101, "P2": 211, "P3": 307}[kind])
    if kind == "P1":
        matrix = np.zeros((n, audio_dim), dtype=np.float64)
        sign = rng.choice((-1.0, 1.0), size=audio_dim)
        for col in range(audio_dim):
            matrix[col % n, col] = sign[col] * math.sqrt(baseline_energy / audio_dim)
        description = "round-robin one-target signed map; fixed Frobenius energy matched to P0 audio gains"
    elif kind == "P2":
        matrix = np.zeros((n, audio_dim), dtype=np.float64)
        per_feature = min(4, n)
        magnitude = math.sqrt(baseline_energy / (audio_dim * per_feature))
        for col in range(audio_dim):
            targets = rng.choice(n, size=per_feature, replace=False)
            matrix[targets, col] = rng.choice((-magnitude, magnitude), size=per_feature)
        description = f"{per_feature}-target sparse signed map; fixed Frobenius energy matched to P0 audio gains"
    elif kind == "P3":
        basis, _ = np.linalg.qr(rng.normal(size=(audio_dim, n)))
        matrix = basis.T * math.sqrt(baseline_energy / n)
        description = "fixed orthogonal random projection to 64 dimensions; no liquid dynamics"
    else:
        raise ValueError(kind)
    return matrix, description


def _new_network(seed, config):
    net = LiquidNetwork(199 + 4 + 16 + 49, seed=seed,
                        config=config, variant="I0", visual_dim=199)
    return net


def _state_stats(state, net, time_seconds):
    active = float(np.mean(np.abs(state) > net.config.resting_threshold)) if net else float(np.mean(np.abs(state) > 0.012))
    if net is None:
        latest_age = ""
        queue = 0
    else:
        changed = np.flatnonzero(net.neuron_update_counts > net._diagnostic_update_start)
        latest_age = (float(time_seconds - np.max(net.last_update_time[changed])) if len(changed) else "")
        queue = net.pending_events
    return {"state_norm": float(np.linalg.norm(state)), "active_neuron_fraction": active,
        "mean_abs_activation": float(np.mean(np.abs(state))),
        "max_abs_activation": float(np.max(np.abs(state))),
        "time_since_latest_neuron_update_seconds": latest_age,
        "pending_queue_events": queue}


def _process_trial(trial, projection, seed, config, matrices):
    net = None if projection == "P3" else _new_network(seed, config)
    if net is not None:
        net._diagnostic_update_start = int(net.neuron_update_counts.sum())
        initial = net.state_dict()
    else:
        initial = None
    matrix = matrices.get(projection)
    if projection in ("P1", "P2"):
        # The adapter supplies 64 projected values to the unchanged inject/event path.
        net._input_targets = np.arange(net.input_dim, dtype=np.int64) % net.config.neurons
        net._input_gains = np.ones(net.input_dim, dtype=np.float64)
    if projection == "P3":
        n = matrix.shape[0]
        state_now = np.zeros(n, dtype=np.float64)
    else:
        state_now = np.zeros(net.config.neurons, dtype=np.float64)
    snapshots = {"before_instruction": state_now.copy()}
    snapshot_times = {"before_instruction": 0.0}
    activity_rows = []
    event_rows = []
    tick_states = []
    ew = np.zeros_like(state_now)
    event_base = asdict(net.stats()) if net is not None else {}
    sequence_base = net._sequence if net is not None else 0
    update_base = net.neuron_update_counts.copy() if net is not None else None
    start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()

    def capture(key, at):
        nonlocal state_now
        if net is not None:
            net.advance_to(at)
            state_now = net.snapshot()
        snapshots[key] = state_now.copy()
        snapshot_times[key] = float(at)
        stat = _state_stats(state_now, net, at)
        activity_rows.append({"seed": seed, "split": trial["split"], "trial_id": trial["trial_id"],
            "position": trial["position"], "projection": projection, "checkpoint": key,
            "time_since_instruction_start_seconds": float(at),
            "time_since_final_token_seconds": float(at - trial["token_end"]),
            **stat})

    # Snapshot reads are observational. The simulator advances only at declared event times.
    # Initial zero state is already stored; do not perform a second read.
    token_boundaries = {span["end_tick"]: span for span in trial["token_spans"]}
    post_token_keys = {"final_token_plus_000ms": 0.0, **{
        f"final_token_plus_{int(delay * 1000):03d}ms": delay for delay in DELAYS[1:]}}
    post_instr_keys = {"instruction_end": 0.0, **{
        f"instruction_end_plus_{int(delay * 1000):03d}ms": delay for delay in POST_INSTRUCTION_DELAYS}}
    pending = {}
    for key, delay in post_token_keys.items():
        pending[key] = trial["token_end"] + delay
    for key, delay in post_instr_keys.items():
        pending[key] = 0.8 + delay
    captured = set(snapshots)
    for tick, obs in enumerate(trial["observations"]):
        t = obs.t
        for key, at in sorted(pending.items(), key=lambda item: item[1]):
            if key not in captured and at <= t + 1e-12:
                capture(key, max(at, net.time_seconds if net is not None else at))
                captured.add(key)
        feature = trial["raw_audio"][tick]
        if projection == "P0":
            net.ingest_observation(obs)
            state_now = net.snapshot()
        elif projection in ("P1", "P2"):
            drive = matrix @ feature
            padded = np.zeros(net.input_dim, dtype=np.float64)
            padded[:net.config.neurons] = drive
            net.inject(padded, t)
            state_now = net.snapshot()
        else:
            state_now = matrix @ feature
        tick_states.append(state_now.copy())
        ew = math.exp(-0.05 / EW_TAU_SECONDS) * ew + (1.0 - math.exp(-0.05 / EW_TAU_SECONDS)) * state_now
        if tick + 1 in token_boundaries:
            boundary = (tick + 1) / VISUAL_RATE
            if net is not None:
                net.advance_to(boundary)
                state_now = net.snapshot()
            span = token_boundaries[tick + 1]
            key = f"token_end_{span['token']}_{tick + 1}"
            capture(key, boundary)
            activity_rows[-1]["token"] = span["token"]
            if span["token"] in ("+", "-"):
                activity_rows[-1]["token_phase"] = "sign"
            else:
                activity_rows[-1]["token_phase"] = "digit"
        # Preserve the real-time ordering of delayed readouts and later sensory chunks.
    if net is not None:
        net.advance_to(0.8)
        state_now = net.snapshot()
    else:
        state_now = np.zeros_like(state_now)
    if "instruction_end" not in captured:
        capture("instruction_end", 0.8)
        captured.add("instruction_end")
    for key, at in sorted(pending.items(), key=lambda item: item[1]):
        if key in captured:
            continue
        if net is not None:
            net.advance_to(at)
            state_now = net.snapshot()
        else:
            state_now = np.zeros_like(state_now)
        capture(key, at)
        captured.add(key)

    # A fixed set of aggregate representations; each has a documented dimension.
    ticks = np.asarray(tick_states)
    trace = ticks.reshape(-1)
    compact = {
        "mean_activation": np.mean(ticks, axis=0),
        "max_abs_activation": np.max(np.abs(ticks), axis=0),
        "time_integral_activation": np.sum(ticks, axis=0) / VISUAL_RATE,
        "event_count_per_neuron": ((net.neuron_update_counts - update_base).astype(np.float64)
                                    if net is not None else np.zeros(len(state_now))),
        "ew_activity_accumulator": ew.copy(),
    }
    full = {"terminal_" + key: value for key, value in snapshots.items()}
    full["ordered_instruction_trajectory"] = trace
    full["raw_acoustic_features"] = trial["raw_audio"].reshape(-1).copy()
    full.update(compact)
    full["sign_retention_trajectory"] = {
        f"final_token_plus_{int(delay * 1000):03d}ms": np.concatenate((
            ticks[:int(round(trial["token_end"] * VISUAL_RATE))].reshape(-1),
            snapshots[f"final_token_plus_{int(delay * 1000):03d}ms"].reshape(-1)))
        for delay in DELAYS
    }
    final_stats = asdict(net.stats()) if net is not None else {}
    if net is not None:
        deltas = {key: final_stats[key] - event_base[key] for key in final_stats}
        scheduled = net._sequence - sequence_base
        event_delta = deltas
        topo = net.topology_summary()
    else:
        deltas = {key: 0 for key in asdict(net.stats())} if False else {}
        scheduled = 0
        event_delta = {}
        topo = {"active_neurons": int(np.count_nonzero(np.abs(state_now) > .012)),
                "mean_out_degree": 0.0, "active_connections": 0}
    wall_ms = (time.perf_counter_ns() - start_wall) / 1e6
    cpu_ms = (time.process_time_ns() - start_cpu) / 1e6
    compute = {"seed": seed, "split": trial["split"], "trial_id": trial["trial_id"],
        "position": trial["position"], "projection": projection,
        "wall_ms": wall_ms, "cpu_ms": cpu_ms,
        "processed_events": event_delta.get("processed_events", 0),
        "propagated_events": event_delta.get("propagated_events", 0),
        "sensory_events": event_delta.get("sensory_events", 0),
        "neuron_updates": event_delta.get("neuron_updates", 0),
        "scheduled_events": scheduled, "peak_queue_observed": max(
            (row["pending_queue_events"] for row in activity_rows), default=0),
        "active_neurons_final": topo["active_neurons"],
        "mean_out_degree": topo["mean_out_degree"],
        "active_connections": topo["active_connections"],
        "weight_updates": event_delta.get("weight_updates", 0),
        "structural_updates": event_delta.get("structural_updates", 0),
    }
    event_metrics = _event_segments(trial, activity_rows, compute)
    return {"trial": trial, "features": full, "snapshots": snapshots,
            "tick_states": ticks, "token_spans": token_boundaries,
            "activity": activity_rows, "events": event_metrics,
            "compute": compute, "initial_snapshot": initial,
            "position_timepoints": {k.removeprefix("terminal_"): v for k, v in full.items()
                                    if k.startswith("terminal_")}}


def _event_segments(trial, activity_rows, compute):
    # The simulator exposes aggregate counters only. Leave phase counts blank
    # rather than copying whole-trial totals into each phase.
    active_ticks = sum(span["end_tick"] - span["start_tick"] for span in trial["token_spans"])
    instruction_seconds = DigitSpatialVocabulary.INSTRUCTION_TICKS / VISUAL_RATE
    rows = []
    for phase, duration in (("acoustic_token_intervals", active_ticks / VISUAL_RATE),
                            ("silence_padding_and_gaps", instruction_seconds - active_ticks / VISUAL_RATE),
                            ("post_instruction_1s", 1.0)):
        rows.append({"seed": trial["seed"], "split": trial["split"], "trial_id": trial["trial_id"],
            "position": trial["position"], "projection": compute["projection"],
            "phase": phase, "duration_seconds": duration,
            "processed_events": "", "propagated_events": "",
            "scheduled_events": "", "neuron_updates": "",
            "mean_propagation_per_sensory_event": "",
            "phase_counts_supported": False})
    return rows


def _fixed_random_projection(dimension, output_dimension, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dimension, output_dimension)))
    return q[:, :output_dimension]


def _representation(record, name, projection=None):
    if name == "ordered_instruction_trajectory_random64":
        return record["features"]["ordered_instruction_trajectory"] @ projection
    if name == "raw_acoustic_features_random64":
        return record["features"]["raw_acoustic_features"] @ projection
    return record["features"][name]


def _fit_validated(train_records, validation_records, test_records,
                   representation, projection=None):
    train_x = np.asarray([_representation(r, representation, projection) for r in train_records])
    validation_x = np.asarray([_representation(r, representation, projection) for r in validation_records])
    test_x = np.asarray([_representation(r, representation, projection) for r in test_records])
    y_train = np.asarray([r["trial"]["position"] for r in train_records])
    y_val = np.asarray([r["trial"]["position"] for r in validation_records])
    y_test = np.asarray([r["trial"]["position"] for r in test_records])
    y_train_oh, y_val_oh = _targets(y_train), _targets(y_val)
    best = None
    for alpha in ALPHAS:
        scores, model = _dual_ridge(train_x, y_train_oh, validation_x, alpha)
        metrics = _accuracy(scores, y_val)
        key = (metrics["exact_accuracy"], -alpha)
        if best is None or key > best[0]:
            best = (key, alpha, model, metrics)
    alpha = best[1]
    test_scores, model = _dual_ridge(train_x, y_train_oh, test_x, alpha)
    validation_scores, _ = _dual_ridge(train_x, y_train_oh, validation_x, alpha)
    return {"model": model, "alpha": alpha, "train_x": train_x,
        "validation_metrics": _accuracy(validation_scores, y_val),
        "test_metrics": _accuracy(test_scores, y_test),
        "test_scores": test_scores, "train_y": y_train, "val_y": y_val,
        "test_y": y_test, "feature_dim": int(train_x.shape[1]),
        "train_count": len(train_x), "test_count": len(test_x)}


def _append_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()
        if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def _score_projection(seed, projection, datasets, matrices, output, config):
    records = {}
    telemetry, compute, events = [], [], []
    initial_saved = False
    for split in ("train", "validation", "test"):
        records[split] = []
        for trial in datasets[split]:
            record = _process_trial(trial, projection, seed, config, matrices)
            records[split].append(record)
            telemetry.extend(record["activity"])
            compute.append(record["compute"])
            events.extend(record["events"])
            if projection == "P0" and split == "train" and not initial_saved:
                _write_json(output / "model_snapshots" / f"seed_{seed}_I0_initial.json",
                            record["initial_snapshot"])
                initial_saved = True
    return records, telemetry, compute, events


def _representation_names(record):
    return ["terminal_before_instruction", "terminal_final_token_plus_000ms",
        "terminal_final_token_plus_050ms", "terminal_final_token_plus_100ms",
        "terminal_final_token_plus_250ms", "terminal_final_token_plus_500ms",
        "terminal_final_token_plus_1000ms", "terminal_instruction_end",
        "terminal_instruction_end_plus_050ms", "terminal_instruction_end_plus_100ms",
        "terminal_instruction_end_plus_250ms", "terminal_instruction_end_plus_500ms",
        "terminal_instruction_end_plus_1000ms", "ordered_instruction_trajectory",
        "mean_activation", "max_abs_activation", "time_integral_activation",
        "event_count_per_neuron", "ew_activity_accumulator", "raw_acoustic_features"]


def _timepoint_rows(seed, projection, records, projections, readouts, config):
    result_rows, readout_rows, selected_models = [], [], {}
    train, val, test = records["train"], records["validation"], records["test"]
    first = train[0]
    random_maps = {
        "ordered_instruction_trajectory_random64": _fixed_random_projection(
            len(first["features"]["ordered_instruction_trajectory"]), 64, seed + 31_337),
        "raw_acoustic_features_random64": _fixed_random_projection(
            len(first["features"]["raw_acoustic_features"]), 64, seed + 31_339),
    }
    representations = _representation_names(first) + list(random_maps)
    p0_candidates = []
    for representation in representations:
        matrix = random_maps.get(representation)
        fit = _fit_validated(train, val, test, representation, matrix)
        metrics = fit["test_metrics"]
        val_metrics = fit["validation_metrics"]
        row = {"seed": seed, "projection": projection, "representation": representation,
            "feature_dimension": fit["feature_dim"], "train_examples": fit["train_count"],
            "validation_examples": len(val), "test_examples": fit["test_count"],
            "readout_parameters": fit["feature_dim"] * 3 + 3,
            "selected_alpha": fit["alpha"], "validation_exact_accuracy": val_metrics["exact_accuracy"],
            "exact_accuracy": metrics["exact_accuracy"],
            "within_one_accuracy": metrics["within_one_accuracy"],
            "mae_position_units": metrics["mae"],
            "median_absolute_error": metrics["median_absolute_error"],
            "p95_absolute_error": metrics["p95_absolute_error"]}
        readout_rows.append(row)
        result_rows.append({**row, "condition": "three_position_audio_only"})
        # Do not use the final test score to select the primary readout.
        if projection == "P0" and representation not in ("terminal_before_instruction",):
            params_penalty = fit["feature_dim"]
            p0_candidates.append((val_metrics["exact_accuracy"], -params_penalty,
                                  representation, fit))
        save_key = f"{projection}_{representation}"
        selected_models[save_key] = fit["model"]

    # Every named snapshot is scored independently to expose retention over time.
    snapshot_names = [name for name in representations if name.startswith("terminal_")]
    for name in snapshot_names:
        fit = next(row for row in readout_rows if row["representation"] == name)
        result_rows.append({**fit, "condition": "temporal_snapshot"})
    selected = max(p0_candidates, key=lambda item: (item[0], item[1])) if projection == "P0" else None
    best_row = None
    if selected:
        best_row = {"seed": seed, "projection": projection,
            "selected_representation": selected[2], "validation_exact_accuracy": selected[0],
            "test_exact_accuracy": selected[3]["test_metrics"]["exact_accuracy"],
            "test_mae_position_units": selected[3]["test_metrics"]["mae"],
            "feature_dimension": selected[3]["feature_dim"],
            "selected_alpha": selected[3]["alpha"]}
    return result_rows, readout_rows, selected_models, best_row, random_maps


def _token_probes(seed, records_by_split, projection):
    classes = ("+", "-", "0", "1")
    rows = []
    train, val, test = (records_by_split[name] for name in ("train", "validation", "test"))
    # Token end states / local four-observation trajectories, with no semantic labels in inputs.
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    def token_examples(record):
        pos = record["trial"]["position"]
        if pos == 0:
            return [("0", record["features"]["terminal_token_end_0_4"],
                     record["tick_states"][0:4].reshape(-1))]
        sign = "+" if pos > 0 else "-"
        return [(sign, record["features"][f"terminal_token_end_{sign}_4"], record["tick_states"][0:4].reshape(-1)),
                ("1", record["features"]["terminal_token_end_1_10"], record["tick_states"][6:10].reshape(-1))]
    for split_name, split_records, target_x, target_y in (
        ("train", train, train_x, train_y), ("validation", val, val_x, val_y),
        ("test", test, test_x, test_y)):
        for record in split_records:
            for label, terminal, trajectory in token_examples(record):
                target_x.append((terminal, trajectory)); target_y.append(label)
    for representation_i, representation_name in ((0, "terminal_token_state"), (1, "ordered_four_tick_token_trajectory")):
        for split_x, split_y in ((train_x, train_y), (val_x, val_y), (test_x, test_y)):
            pass
        xx_train = np.asarray([item[representation_i] for item in train_x])
        xx_val = np.asarray([item[representation_i] for item in val_x])
        xx_test = np.asarray([item[representation_i] for item in test_x])
        def onehot(labels):
            result = np.zeros((len(labels), len(classes)))
            idx = {c: i for i, c in enumerate(classes)}
            for i, label in enumerate(labels): result[i, idx[label]] = 1.0
            return result
        best = None
        for alpha in ALPHAS:
            scores, model = _dual_ridge(xx_train, onehot(train_y), xx_val, alpha)
            acc = float(np.mean(np.asarray(classes)[np.argmax(scores, axis=1)] == np.asarray(val_y)))
            if best is None or (acc, -alpha) > (best[0], -best[1]):
                best = (acc, alpha, model)
        scores, model = _dual_ridge(xx_train, onehot(train_y), xx_test, best[1])
        pred = np.asarray(classes)[np.argmax(scores, axis=1)]
        truth = np.asarray(test_y)
        per_class = [float(np.mean(pred[truth == c] == c)) for c in classes if np.any(truth == c)]
        rows.append({"seed": seed, "projection": projection,
            "probe": representation_name, "selected_alpha": best[1],
            "validation_accuracy": best[0], "test_accuracy": float(np.mean(pred == truth)),
            "macro_class_accuracy": float(np.mean(per_class)), "test_examples": len(truth),
            "feature_dimension": int(xx_train.shape[1])})
    # Sign-specific retention: compare recoverability after the full cue at fixed delays.
    for rep_index, rep_name in ((0, "terminal"), (1, "trajectory")):
        def sign_target(labels):
            return np.asarray([[float(x == "+"), float(x == "-")] for x in labels])
        for delay in DELAYS:
            key = f"final_token_plus_{int(delay*1000):03d}ms"
            matrices, labels = [], []
            for split_records in (train, val, test):
                sx, sy = [], []
                for record in split_records:
                    pos = record["trial"]["position"]
                    if pos == 0: continue
                    value = (record["features"]["terminal_" + key] if rep_index == 0
                             else record["features"]["sign_retention_trajectory"][key])
                    sx.append(value); sy.append("+" if pos > 0 else "-")
                matrices.append(np.asarray(sx)); labels.append(sy)
            sign_train_x, sign_val_x, test_matrix = matrices
            sign_train_y, sign_val_y, sign_test_y = labels
            best = None
            for alpha in ALPHAS:
                scores, _ = _dual_ridge(sign_train_x, sign_target(sign_train_y), sign_val_x, alpha)
                val_pred = np.asarray(("+", "-"))[np.argmax(scores, axis=1)]
                acc = float(np.mean(val_pred == np.asarray(sign_val_y)))
                if best is None or (acc, -alpha) > (best[0], -best[1]): best = (acc, alpha)
            scores, _ = _dual_ridge(sign_train_x, sign_target(sign_train_y), test_matrix, best[1])
            pred = np.asarray(("+", "-"))[np.argmax(scores, axis=1)]
            rows.append({"seed": seed, "projection": projection,
                "probe": f"sign_retention_{rep_name}_plus_{int(delay*1000):03d}ms",
                "selected_alpha": best[1], "validation_accuracy": best[0],
                "test_accuracy": float(np.mean(pred == np.asarray(sign_test_y))),
                "macro_class_accuracy": float(np.mean([
                    np.mean(pred[np.asarray(sign_test_y) == c] == c) for c in ("+", "-")])),
                "test_examples": len(sign_test_y), "feature_dimension": int(test_matrix.shape[1])})
    return rows


def _write_report(output, config, results, projection_rows, event_rows):
    lines = ["# Temporal Information Retention and Liquid-State Readout Report", "",
        "## Timing and input audit", "",
        "Acoustic tokens are 4 ticks (0.2 s) at 2,000 samples/s; silence gaps are 2 ticks (0.1 s). Instructions are 16 ticks (0.8 s), chunked into 100 samples at 20 Hz. The three instructions are `-`→`1`, `0`, and `+`→`1`; sign and digit tokens end at 0.2 s and 0.5 s for signed trials, while zero ends at 0.2 s and is followed by 0.6 s padding.",
        "", "Each chunk is transformed by `sensory_features`: 199 zero-valued visual inputs, 4 waveform statistics, 16 signed samples, and 49 FFT magnitudes (268 total). For P0, each input coordinate has one seed-fixed target neuron and signed gain; the drive is aggregated per neuron and capped to 8 sensory targets. Default liquid parameters include tau=0.2 s, adaptation tau=0.6 s, internal interval=0.05 s, transmission delay=0.005 s, resting threshold=0.012, and emission threshold=0.035. Neuron states decay lazily between scheduled events; `advance_to` processes events through the requested simulated time. A snapshot returns analytically decayed activation without propagation or state mutation; it increments only the readout-scan counter.",
        "", "Terminal timing is explicitly separated into time since instruction start, final token end, instruction-window end, and latest neuron update. Signed cues end at 0.5 s; zero ends at 0.2 s. The common instruction window ends at 0.8 s. Active simulation continues through each requested post-cue and post-instruction checkpoint.",
        "", "## Protocol", "", 
        f"Seeds: {', '.join(map(str, config['seeds']))}. Per seed, each of -1, 0, +1 has {config['train_per_position']} train, {config['validation_per_position']} validation, and {config['test_per_position']} test waveform realizations, independently perturbed by gain, circular delay, and 0.01-SD additive noise. Train, validation, and test trials are disjoint. I0 weights and topology are frozen. Probe labels are used only to fit supervised readouts; no position is passed to the liquid.",
        "", "Readout regularization is selected from the predeclared alpha grid by validation accuracy only. The report compares raw acoustic features, time-resolved states, standardized terminal states, compact statistics, fixed random 64-D projections, and a causal 250-ms EW accumulator. Feature dimensions and fitted parameter counts are recorded. No position-accuracy test score is used to select the primary readout.",
        "", "## Checkpoints", ""]
    for name, passed, detail in results:
        lines.append(f"- **{name}: {'PASS' if passed else 'FAIL'}.** {detail}")
    lines += ["", "## Projection comparison", "", "| Projection | Mean test accuracy | Mean MAE | Mean episode CPU ms |", "|---|---:|---:|---:|"]
    for row in projection_rows:
        lines.append(f"| {row['projection']} | {row['exact_accuracy']:.3f} | {row['mae']:.3f} | {row['cpu_ms']:.2f} |")
    lines += ["", "## Limits and next step", "",
        "The projections are fixed diagnostic input maps; they do not alter the neuron equations, weights, or topology. P3 is a projection-only baseline without liquid dynamics. Event counts are scheduler operations, not FLOPs. The observed queue peak is sampled at readout checkpoints and is therefore a lower bound on any unsampled transient peak. Traced Python allocation is omitted from architecture claims; the report does not infer physical information destruction solely from declining linear-probe accuracy.",
        "", "The report separates measured recoverability from interpretation. A pass demonstrates fixed-liquid representational capacity with the selected supervised readout, not unsupervised or multimodal learning. If the 80% three-position threshold fails, do not expand to the 199-position task or enable I1/I2; use the best validated projection/readout result to define the next single-factor experiment.", ""]
    (Path(output) / "TEMPORAL_INFORMATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(output, seeds=SEEDS, train_per_position=12, validation_per_position=6,
        test_per_position=12):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "model_snapshots").mkdir(exist_ok=True)
    seeds = tuple(int(x) for x in seeds)
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("at least five unique seeds are required")
    if min(train_per_position, validation_per_position, test_per_position) < 2:
        raise ValueError("each split needs at least two examples per position")
    config = LiquidConfig()
    config_data = {"seeds": list(seeds), "positions": list(POSITIONS),
        "train_per_position": train_per_position,
        "validation_per_position": validation_per_position,
        "test_per_position": test_per_position,
        "splits_are_disjoint_episodes": True, "liquid_variant": "I0",
        "liquid_config": asdict(config), "feature_dimensions": {"vision": 199,
            "audio_summary": 4, "audio_waveform_samples": 16, "audio_fft_bins": 49,
            "full_input": 268, "audio_only_input": 69},
        "sample_rate_hz": SAMPLE_RATE, "chunk_samples": CHUNK_SIZE,
        "visual_event_rate_hz": VISUAL_RATE, "token_ticks": 4,
        "token_seconds": 0.2, "silence_gap_ticks": 2, "silence_gap_seconds": 0.1,
        "instruction_ticks": 16, "instruction_seconds": 0.8,
        "alphas": list(ALPHAS), "alpha_selection": "validation exact accuracy; ties prefer smaller alpha",
        "waveform_variation": {"gain": [0.85, 1.15], "circular_shift_samples": [-5, 5],
            "additive_gaussian_noise_sd": 0.01},
        "readouts": ["raw acoustic features", "snapshot timepoints", "ordered full trajectory",
            "mean", "max absolute activation", "time integral", "per-neuron event counts",
            "fixed 64-dimensional trajectory random projection", "250-ms causal EW accumulator"],
        "projection_conditions": {"P0": "existing one-target signed random mapping",
            "P1": "round-robin one-target signed map; Frobenius energy matched",
            "P2": "four-target sparse signed map; Frobenius energy matched",
            "P3": "orthogonal fixed 64-D projection without liquid dynamics"},
        "event_sweep": ["default", "tau=0.10", "tau=0.40", "emit=0.020", "emit=0.060"],
        "checkpoint_C": {"mean_test_accuracy_min": 0.8, "per_seed_test_accuracy_min": 0.7}}
    _write_json(output / "temporal_diagnostics_config.json", config_data)

    position_rows, readout_rows, token_rows = [], [], []
    activity_rows, compute_rows, event_rows, projection_rows = [], [], [], []
    partitions = []
    projection_matrices = {}
    best_seed_rows = []
    for seed in seeds:
        datasets, part_rows, vocab = _make_trials(seed, {
            "train": train_per_position,
            "validation": validation_per_position,
            "test": test_per_position})
        partitions.extend(part_rows)
        per_seed_projection = {}
        for projection in PROJECTIONS:
            # Construct a canonical network to derive energy-matched input maps.
            canon = _new_network(seed, config)
            matrices = {}
            for kind in ("P1", "P2", "P3"):
                matrices[kind], _ = _projection_matrix(kind, canon, seed)
            if seed not in projection_matrices:
                projection_matrices[seed] = matrices
            records, telemetry, costs, events = _score_projection(
                seed, projection, datasets, matrices, output, config)
            activity_rows.extend(telemetry); compute_rows.extend(costs); event_rows.extend(events)
            time_rows, reads, models, best_row, random_maps = _timepoint_rows(
                seed, projection, records, matrices, {}, config)
            position_rows.extend(time_rows); readout_rows.extend(reads)
            if projection == "P0":
                best_seed_rows.append(best_row)
            token_rows.extend(_token_probes(seed, records, projection))
            per_seed_projection[projection] = records
            # Keep readout weights, selected alpha, train normalization and fixed projections.
            if projection == "P0":
                np.savez_compressed(output / f"readouts_seed_{seed}.npz",
                    **{key: value for key, model in models.items()
                       for value in ((key + "_mean", model["mean"]),
                                     (key + "_scale", model["scale"]),
                                     (key + "_coef", model["coef"]),
                                     (key + "_intercept", model["intercept"]))})
        for projection in PROJECTIONS:
            records = per_seed_projection[projection]
            train, val, test = (records[name] for name in ("train", "validation", "test"))
            # P0/P1/P2/P3 summary uses the strongest validation-selected representation.
            candidate_rows = [row for row in readout_rows if row["seed"] == seed and row["projection"] == projection]
            liquids = [row for row in candidate_rows if row["representation"] not in (
                "raw_acoustic_features", "raw_acoustic_features_random64")]
            best = max(liquids, key=lambda row: (row["validation_exact_accuracy"],
                                                  -row["feature_dimension"]))
            compute_for_proj = [row for row in compute_rows if row["seed"] == seed and row["projection"] == projection and row["split"] == "test"]
            projection_rows.append({"seed": seed, "projection": projection,
                "selected_representation": best["representation"],
                "validation_accuracy": best["validation_exact_accuracy"],
                "exact_accuracy": best["exact_accuracy"], "mae": best["mae_position_units"],
                "feature_dimension": best["feature_dimension"],
                "cpu_ms": float(np.mean([r["cpu_ms"] for r in compute_for_proj])),
                "wall_ms": float(np.mean([r["wall_ms"] for r in compute_for_proj]))})
        print(f"seed {seed}: collected I0 temporal and projection diagnostics", flush=True)

    _append_csv(output / "temporal_position_accuracy.csv", position_rows)
    _append_csv(output / "temporal_readout_comparison.csv", readout_rows)
    _append_csv(output / "temporal_token_information.csv", token_rows)
    _append_csv(output / "temporal_state_activity.csv", activity_rows)
    _append_csv(output / "temporal_compute_results.csv", compute_rows)
    _append_csv(output / "temporal_event_analysis.csv", event_rows)
    _append_csv(output / "temporal_projection_comparison.csv", projection_rows)
    _append_csv(output / "temporal_partitions.csv", partitions)
    np.savez_compressed(output / "projection_matrices.npz",
        **{f"seed_{seed}_{name}": matrix for seed, matrices in projection_matrices.items()
           for name, matrix in matrices.items()})

    # Aggregate only by seed; report both the 5-seed mean and each run.
    p0_rows = [row for row in projection_rows if row["projection"] == "P0"]
    raw_rows = [row for row in readout_rows if row["projection"] == "P0"
                and row["representation"] == "raw_acoustic_features"]
    token_traj = [row for row in token_rows if row["projection"] == "P0"
                  and row["probe"] == "ordered_four_tick_token_trajectory"]
    c_mean = float(np.mean([row["exact_accuracy"] for row in p0_rows]))
    c_min = float(np.min([row["exact_accuracy"] for row in p0_rows]))
    a_mean = float(np.mean([row["exact_accuracy"] for row in raw_rows]))
    b_mean = float(np.mean([row["test_accuracy"] for row in token_traj]))
    checkpoint_results = [
        ("A — raw acoustic recoverability", a_mean >= 0.80,
         f"raw full-instruction feature mean accuracy {a_mean:.3f}; per-seed values are in temporal_readout_comparison.csv"),
        ("B — liquid token trajectory", b_mean >= 0.80 and min(row["test_accuracy"] for row in token_traj) >= 0.70,
         f"four-tick token-state trajectory mean accuracy {b_mean:.3f}, minimum seed {min(row['test_accuracy'] for row in token_traj):.3f}"),
        ("C — three-position I0 prediction", c_mean >= 0.80 and c_min >= 0.70,
         f"P0 validation-selected readout mean test accuracy {c_mean:.3f}, minimum seed {c_min:.3f}"),
    ]
    summary_proj = []
    for projection in PROJECTIONS:
        rows = [r for r in projection_rows if r["projection"] == projection]
        summary_proj.append({"projection": projection,
            "exact_accuracy": float(np.mean([r["exact_accuracy"] for r in rows])),
            "mae": float(np.mean([r["mae"] for r in rows])),
            "cpu_ms": float(np.mean([r["cpu_ms"] for r in rows]))})
    _write_plots(output, position_rows, readout_rows, token_rows, activity_rows, projection_rows)
    _write_report(output, config_data, checkpoint_results, summary_proj, event_rows)
    return checkpoint_results


def _write_plots(output, position_rows, readout_rows, token_rows, activity_rows, projection_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p0 = [row for row in position_rows if row["projection"] == "P0" and row["condition"] == "temporal_snapshot"]
    keys = sorted({row["representation"] for row in p0}, key=lambda x: float(x.split("_")[-1]) if x.split("_")[-1].replace("ms", "").isdigit() else -1)
    token_delays = [row for row in token_rows if row["projection"] == "P0" and row["probe"].startswith("sign_retention")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for projection in PROJECTIONS:
        rows = [r for r in projection_rows if r["projection"] == projection]
        ax.plot([r["seed"] for r in rows], [r["exact_accuracy"] for r in rows], marker="o", label=projection)
    ax.axhline(.8, color="grey", linestyle="--"); ax.set_ylabel("Validation-selected test accuracy")
    ax.set_title("Three-position accuracy by fixed input projection"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "temporal_position_retention.png", dpi=150); plt.close(fig)

    means = []
    for probe in ("ordered_four_tick_token_trajectory",):
        means = [np.mean([r["test_accuracy"] for r in token_rows if r["projection"] == "P0" and r["probe"] == probe])]
    delays = [0, 50, 100, 250, 500, 1000]
    retention_means = []
    for delay in delays:
        needle = f"plus_{delay:03d}ms"
        values = [r["test_accuracy"] for r in token_delays if needle in r["probe"]]
        retention_means.append(float(np.mean(values)) if values else float("nan"))
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(delays, retention_means, marker="o")
    ax.set_xlabel("Time after final token (ms)"); ax.set_ylabel("Sign classification accuracy")
    ax.set_ylim(0, 1); ax.set_title("Sign information recoverability in I0"); fig.tight_layout()
    fig.savefig(output / "temporal_token_retention.png", dpi=150); plt.close(fig)

    rep_rows = [r for r in readout_rows if r["projection"] == "P0"]
    focus = ["raw_acoustic_features", "ordered_instruction_trajectory",
             "ordered_instruction_trajectory_random64", "mean_activation",
             "max_abs_activation", "time_integral_activation", "event_count_per_neuron",
             "ew_activity_accumulator"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals = [np.mean([r["exact_accuracy"] for r in rep_rows if r["representation"] == name]) for name in focus]
    ax.bar(range(len(focus)), vals, color="#388b83"); ax.axhline(1/3, color="#bd6b44", linestyle="--", label="chance")
    ax.set_xticks(range(len(focus)), focus, rotation=28, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("Exact position accuracy"); ax.legend(); ax.set_title("Readout representations on identical trials")
    fig.tight_layout(); fig.savefig(output / "temporal_readout_accuracy.png", dpi=150); plt.close(fig)

    state_rows = [r for r in activity_rows if r["projection"] == "P0" and r["split"] == "test"]
    time_order = ["before_instruction", "token_end_+_4", "token_end_-_4", "token_end_0_4",
        "token_end_1_10", "final_token_plus_000ms", "final_token_plus_050ms",
        "final_token_plus_100ms", "final_token_plus_250ms", "final_token_plus_500ms",
        "final_token_plus_1000ms", "instruction_end", "instruction_end_plus_050ms",
        "instruction_end_plus_100ms", "instruction_end_plus_250ms",
        "instruction_end_plus_500ms", "instruction_end_plus_1000ms"]
    xvals, norms, active = [], [], []
    for key in time_order:
        values = [r for r in state_rows if r["checkpoint"] == key]
        if values:
            xvals.append(float(np.mean([r["time_since_instruction_start_seconds"] for r in values])))
            norms.append(float(np.mean([r["state_norm"] for r in values])))
            active.append(float(np.mean([r["active_neuron_fraction"] for r in values])))
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(xvals, norms, marker="o", label="state norm")
    ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("Mean state norm"); ax2 = ax.twinx()
    ax2.plot(xvals, active, color="#bd6b44", marker="s", label="active fraction")
    ax2.set_ylabel("Active neuron fraction"); ax.set_title("I0 state activity and decay"); fig.tight_layout()
    fig.savefig(output / "temporal_state_activity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for projection in PROJECTIONS:
        rows = [r for r in projection_rows if r["projection"] == projection]
        ax.scatter([r["cpu_ms"] for r in rows], [r["exact_accuracy"] for r in rows], label=projection)
    ax.set_xlabel("Mean CPU time per test episode (ms)"); ax.set_ylabel("Exact position accuracy")
    ax.axhline(.8, color="grey", linestyle="--"); ax.legend(); ax.set_title("Accuracy versus simulation cost")
    fig.tight_layout(); fig.savefig(output / "temporal_accuracy_vs_compute.png", dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/temporal-diagnostics")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--train-per-position", type=int, default=12)
    parser.add_argument("--validation-per-position", type=int, default=6)
    parser.add_argument("--test-per-position", type=int, default=12)
    args = parser.parse_args(argv)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    try:
        results = run(args.output, seeds, args.train_per_position,
                      args.validation_per_position, args.test_per_position)
    except ValueError as exc:
        parser.error(str(exc))
    print("Checkpoints:")
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")


if __name__ == "__main__":
    main()
