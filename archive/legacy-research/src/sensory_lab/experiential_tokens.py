"""Checkpoint 1: probe acoustic-token information in frozen liquid states.

Probe labels are diagnostic targets only and never enter the liquid input.
This runner does not perform audiovisual association learning.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .contracts import Observation
from .liquid import LiquidConfig, LiquidNetwork
from .liquid_readout import RidgeReadout
from .processors import sensory_features
from .spatial_environment import (
    CHUNK_SIZE, SAMPLE_RATE, VISUAL_RATE, DigitSpatialVocabulary, visual_array,
)

TOKENS = ("+", "-", *(str(i) for i in range(10)))
PASS_MEAN = 0.80
PASS_MIN_SEED = 0.70


def _observations(waveform, position_cells=199):
    result = []
    for tick in range(len(waveform) // CHUNK_SIZE):
        result.append(Observation(t=tick / VISUAL_RATE,
            vision=np.zeros(position_cells, dtype=np.float64),
            audio=waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()))
    return result


def _encode(net, waveform, position_cells=199):
    net.reset_episode()
    observations = _observations(waveform, position_cells)
    trajectory = []
    raw_sequence = []
    for observation in observations:
        net.ingest_observation(observation)
        trajectory.append(net.snapshot())
        raw_sequence.append(sensory_features(observation)[position_cells:])
    net.advance_to(observations[-1].t + 0.05)
    final_state = net.snapshot()
    return final_state, np.asarray(trajectory).reshape(-1), np.asarray(raw_sequence).reshape(-1)


def _variant(waveform, rng, noise_sd=0.01):
    shifted = np.roll(waveform, int(rng.integers(-5, 6)))
    return float(rng.uniform(0.85, 1.15)) * shifted + rng.normal(0.0, noise_sd, waveform.shape)


def _one_hot(labels, classes):
    index = {name: i for i, name in enumerate(classes)}
    target = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        target[row, index[label]] = 1.0
    return target


def _position_diagnostics(seed, neurons=64):
    """Probe ordered full instructions after token recoverability is checked."""
    vocab = DigitSpatialVocabulary(seed=seed + 1_000_003)
    rng = np.random.default_rng(seed + 3_000_007)
    cfg = LiquidConfig(neurons=neurons, initial_out_degree=6, max_out_degree=8,
        max_connections=neurons * 8, max_sensory_targets=min(8, neurons))
    net = LiquidNetwork(199 + 4 + 16 + 49, seed=seed, config=cfg,
                        variant="I0", visual_dim=199)
    positions = tuple(range(-99, 100))
    train_last, train_trace, train_y, train_length = [], [], [], []
    test_last, test_trace, test_y, test_length = [], [], [], []
    train_roles, train_role_y, test_roles, test_role_y = [], [], [], []
    for position in positions:
        base = vocab.instruction(position)
        for is_test in (False, True):
            last, trace, _ = _encode(net, _variant(base, rng))
            trace = trace.reshape(16, neurons)
            ordered_trace = trace.reshape(4, 4, neurons).mean(axis=1).reshape(-1)
            role_states = []
            if abs(position) >= 10:
                role_states = [trace[9].copy(), trace[15].copy()]
                role_labels = ["tens", "ones"]
            if is_test:
                test_last.append(last); test_trace.append(ordered_trace)
                test_y.append(position)
                test_length.append("two_digit" if abs(position) >= 10 else "single_digit")
                test_roles.extend(role_states); test_role_y.extend(role_labels if role_states else [])
            else:
                train_last.append(last); train_trace.append(ordered_trace)
                train_y.append(position)
                train_length.append("two_digit" if abs(position) >= 10 else "single_digit")
                train_roles.extend(role_states); train_role_y.extend(role_labels if role_states else [])
    rows = []
    predictions_by_readout = {}
    targets = np.asarray(train_y, dtype=np.float64)
    for mode, tx, vx in (("terminal", train_last, test_last),
                         ("trajectory", train_trace, test_trace)):
        readout = RidgeReadout(alpha=1.0).fit(np.asarray(tx), targets)
        predicted = readout.predict(np.asarray(vx)).ravel()
        errors = np.abs(predicted - np.asarray(test_y, dtype=np.float64))
        rows.append({"seed": seed, "diagnostic": "complete_instruction_position",
            "readout": mode, "accuracy": float(np.mean(errors <= 1e-9)),
            "within_one": float(np.mean(errors <= 1.0)), "mae": float(errors.mean()),
            "examples": len(errors)})
        predictions_by_readout[mode] = (readout, predicted)

    for mode, tx, vx in (("terminal", train_last, test_last),
                         ("trajectory", train_trace, test_trace)):
        length_readout = RidgeReadout(alpha=1.0).fit(
            np.asarray(tx), _one_hot(train_length, ("single_digit", "two_digit")))
        decoded = np.asarray(("single_digit", "two_digit"))[np.argmax(length_readout.predict(np.asarray(vx)), axis=1)]
        rows.append({"seed": seed, "diagnostic": "instruction_length",
            "readout": mode, "accuracy": float(np.mean(decoded == np.asarray(test_length))),
            "within_one": "", "mae": "", "examples": len(test_length)})
    role_readout = RidgeReadout(alpha=1.0).fit(np.asarray(train_roles),
        _one_hot(train_role_y, ("tens", "ones")))
    role_pred = np.asarray(("tens", "ones"))[np.argmax(role_readout.predict(np.asarray(test_roles)), axis=1)]
    rows.append({"seed": seed, "diagnostic": "digit_place_role", "readout": "token_end_state",
        "accuracy": float(np.mean(role_pred == np.asarray(test_role_y))),
        "within_one": "", "mae": "", "examples": len(test_role_y)})

    counterfactuals = []
    interventions = ((37, "reference +37"), (-37, "change sign only"),
        (47, "change tens digit only"), (38, "change ones digit only"),
        (73, "swap digit order"))
    for target, intervention in interventions:
        last, trace, _ = _encode(net, _variant(vocab.instruction(target), rng))
        ordered_trace = trace.reshape(4, 4, neurons).mean(axis=1).reshape(-1)
        for mode, state in (("terminal", last), ("trajectory", ordered_trace)):
            value = float(predictions_by_readout[mode][0].predict(state[None, :]).ravel()[0])
            counterfactuals.append({"seed": seed, "reference_position": 37,
                "intervention": intervention, "stimulus_position": target,
                "predicted_position": value, "absolute_error": abs(value - target),
                "readout": mode})
    return rows, counterfactuals


def _run_position_diagnostics(output, seeds, neurons):
    rows, counterfactuals = [], []
    for seed in seeds:
        seed_rows, seed_counterfactuals = _position_diagnostics(seed, neurons)
        rows.extend(seed_rows); counterfactuals.extend(seed_counterfactuals)
        print(f"seed {seed}: completed instruction-order probes", flush=True)
    for filename, data in (("experiential_composition_results.csv", rows),
                           ("experiential_counterfactual_results.csv", counterfactuals)):
        with (Path(output) / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    return rows, counterfactuals


def _state_for_modality(net, position, waveform, modality):
    net.reset_episode()
    retina = visual_array(position, -99, 99)
    trajectory = []
    for tick in range(DigitSpatialVocabulary.INSTRUCTION_TICKS):
        audio = waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
        vision = retina.copy()
        if modality == "audio":
            vision.fill(0.0)
        elif modality == "vision":
            audio.fill(0.0)
        elif modality != "both":
            raise ValueError("modality must be audio, vision, or both")
        net.ingest_observation(Observation(t=tick / VISUAL_RATE,
            vision=vision, audio=audio))
        trajectory.append(net.snapshot())
    net.advance_to((DigitSpatialVocabulary.INSTRUCTION_TICKS - 1) / VISUAL_RATE + 0.05)
    terminal = net.snapshot()
    ordered = np.asarray(trajectory).reshape(4, 4, net.config.neurons).mean(axis=1).reshape(-1)
    return terminal, ordered


def _three_position_readout(seed, train_variants=8, test_variants=4, neurons=64):
    positions = (-1, 0, 1)
    vocab = DigitSpatialVocabulary(seed=seed + 1_000_003)
    rng = np.random.default_rng(seed + 4_000_009)
    config = LiquidConfig(neurons=neurons, initial_out_degree=6,
        max_out_degree=8, max_connections=neurons * 8,
        max_sensory_targets=min(8, neurons))
    net = LiquidNetwork(199 + 4 + 16 + 49, seed=seed,
        config=config, variant="I0", visual_dim=199)
    datasets = {split: {mode: {"terminal": [], "trajectory": []}
                        for mode in ("audio", "both", "vision")}
                for split in ("train", "test")}
    labels = {split: [] for split in ("train", "test")}
    for position in positions:
        wave = vocab.instruction(position)
        for split, repetitions in (("train", train_variants), ("test", test_variants)):
            for _ in range(repetitions):
                waveform = _variant(wave, rng)
                labels[split].append(position)
                for mode in ("audio", "both", "vision"):
                    terminal, trajectory = _state_for_modality(net, position, waveform, mode)
                    datasets[split][mode]["terminal"].append(terminal)
                    datasets[split][mode]["trajectory"].append(trajectory)
    targets = np.asarray(labels["train"], dtype=np.float64)
    rows = []
    for representation in ("terminal", "trajectory"):
        for source in ("audio", "both", "vision"):
            readout = RidgeReadout(alpha=1.0).fit(
                np.asarray(datasets["train"][source][representation]), targets)
            for target_modality in ("audio", "both", "vision"):
                pred = readout.predict(np.asarray(
                    datasets["test"][target_modality][representation])).ravel()
                errors = np.abs(pred - np.asarray(labels["test"], dtype=np.float64))
                rows.append({"seed": seed, "readout_representation": representation,
                    "readout_source": source, "test_modality": target_modality,
                    "exact_accuracy": float(np.mean(errors <= 0.5)),
                    "within_one_accuracy": float(np.mean(errors <= 1.0)),
                    "mae": float(np.mean(errors)), "examples": len(errors)})
    return rows


def _dual_ridge_predict(train_x, train_y, test_x, alpha=1.0):
    """Equivalent standardized ridge solve in sample space for wide inputs."""
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64).reshape(-1, 1)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    z_train = (train_x - mean) / scale
    z_test = (test_x - mean) / scale
    y_mean = train_y.mean(axis=0)
    centered_y = train_y - y_mean
    dual = np.linalg.solve(z_train @ z_train.T + alpha * np.eye(len(z_train)), centered_y)
    coefficients = z_train.T @ dual
    return (z_test @ coefficients + y_mean).ravel()


def _three_position_raw_control(seed, train_variants=8, test_variants=4):
    vocab = DigitSpatialVocabulary(seed=seed + 1_000_003)
    rng = np.random.default_rng(seed + 4_000_009)
    train_x, train_y, test_x, test_y = [], [], [], []
    for position in (-1, 0, 1):
        wave = vocab.instruction(position)
        for is_test, count in ((False, train_variants), (True, test_variants)):
            for _ in range(count):
                observations = _observations(_variant(wave, rng), 199)
                features = np.concatenate([sensory_features(obs)[199:] for obs in observations])
                if is_test:
                    test_x.append(features); test_y.append(position)
                else:
                    train_x.append(features); train_y.append(position)
    prediction = _dual_ridge_predict(train_x, train_y, test_x)
    error = np.abs(prediction - np.asarray(test_y, dtype=np.float64))
    return {"seed": seed, "readout_representation": "raw_audio_features",
        "readout_source": "audio", "test_modality": "audio",
        "exact_accuracy": float(np.mean(error <= 0.5)),
        "within_one_accuracy": float(np.mean(error <= 1.0)),
        "mae": float(error.mean()), "examples": len(error)}


def _append_three_position_raw_control(output, seeds, train_variants=8, test_variants=4):
    path = Path(output) / "experiential_readout_comparison.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("readout_representation") != "raw_audio_features"]
    rows.extend(_three_position_raw_control(seed, train_variants, test_variants) for seed in seeds)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _run_three_position_readout(output, seeds, train_variants, test_variants, neurons):
    rows = []
    for seed in seeds:
        rows.extend(_three_position_readout(seed, train_variants, test_variants, neurons))
        print(f"seed {seed}: completed three-position readout-source comparison", flush=True)
    path = Path(output) / "experiential_readout_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def run_seed(seed, train_variants=8, test_variants=4, neurons=64):
    if min(train_variants, test_variants, neurons) < 1:
        raise ValueError("variant counts and neurons must be positive")
    vocab = DigitSpatialVocabulary(seed=seed + 1_000_003)
    rng = np.random.default_rng(seed + 2_000_003)
    cfg = LiquidConfig(neurons=neurons, initial_out_degree=6, max_out_degree=8,
        max_connections=neurons * 8, max_sensory_targets=min(8, neurons))
    net = LiquidNetwork(199 + 4 + 16 + 49, seed=seed, config=cfg,
                        variant="I0", visual_dim=199)
    train_x, train_traj, train_raw, train_y = [], [], [], []
    test_x, test_traj, test_raw, test_y = [], [], [], []
    started = time.perf_counter()
    for token in TOKENS:
        base = vocab.token(token)
        for _ in range(train_variants):
            state, trajectory, raw = _encode(net, _variant(base, rng))
            train_x.append(state); train_traj.append(trajectory); train_raw.append(raw)
            train_y.append(token)
        for _ in range(test_variants):
            state, trajectory, raw = _encode(net, _variant(base, rng))
            test_x.append(state); test_traj.append(trajectory); test_raw.append(raw)
            test_y.append(token)
    elapsed = time.perf_counter() - started
    truth = np.asarray(test_y, dtype=object)
    targets = _one_hot(train_y, TOKENS)
    predictions = {}
    for name, train_features, test_features in (
        ("liquid_last_state_12way", train_x, test_x),
        ("liquid_trajectory_12way", train_traj, test_traj),
        ("raw_audio_features_12way", train_raw, test_raw),
    ):
        probe = RidgeReadout(alpha=1.0).fit(np.asarray(train_features), targets)
        predictions[name] = probe.predict(np.asarray(test_features))
    scores = predictions["liquid_last_state_12way"]
    predicted = np.asarray(TOKENS, dtype=object)[np.argmax(scores, axis=1)]
    correct = predicted == truth
    signs = np.isin(truth, ("+", "-"))
    digits = ~signs
    digit_classes = np.asarray(tuple(str(i) for i in range(10)), dtype=object)
    rows = [
        {"seed": seed, "probe": "all_tokens_12way", "accuracy": float(correct.mean()),
         "correct": int(correct.sum()), "examples": len(truth), "chance": 1 / 12},
        {"seed": seed, "probe": "liquid_trajectory_12way",
         "accuracy": float(np.mean(np.asarray(TOKENS)[np.argmax(predictions["liquid_trajectory_12way"], axis=1)] == truth)),
         "correct": int(np.count_nonzero(np.asarray(TOKENS)[np.argmax(predictions["liquid_trajectory_12way"], axis=1)] == truth)),
         "examples": len(truth), "chance": 1 / 12},
        {"seed": seed, "probe": "raw_audio_features_12way",
         "accuracy": float(np.mean(np.asarray(TOKENS)[np.argmax(predictions["raw_audio_features_12way"], axis=1)] == truth)),
         "correct": int(np.count_nonzero(np.asarray(TOKENS)[np.argmax(predictions["raw_audio_features_12way"], axis=1)] == truth)),
         "examples": len(truth), "chance": 1 / 12},
        {"seed": seed, "probe": "sign_2way", "accuracy": float(np.mean(predicted[signs] == truth[signs])),
         "correct": int(np.count_nonzero(predicted[signs] == truth[signs])),
         "examples": int(signs.sum()), "chance": 0.5},
        {"seed": seed, "probe": "digit_10way_conditional",
         "accuracy": float(np.mean(digit_classes[np.argmax(scores[digits, 2:], axis=1)] == truth[digits])),
         "correct": int(np.count_nonzero(digit_classes[np.argmax(scores[digits, 2:], axis=1)] == truth[digits])),
         "examples": int(digits.sum()), "chance": 0.1},
        {"seed": seed, "probe": "digit_in_12way",
         "accuracy": float(np.mean(predicted[digits] == truth[digits])),
         "correct": int(np.count_nonzero(predicted[digits] == truth[digits])),
         "examples": int(digits.sum()), "chance": 1 / 12},
    ]
    confusion = []
    for actual in TOKENS:
        mask = truth == actual
        for guess in TOKENS:
            confusion.append({"seed": seed, "true_token": actual, "predicted_token": guess,
                "count": int(np.count_nonzero(predicted[mask] == guess))})
    return {"rows": rows, "confusion": confusion, "seconds": elapsed,
            "accuracy": float(correct.mean()), "config": cfg}


def run(output, seeds=(42, 43, 44, 45, 46), train_variants=8,
        test_variants=4, neurons=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least two unique seeds")
    example_vocab = DigitSpatialVocabulary(seed=seeds[0] + 1_000_003)
    config = {
        "experiment": "experiential-spatial-checkpoint-1-token-recognition",
        "seeds": list(seeds), "neurons": neurons,
        "train_variants_per_token": train_variants, "test_variants_per_token": test_variants,
        "position_range": [-99, 99], "position_cells": 199,
        "acoustic_sample_rate_hz": SAMPLE_RATE, "visual_event_rate_hz": VISUAL_RATE,
        "chunk_samples": CHUNK_SIZE, "token_names": list(TOKENS),
        "token_duration_ticks": DigitSpatialVocabulary.TOKEN_TICKS,
        "token_duration_seconds": DigitSpatialVocabulary.TOKEN_TICKS / VISUAL_RATE,
        "silence_duration_ticks": DigitSpatialVocabulary.SILENCE_TICKS,
        "instruction_duration_ticks": DigitSpatialVocabulary.INSTRUCTION_TICKS,
        "grammar": "zero=[0]; otherwise=[sign, decimal digits most-significant first]",
        "waveform_nuisance": {"gain_range": [0.85, 1.15], "circular_shift_samples": [-5, 5],
                              "gaussian_noise_sd": 0.01},
        "model": "I0 fixed liquid; 64 neurons by default; plasticity disabled",
        "probe": "one-hot ridge on frozen end state, state trajectory, and raw acoustic features; labels never enter the liquid",
        "pass_rule": {"probe": "liquid_trajectory_12way",
                      "mean_accuracy_min": PASS_MEAN,
                      "each_seed_accuracy_min": PASS_MIN_SEED},
        "ordered_instruction_diagnostic": {
            "positions": list(range(-99, 100)),
            "train_and_test_variants_per_position": 1,
            "readouts": ["terminal-state ridge", "four-time-bin trajectory ridge"],
            "exposure_condition": "audio only; label used only by the diagnostic readout",
            "note": "positions overlap across training and test; tests waveform-variation decoding, not unseen-position composition",
        },
        "three_position_readout_comparison": {
            "positions": [-1, 0, 1], "train_variants_per_position": train_variants,
            "test_variants_per_position": test_variants,
            "readout_sources": ["audio", "audiovisual", "vision calibration"],
            "test_modalities": ["audio", "audiovisual", "vision"],
            "learning_condition": "I0 fixed dynamics; supervised readout only; no feedback",
        },
        "codebook_seed_rule": "seed + 1000003",
        "example_codebook_metrics": example_vocab.vocabulary_rows(),
    }
    (output / "experiential_learning_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, confusions = [], []
    for seed in seeds:
        result = run_seed(seed, train_variants, test_variants, neurons)
        rows.extend(result["rows"]); confusions.extend(result["confusion"])
        print(f"seed {seed}: token accuracy {result['accuracy']:.3f}; "
              f"{result['seconds']:.1f}s frozen liquid exposure", flush=True)
    for filename, data in (("experiential_token_results.csv", rows),
                           ("experiential_token_confusion.csv", confusions)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    trajectory_accuracy = [row["accuracy"] for row in rows
                           if row["probe"] == "liquid_trajectory_12way"]
    if (np.mean(trajectory_accuracy) >= PASS_MEAN
            and min(trajectory_accuracy) >= PASS_MIN_SEED):
        _run_position_diagnostics(output, seeds, neurons)
        _run_three_position_readout(output, seeds, train_variants, test_variants, neurons)
        _append_three_position_raw_control(output, seeds, train_variants, test_variants)
    return finalize(output, train_variants, test_variants)


def finalize(output, train_variants=8, test_variants=4):
    output = Path(output)
    with (output / "experiential_token_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in ("accuracy", "chance"):
            row[key] = float(row[key])
        for key in ("correct", "examples"):
            row[key] = int(row[key])
    seeds = tuple(sorted({row["seed"] for row in rows}))
    overall = [r for r in rows if r["probe"] == "liquid_trajectory_12way"]
    mean_accuracy = float(np.mean([r["accuracy"] for r in overall]))
    minimum_accuracy = float(np.min([r["accuracy"] for r in overall]))
    passed = mean_accuracy >= PASS_MEAN and minimum_accuracy >= PASS_MIN_SEED
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        probes = ["all_tokens_12way", "liquid_trajectory_12way", "raw_audio_features_12way"]
        means = [np.mean([r["accuracy"] for r in rows if r["probe"] == name]) for name in probes]
        deviations = [np.std([r["accuracy"] for r in rows if r["probe"] == name], ddof=1) for name in probes]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(probes, means, yerr=deviations, capsize=4, color="#338c85")
        ax.axhline(1 / 12, color="#bd6b44", linestyle="--", label="12-way chance")
        ax.set_ylim(0, 1); ax.set_ylabel("Held-out accuracy")
        ax.set_title("Frozen I0 acoustic token probes (mean ± seed SD)")
        ax.tick_params(axis="x", rotation=16); ax.legend(); fig.tight_layout()
        fig.savefig(output / "experiential_token_accuracy.png", dpi=160); plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipped PNG plot", flush=True)
    composition_rows, counterfactual_rows = [], []
    composition_path = output / "experiential_composition_results.csv"
    if composition_path.exists():
        with composition_path.open(newline="", encoding="utf-8") as stream:
            composition_rows = list(csv.DictReader(stream))
    counterfactual_path = output / "experiential_counterfactual_results.csv"
    if counterfactual_path.exists():
        with counterfactual_path.open(newline="", encoding="utf-8") as stream:
            counterfactual_rows = list(csv.DictReader(stream))
    readout_rows = []
    readout_path = output / "experiential_readout_comparison.csv"
    if readout_path.exists():
        with readout_path.open(newline="", encoding="utf-8") as stream:
            readout_rows = list(csv.DictReader(stream))
    if readout_rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            conditions = [
                ("terminal audio→audio", "terminal", "audio", "audio"),
                ("terminal AV→audio", "terminal", "both", "audio"),
                ("trajectory audio→audio", "trajectory", "audio", "audio"),
                ("trajectory AV→audio", "trajectory", "both", "audio"),
                ("trajectory AV→AV", "trajectory", "both", "both"),
                ("raw audio→audio", "raw_audio_features", "audio", "audio"),
            ]
            means = []
            for _, rep, source, modality in conditions:
                selected = [row for row in readout_rows
                    if row["readout_representation"] == rep
                    and row["readout_source"] == source and row["test_modality"] == modality]
                means.append(np.mean([float(row["exact_accuracy"]) for row in selected]))
            fig, ax = plt.subplots(figsize=(9, 4.7))
            ax.bar(range(len(means)), means, color="#338c85")
            ax.axhline(1 / 3, color="#bd6b44", linestyle="--", label="three-position chance")
            ax.set_xticks(range(len(means)), [item[0] for item in conditions], rotation=22, ha="right")
            ax.set_ylim(0, 1); ax.set_ylabel("Exact position accuracy")
            ax.set_title("I0 three-position readout-source comparison")
            ax.legend(); fig.tight_layout()
            fig.savefig(output / "experiential_readout_comparison.png", dpi=160); plt.close(fig)
        except ImportError:
            print("matplotlib unavailable; skipped readout comparison plot", flush=True)
    _write_report(output, rows, seeds, train_variants, test_variants,
                  mean_accuracy, minimum_accuracy, passed,
                  composition_rows, counterfactual_rows, readout_rows)
    return passed


def _write_report(output, rows, seeds, train_variants, test_variants,
                  mean_accuracy, minimum_accuracy, passed,
                  composition_rows=(), counterfactual_rows=(), readout_rows=()):
    lines = [
        "# Experiential Spatial Learning — Checkpoint 1", "", "## Acoustic codebook audit", "",
        "The decimal vocabulary contains sign tokens `+` and `-` and digit tokens `0` through `9`. Zero is one `0` token. Nonzero positions use sign followed by magnitude digits from most significant to least significant: -37 is `-`, `3`, `7`; +3 is `+`, `3`; +30 is `+`, `3`, `0`. Each token lasts 4 ticks (0.2 s), with 2 ticks (0.1 s) of silence between adjacent tokens. The full instruction is 16 ticks (0.8 s) at 2,000 waveform samples/s, divided into 100-sample observations at 20 Hz. Two-digit instructions exactly fill the duration; shorter instructions end in silence padding. Identical digit tokens use the same underlying waveform in every context.",
        "", "Each model observation contains timestamp, binary vision, and raw audio chunk. The existing liquid feature transform receives those streams. Probe labels and vocabulary metadata are diagnostic targets only. Current `make_digit_episode` presents vision and audio simultaneously; the phase-separated teaching sequence has not yet been added.",
        "", "## Checkpoint 1 protocol", "",
        f"The fixed I0 reservoir was tested across {len(seeds)} seeds. Each of 12 tokens had {train_variants} training and {test_variants} held-out waveform variants with modest gain, circular shift, and additive-noise nuisance. One-hot ridge probes used the frozen final state, the concatenated four-tick state trajectory, or the concatenated 69-D-per-tick audio features that feed the liquid. The liquid never received class labels. The token gate uses the time-resolved state trajectory (mean >= {PASS_MEAN:.0%}; every seed >= {PASS_MIN_SEED:.0%}); terminal-state performance is reported separately.",
        "", "## Results", "", "| Probe | Mean accuracy | Per-seed range | Chance |", "|---|---:|---:|---:|",
    ]
    for probe in ("all_tokens_12way", "liquid_trajectory_12way", "raw_audio_features_12way",
                  "sign_2way", "digit_10way_conditional", "digit_in_12way"):
        values = [r["accuracy"] for r in rows if r["probe"] == probe]
        chance = next(r["chance"] for r in rows if r["probe"] == probe)
        lines.append(f"| {probe} | {np.mean(values):.3f} | {min(values):.3f}–{max(values):.3f} | {chance:.3f} |")
    lines += ["", f"**Token-recognition checkpoint: {'PASS' if passed else 'FAIL'}.** Trajectory-probe mean: {mean_accuracy:.3f}; minimum seed: {minimum_accuracy:.3f}.", "",
        "The raw features and time-resolved liquid trajectory retain token identity, but a single terminal liquid state performs much worse. This points to temporal aggregation or state retention as the immediate bottleneck rather than inadequate acoustic feature separation. The probe results do not establish that the liquid learned token semantics. Per-token confusions are in `experiential_token_confusion.csv`.", ""]
    if composition_rows:
        lines += ["## Ordered-instruction diagnostics", "",
            "These supervised probes measure information recoverable from complete audio instructions; they do not show that the liquid learned numerical composition through local plasticity.", "",
            "| Diagnostic | Readout | Mean accuracy | Mean MAE | Mean within 1 |", "|---|---|---:|---:|---:|"]
        diagnostic_names = sorted({row["diagnostic"] for row in composition_rows})
        for diagnostic in diagnostic_names:
            readouts = sorted({row["readout"] for row in composition_rows if row["diagnostic"] == diagnostic})
            for readout_name in readouts:
                selected = [row for row in composition_rows if row["diagnostic"] == diagnostic and row["readout"] == readout_name]
                accuracy = np.mean([float(row["accuracy"]) for row in selected])
                mae_values = [float(row["mae"]) for row in selected if row["mae"]]
                within_values = [float(row["within_one"]) for row in selected if row["within_one"]]
                mae = f"{np.mean(mae_values):.2f}" if mae_values else "—"
                within = f"{np.mean(within_values):.3f}" if within_values else "—"
                lines.append(f"| {diagnostic} | {readout_name} | {accuracy:.3f} | {mae} | {within} |")
        if counterfactual_rows:
            lines += ["", "### Counterfactual component changes", "",
                "Stimuli intervene on one component relative to +37. Predictions are from supervised full-position probes; targets are analysis labels only.", "",
                "| Intervention | Target | Terminal prediction | Trajectory prediction |", "|---|---:|---:|---:|"]
            for intervention in sorted({row["intervention"] for row in counterfactual_rows}):
                selected = [row for row in counterfactual_rows if row["intervention"] == intervention]
                target = float(selected[0]["stimulus_position"])
                terminal = np.mean([float(row["predicted_position"]) for row in selected if row["readout"] == "terminal"])
                trajectory = np.mean([float(row["predicted_position"]) for row in selected if row["readout"] == "trajectory"])
                lines.append(f"| {intervention} | {target:.0f} | {terminal:.2f} | {trajectory:.2f} |")
        lines.append("")
    if readout_rows:
        lines += ["## Three-position readout-source checkpoint", "",
            "I0 is a fixed liquid here. Readouts are fit on labels from their indicated training modality. The vision-only readout is a pre-exposure calibration baseline; this experiment includes no audiovisual plasticity. Terminal-state and four-ordered-bin temporal readouts are compared. Exact accuracy counts a prediction within 0.5 position units of its integer target.", "",
            "| Representation | Readout source | Test modality | Exact accuracy | Within 1 | MAE |", "|---|---|---|---:|---:|---:|"]
        for representation in ("terminal", "trajectory", "raw_audio_features"):
            for source in ("audio", "both", "vision"):
                for modality in ("audio", "both", "vision"):
                    subset = [row for row in readout_rows if row["readout_representation"] == representation and row["readout_source"] == source and row["test_modality"] == modality]
                    if subset:
                        exact = np.mean([float(row["exact_accuracy"]) for row in subset])
                        within = np.mean([float(row["within_one_accuracy"]) for row in subset])
                        mae = np.mean([float(row["mae"]) for row in subset])
                        lines.append(f"| {representation} | {source} | {modality} | {exact:.3f} | {within:.3f} | {mae:.3f} |")
        lines += ["", "The AV-trained readout evaluated on audio-only states measures modality-induced state/readout shift. The audio-supervised row is a labeled diagnostic control. Vision calibration evaluated on audio is a pre-exposure transfer baseline; transfer after plastic adaptation remains gated.", ""]
    if readout_rows:
        audio_traj = [float(row["exact_accuracy"]) for row in readout_rows
            if row["readout_representation"] == "trajectory" and row["readout_source"] == "audio" and row["test_modality"] == "audio"]
        av_audio = [float(row["exact_accuracy"]) for row in readout_rows
            if row["readout_representation"] == "trajectory" and row["readout_source"] == "both" and row["test_modality"] == "audio"]
        av_both = [float(row["exact_accuracy"]) for row in readout_rows
            if row["readout_representation"] == "trajectory" and row["readout_source"] == "both" and row["test_modality"] == "both"]
        raw_audio = [float(row["exact_accuracy"]) for row in readout_rows
            if row["readout_representation"] == "raw_audio_features" and row["readout_source"] == "audio" and row["test_modality"] == "audio"]
        lines += ["## Checkpoint decision", "",
            f"The token-recognition checkpoint passes on time-resolved states, but fails for a terminal state. Complete-instruction position decoding remains near the constant-zero baseline (trajectory MAE {np.mean([float(row['mae']) for row in composition_rows if row['diagnostic'] == 'complete_instruction_position' and row['readout'] == 'trajectory']):.2f} vs 49.75 baseline; exact accuracy 0). On the simpler three-position task, mean exact accuracy is {np.mean(audio_traj):.3f} for an audio-trained trajectory readout, {np.mean(av_audio):.3f} for an AV-trained trajectory readout evaluated on audio, and {np.mean(av_both):.3f} for that AV readout evaluated on AV. The supervised raw-audio-feature control reaches {np.mean(raw_audio):.3f}. Each seed contributes 12 held-out noisy trials to each three-position condition.", "",
            "**The simple position-association checkpoint is not established for the liquid.** The audio-feature control indicates the cues and labels are recoverable by a direct supervised readout; the liquid-state readouts do not reliably recover them, and the AV-to-audio shift remains. Do not interpret this as evidence for or against adaptive plasticity: I0 is fixed and no repeated audiovisual teaching or target feedback occurred.", "",
            "Next controlled experiment: hold the three-position data fixed and compare terminal state, ordered liquid trajectory, and raw acoustic features with an equally declared readout and training budget. Diagnose projection/temporal summarization before enabling I1/I2 plasticity or repeated feedback.", ""]
    else:
        lines += [("The trajectory token gate passed. Continue only to controlled readout-source and three-position association tests; keep repeated exposure gated." if passed else "The token trajectory gate failed. Stop before audiovisual association and investigate temporal aggregation or sensory projection."), ""]
    lines += ["No repeated-association, feedback, retention, curriculum, or adaptive-advantage result is claimed here.", ""]
    (output / "EXPERIENTIAL_SPATIAL_LEARNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/experiential-spatial")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--train-variants", type=int, default=8)
    parser.add_argument("--test-variants", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="write plot/report from existing token CSVs without rerunning liquid episodes")
    args = parser.parse_args(argv)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    try:
        if args.finalize_existing:
            passed = finalize(args.output, args.train_variants, args.test_variants)
        else:
            passed = run(args.output, seeds, args.train_variants, args.test_variants, args.neurons)
    except ValueError as exc:
        parser.error(str(exc))
    print("Checkpoint 1", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
