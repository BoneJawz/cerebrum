"""Controlled acoustic-to-visual multiple-choice study for Model I.

Coordinates and answer indices live only in the experiment controller.  Every
liquid observation contains a timestamp, binary retina, and waveform chunk.
The fixed matcher receives two liquid vectors and never a coordinate or label.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from .contracts import Observation
from .experiential_tokens import _variant
from .liquid import LiquidConfig, LiquidNetwork
from .processors import sensory_features
from .spatial_environment import (
    CHUNK_SIZE, SAMPLE_RATE, VISUAL_RATE, DigitSpatialVocabulary, visual_array,
)


POSITIONS = (-1, 0, 1)
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("I0", "I1", "I2", "I2_rewire_fixed")
EXPOSURES = (0, 1, 2, 5, 10, 20, 50)
PAIRINGS = ("correct", "mismatched", "audio_only", "vision_only")
DELAYS = (0.0, 0.1, 0.5, 1.0, 5.0)
FIELD_CELLS = 199
TICK_SECONDS = 1.0 / VISUAL_RATE


@dataclass(frozen=True)
class RecognitionConfig:
    seeds: tuple[int, ...] = SEEDS
    exposures: tuple[int, ...] = EXPOSURES
    variants: tuple[str, ...] = VARIANTS
    pairings: tuple[str, ...] = PAIRINGS
    neurons: int = 24
    calibration_variants: int = 6
    test_variants: int = 3
    candidate_ticks: int = 4
    temperature: float = 1.0
    projection: str = "P0"
    output: str = "results/recognition-learning"

    def __post_init__(self):
        if len(self.seeds) < 1 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.exposures[0] != 0 or any(a >= b for a, b in zip(self.exposures, self.exposures[1:])):
            raise ValueError("exposures must start at zero and increase")
        if any(v not in VARIANTS for v in self.variants):
            raise ValueError("unknown liquid variant")
        if any(p not in PAIRINGS for p in self.pairings):
            raise ValueError("unknown exposure pairing")
        if self.calibration_variants < 2 or self.test_variants < 1:
            raise ValueError("invalid calibration or test size")
        if self.projection not in ("P0", "P1"):
            raise ValueError("projection must be P0 or P1")


def liquid_config(neurons: int) -> LiquidConfig:
    """Same equations, smaller CPU profile for all four matched variants."""
    return LiquidConfig(neurons=neurons, initial_out_degree=3, max_out_degree=5,
                        max_connections=neurons * 5, max_sensory_targets=min(8, neurons),
                        candidate_pairs_per_update=32)


def _new_network(seed, variant, cfg):
    net = LiquidNetwork(268, seed=seed, config=liquid_config(cfg.neurons),
                        variant=variant, visual_dim=FIELD_CELLS)
    if cfg.projection == "P1":
        # Global, label-free floor for *all* visual cells, then match P0's
        # total visual projection energy. Graph, equations, and audio stay fixed.
        gains = net._input_gains[:FIELD_CELLS]
        original_energy = float(np.sum(gains ** 2))
        adjusted = np.where(gains < 0, -1.0, 1.0) * np.maximum(np.abs(gains), 0.05)
        net._input_gains[:FIELD_CELLS] = adjusted * np.sqrt(original_energy / np.sum(adjusted ** 2))
    return net


def _input_observations(vocab, position, modality, rng=None, base_time=0.0):
    """The position is used only by the controller to construct raw stimuli."""
    waveform = vocab.instruction(position)
    if rng is not None:
        waveform = _variant(waveform, rng)
    retina = visual_array(position, -99, 99)
    observations = []
    for tick in range(DigitSpatialVocabulary.INSTRUCTION_TICKS):
        audio = waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
        vision = retina.copy()
        if modality == "audio":
            vision.fill(0.0)
        elif modality == "vision":
            audio.fill(0.0)
        elif modality != "both":
            raise ValueError(modality)
        observations.append(Observation(t=base_time + tick * TICK_SECONDS,
                                        vision=vision, audio=audio))
    return observations


def _teach(net, vocab, cue_position, pairing, rng):
    """One counted audiovisual or single-modality sensory experience."""
    if pairing == "correct":
        shown = cue_position
        modality = "both"
    elif pairing == "mismatched":
        # A seed-controlled derangement preserves visual position frequencies.
        shift = 1 + (net.seed % 2)
        shown = POSITIONS[(POSITIONS.index(cue_position) + shift) % len(POSITIONS)]
        modality = "both"
    elif pairing == "audio_only":
        shown = cue_position
        modality = "audio"
    elif pairing == "vision_only":
        shown = cue_position
        modality = "vision"
    else:
        raise ValueError(pairing)
    base = net.time_seconds + TICK_SECONDS
    # A mismatched pairing still uses the cue waveform for the intended cue,
    # with a separately constructed binary retina for the shown position.
    observations = _input_observations(vocab, cue_position, "audio", rng, base)
    visual = visual_array(shown, -99, 99) if modality != "audio" else np.zeros(FIELD_CELLS)
    for obs in observations:
        sound = obs.audio if modality != "vision" else np.zeros(CHUNK_SIZE)
        net.ingest_observation(Observation(t=obs.t, vision=visual.copy(), audio=sound))
    net.advance_to(base + DigitSpatialVocabulary.INSTRUCTION_TICKS * TICK_SECONDS)
    return shown


def _audio_assessment(net, waveform):
    """Return a frozen post-cue snapshot and fixed causal mean-state summary."""
    probe = LiquidNetwork.from_state_dict(net.state_dict())
    probe.weight_plasticity = False
    probe.structural_plasticity = False
    base = probe.time_seconds + TICK_SECONDS
    states = []
    for tick in range(DigitSpatialVocabulary.INSTRUCTION_TICKS):
        chunk = waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
        probe.ingest_observation(Observation(t=base + tick * TICK_SECONDS,
            vision=np.zeros(FIELD_CELLS), audio=chunk))
        states.append(probe.snapshot())
    probe.advance_to(base + DigitSpatialVocabulary.INSTRUCTION_TICKS * TICK_SECONDS)
    acoustic_summary = np.mean(states, axis=0)
    return probe.state_dict(), acoustic_summary


def _visual_branch(snapshot, visual, candidate_ticks):
    """Advance an identical frozen clone for every visual candidate."""
    branch = LiquidNetwork.from_state_dict(snapshot)
    branch.weight_plasticity = False
    branch.structural_plasticity = False
    base = branch.time_seconds + TICK_SECONDS
    for tick in range(candidate_ticks):
        branch.ingest_observation(Observation(t=base + tick * TICK_SECONDS,
            vision=np.asarray(visual, dtype=np.float64).copy(),
            audio=np.zeros(CHUNK_SIZE)))
    end = base + candidate_ticks * TICK_SECONDS
    branch.advance_to(end)
    return branch.snapshot(), branch


def _branch_features(snapshot, acoustic_summary, visuals, candidate_ticks):
    blank, _ = _visual_branch(snapshot, np.zeros(FIELD_CELLS), candidate_ticks)
    features = []
    event_counts = []
    queue_peaks = []
    for visual in visuals:
        response, branch = _visual_branch(snapshot, visual, candidate_ticks)
        features.append(acoustic_summary * (response - blank))
        event_counts.append(branch.stats().processed_events - snapshot["stats"]["processed_events"])
        queue_peaks.append(branch.pending_events)
    return np.asarray(features), event_counts, queue_peaks


class CandidateMatcher:
    """Replaceable fixed comparison rule; scoring accepts features only."""

    def __init__(self, weights=None, bias=0.0):
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64).copy()
        self.bias = float(bias)

    def score(self, pair_features):
        values = np.asarray(pair_features, dtype=np.float64)
        if self.weights is None:
            # Fixed diagonal cross-modal similarity. No fitted mapping.
            return np.sum(values, axis=1)
        return values @ self.weights + self.bias


def _softmax(scores, temperature):
    z = (np.asarray(scores) - np.max(scores)) / temperature
    e = np.exp(z)
    return e / np.sum(e)


def _stable_argmax(scores, tie_keys):
    """Break exact score ties by stimulus bytes, independent of display order."""
    highest = np.max(scores)
    choices = np.flatnonzero(np.asarray(scores) == highest)
    return int(min(choices, key=lambda i: tie_keys[int(i)]))


def _candidate_order(candidates, seed, trial_index):
    """A seeded rotation counterbalances every candidate across display slots."""
    values = tuple(candidates)
    rng = np.random.default_rng(seed)
    base = tuple(np.asarray(values)[rng.permutation(len(values))])
    offset = trial_index % len(values)
    return base[offset:] + base[:offset]


def _recognize(net, waveform, candidates, matcher, cfg, seed, trial_index,
               stochastic=False):
    """Assessment receives raw cues and candidate positions, never the answer."""
    start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
    snapshot, acoustic = _audio_assessment(net, waveform)
    ordered = _candidate_order(candidates, seed, trial_index)
    visuals = [visual_array(int(position), -99, 99) for position in ordered]
    tie_keys = tuple(hashlib.sha256(visual.tobytes()).hexdigest() for visual in visuals)
    features, event_counts, queue_peaks = _branch_features(
        snapshot, acoustic, visuals, cfg.candidate_ticks)
    scores = matcher.score(features)
    probabilities = _softmax(scores, cfg.temperature)
    if stochastic:
        rng = np.random.default_rng(seed + 13_133 * (trial_index + 1))
        choice = int(rng.choice(len(ordered), p=probabilities))
    else:
        choice = _stable_argmax(scores, tie_keys)
    margin = float(np.sort(scores)[-1] - np.sort(scores)[-2])
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))
    elapsed = {"wall_ms": (time.perf_counter_ns() - start_wall) / 1e6,
               "cpu_ms": (time.process_time_ns() - start_cpu) / 1e6,
               "candidate_processed_events": int(sum(event_counts)),
               "peak_queue_observed": int(max(queue_peaks, default=0)),
               "assessment_simulation_seconds": float(snapshot["time_seconds"]),
               "candidate_simulation_seconds": cfg.candidate_ticks * TICK_SECONDS}
    result = {"choice_index": choice, "candidate_order": ordered,
              "scores": scores, "probabilities": probabilities,
              "score_margin": margin, "choice_entropy": entropy,
              "tie_keys": tie_keys,
              "features": features, "snapshot": snapshot, **elapsed}
    return result


def _calibrate_matcher(seed, vocab, cfg):
    """Supervised diagnostic comparator, fitted before and frozen during exposure.

    It uses separate noisy waveform realizations and every audio/visual pairing.
    This scorer can itself learn the three concepts, so its accuracy cannot be
    attributed to liquid plasticity.
    """
    net = _new_network(seed, "I0", cfg)
    rng = np.random.default_rng(seed + 110_003)
    features, targets = [], []
    for position in POSITIONS:
        for _ in range(cfg.calibration_variants):
            wave = _variant(vocab.instruction(position), rng)
            snapshot, acoustic = _audio_assessment(net, wave)
            rows, _, _ = _branch_features(snapshot, acoustic,
                [visual_array(candidate, -99, 99) for candidate in POSITIONS], cfg.candidate_ticks)
            features.extend(rows)
            targets.extend(float(candidate == position) for candidate in POSITIONS)
    x = np.asarray(features)
    y = np.asarray(targets)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    z = (x - mean) / scale
    alpha = 1.0
    centered = y - y.mean()
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ centered)
    weights = coef / scale
    bias = float(y.mean() - mean @ weights)
    return CandidateMatcher(weights, bias), {
        "weights": weights, "bias": bias, "normalization_mean": mean,
        "normalization_scale": scale, "alpha": alpha,
        "labeled_calibration_pairs": len(y), "optimizer_updates": 1}


def _test_waveforms(seed, vocab, variants):
    rng = np.random.default_rng(seed + 220_007)
    return [(position, _variant(vocab.instruction(position), rng), replicate)
            for position in POSITIONS for replicate in range(variants)]


def _exposure_order(seed, count):
    rng = np.random.default_rng(seed + 330_017)
    order = []
    while len(order) < count:
        order.extend(np.asarray(POSITIONS)[rng.permutation(3)].tolist())
    return order[:count]


def _all_scores(result, matcher, temperature):
    scores = matcher.score(result["features"])
    probabilities = _softmax(scores, temperature)
    choice = _stable_argmax(scores, result["tie_keys"])
    return scores, probabilities, choice


def _frozen_view(net, mode, delay=0.0, interference=None, vocab=None, rng=None):
    probe = LiquidNetwork.from_state_dict(net.state_dict())
    probe.weight_plasticity = False
    probe.structural_plasticity = False
    if mode == "independent":
        probe.reset_episode()
    elif mode != "continuous":
        raise ValueError(mode)
    if interference is not None:
        # Unrelated raw audiovisual input, with no plasticity and no feedback.
        _teach(probe, vocab, interference, "correct", rng)
    if delay:
        probe.advance_to(probe.time_seconds + delay)
    return probe


def _assess_set(net, test_waveforms, cfg, seed, variant, pairing, exposures,
                matchers, mode, choice_count=3, delay=0.0, interference=None):
    trial_rows, score_rows, compute_rows = [], [], []
    for index, (position, waveform, replicate) in enumerate(test_waveforms):
        # The test set has no feedback. Every trial starts from the same learned
        # checkpoint, so test order cannot change later responses.
        interference_position = (POSITIONS[(POSITIONS.index(position) + 1) % 3]
                                 if interference is not None else None)
        probe = _frozen_view(net, mode, delay, interference_position,
            None if interference is None else DigitSpatialVocabulary(seed + 1_000_003),
            np.random.default_rng(seed + 540_011 + index))
        if choice_count == 3:
            candidates = POSITIONS
        else:
            others = [p for p in POSITIONS if p != position]
            candidates = (position, others[replicate % len(others)])
        memory_probe = index == 0 and pairing == "correct" and choice_count == 3 and exposures == cfg.exposures[-1] and delay == 0
        if memory_probe:
            tracemalloc.start()
        result = _recognize(probe, waveform, candidates, matchers["fixed"],
            cfg, seed + 7001, index)
        traced_peak_kib = ""
        if memory_probe:
            traced_peak_kib = tracemalloc.get_traced_memory()[1] / 1024.0
            tracemalloc.stop()
        for matcher_name, matcher in matchers.items():
            scores, probabilities, choice = _all_scores(result, matcher, cfg.temperature)
            chosen = result["candidate_order"][choice]
            truth_index = result["candidate_order"].index(position)
            stochastic_rng = np.random.default_rng(seed + 910_003 + 37 * index +
                (0 if matcher_name == "fixed" else 1))
            sampled_choice = int(stochastic_rng.choice(len(candidates), p=probabilities))
            row = {"seed": seed, "variant": variant, "pairing": pairing,
                "exposures": exposures, "matcher": matcher_name, "mode": mode,
                "delay_seconds": delay, "interference": interference is not None,
                "choice_count": choice_count, "difficulty": "three_position_standard" if choice_count == 3 else "two_position_standard",
                "trial_id": f"test_{position}_{replicate}", "position": position,
                "correct_index": truth_index, "selected_index": choice,
                "selected_position": int(chosen), "correct": int(chosen == position),
                "stochastic_selected_index": sampled_choice,
                "stochastic_correct": int(sampled_choice == truth_index),
                "score_margin": float(np.sort(scores)[-1] - np.sort(scores)[-2]),
                "choice_entropy": float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)))),
                "stochastic_expected_accuracy": float(probabilities[truth_index]),
                "candidate_order": "/".join(map(str, result["candidate_order"]))}
            trial_rows.append(row)
            for slot, candidate in enumerate(result["candidate_order"]):
                score_rows.append({**{key: row[key] for key in (
                    "seed", "variant", "pairing", "exposures", "matcher", "mode",
                    "delay_seconds", "interference", "choice_count", "trial_id")},
                    "candidate_index": slot, "candidate_position": int(candidate),
                    "score": float(scores[slot]), "choice_weight": float(probabilities[slot]),
                    "is_correct": int(slot == truth_index)})
        compute_rows.append({"seed": seed, "variant": variant, "pairing": pairing,
            "exposures": exposures, "mode": mode, "choice_count": choice_count,
            "trial_id": f"test_{position}_{replicate}",
            "traced_python_peak_kib": traced_peak_kib,
            "memory_instrumented": memory_probe,
            **{key: result[key] for key in ("wall_ms", "cpu_ms", "candidate_processed_events",
                                          "peak_queue_observed", "assessment_simulation_seconds",
                                          "candidate_simulation_seconds")}})
    return trial_rows, score_rows, compute_rows


def _weights(net):
    return {(edge.source, edge.destination): edge.weight
            for row in net.adjacency for edge in row.values()}


def _plasticity_row(net, initial_weights, seed, variant, pairing, exposures,
                    learning_wall_ms, learning_cpu_ms, baseline_stats):
    now = _weights(net)
    shared = set(now) & set(initial_weights)
    changes = [abs(now[edge] - initial_weights[edge]) for edge in shared]
    stats = asdict(net.stats())
    topology = net.topology_summary()
    return {"seed": seed, "variant": variant, "pairing": pairing,
        "exposures": exposures, "weight_updates": stats["weight_updates"] - baseline_stats["weight_updates"],
        "formed_connections": stats["formed_connections"] - baseline_stats["formed_connections"],
        "pruned_connections": stats["pruned_connections"] - baseline_stats["pruned_connections"],
        "mean_abs_shared_weight_change": float(np.mean(changes)) if changes else 0.0,
        "active_connections": topology["active_connections"],
        "mean_out_degree": topology["mean_out_degree"],
        "active_neurons": topology["active_neurons"],
        "processed_events": stats["processed_events"] - baseline_stats["processed_events"],
        "propagated_events": stats["propagated_events"] - baseline_stats["propagated_events"],
        "neuron_updates": stats["neuron_updates"] - baseline_stats["neuron_updates"],
        "learning_wall_ms": learning_wall_ms, "learning_cpu_ms": learning_cpu_ms}


def _write_csv(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_numeric_csv(path, integer=(), floating=(), boolean=()):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in integer:
            if row.get(key) not in ("", None): row[key] = int(row[key])
        for key in floating:
            if row.get(key) not in ("", None): row[key] = float(row[key])
        for key in boolean:
            if row.get(key) not in ("", None): row[key] = row[key] == "True"
    return rows


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item)),
        encoding="utf-8")


def reconstruct_artifacts(cfg: RecognitionConfig):
    """Reproduce held-out partitions and post-learning acoustic state traces."""
    output = Path(cfg.output)
    partitions, trajectories, projection_rows, experience_rows = [], [], [], []
    for seed in cfg.seeds:
        vocab = DigitSpatialVocabulary(seed + 1_000_003)
        interface = _new_network(seed, "I0", cfg)
        for position in POSITIONS:
            feature_index = position + 99
            gain = float(interface._input_gains[feature_index])
            projection_rows.append({"seed": seed, "projection": cfg.projection,
                "position": position, "visual_feature_index": feature_index,
                "target_neuron": int(interface._input_targets[feature_index]),
                "signed_gain": gain, "absolute_gain": abs(gain),
                "resting_threshold": interface.config.resting_threshold,
                "passes_sensory_threshold": int(abs(gain) > interface.config.resting_threshold),
                "total_visual_projection_energy": float(np.sum(interface._input_gains[:FIELD_CELLS] ** 2))})
        test = _test_waveforms(seed, vocab, cfg.test_variants)
        for position, _, replicate in test:
            partitions.append({"seed": seed, "split": "test", "position": position,
                               "replicate": replicate, "trial_id": f"test_{position}_{replicate}"})
        for position in POSITIONS:
            for replicate in range(cfg.calibration_variants):
                partitions.append({"seed": seed, "split": "comparator_calibration",
                                   "position": position, "replicate": replicate,
                                   "trial_id": f"calibration_{position}_{replicate}"})
        for exposure, position in enumerate(_exposure_order(seed, cfg.exposures[-1])):
            partitions.append({"seed": seed, "split": "experience",
                               "position": position, "replicate": exposure,
                               "trial_id": f"experience_{exposure}"})
            for variant in cfg.variants:
                for pairing in cfg.pairings:
                    shown = (POSITIONS[(POSITIONS.index(position) + 1 + (seed % 2)) % 3]
                             if pairing == "mismatched" else position)
                    experience_rows.append({"seed": seed, "variant": variant,
                        "pairing": pairing, "experience_index": exposure,
                        "experience_count_after": exposure + 1,
                        "preceding_prediction_trial": (f"online_{exposure}" if pairing == "correct" else ""),
                        "cue_position_controller_only": position,
                        "shown_visual_position_controller_only": (shown if pairing != "audio_only" else ""),
                        "audio_present": pairing != "vision_only",
                        "vision_present": pairing != "audio_only",
                        "phase": "feedback_after_recorded_choice" if pairing == "correct" else "control_exposure"})
        for variant in cfg.variants:
            for pairing in cfg.pairings:
                path = output / "model_checkpoints" / f"seed_{seed}_{variant}_{pairing}_after_{cfg.exposures[-1]}.json"
                if not path.exists():
                    continue
                learned = LiquidNetwork.from_state_dict(json.loads(path.read_text(encoding="utf-8")))
                learned.weight_plasticity = False
                learned.structural_plasticity = False
                for position, waveform, replicate in test:
                    if replicate != 0:
                        continue
                    probe = LiquidNetwork.from_state_dict(learned.state_dict())
                    probe.reset_episode()
                    before = asdict(probe.stats())
                    for tick in range(DigitSpatialVocabulary.INSTRUCTION_TICKS):
                        chunk = waveform[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE].copy()
                        probe.ingest_observation(Observation(t=tick * TICK_SECONDS,
                            vision=np.zeros(FIELD_CELLS), audio=chunk))
                        state = probe.snapshot()
                        stats = asdict(probe.stats())
                        trajectories.append({"seed": seed, "variant": variant,
                            "pairing": pairing, "exposures": cfg.exposures[-1],
                            "position": position, "tick": tick,
                            "simulation_seconds": tick * TICK_SECONDS,
                            "state_norm": float(np.linalg.norm(state)),
                            "active_fraction": float(np.mean(np.abs(state) > probe.config.resting_threshold)),
                            "mean_abs_activation": float(np.mean(np.abs(state))),
                            "propagated_events_cumulative": stats["propagated_events"] - before["propagated_events"],
                            "processed_events_cumulative": stats["processed_events"] - before["processed_events"],
                            "queue_events_observed": probe.pending_events})
    _write_csv(output / "recognition_partitions.csv", partitions)
    _write_csv(output / "recognition_experience_log.csv", experience_rows)
    _write_csv(output / "recognition_state_trajectories.csv", trajectories)
    _write_csv(output / "recognition_projection_audit.csv", projection_rows)
    score_path = output / "recognition_candidate_scores.csv"
    if score_path.exists():
        with score_path.open(newline="", encoding="utf-8") as stream:
            raw_scores = list(csv.DictReader(stream))
        lookup = {(int(row["seed"]), row["variant"], row["pairing"],
                   row["trial_id"], int(row["candidate_position"])): float(row["score"])
                  for row in raw_scores if row["matcher"] == "fixed" and
                  row["mode"] == "independent" and row["choice_count"] == "3" and
                  row["exposures"] == str(cfg.exposures[-1]) and
                  row["delay_seconds"] == "0.0" and row["interference"] == "False"}
        paired_rows = []
        for seed in cfg.seeds:
            for variant in cfg.variants:
                correct_path = output / "model_checkpoints" / f"seed_{seed}_{variant}_correct_after_{cfg.exposures[-1]}.json"
                if not correct_path.exists(): continue
                correct = json.loads(correct_path.read_text(encoding="utf-8"))
                correct_edges = {(e["source"], e["destination"]): e["weight"] for e in correct["edges"]}
                for control in ("mismatched", "audio_only", "vision_only"):
                    control_path = output / "model_checkpoints" / f"seed_{seed}_{variant}_{control}_after_{cfg.exposures[-1]}.json"
                    if not control_path.exists(): continue
                    other = json.loads(control_path.read_text(encoding="utf-8"))
                    other_edges = {(e["source"], e["destination"]): e["weight"] for e in other["edges"]}
                    common = set(correct_edges) & set(other_edges)
                    differences = [abs(value - lookup[(seed, variant, control, trial_id, candidate)])
                        for (s, v, pairing, trial_id, candidate), value in lookup.items()
                        if s == seed and v == variant and pairing == "correct" and
                        (seed, variant, control, trial_id, candidate) in lookup]
                    paired_rows.append({"seed": seed, "variant": variant,
                        "control": control, "shared_edges": len(common),
                        "different_edge_count": len(set(correct_edges) ^ set(other_edges)),
                        "mean_abs_shared_weight_difference": (float(np.mean([
                            abs(correct_edges[k] - other_edges[k]) for k in common])) if common else ""),
                        "mean_abs_candidate_score_difference": (float(np.mean(differences)) if differences else ""),
                        "matched_candidate_scores": len(differences)})
        _write_csv(output / "recognition_pairing_diagnostics.csv", paired_rows,
            ["seed", "variant", "control", "shared_edges", "different_edge_count",
             "mean_abs_shared_weight_difference", "mean_abs_candidate_score_difference",
             "matched_candidate_scores"])


def finalize_existing(cfg: RecognitionConfig):
    """Refresh analysis/figures from an already completed raw run, without retraining."""
    output = Path(cfg.output)
    trial_rows = _read_numeric_csv(output / "recognition_trial_results.csv",
        integer=("seed", "exposures", "choice_count", "position", "selected_position",
                 "correct", "stochastic_correct"),
        floating=("delay_seconds", "score_margin", "choice_entropy", "stochastic_expected_accuracy"),
        boolean=("interference",))
    compute = _read_numeric_csv(output / "recognition_compute_results.csv",
        integer=("seed", "exposures", "choice_count"),
        floating=("cpu_ms", "wall_ms", "traced_python_peak_kib"))
    plasticity = _read_numeric_csv(output / "recognition_plasticity_results.csv",
        integer=("seed", "exposures"),
        floating=("propagated_events", "weight_updates", "formed_connections", "pruned_connections"))
    production = _read_numeric_csv(output / "recognition_production_reference.csv",
        integer=("seed",), floating=("exact_accuracy_within_half_unit", "mean_absolute_error"))
    curves = _summaries(trial_rows)
    _write_csv(output / "recognition_learning_curves.csv", curves)
    _write_csv(output / "recognition_distractor_results.csv",
        [r for r in curves if r["choice_count"] == 2 or
         (r["exposures"] == cfg.exposures[-1] and r["delay_seconds"] == 0 and not r["interference"])])
    _write_csv(output / "recognition_retention_results.csv",
        [r for r in curves if r["delay_seconds"] > 0 or r["interference"] or r["mode"] == "continuous"])
    reconstruct_artifacts(cfg)
    _plot_results(output, curves, trial_rows, compute)
    _report(output, cfg, curves, plasticity, production, compute)
    config_path = output / "recognition_experiment_config.json"
    details = json.loads(config_path.read_text(encoding="utf-8"))
    details["projection"] = cfg.projection
    details["waveform_sample_rate_hz"] = SAMPLE_RATE
    details["waveform_chunk_samples"] = CHUNK_SIZE
    details["visual_observation_rate_hz"] = VISUAL_RATE
    details["projection_profile"] = ("original sensory map" if cfg.projection == "P0" else
        "fixed label-free visual-gain floor, energy matched to P0, on all 199 cells; audio map unchanged")
    _write_json(config_path, details)
    return curves


def compare_projection_runs(p0_output, p1_output):
    """Write paired seed comparison between completed P0 and P1 I0 studies."""
    p0_output, p1_output = Path(p0_output), Path(p1_output)
    def load_final(path):
        rows = _read_numeric_csv(path / "recognition_learning_curves.csv",
            integer=("seed", "exposures", "choice_count"),
            floating=("accuracy", "delay_seconds"), boolean=("interference",))
        return {(r["seed"], r["matcher"]): r["accuracy"] for r in rows if
            r["variant"] == "I0" and r["pairing"] == "correct" and
            r["mode"] == "independent" and r["choice_count"] == 3 and
            r["exposures"] == 50 and r["delay_seconds"] == 0 and not r["interference"]}
    original, corrected = load_final(p0_output), load_final(p1_output)
    rows = [{"seed": seed, "matcher": matcher, "P0_accuracy": original[seed, matcher],
             "P1_accuracy": corrected[seed, matcher],
             "P1_minus_P0": corrected[seed, matcher] - original[seed, matcher]}
            for seed, matcher in sorted(set(original) & set(corrected))]
    _write_csv(p0_output / "recognition_projection_control_comparison.csv", rows)
    if rows:
        lines = ["", "## Paired visual-projection control", "",
            "P1 changes the fixed visual input gains globally and matches P0's total visual projection energy. Audio features, graph initialization, task examples, and readout procedures are unchanged. It corrects threshold-silent visual cells in seeds 43 and 45. The following values use the same five seeds and held-out trial IDs; each comparator is frozen within its exposure run.",
            "", "| Matcher | P0 mean | P1 mean | Paired difference |", "|---|---:|---:|---:|"]
        for matcher in ("fixed", "calibrated"):
            selected = [r for r in rows if r["matcher"] == matcher]
            if selected:
                lines.append(f"| {matcher} | {np.mean([r['P0_accuracy'] for r in selected]):.3f} | {np.mean([r['P1_accuracy'] for r in selected]):.3f} | {np.mean([r['P1_minus_P0'] for r in selected]):+.3f} |")
        lines += ["", "P1 does not meet the predeclared recognition gate. Correcting the visual sensory-event threshold is necessary for input validity but is insufficient for this candidate scorer to recover the association. The separate P1 report and per-seed comparison CSV retain the exact results.", ""]
        report = p0_output / "RECOGNITION_LEARNING_REPORT.md"
        report.write_text(report.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")
    return rows


def _summaries(trials):
    groups = {}
    for row in trials:
        if row.get("stage") != "frozen_test":
            continue
        key = tuple(row[k] for k in ("seed", "variant", "pairing", "exposures",
                                     "matcher", "mode", "choice_count", "delay_seconds",
                                     "interference"))
        groups.setdefault(key, []).append(row)
    columns = ("seed", "variant", "pairing", "exposures", "matcher", "mode",
               "choice_count", "delay_seconds", "interference")
    rows = []
    for key, values in groups.items():
        accuracy = float(np.mean([row["correct"] for row in values]))
        chance = 1.0 / key[6]
        signed = [row for row in values if row["position"] != 0]
        rows.append({**dict(zip(columns, key)), "trials": len(values),
            "accuracy": accuracy, "chance": chance,
            "conditional_sign_accuracy": (float(np.mean([row["selected_position"] == row["position"]
                for row in signed])) if signed else ""),
            "zero_vs_nonzero_accuracy": float(np.mean([(row["selected_position"] == 0) ==
                (row["position"] == 0) for row in values])),
            "stochastic_accuracy": float(np.mean([row["stochastic_correct"] for row in values])),
            "chance_adjusted_accuracy": (accuracy - chance) / (1.0 - chance),
            "mean_score_margin": float(np.mean([row["score_margin"] for row in values])),
            "mean_choice_entropy": float(np.mean([row["choice_entropy"] for row in values])),
            "stochastic_expected_accuracy": float(np.mean([
                row["stochastic_expected_accuracy"] for row in values]))})
    return rows


def _production_reference(seed, vocab, cfg, test_waveforms):
    """Separately supervised scalar production control using fixed I0 states."""
    from .liquid_readout import RidgeReadout
    net = _new_network(seed, "I0", cfg)
    rng = np.random.default_rng(seed + 440_019)
    x_train, y_train = [], []
    for position in POSITIONS:
        for _ in range(cfg.calibration_variants):
            _, acoustic = _audio_assessment(net, _variant(vocab.instruction(position), rng))
            x_train.append(acoustic)
            y_train.append(position)
    readout = RidgeReadout(alpha=1.0).fit(np.asarray(x_train), np.asarray(y_train)[:, None])
    x_test = np.asarray([_audio_assessment(net, waveform)[1]
                         for _, waveform, _ in test_waveforms])
    prediction = readout.predict(x_test).ravel()
    target = np.asarray([p for p, _, _ in test_waveforms])
    return {"seed": seed, "model": "I0_supervised_audio_scalar_readout",
        "train_labels": len(y_train), "test_trials": len(target),
        "exact_accuracy_within_half_unit": float(np.mean(np.abs(prediction - target) <= 0.5)),
        "mean_absolute_error": float(np.mean(np.abs(prediction - target))),
        "readout_supervised": True}


def _plot_results(output, curves, trials, compute):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final_exposure = max((r["exposures"] for r in curves), default=0)

    def save(name, title, draw):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not draw(ax):
            ax.text(.5, .5, "Not measured at this checkpoint", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    def learning(ax):
        rows = [r for r in curves if r["mode"] == "independent" and
                r["pairing"] == "correct" and r["choice_count"] == 3]
        if not rows: return False
        for variant in sorted({r["variant"] for r in rows}):
            for matcher in sorted({r["matcher"] for r in rows}):
                data = [r for r in rows if r["variant"] == variant and r["matcher"] == matcher]
                x = sorted({r["exposures"] for r in data})
                y = [np.mean([r["accuracy"] for r in data if r["exposures"] == n]) for n in x]
                ax.plot(x, y, marker="o", label=f"{variant} {matcher}")
        ax.axhline(1/3, color="gray", linestyle="--")
        ax.set_xlabel("Counted sensory experiences"); ax.set_ylabel("Three-choice accuracy")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        return True
    save("recognition_learning_curve.png", "Independent-episode recognition", learning)

    def confusion(ax):
        rows = [r for r in trials if r.get("stage") == "frozen_test" and
                r["pairing"] == "correct" and r["mode"] == "independent" and
                r["choice_count"] == 3 and r["exposures"] == final_exposure and
                r["matcher"] == "fixed"]
        if not rows: return False
        matrix = np.zeros((3, 3), dtype=int)
        for row in rows:
            matrix[POSITIONS.index(row["position"]), POSITIONS.index(row["selected_position"])] += 1
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(3), POSITIONS); ax.set_yticks(range(3), POSITIONS)
        ax.set_xlabel("Selected position"); ax.set_ylabel("Correct position")
        for i in range(3):
            for j in range(3): ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        return True
    save("recognition_confusion_matrix.png", "Three-position confusion", confusion)

    def difficulty(ax):
        rows = [r for r in curves if r["pairing"] == "correct" and
                r["mode"] == "independent" and r["exposures"] == final_exposure]
        if not rows: return False
        for matcher in sorted({r["matcher"] for r in rows}):
            for count in (2, 3):
                values = [r["accuracy"] for r in rows if r["matcher"] == matcher and r["choice_count"] == count]
                if values: ax.bar(count + (-.13 if matcher == "fixed" else .13), np.mean(values),
                                  width=.25, label=matcher if count == 2 else None)
        ax.set_xticks((2, 3)); ax.set_xlabel("Choices"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.legend()
        return True
    save("recognition_difficulty_comparison.png", "Candidate count", difficulty)

    def retention(ax):
        rows = [r for r in curves if r["pairing"] == "correct" and
                r["choice_count"] == 3 and r["exposures"] == final_exposure and
                r["mode"] == "continuous" and r["matcher"] == "fixed"]
        if not rows: return False
        for variant in sorted({r["variant"] for r in rows}):
            data = [r for r in rows if r["variant"] == variant and not r["interference"]]
            x = sorted({r["delay_seconds"] for r in data})
            ax.plot(x, [np.mean([r["accuracy"] for r in data if r["delay_seconds"] == d])
                        for d in x], marker="o", label=variant)
        ax.axhline(1/3, color="gray", linestyle="--")
        ax.set_xlabel("Delay in simulation seconds"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.legend()
        return True
    save("recognition_retention.png", "Independent-episode retention", retention)

    def cost(ax):
        rows = [r for r in curves if r["pairing"] == "correct" and
                r["choice_count"] == 3 and r["exposures"] == final_exposure and
                r["mode"] == "independent" and r["delay_seconds"] == 0]
        if not rows: return False
        for variant in sorted({r["variant"] for r in rows}):
            quality = np.mean([r["accuracy"] for r in rows if r["variant"] == variant and r["matcher"] == "fixed"])
            matches = [r for r in compute if r["variant"] == variant and r["pairing"] == "correct" and
                       r["exposures"] == final_exposure and r["mode"] == "independent" and r["choice_count"] == 3]
            if matches:
                ax.scatter(np.mean([r["cpu_ms"] for r in matches]), quality, label=variant)
        ax.set_xlabel("Assessment CPU ms per trial"); ax.set_ylabel("Three-choice accuracy")
        ax.set_ylim(0, 1); ax.legend()
        return True
    save("recognition_accuracy_vs_compute.png", "Accuracy versus assessment CPU", cost)


def run(cfg: RecognitionConfig):
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output / "model_checkpoints"
    snapshots_dir.mkdir(exist_ok=True)
    _write_json(output / "recognition_experiment_config.json", {
        **asdict(cfg), "liquid_config": asdict(liquid_config(cfg.neurons)),
        "vocabulary_seed_rule": "seed + 1000003", "calibration_seed_rule": "seed + 110003",
        "test_seed_rule": "seed + 220007", "exposure_seed_rule": "seed + 330017",
        "instruction_ticks": 16, "token_ticks": 4, "silence_ticks": 2,
        "waveform_sample_rate_hz": SAMPLE_RATE, "waveform_chunk_samples": CHUNK_SIZE,
        "visual_observation_rate_hz": VISUAL_RATE,
        "candidate_ticks": cfg.candidate_ticks, "field_min": -99, "field_max": 99,
        "fixed_score": "sum(acoustic_mean_state * candidate_visual_response_delta)",
        "projection_profile": ("original sensory map" if cfg.projection == "P0" else
            "fixed label-free visual-gain floor, energy matched to P0, on all 199 cells; audio map unchanged"),
        "calibrated_score": "train-standardized diagonal interaction ridge, alpha=1, frozen during exposure",
        "stochastic_mode": "softmax expected accuracy; deterministic argmax is primary",
        "checkpoint_thresholds": {"mean": 0.8, "minimum_seed": 0.7},
        "experience_definition": "one audiovisual or single-modality teaching/feedback interaction",
        "assessment": "independent frozen clone per trial; candidate branches from identical post-cue state"})
    trials, scores, computes, plasticity, production = [], [], [], [], []
    stimulus_rows = []
    for seed in cfg.seeds:
        vocab = DigitSpatialVocabulary(seed + 1_000_003)
        stimulus_rows.extend({"seed": seed, **row} for row in vocab.vocabulary_rows())
        calibrated, calibration_meta = _calibrate_matcher(seed, vocab, cfg)
        _write_json(output / f"calibrated_matcher_seed_{seed}.json", calibration_meta)
        matchers = {"fixed": CandidateMatcher(), "calibrated": calibrated}
        test = _test_waveforms(seed, vocab, cfg.test_variants)
        production.append(_production_reference(seed, vocab, cfg, test))
        for variant in cfg.variants:
            for pairing in cfg.pairings:
                net = _new_network(seed, variant, cfg)
                initial_weights = _weights(net)
                initial_stats = asdict(net.stats())
                order = _exposure_order(seed, cfg.exposures[-1])
                exposure_rng = np.random.default_rng(seed + 440_039)
                learning_wall_ms = learning_cpu_ms = 0.0
                for exposure in range(cfg.exposures[-1] + 1):
                    # The full learning curve is measured for correct pairing.
                    # Controls use the same exposure budget, with endpoint
                    # assessments to limit repeated frozen test computation.
                    assess_checkpoint = (exposure in cfg.exposures and
                        (pairing == "correct" or exposure in (0, cfg.exposures[-1])))
                    if assess_checkpoint:
                        batch = _assess_set(net, test, cfg, seed, variant, pairing,
                            exposure, matchers, "independent")
                        for row in batch[0]: row["stage"] = "frozen_test"
                        trials.extend(batch[0]); scores.extend(batch[1]); computes.extend(batch[2])
                        plasticity.append(_plasticity_row(net, initial_weights, seed,
                            variant, pairing, exposure, learning_wall_ms,
                            learning_cpu_ms, initial_stats))
                        if pairing == "correct" and exposure in (0, cfg.exposures[-1]):
                            for mode in ("independent", "continuous"):
                                extra = _assess_set(net, test, cfg, seed, variant,
                                    pairing, exposure, matchers, mode, choice_count=2)
                                for row in extra[0]: row["stage"] = "frozen_test"
                                trials.extend(extra[0]); scores.extend(extra[1]); computes.extend(extra[2])
                            extra = _assess_set(net, test, cfg, seed, variant,
                                pairing, exposure, matchers, "continuous", choice_count=3)
                            for row in extra[0]: row["stage"] = "frozen_test"
                            trials.extend(extra[0]); scores.extend(extra[1]); computes.extend(extra[2])
                    if exposure == cfg.exposures[-1]:
                        break
                    position = int(order[exposure])
                    waveform = _variant(vocab.instruction(position), exposure_rng)
                    # Prediction is recorded before teaching/feedback. Neither
                    # assessment branch nor comparator modifies the main net.
                    if pairing == "correct":
                        assessment = _recognize(net, waveform, POSITIONS,
                            matchers["fixed"], cfg, seed + 901, exposure)
                        for name, matcher in matchers.items():
                            estimates, probabilities, chosen = _all_scores(
                                assessment, matcher, cfg.temperature)
                            trials.append({"seed": seed, "variant": variant,
                                "pairing": pairing, "exposures": exposure,
                                "matcher": name, "mode": "continuous",
                                "choice_count": 3, "difficulty": "three_position_standard",
                                "trial_id": f"online_{exposure}", "stage": "online_pre_feedback",
                                "position": position, "selected_position": int(assessment["candidate_order"][chosen]),
                                "correct": int(assessment["candidate_order"][chosen] == position),
                                "score_margin": float(np.sort(estimates)[-1] - np.sort(estimates)[-2]),
                                "choice_entropy": float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)))),
                                "correct_index": assessment["candidate_order"].index(position), "selected_index": chosen,
                                "candidate_order": "/".join(map(str, assessment["candidate_order"])),
                                "stochastic_expected_accuracy": float(probabilities[assessment["candidate_order"].index(position)])})
                    before_wall, before_cpu = time.perf_counter_ns(), time.process_time_ns()
                    _teach(net, vocab, position, pairing, exposure_rng)
                    learning_wall_ms += (time.perf_counter_ns() - before_wall) / 1e6
                    learning_cpu_ms += (time.process_time_ns() - before_cpu) / 1e6
                if pairing == "correct":
                    absent = [(p, np.zeros_like(wave), replicate)
                              for p, wave, replicate in test]
                    batch = _assess_set(net, absent, cfg, seed, variant,
                        "cue_absent", cfg.exposures[-1], matchers, "independent")
                    for row in batch[0]: row["stage"] = "frozen_test"
                    trials.extend(batch[0]); scores.extend(batch[1]); computes.extend(batch[2])
                    for delay in DELAYS:
                        if delay == 0: continue  # already in the main checkpoint
                        batch = _assess_set(net, test, cfg, seed, variant, pairing,
                            cfg.exposures[-1], matchers, "continuous", delay=delay)
                        for row in batch[0]: row["stage"] = "frozen_test"
                        trials.extend(batch[0]); scores.extend(batch[1]); computes.extend(batch[2])
                    batch = _assess_set(net, test, cfg, seed, variant, pairing,
                        cfg.exposures[-1], matchers, "independent", interference=True)
                    for row in batch[0]: row["stage"] = "frozen_test"
                    trials.extend(batch[0]); scores.extend(batch[1]); computes.extend(batch[2])
                _write_json(snapshots_dir / f"seed_{seed}_{variant}_{pairing}_after_{cfg.exposures[-1]}.json",
                            net.state_dict())
                print(f"seed {seed}: {variant}, {pairing}, {cfg.exposures[-1]} experiences", flush=True)

    curves = _summaries(trials)
    _write_csv(output / "recognition_trial_results.csv", trials)
    _write_csv(output / "recognition_learning_curves.csv", curves)
    _write_csv(output / "recognition_candidate_scores.csv", scores)
    _write_csv(output / "recognition_distractor_results.csv",
               [r for r in curves if r["choice_count"] == 2 or r["exposures"] == cfg.exposures[-1]])
    _write_csv(output / "recognition_retention_results.csv",
               [r for r in curves if r["delay_seconds"] > 0 or r["interference"] or r["mode"] == "continuous"])
    _write_csv(output / "recognition_composition_results.csv", [],
               ["status", "reason", "seed", "held_out_combination", "accuracy"])
    _write_csv(output / "recognition_plasticity_results.csv", plasticity)
    _write_csv(output / "recognition_compute_results.csv", computes)
    _write_csv(output / "recognition_production_reference.csv", production)
    _write_csv(output / "recognition_stimuli.csv", stimulus_rows)
    reconstruct_artifacts(cfg)
    _plot_results(output, curves, trials, computes)
    _report(output, cfg, curves, plasticity, production, computes)
    return curves


def _report(output, cfg, curves, plasticity, production, compute):
    final = cfg.exposures[-1]
    def select(variant, pairing, matcher="fixed", exposure=final):
        return [r for r in curves if r["variant"] == variant and
                r["pairing"] == pairing and r["matcher"] == matcher and
                r["exposures"] == exposure and r["mode"] == "independent" and
                r["choice_count"] == 3 and r["delay_seconds"] == 0 and
                not r["interference"]]
    lines = ["# Experiential Visual Recognition Report", "",
        "## Protocol", "",
        f"Seeds: {', '.join(map(str, cfg.seeds))}. The retina has 199 fixed cells spanning -99..99; only -1, 0, +1 are assessed. One cell is on at a time. Each sign/digit waveform is seed-fixed across contexts. Independent noisy waveform variants are used for comparator calibration and evaluation. Every candidate runs for {cfg.candidate_ticks / VISUAL_RATE:.2f} simulation seconds from an identical frozen post-cue snapshot. Candidate order uses seeded rotations; the three-choice chance level is 1/3.",
        "", f"The fixed matcher sums elementwise products of the causal mean acoustic state and candidate-induced visual response. The calibrated matcher fits a {cfg.neurons}-weight diagonal ridge compatibility function (alpha 1.0) on {cfg.calibration_variants} independent noisy audio variants per position times three visual candidates, or {cfg.calibration_variants * 9} labeled pairs per seed. It is frozen during later experiences. Its supervised fit can learn the three associations externally, so only changes with the fixed matcher and paired-versus-control contrasts are relevant to internal-learning claims. Plasticity is frozen in all assessment clones. Online responses are recorded before the corresponding feedback interaction; every teaching/feedback presentation increments the exposure count once.",
        "", "## Sensory projection validity", "",
        f"Projection profile: {cfg.projection}. P0 is the original fixed map. P1 applies a fixed absolute gain floor of 0.05 to every visual cell and rescales all visual gains to match P0's total projection energy; audio gains and liquid equations remain unchanged. It is a separate sensory-interface intervention, so P0 and P1 results must not be attributed solely to plasticity.",
        "", "| Seed | Visual positions below sensory threshold |", "|---:|---|"]
    for seed in cfg.seeds:
        interface = _new_network(seed, "I0", cfg)
        invisible = [position for position in POSITIONS if
            abs(interface._input_gains[position + 99]) <= interface.config.resting_threshold]
        lines.append(f"| {seed} | {', '.join(map(str, invisible)) if invisible else 'none'} |")
    lines += ["", "## Measured three-choice recognition", "",
        "| Variant | Pairing | Fixed before | Fixed after | Calibrated after |", "|---|---|---:|---:|---:|"]
    for variant in cfg.variants:
        for pairing in cfg.pairings:
            before = select(variant, pairing, exposure=0)
            after = select(variant, pairing)
            calibrated = select(variant, pairing, "calibrated")
            if after:
                mean = lambda rows: float(np.mean([r["accuracy"] for r in rows]))
                lines.append(f"| {variant} | {pairing} | {mean(before):.3f} | {mean(after):.3f} | {mean(calibrated):.3f} |")
    lines += ["", "## Checkpoints", ""]
    available = [variant for variant in cfg.variants if select(variant, "correct")]
    if available:
        for variant in available:
            rows = select(variant, "correct")
            mean, minimum = np.mean([r["accuracy"] for r in rows]), min(r["accuracy"] for r in rows)
            lines.append(f"- **{variant} three-choice gate:** {'PASS' if mean >= .8 and minimum >= .7 and len(rows) >= 5 else 'FAIL/INCOMPLETE'}; mean {mean:.3f}, minimum seed {minimum:.3f}, {len(rows)} seed(s).")
    else:
        lines.append("- No five-seed three-choice gate was evaluated.")
    passed = [variant for variant in available if len(select(variant, "correct")) >= 5 and
        np.mean([r["accuracy"] for r in select(variant, "correct")]) >= .8 and
        min(r["accuracy"] for r in select(variant, "correct")) >= .7]
    if len(cfg.seeds) >= 5 and not passed:
        lines.append("- **Progression decision:** the predeclared three-position accuracy threshold was not met. Larger position sets, four/five-choice tasks, and unseen sign-digit composition are not interpreted from this run.")
    lines += ["", "## Scoring and control diagnostics", ""]
    for variant in cfg.variants:
        initial = select(variant, "correct", exposure=0)
        paired = select(variant, "correct")
        masked = select(variant, "cue_absent")
        if paired:
            avg = lambda values: float(np.mean([r["accuracy"] for r in values])) if values else float("nan")
            lines.append(f"- {variant}: fixed scorer {avg(initial):.3f} before exposure, {avg(paired):.3f} after {final} correct experiences; cue-absent {avg(masked):.3f}.")
        if "mismatched" in cfg.pairings:
            unpaired = select(variant, "mismatched")
            if paired and unpaired:
                by_seed = {r["seed"]: r["accuracy"] for r in unpaired}
                differences = np.asarray([r["accuracy"] - by_seed[r["seed"]]
                                          for r in paired if r["seed"] in by_seed])
                if len(differences):
                    lines.append(f"- {variant} paired minus mismatched fixed-scorer accuracy: mean {differences.mean():+.3f}; per-seed differences {', '.join(f'{d:+.3f}' for d in differences)}.")
    pairing_file = output / "recognition_pairing_diagnostics.csv"
    if pairing_file.exists():
        with pairing_file.open(newline="", encoding="utf-8") as stream:
            pairing_rows = list(csv.DictReader(stream))
        for variant in cfg.variants:
            controls = [r for r in pairing_rows if r["variant"] == variant and r["control"] == "mismatched"]
            if controls:
                mean_weight = np.mean([float(r["mean_abs_shared_weight_difference"]) for r in controls])
                mean_score = np.mean([float(r["mean_abs_candidate_score_difference"]) for r in controls])
                changed_edges = np.mean([int(r["different_edge_count"]) for r in controls])
                lines.append(f"- {variant} correct-versus-mismatched final graph: mean absolute shared-weight difference {mean_weight:.6f}, different edge count {changed_edges:.1f}, mean absolute candidate-score difference {mean_score:.6f}.")
        mismatch = [r for r in pairing_rows if r["control"] == "mismatched"]
        if mismatch and all(float(r["mean_abs_candidate_score_difference"]) == 0.0 for r in mismatch):
            lines.append("- **Measured pairing result:** every matched final candidate score was identical after correct and mismatched exposure, even where I1/I2 shared-edge weights differed. This scorer reveals no pairing-specific benefit from the observed plasticity.")
    lines += ["", "Stochastic sampled accuracy and deterministic accuracy are separate columns in the learning-curve CSV. Softmax choice weights use fixed temperature 1.0 and have not been calibrated as probabilities. Blank-audio trials keep the same balanced candidate roster and test for cue-free shortcuts.",
        "", "## Retention and interference", ""]
    for variant in cfg.variants:
        base = select(variant, "correct")
        delayed = [r for r in curves if r["variant"] == variant and r["pairing"] == "correct" and
            r["matcher"] == "fixed" and r["mode"] == "continuous" and
            r["choice_count"] == 3 and r["exposures"] == final and not r["interference"]]
        interference = [r for r in curves if r["variant"] == variant and r["pairing"] == "correct" and
            r["matcher"] == "fixed" and r["interference"] and r["exposures"] == final]
        if base and delayed:
            delay_text = ", ".join(
                f"{delay:g}s={np.mean([r['accuracy'] for r in delayed if r['delay_seconds'] == delay]):.3f}"
                for delay in sorted({r["delay_seconds"] for r in delayed}))
            interference_mean = np.mean([r["accuracy"] for r in interference]) if interference else float("nan")
            lines.append(f"- {variant}: independent reset {np.mean([r['accuracy'] for r in base]):.3f}; continuous delay scores ({delay_text}); interference {interference_mean:.3f}. Seed-level rows are in the retention CSV.")
    lines += ["", "## Production reference", "",
        f"The separately supervised fixed-I0 scalar readout achieved mean exact-within-half-unit accuracy {np.mean([r['exact_accuracy_within_half_unit'] for r in production]):.3f} and MAE {np.mean([r['mean_absolute_error'] for r in production]):.3f}. Its labels are outside the liquid and its result is not evidence of internal association learning. This production diagnostic has different supervision from the fixed recognition matcher, so their scores do not by themselves establish which task is intrinsically easier.",
        "", "## Online pre-feedback choices", ""]
    trial_file = output / "recognition_trial_results.csv"
    if trial_file.exists():
        with trial_file.open(newline="", encoding="utf-8") as stream:
            online_rows = [r for r in csv.DictReader(stream) if r["stage"] == "online_pre_feedback" and
                           r["pairing"] == "correct" and r["matcher"] == "fixed"]
        for variant in cfg.variants:
            early = [int(r["correct"]) for r in online_rows if r["variant"] == variant and int(r["exposures"]) < min(10, final)]
            late = [int(r["correct"]) for r in online_rows if r["variant"] == variant and int(r["exposures"]) >= max(0, final - 10)]
            if early and late:
                lines.append(f"- {variant}: first {min(10, final)} choices per seed ({len(early)} total) {np.mean(early):.3f} accuracy; last {min(10, final)} per seed ({len(late)} total) {np.mean(late):.3f}. These pre-feedback choices include transient state and must be compared with independent-episode retention before calling them learned associations.")
    lines += [
        "", "## Dynamics and compute", ""]
    for variant in cfg.variants:
        data = [r for r in plasticity if r["variant"] == variant and r["pairing"] == "correct" and r["exposures"] == final]
        timed = [r for r in compute if r["variant"] == variant and r["pairing"] == "correct" and r["exposures"] == final and r["choice_count"] == 3 and r["mode"] == "independent"]
        if data and timed:
            lines.append(f"- {variant}: mean propagated events during learning {np.mean([r['propagated_events'] for r in data]):.0f}; weight updates {np.mean([r['weight_updates'] for r in data]):.0f}; formed/pruned connections {np.mean([r['formed_connections'] for r in data]):.0f}/{np.mean([r['pruned_connections'] for r in data]):.0f}; assessment CPU {np.mean([r['cpu_ms'] for r in timed]):.2f} ms/trial.")
            memory = [float(r["traced_python_peak_kib"]) for r in timed if r.get("traced_python_peak_kib") not in ("", None)]
            if memory:
                lines.append(f"  First-trial traced Python allocation peak: mean {np.mean(memory):.1f} KiB across seeds (native NumPy allocation and process RSS excluded). These traced trials are included in the CPU mean and carry instrumentation overhead.")
    process_memory_path = output / "recognition_process_memory_observation.json"
    if process_memory_path.exists():
        observation = json.loads(process_memory_path.read_text(encoding="utf-8"))
        lines.append(f"- Observed whole-process peak working set before completion: {observation['peak_working_set_bytes_observed'] / (1024 ** 2):.1f} MiB. This is a lower bound for the sequential run, not a model-specific peak.")
    lines += ["", "## Research questions", "",
        "1. **Visual recognition from sound:** the three-choice table and seed-level CSV give the measured answer. No tested variant passed the predeclared gate.",
        "2. **Recognition versus coordinate production:** the separate scalar production reference has different supervision, so this run does not identify which task is intrinsically easier.",
        "3. **Repeated experience:** the fixed-scorer curves and paired controls show whether accuracy rises with counted presentations. A rise without a correct-pairing advantage is not association evidence.",
        "4. **Location of any improvement:** the external calibrated comparator is labeled and frozen; fixed-comparator changes can be attributed to liquid-state or graph changes, but require pairing controls to show useful association.",
        "5. **Weight adaptation:** compare I1 with I0 and their correct-versus-mismatched final scores; weight updates alone are reported separately from recognition.",
        "6. **Structural adaptation:** compare I2 with I1 and I2_rewire_fixed using accuracy, formed/pruned edges, and event cost. Rewiring alone is not a benefit without a pairing-specific gain.",
        "7. **Choice difficulty:** two- and three-choice scores are measured separately. Four/five-choice and broad distractor difficulty are gated off in the three-position subset.",
        "8. **Retention:** independent reset, continuous simulated delays, and interference results are separate. Scores near chance cannot establish retained association.",
        "9. **Continuous learning:** main liquid plasticity operates during the counted feedback experiences with no global optimizer epochs; online choices precede feedback. This is continuous updating, not proof of useful learning.",
        "10. **Efficiency and conventional controls:** liquid CPU, events, and traced memory are measured. FF_direct/F/G are not compared after the candidate-assessment gate fails, so no architecture superiority claim is possible.",
        "11. **Component composition:** unseen sign-digit combinations were not assessed because earlier gates failed; familiar three-position performance cannot establish composition.",
        "12. **Next capability:** audit audio-state information, candidate visual separation, and plasticized graph paths at the same three positions before expanding the spatial range.", ""]
    lines += ["", "## Interpretation and next experiment", "",
        "A rise in fixed-matcher accuracy after correctly paired exposure, exceeding mismatched and single-modality controls on paired seeds, would support an effect of internal adaptation under this specific scoring rule. The calibrated comparator has label supervision and cannot independently establish that claim. Independent-episode assessment clears transient state and queue while retaining graph and weights; continuous assessments are labeled separately. Delay values are simulation time, while CPU and wall times are machine measurements. Event counts are scheduler events, not FLOPs. Observed queue size at the end of candidate branches is a lower bound on transient queue peaks. The 24-neuron CPU profile differs from the historical 64-neuron experiential report, so cross-report score differences cannot be attributed to assessment format alone. Control pairings are assessed at exposure 0 and the final count; the full intermediate learning curve is measured for correctly paired exposure.",
        "", "The three-position experiment cannot test unseen sign-digit composition, two-digit grammar, broad distractor difficulty, or four/five-choice recognition. The composition CSV is intentionally empty until the three-position gate and paired-exposure advantage are established. FF_direct/F/G controls remain behind the same gate; comparing architectures using an invalid matcher would be difficult to interpret. Candidate waveforms and visual arrays retain the historical stimulus grammar. No new neuron or plasticity equation was added.",
        "", "The smallest justified next experiment is an information-path audit in the same three-position task: measure separate held-out decodability of the audio trajectory and each visual branch, then check whether plasticized edges lie on paths that affect the frozen candidate score. Predeclare one fixed, label-free temporal summary or cross-modal normalization and repeat the five-seed paired-versus-mismatched test. Expand to -9..+9 only after that competence and pairing-advantage checkpoint passes.", ""]
    (output / "RECOGNITION_LEARNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/recognition-learning")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--pairings", default=",".join(PAIRINGS))
    parser.add_argument("--exposures", default=",".join(map(str, EXPOSURES)))
    parser.add_argument("--test-variants", type=int, default=3)
    parser.add_argument("--calibration-variants", type=int, default=6)
    parser.add_argument("--projection", choices=("P0", "P1"), default="P0")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="refresh analysis from existing trial CSVs without retraining")
    args = parser.parse_args(argv)
    cfg = RecognitionConfig(seeds=tuple(map(int, args.seeds.split(","))),
        variants=tuple(args.variants.split(",")), pairings=tuple(args.pairings.split(",")),
        exposures=tuple(map(int, args.exposures.split(","))), output=args.output,
        test_variants=args.test_variants, calibration_variants=args.calibration_variants,
        projection=args.projection)
    if args.finalize_existing:
        finalize_existing(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
