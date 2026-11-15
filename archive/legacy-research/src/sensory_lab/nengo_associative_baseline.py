"""Nengo Voja/PES associative-learning baseline.

This module is deliberately separate from Model I.  It reproduces the official
Nengo heteroassociative-memory example, then applies the same documented
Voja/PES mechanism to raw waveform-derived features and 199-cell visual
patterns for positions -1, 0, and +1.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np

from .processors import sensory_features
from .spatial_environment import CHUNK_SIZE, VISUAL_RATE, DigitSpatialVocabulary, visual_array
from .contracts import Observation


POSITIONS = (-1, 0, 1)
SEEDS = (42, 43, 44, 45, 46)
EXPOSURES = (0, 1, 5, 10, 20, 50)


def _require_nengo():
    try:
        import nengo
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install the research extra with: pip install -e '.[nengo]'") from exc
    return nengo


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()
        if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def _cycle_node(nengo, state, key, dimension):
    return nengo.Node(output=lambda t: np.asarray(state[key], dtype=np.float64), size_out=dimension)


def _reference_model(seed=7, output=None, dt=0.001, n_neurons=200):
    """Run the official five-item Voja/PES pattern with learning then recall."""
    nengo = _require_nengo()
    rng = np.random.RandomState(seed)
    num_items, d_key, d_value, period = 5, 2, 4, 0.3
    keys = nengo.dists.UniformHypersphere(surface=True).sample(num_items, d_key, rng=rng)
    values = nengo.dists.UniformHypersphere(surface=False).sample(num_items, d_value, rng=rng)
    intercept = float((np.dot(keys, keys.T) - np.eye(num_items)).flatten().max())
    state = {"key": np.zeros(d_key), "value": np.zeros(d_value), "learning": 0.0}
    model = nengo.Network(seed=seed)
    with model:
        stim_keys = _cycle_node(nengo, state, "key", d_key)
        stim_values = _cycle_node(nengo, state, "value", d_value)
        learning = _cycle_node(nengo, state, "learning", 1)
        recall = nengo.Node(size_in=d_value)
        memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons)
        voja = nengo.Voja(learning_rate=5e-2, post_synapse=None)
        conn_in = nengo.Connection(stim_keys, memory, synapse=None, learning_rule_type=voja)
        nengo.Connection(learning, conn_in.learning_rule, synapse=None)
        conn_out = nengo.Connection(memory, recall, learning_rule_type=nengo.PES(1e-3),
                                    function=lambda x: np.zeros(d_value))
        error = nengo.Ensemble(n_neurons, d_value)
        nengo.Connection(learning, error.neurons, transform=[[10.0]] * n_neurons, synapse=None)
        nengo.Connection(stim_values, error, transform=-1, synapse=None)
        nengo.Connection(recall, error, synapse=None)
        p_keys = nengo.Probe(stim_keys, synapse=None)
        p_values = nengo.Probe(stim_values, synapse=None)
        p_learning = nengo.Probe(learning, synapse=None)
        p_error = nengo.Probe(error, synapse=0.005)
        p_recall = nengo.Probe(recall, synapse=None)
        p_error = nengo.Probe(error, synapse=None)
        p_delta = nengo.Probe(conn_out.learning_rule, "delta")
        p_encoders = nengo.Probe(conn_in.learning_rule, "scaled_encoders")
    started = time.perf_counter()
    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run_steps(1, progress_bar=False)
        initial = sim.data[p_encoders][0].copy()
        steps = int(round(period / dt))
        for phase in ("train", "recall"):
            state["learning"] = 0.0 if phase == "train" else -1.0
            for index in range(num_items):
                state["key"] = keys[index]
                state["value"] = values[index]
                sim.run_steps(steps, progress_bar=False)
        elapsed = time.perf_counter() - started
        final = sim.data[p_encoders][-1].copy()
        t = sim.trange()
        recall_error = np.linalg.norm(sim.data[p_recall] - sim.data[p_values], axis=1)
        train_mask = t <= period * num_items
        recall_mask = ~train_mask
        rows = [{"seed": seed, "stage": "reference", "metric": "recall_rmse",
                 "value": float(np.sqrt(np.mean(recall_error[recall_mask] ** 2)))},
                {"seed": seed, "stage": "reference", "metric": "recall_norm_mean",
                 "value": float(np.mean(np.linalg.norm(sim.data[p_recall], axis=1)))},
                {"seed": seed, "stage": "reference", "metric": "training_error_rmse",
                 "value": float(np.sqrt(np.mean(recall_error[train_mask] ** 2)))},
                {"seed": seed, "stage": "reference", "metric": "encoder_change_l2",
                 "value": float(np.linalg.norm(final - initial))},
                {"seed": seed, "stage": "reference", "metric": "cpu_seconds",
                 "value": float(elapsed)}]
        if output is not None:
            _plot_reference(output, t, sim.data[p_keys], sim.data[p_values],
                            sim.data[p_recall], recall_error, initial, final, keys)
    return rows, {"keys": keys, "values": values, "intercept": intercept,
                  "n_neurons": n_neurons, "dt": dt, "period": period}


def _plot_reference(output, t, keys, values, recall, error, initial, final, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, keys); axes[0].set_title("Reference input patterns")
    axes[1].plot(t, values, label="target"); axes[1].plot(t, recall, linestyle="--", label="recalled")
    axes[1].set_title("Reference target and recalled patterns"); axes[1].legend()
    axes[2].plot(t, error); axes[2].set_title("Recall error across simulation time"); axes[2].set_xlabel("Simulation time (s)")
    fig.tight_layout(); fig.savefig(output / "nengo_reference_recall.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(initial[:, 0], initial[:, 1], label="before")
    axes[0].scatter(final[:, 0], final[:, 1], label="after"); axes[0].set_title("Voja encoder changes"); axes[0].legend()
    axes[1].plot(targets[:, 0], targets[:, 1], "o-"); axes[1].set_title("Reference keys")
    fig.tight_layout(); fig.savefig(output / "nengo_reference_encoders.png", dpi=150); plt.close(fig)


def _audio_features(vocab, position, rng):
    wave = vocab.instruction(position)
    wave = rng.uniform(0.85, 1.15) * np.roll(wave, int(rng.integers(-5, 6)))
    result = []
    for tick in range(len(wave) // CHUNK_SIZE):
        audio = wave[tick * CHUNK_SIZE:(tick + 1) * CHUNK_SIZE]
        result.append(sensory_features(Observation(t=tick / VISUAL_RATE,
            vision=np.zeros(199), audio=audio))[199:])
    return np.mean(result, axis=0)


def _projection(seed, dimension=69, output_dimension=16):
    rng = np.random.default_rng(seed + 910_003)
    matrix = rng.normal(size=(dimension, output_dimension))
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
    return matrix


def _normalize_key(key):
    key = np.asarray(key, dtype=np.float64)
    return key / max(float(np.linalg.norm(key)), 1e-12)


def _spatial_model(seed, cfg, condition, output):
    nengo = _require_nengo()
    rng = np.random.default_rng(seed + 17_003)
    vocab = DigitSpatialVocabulary(seed + 1_000_003)
    proj = _projection(seed, 69, cfg["key_dim"])
    key_train = {position: _normalize_key(_audio_features(vocab, position, rng) @ proj)
                 for position in POSITIONS}
    raw_targets = {position: visual_array(position, -99, 99) for position in POSITIONS}
    target_projection = _projection(seed + 771_007, 199, cfg["output_dim"])
    targets = {position: raw_targets[position] @ target_projection for position in POSITIONS}
    input_dim, output_dim, period = cfg["key_dim"], cfg["output_dim"], cfg["period"]
    order_rng = np.random.default_rng(seed + 23_003)
    state = {"key": np.zeros(input_dim), "target": np.zeros(output_dim), "learning": 0.0}
    model = nengo.Network(seed=seed)
    with model:
        key_node = _cycle_node(nengo, state, "key", input_dim)
        target_node = _cycle_node(nengo, state, "target", output_dim)
        learning = _cycle_node(nengo, state, "learning", 1)
        recall = nengo.Node(size_in=output_dim)
        memory = nengo.Ensemble(cfg["neurons"], input_dim, intercepts=[cfg["intercept"]] * cfg["neurons"])
        voja = nengo.Voja(learning_rate=cfg["voja_rate"], post_synapse=None)
        conn_in = nengo.Connection(key_node, memory, synapse=None, learning_rule_type=voja)
        nengo.Connection(learning, conn_in.learning_rule, synapse=None)
        conn_out = nengo.Connection(memory, recall, learning_rule_type=nengo.PES(cfg["pes_rate"]),
                                    function=lambda x: np.zeros(output_dim))
        error = nengo.Ensemble(cfg["neurons"], output_dim)
        nengo.Connection(learning, error.neurons, transform=[[10.0]] * cfg["neurons"], synapse=None)
        nengo.Connection(target_node, error, transform=-1, synapse=None)
        nengo.Connection(recall, error, synapse=None)
        p_recall = nengo.Probe(recall, synapse=None)
        p_error = nengo.Probe(error, synapse=0.005)
        p_error_raw = nengo.Probe(error, synapse=None)
        p_delta = nengo.Probe(conn_out.learning_rule, "delta")
        p_encoders = nengo.Probe(conn_in.learning_rule, "scaled_encoders")
    rows, scores, confusion, curves = [], [], [], []
    test_rng = np.random.default_rng(seed + 45_007)
    with nengo.Simulator(model, dt=cfg["dt"], progress_bar=False) as sim:
        sim.run_steps(1, progress_bar=False)
        initial_encoders = sim.data[p_encoders][0].copy()
        maximum = cfg["max_exposures"] * len(POSITIONS)
        checkpoint_steps = {int(value) * len(POSITIONS): int(value)
                            for value in cfg["checkpoints"]}
        for exposure in range(maximum + 1):
            if exposure in checkpoint_steps:
                for position in POSITIONS:
                    key = _normalize_key(_audio_features(vocab, position, test_rng) @ proj)
                    state["key"], state["target"], state["learning"] = key, np.zeros(output_dim), -1.0
                    sim.run_steps(int(round(period / cfg["dt"])), progress_bar=False)
                    recalled = sim.data[p_recall][-1].copy()
                    candidate_scores = {candidate: float(np.dot(recalled, targets[candidate]) /
                        max(np.linalg.norm(recalled) * np.linalg.norm(targets[candidate]), 1e-12))
                        for candidate in POSITIONS}
                    chosen = max(candidate_scores, key=candidate_scores.get)
                    rows.append({"seed": seed, "condition": condition, "exposures": checkpoint_steps[exposure],
                        "position": position, "selected_position": chosen, "correct": int(chosen == position),
                            "recall_error_l2": float(np.linalg.norm(recalled - targets[position])),
                        "recall_norm": float(np.linalg.norm(recalled)),
                        "error_norm": float(np.linalg.norm(sim.data[p_error_raw][-1])),
                        "decoder_delta_norm": float(np.linalg.norm(sim.data[p_delta][-1])),
                        "score_neg1": candidate_scores[-1], "score_zero": candidate_scores[0], "score_pos1": candidate_scores[1]})
                    for candidate, score in candidate_scores.items():
                        scores.append({"seed": seed, "condition": condition, "exposures": checkpoint_steps[exposure],
                            "position": position, "candidate": candidate, "score": score, "is_correct": int(candidate == position)})
            if exposure == maximum:
                break
            position = POSITIONS[int(order_rng.integers(0, len(POSITIONS)))]
            state["key"], state["target"] = key_train[position], targets[position]
            state["learning"] = 0.0
            if condition == "mismatched":
                state["target"] = targets[POSITIONS[(POSITIONS.index(position) + 1) % 3]]
            if condition == "no_learning":
                state["learning"] = -1.0
            sim.run_steps(int(round(period / cfg["dt"])), progress_bar=False)
        final_encoders = sim.data[p_encoders][-1].copy()
        max_error_norm = float(np.max(np.linalg.norm(sim.data[p_error_raw], axis=1)))
        max_decoder_delta = float(np.max(np.linalg.norm(sim.data[p_delta], axis=1)))
        for checkpoint in cfg["checkpoints"]:
            subset = [row for row in rows if row["exposures"] == checkpoint]
            curves.append({"seed": seed, "condition": condition, "exposures": checkpoint,
                "accuracy": float(np.mean([row["correct"] for row in subset])),
                "mean_recall_error_l2": float(np.mean([row["recall_error_l2"] for row in subset])),
                "mean_recall_norm": float(np.mean([row["recall_norm"] for row in subset])),
                "mean_error_norm": float(np.mean([row["error_norm"] for row in subset])),
                "mean_decoder_delta_norm": float(np.mean([row["decoder_delta_norm"] for row in subset])),
                "max_training_error_norm": max_error_norm,
                "max_decoder_delta_norm": max_decoder_delta,
                "encoder_change_l2": float(np.linalg.norm(final_encoders - initial_encoders))})
        _write_json(output / f"nengo_spatial_state_seed_{seed}_{condition}.json", {
            "initial_encoders": initial_encoders, "final_encoders": final_encoders,
            "projection": proj, "target_projection": target_projection,
            "keys": key_train, "targets": targets, "raw_target_shape": [199]})
    return rows, scores, curves


def _plots(output, rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition in sorted({r["condition"] for r in curves}):
        data = [r for r in curves if r["condition"] == condition]
        x = sorted({r["exposures"] for r in data})
        ax.plot(x, [np.mean([r["accuracy"] for r in data if r["exposures"] == e]) for e in x], marker="o", label=condition)
    ax.axhline(1/3, color="gray", linestyle="--"); ax.set_ylim(0, 1); ax.set_xlabel("Exposures per position"); ax.set_ylabel("Three-choice accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "nengo_learning_progress.png", dpi=150); plt.close(fig)
    final = max(r["exposures"] for r in curves)
    data = [r for r in rows if r["exposures"] == final]
    matrix = np.zeros((3, 3), dtype=int)
    for row in data:
        matrix[POSITIONS.index(row["position"]), POSITIONS.index(row["selected_position"])] += 1
    fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(matrix, cmap="Blues"); ax.set_xticks(range(3), POSITIONS); ax.set_yticks(range(3), POSITIONS); ax.set_xlabel("Selected"); ax.set_ylabel("Correct"); fig.tight_layout(); fig.savefig(output / "nengo_recognition_confusion.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot([r["recall_error_l2"] for r in data], marker="."); ax.set_title("Recall visual-pattern error at final exposure"); ax.set_xlabel("Assessment trial"); ax.set_ylabel("L2 error"); fig.tight_layout(); fig.savefig(output / "nengo_recall_example.png", dpi=150); plt.close(fig)


def run(output="results/nengo-associative", seeds=SEEDS, neurons=100, dt=0.002):
    nengo = _require_nengo()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    reference_rows, reference_meta = _reference_model(seeds[0], output, dt=min(dt, 0.001), n_neurons=200)
    cfg = {"seeds": list(seeds), "neurons": neurons, "key_dim": 16, "output_dim": 16,
        "dt": dt, "period": 0.08, "max_exposures": 50, "checkpoints": [0, 1, 5, 10, 20, 50],
        "intercept": 0.5,
        "voja_rate": 0.05, "pes_rate": 0.001, "conditions": ["no_learning", "correct", "mismatched"],
        "nengo_version": nengo.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "seed_rule": "seed-specific vocabulary, projection, order, and waveform nuisance RNGs",
        "input_semantics": "69-D audio features from existing raw waveform pipeline, fixed random 69x16 projection fit without labels",
        "target_semantics": "fixed 199x16 projection of visual_array(position,-99,99); candidate images remain raw 199-cell arrays and are projected by this frozen matrix",
        "learning": "documented Nengo Voja encoder adaptation and PES error-modulated decoder learning; learning signal 0 during teaching and -1 during recall",
        "supervision": "PES uses the visual target during teaching; recall removes target and learning error input"}
    _write_json(output / "nengo_association_config.json", cfg)
    spatial_rows, candidate_scores, curves = [], [], []
    for seed in seeds:
        for condition in cfg["conditions"]:
            rows, scores, curve = _spatial_model(seed, cfg, condition, output)
            spatial_rows.extend(rows); candidate_scores.extend(scores); curves.extend(curve)
    _write_csv(output / "nengo_reference_results.csv", reference_rows)
    _write_csv(output / "nengo_spatial_recognition.csv", spatial_rows)
    _write_csv(output / "nengo_learning_curves.csv", curves)
    _write_csv(output / "nengo_candidate_scores.csv", candidate_scores)
    _plots(output, spatial_rows, curves)
    _write_report(output, cfg, reference_rows, curves, spatial_rows)
    return cfg


def _write_report(output, cfg, reference, curves, rows):
    reference_recall = [r["value"] for r in reference if r["metric"] == "recall_rmse"]
    lines = ["# Nengo Associative Baseline Report", "", "## Reference reproduction", "",
        f"Nengo {cfg['nengo_version']} was used with the documented Voja/PES architecture: a 200-neuron key ensemble, Voja encoder adaptation, PES decoder learning, sequential five-key presentation, and a negative learning signal during recall. The reference run used seed {cfg['seeds'][0]} and dt={min(cfg['dt'], 0.001)} s. Recall RMSE was {np.mean(reference_recall):.4f}; encoder change L2 was {np.mean([r['value'] for r in reference if r['metric'] == 'encoder_change_l2']):.4f}. This reproduces the mechanism and produces finite recall; the full time-series plots are saved separately.",
        "", "## Spatial adaptation", "",
        "The adapted task uses only -1, 0, and +1. Each 4-tick acoustic instruction is transformed through the existing 69-D waveform-derived audio feature interface and a fixed 69x16 random projection. The PES target is a fixed 199x16 projection of the actual one-cell visual array; candidate images are still the raw 199-cell arrays and are passed through the same frozen projection before cosine scoring. This intermediate control keeps the image provenance explicit while making the decoder CPU-sized. Coordinates, token IDs, candidate indices, and answer labels remain controller-side. Training presents the projected visual target to the PES error pathway; recognition presents only the audio cue, runs the neural dynamics, and scores cosine similarity between recalled output and each projected candidate image. Learning is disabled during recall.",
        "", "| Condition | Exposure 0 accuracy | Exposure 50 accuracy |", "|---|---:|---:|"]
    for condition in cfg["conditions"]:
        values = [r for r in curves if r["condition"] == condition]
        zero = np.mean([r["accuracy"] for r in values if r["exposures"] == 0])
        final = np.mean([r["accuracy"] for r in values if r["exposures"] == 50])
        lines.append(f"| {condition} | {zero:.3f} | {final:.3f} |")
    lines += ["", "Chance for the three-choice assessment is 0.333. Results are five-seed means; seed-level rows remain in `nengo_learning_curves.csv`. The no-learning and mismatched conditions use the same number and duration of presentations as correct pairing. Independent nuisance waveform realizations are used for recognition.",
        "", "## Interpretation", "",
        "Voja/PES is supervised error-modulated associative learning. A successful spatial result would establish a conventional reference mechanism, not autonomous discovery of meaning. The engineered reference vectors are unlike the raw waveform interface; the spatial adaptation therefore tests a materially harder sensor-to-pattern mapping. Input projection and Nengo neuron counts are fixed before evaluation; no test score selects them.",
        "", "Model I was not imported or modified. Future comparison should match output representation, candidate scorer, exposure schedule, CPU budget, and learning supervision. Only after this baseline passes its own recognition check should the liquid be compared against it.", ""]
    (output / "NENGO_ASSOCIATIVE_BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/nengo-associative")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.002)
    args = parser.parse_args(argv)
    run(args.output, tuple(int(v) for v in args.seeds.split(",")), args.neurons, args.dt)


if __name__ == "__main__":
    main()
