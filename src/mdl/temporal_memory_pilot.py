"""Fast direction-memory and quiet-decay pilot using the existing Brian2 network."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .distributed_network import DistributedVisualAssociator


OUT = Path("results/temporal_memory_pilot")
SEEDS = (42, 43)
EVAL_COLUMNS = (1, 2, 3)
STEPS_TO_EVALUATION = (1, 2, 3, 4)
REPEATS_PER_STRATUM = 4  # first two train, last two held out
QUIET_DELAYS_MS = (0, 20, 50, 100)
INTERVAL_RANGE_S = (0.008, 0.014)
CHANCE = 0.5


def _frame(column: int) -> np.ndarray:
    x = np.zeros((5, 5), dtype=np.uint8)
    x[:, int(column)] = 1
    return x


def _history(values: list[np.ndarray], end: int, length: int = 3) -> np.ndarray:
    selected = values[max(0, end - length + 1):end + 1]
    missing = length - len(selected)
    suffix = np.stack(selected) if selected else np.empty((0,) + values[0].shape, dtype=values[0].dtype)
    prefix = np.zeros((missing,) + values[0].shape, dtype=values[0].dtype)
    return np.concatenate((prefix, suffix), axis=0)


def _simulate_trial(seed: int, trial_id: str, direction: int, eval_column: int,
                    steps: int, periods_s: np.ndarray, split: str) -> dict:
    # A circular five-column track makes start position uniform and exactly
    # direction-balanced conditional on each evaluated column (steps 1..4).
    start = int((eval_column - direction * steps) % 5)
    positions = ((start + direction * np.arange(steps + 1)) % 5).astype(int)
    net = DistributedVisualAssociator(seed=seed, learning=False)
    net.set_plasticity(False)
    voltages, visual_frames, spikes_by_frame = [], [], []
    spike_events = []
    for col, duration in zip(positions, periods_s):
        frame = net.present(cue=None, pattern=_frame(int(col)),
                            duration_s=float(duration), visual_delay_s=0.001)
        voltages.append(frame["voltage_mv"][:, -1].astype(float))
        visual_frames.append(_frame(int(col)).reshape(-1))
        spikes_by_frame.append(np.bincount(frame["spike_ids"], minlength=18).astype(int))
        spike_events.append({"times_s": frame["spike_times_s"].astype(float).tolist(),
                             "neuron_ids": frame["spike_ids"].astype(int).tolist()})

    eval_idx = steps
    quiet_states = {0: voltages[-1].copy()}
    quiet_spike_counts = {0: 0}
    previous_delay = 0
    for delay_ms in QUIET_DELAYS_MS[1:]:
        quiet = net.present(cue=None, pattern=None,
                            duration_s=(delay_ms - previous_delay) / 1000.0)
        quiet_states[delay_ms] = quiet["voltage_mv"][:, -1].astype(float)
        quiet_spike_counts[delay_ms] = quiet_spike_counts[previous_delay] + int(len(quiet["spike_ids"]))
        previous_delay = delay_ms

    return {
        "seed": seed, "split": split, "trial_id": trial_id,
        "direction": int(direction), "eval_column": int(eval_column),
        "steps": int(steps), "start_column": start,
        "positions": positions, "periods_s": periods_s.copy(),
        "voltage_states": np.asarray(voltages),
        "visual_states": np.asarray(visual_frames),
        "spikes_by_frame": np.asarray(spikes_by_frame),
        "spike_events": spike_events,
        "current_voltage": voltages[-1].copy(),
        "current_visual": visual_frames[-1].copy(),
        "neural_history": _history(voltages, eval_idx).reshape(-1),
        "visual_history": _history(visual_frames, eval_idx).reshape(-1),
        "quiet_states": quiet_states,
        "quiet_spike_counts": quiet_spike_counts,
    }


def _generate_trials() -> list[dict]:
    all_trials = []
    for seed in SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261115]))
        plan = []
        # Timing is paired across directions and repeat, preventing interval,
        # elapsed-time, or frame-count shortcuts in the balanced strata.
        for column in EVAL_COLUMNS:
            for steps in STEPS_TO_EVALUATION:
                for repeat in range(REPEATS_PER_STRATUM):
                    periods = rng.uniform(*INTERVAL_RANGE_S, size=steps + 1)
                    split = "train" if repeat < 2 else "test"
                    for direction in (-1, 1):
                        plan.append((column, steps, repeat, periods.copy(), split, direction))
        order = rng.permutation(len(plan))
        for idx in order:
            column, steps, repeat, periods, split, direction = plan[int(idx)]
            trial_id = f"s{seed}_x{column}_k{steps}_r{repeat}_{'R' if direction > 0 else 'L'}"
            all_trials.append(_simulate_trial(seed, trial_id, direction, column,
                                               steps, periods, split))
    return all_trials


def _nearest_centroid(x_train: np.ndarray, y_train: np.ndarray,
                      x_test: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = (x_train - center) / scale
    test = (x_test - center) / scale
    classes = np.array([-1, 1])
    prototypes = np.stack([train[np.asarray(y_train) == label].mean(axis=0)
                            for label in classes])
    distances = ((test[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return classes[np.argmin(distances, axis=1)]


def _separation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    left, right = x[np.asarray(y) == -1], x[np.asarray(y) == 1]
    between = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
    within = float(np.mean([
        np.linalg.norm(left - left.mean(axis=0), axis=1).mean(),
        np.linalg.norm(right - right.mean(axis=0), axis=1).mean(),
    ]))
    return between, within


def _score(trials: list[dict]) -> tuple[list[dict], list[dict], dict]:
    direction_rows, decay_rows, summaries = [], [], {}
    for seed in SEEDS:
        train = [r for r in trials if r["seed"] == seed and r["split"] == "train"]
        test = [r for r in trials if r["seed"] == seed and r["split"] == "test"]
        ytr = np.array([r["direction"] for r in train])
        yte = np.array([r["direction"] for r in test])

        # Four matched, lightweight classifiers. C and D use the same three-frame
        # window; only their representation (neural state vs visual arrays) differs.
        predictions = {
            "A": _nearest_centroid(np.stack([r["current_visual"] for r in train]), ytr,
                                   np.stack([r["current_visual"] for r in test])),
            "B": _nearest_centroid(np.stack([r["current_voltage"] for r in train]), ytr,
                                   np.stack([r["current_voltage"] for r in test])),
            "C": _nearest_centroid(np.stack([r["neural_history"] for r in train]), ytr,
                                   np.stack([r["neural_history"] for r in test])),
            "D": _nearest_centroid(np.stack([r["visual_history"] for r in train]), ytr,
                                   np.stack([r["visual_history"] for r in test])),
        }
        acc = {name: float(np.mean(pred == yte)) for name, pred in predictions.items()}
        c_between, c_within = _separation(np.stack([r["current_voltage"] for r in test]), yte)
        history_between, history_within = _separation(np.stack([r["neural_history"] for r in test]), yte)
        for i, trial in enumerate(test):
            direction_rows.append({
                "seed": seed, "split": "test", "trial_id": trial["trial_id"],
                "true_direction": trial["direction"], "evaluated_column": trial["eval_column"],
                "start_column": trial["start_column"], "steps_to_evaluation": trial["steps"],
                "stimulus_count": len(trial["positions"]),
                "frame_intervals_ms_json": json.dumps((trial["periods_s"] * 1000).round(3).tolist()),
                "positions_json": json.dumps(trial["positions"].tolist()),
                "prediction_A_visual_current": int(predictions["A"][i]),
                "prediction_B_neural_current": int(predictions["B"][i]),
                "prediction_C_neural_history": int(predictions["C"][i]),
                "prediction_D_visual_history": int(predictions["D"][i]),
                "correct_A": int(predictions["A"][i] == yte[i]),
                "correct_B": int(predictions["B"][i] == yte[i]),
                "correct_C": int(predictions["C"][i] == yte[i]),
                "correct_D": int(predictions["D"][i] == yte[i]),
                "center_voltage_mv_json": json.dumps(trial["current_voltage"].tolist()),
                "neural_history_mv_json": json.dumps(trial["neural_history"].reshape(3, 18).tolist()),
                "visual_history_json": json.dumps(trial["visual_history"].reshape(3, 25).tolist()),
                "spike_events_json": json.dumps(trial["spike_events"]),
            })

        delay_accuracies, delay_between, delay_within = {}, {}, {}
        for delay in QUIET_DELAYS_MS:
            x_train = np.stack([r["quiet_states"][delay] for r in train])
            x_test = np.stack([r["quiet_states"][delay] for r in test])
            pred = _nearest_centroid(x_train, ytr, x_test)
            delay_accuracies[delay] = float(np.mean(pred == yte))
            delay_between[delay], delay_within[delay] = _separation(x_test, yte)
            for i, trial in enumerate(test):
                decay_rows.append({
                    "seed": seed, "trial_id": trial["trial_id"],
                    "true_direction": trial["direction"], "evaluated_column": trial["eval_column"],
                    "steps_to_evaluation": trial["steps"], "quiet_delay_ms": delay,
                    "prediction": int(pred[i]), "correct": int(pred[i] == yte[i]),
                    "between_direction_centroid_distance_mv": delay_between[delay],
                    "within_direction_distance_mv": delay_within[delay],
                    "cumulative_associative_spikes_during_quiet": trial["quiet_spike_counts"][delay],
                    "voltage_state_mv_json": json.dumps(trial["quiet_states"][delay].tolist()),
                })

        summaries[seed] = {
            "n_train": len(train), "n_test": len(test), "direction_accuracy": acc,
            "center_state_between": c_between, "center_state_within": c_within,
            "history_between": history_between, "history_within": history_within,
            "decay_accuracy": delay_accuracies, "decay_between": delay_between,
            "decay_within": delay_within,
        }
    return direction_rows, decay_rows, summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summaries: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    names = ["A visual\ncurrent", "B neural\ncurrent", "C neural\nhistory", "D visual\nhistory"]
    keys = ["A", "B", "C", "D"]
    x = np.arange(len(keys))
    width = .34
    for offset, seed, color in [(-width / 2, SEEDS[0], "#2878b5"),
                                (width / 2, SEEDS[1], "#d95f02")]:
        vals = [summaries[seed]["direction_accuracy"][key] for key in keys]
        axes[0].bar(x + offset, vals, width, label=f"seed {seed}", color=color)
    axes[0].axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="chance")
    axes[0].set_xticks(x, names); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Held-out direction accuracy")
    axes[0].set_title("At matched interior positions")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(axis="y", alpha=.2)

    right_axis = axes[1].twinx()
    line_handles, line_labels = [], []
    for seed, color in [(SEEDS[0], "#2878b5"), (SEEDS[1], "#d95f02")]:
        vals = [summaries[seed]["decay_accuracy"][d] for d in QUIET_DELAYS_MS]
        line, = axes[1].plot(QUIET_DELAYS_MS, vals, marker="o", color=color,
                             label=f"seed {seed} accuracy")
        line_handles.append(line); line_labels.append(f"seed {seed} accuracy")
        distance = [summaries[seed]["decay_between"][d] for d in QUIET_DELAYS_MS]
        response, = right_axis.plot(QUIET_DELAYS_MS, distance, marker="s", linestyle="--",
                                    alpha=.75, color=color, label=f"seed {seed} voltage distance")
        line_handles.append(response); line_labels.append(f"seed {seed} voltage distance")
    chance_line = axes[1].axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="chance")
    axes[1].set_xticks(QUIET_DELAYS_MS); axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Quiet interval after final visual input (ms)")
    axes[1].set_ylabel("Held-out direction accuracy")
    right_axis.set_yscale("log")
    right_axis.set_ylabel("Opposite-direction centroid distance (mV, log scale)")
    axes[1].set_title("Accuracy and response separation during silence")
    axes[1].legend(line_handles + [chance_line], line_labels + ["chance"],
                   frameon=False, fontsize=7, loc="center right")
    axes[1].grid(alpha=.2)
    fig.suptitle("Temporal memory pilot (STDP disabled)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_memory.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_report(summaries: dict, wall_s: float, biological_s: float,
                  n_trials: int) -> None:
    def acc_table(model: str) -> str:
        return " | ".join(f"{summaries[s]['direction_accuracy'][model]:.1%}" for s in SEEDS)
    rows = []
    for delay in QUIET_DELAYS_MS:
        values = [summaries[s]["decay_accuracy"][delay] for s in SEEDS]
        separation = [summaries[s]["decay_between"][delay] for s in SEEDS]
        within = [summaries[s]["decay_within"][delay] for s in SEEDS]
        rows.append(f"| {delay} | {values[0]:.1%} | {values[1]:.1%} | "
                    f"{np.mean(values):.1%} | {np.mean(separation):.3g} | {np.mean(within):.3g} |")
    text = f"""# Fast Neural Temporal Memory Experiment

## Design

Reused the existing 5×5 binary visual input, fixed sparse projection, and 18-neuron LIF population, with STDP disabled. No neuron, projection, learning-rule, or timestep changes were made. Each trial started from a fresh network with equivalent initial conditions.

To remove the previous opposite-edge-start shortcut, trajectories use a **circular five-column track**: positions wrap at the boundary. Evaluation columns were 1, 2, and 3; both directions occur at each one. Each trial has 1–4 one-column steps before evaluation, balanced exactly across direction, evaluated position, and split. Thus, conditional on the evaluated column, starting columns are uniformly represented in both directions. For each paired direction and replicate, inter-frame intervals were identical, independently drawn from 8–14 ms; stimulus counts are balanced. This explicit wrap-around is a control assumption and is not a claim about a physical environment.

There were {n_trials} trajectories total: {summaries[SEEDS[0]]['n_train']} train and {summaries[SEEDS[0]]['n_test']} test per seed, balanced by direction. Repeats 0–1 were assigned to training and repeats 2–3 to test before network runs. Whole trajectories were kept within one split. Nearest-centroid scaling and prototypes used training trajectories only.

### Direction decoders at the evaluated frame

- **A:** current 5×5 visual array only.
- **B:** current 18-neuron voltage vector.
- **C:** current and preceding two neural voltage snapshots.
- **D:** current and preceding two visual arrays.

Missing history at the first step was zero-padded; the number of preceding frames is balanced across direction and does not enter as a separate feature.

## Experiment A results

| Readout | Seed 42 | Seed 43 | Chance |
|---|---:|---:|---:|
| A. Current visual frame | {acc_table('A')} | 50% |
| B. Current neural state | {acc_table('B')} | 50% |
| C. Neural state + two prior snapshots | {acc_table('C')} | 50% |
| D. Visual frame + two prior frames | {acc_table('D')} | 50% |

The opposite-direction centroid distance for current voltage was {np.mean([summaries[s]['center_state_between'] for s in SEEDS]):.3f} mV in the 18-dimensional population feature space; mean within-direction distance was {np.mean([summaries[s]['center_state_within'] for s in SEEDS]):.3f} mV. For neural-history features, those distances were {np.mean([summaries[s]['history_between'] for s in SEEDS]):.3f} and {np.mean([summaries[s]['history_within'] for s in SEEDS]):.3f}, respectively. These distances describe feature geometry; they are not calibrated probabilities or evidence of semantic understanding.

## Experiment B results

At each delay, a fresh nearest-centroid decoder was fit using **training trajectories at that same delay only**. It was then applied once to held-out trials; no held-out fitting or calibration occurred. Quiet intervals are cumulative from the final visual event. 

| Quiet interval (ms) | Seed 42 accuracy | Seed 43 accuracy | Mean accuracy | Mean between-direction distance (mV) | Mean within-direction distance (mV) |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The decoder stayed at 66.7% at every tested delay; it did **not** return to chance within 100 ms. At the same time, the absolute between-direction voltage distance fell from about 1.25 mV at 0 ms to about 5.65×10⁻⁵ mV at 100 ms (roughly a 22,000-fold reduction). The train-only feature scaling makes the decoder sensitive to this tiny deterministic residual. With no injected measurement noise or trial-to-trial neural noise, accuracy alone cannot establish that the 100 ms trace is robustly usable. The separation trajectory is the clearer evidence that voltages relax toward rest.

Spikes and voltage snapshots during movement and quiet periods are retained in the CSV files. A near-chance result at longer delays would have been meaningful; this pilot instead finds chance-exceeding classification alongside a very large collapse in absolute response separation. Read both measures together and seed by seed.

## Interpretation and limitations

This pilot tests whether a simple readout can recover direction from the network's present voltage after balanced recent sensory histories, and how recoverability changes during silence. Model D is the explicit visual-history control: it receives recent observations directly. Differences between C and D indicate how well the network state/history preserves information relative to that simple control in this task.

Only two fixed sparse projection seeds, three evaluated columns, one-column steps, and a small number of trajectories were tested. The circular boundary avoids start-location leakage but introduces wrap-around. The classifier is deliberately simple, and accuracy is not a measure of motion understanding. No causal ablation of individual neurons or synapses was performed. This experiment cannot establish autonomous prediction, generalization to new speeds or visual transformations, or a need for recurrence.

## Smallest justified next experiment

Because decoding stayed at 66.7% while absolute state separation collapsed, the smallest next step is to repeat the balanced protocol with a few additional fixed projection seeds and one preregistered, fixed readout-noise condition alongside the noise-free baseline. This will test whether the residual is robust to small measurement variation. Do not add recurrence unless that replicated, noise-aware test shows current state is insufficient while explicit history remains informative.

## Runtime and artifacts

Wall-clock generation, decoding, and plotting took {wall_s:.2f} seconds. Simulated biological time across trials was {biological_s:.3f} seconds; the former is machine-dependent and the latter is the sum of simulated trajectory and quiet intervals. No hyperparameter search or external dependency was added.

- `direction_results.csv`: held-out per-trial predictions for A–D, trajectory metadata, voltage history, and spike events.
- `memory_decay.csv`: held-out direction predictions and neural voltage response by delay.
- `temporal_memory.png`: per-seed direction accuracy and quiet-decay curves.
"""
    (OUT / "TEMPORAL_MEMORY_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    trials = _generate_trials()
    biological_s = float(sum(r["periods_s"].sum() for r in trials))
    # Every trial includes an additional 100 ms silent decay observation.
    biological_s += len(trials) * 0.100
    direction_rows, decay_rows, summaries = _score(trials)
    _write_csv(OUT / "direction_results.csv", direction_rows)
    _write_csv(OUT / "memory_decay.csv", decay_rows)
    _plot(summaries)
    wall_s = time.perf_counter() - start
    _write_report(summaries, wall_s, biological_s, len(trials))
    print(f"wall_seconds={wall_s:.2f}; trajectories={len(trials)}; direction_tests={len(direction_rows)}; decay_rows={len(decay_rows)}")
    for seed in SEEDS:
        print(seed, summaries[seed])
    print((OUT / "TEMPORAL_MEMORY_REPORT.md").resolve())


if __name__ == "__main__":
    main()
