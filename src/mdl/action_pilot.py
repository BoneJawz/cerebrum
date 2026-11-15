"""Fast, standalone pilot for motion history in the distributed Brian2 pathway."""
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


OUT = Path("results/action_pilot")
SEEDS = (42, 43)
N_TRAIN_PER_DIRECTION = 12
N_TEST_PER_DIRECTION = 8
FRAME_PERIOD_RANGE_S = (0.008, 0.014)
ACTION_LAMBDA = 0.5


def _pattern(column: int) -> np.ndarray:
    visual = np.zeros((5, 5), dtype=np.uint8)
    visual[:, int(column)] = 1
    return visual


def _trial(seed: int, trial_id: str, direction: int, periods: np.ndarray) -> dict:
    model = DistributedVisualAssociator(seed=seed, learning=False)
    model.set_plasticity(False)
    positions = np.arange(5) if direction > 0 else np.arange(4, -1, -1)
    voltage, spikes, trace_times, voltage_trace, spike_events = [], [], [], [], []
    for col, period in zip(positions, periods):
        frame = model.present(cue=None, pattern=_pattern(int(col)),
                              duration_s=float(period), visual_delay_s=0.001)
        voltage.append(frame["voltage_mv"][:, -1].astype(float))
        spikes.append(np.bincount(frame["spike_ids"], minlength=18).astype(int))
        times = frame["voltage_times_s"]
        values = frame["voltage_mv"].T
        stride = max(1, int(round(0.001 / np.median(np.diff(times))))) if len(times) > 1 else 1
        trace_times.append(times[::stride].astype(float))
        voltage_trace.append(values[::stride].astype(float))
        spike_events.append({"times_s": frame["spike_times_s"].astype(float).tolist(),
                             "neuron_ids": frame["spike_ids"].astype(int).tolist()})
    return {"trial_id": trial_id, "direction": int(direction),
            "positions": positions.astype(int), "periods": periods.copy(),
            "voltage": np.asarray(voltage), "spikes": np.asarray(spikes),
            "trace_times": trace_times, "voltage_trace": voltage_trace,
            "spike_events": spike_events}


def _nearest_centroid_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                                  x_test: np.ndarray) -> np.ndarray:
    """Standardized two-class nearest centroid, with scaling fit only on train."""
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    tr = (x_train - mean) / scale
    te = (x_test - mean) / scale
    classes = np.array([-1, 1])
    centroids = np.stack([tr[np.asarray(y_train) == label].mean(axis=0)
                          for label in classes])
    return classes[np.argmin(((te[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)]


def _make_data() -> list[dict]:
    records = []
    for seed in SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261115]))
        for direction in (-1, 1):
            n = N_TRAIN_PER_DIRECTION + N_TEST_PER_DIRECTION
            # Split whole trajectories before simulation; no frame from a held-out
            # trajectory can enter fitting or normalization.
            permutation = rng.permutation(n)
            split_by_index = {int(idx): ("train" if rank < N_TRAIN_PER_DIRECTION else "test")
                              for rank, idx in enumerate(permutation)}
            for index in range(n):
                periods = rng.uniform(*FRAME_PERIOD_RANGE_S, size=5)
                trial_id = f"seed{seed}_{'R' if direction > 0 else 'L'}_{index:02d}"
                trial = _trial(seed, trial_id, direction, periods)
                trial.update(seed=seed, split=split_by_index[index])
                records.append(trial)
    return records


def _classify_and_score(records: list[dict]) -> tuple[list[dict], dict]:
    scored, summaries = [], {}
    for seed in SEEDS:
        group = [r for r in records if r["seed"] == seed]
        train = [r for r in group if r["split"] == "train"]
        test = [r for r in group if r["split"] == "test"]
        y_train = np.array([r["direction"] for r in train])
        y_test = np.array([r["direction"] for r in test])

        # Every center image is identical; this control has no direction cue.
        x_visual_train = np.stack([_pattern(2).reshape(-1) for _ in train])
        x_visual_test = np.stack([_pattern(2).reshape(-1) for _ in test])
        visual_pred = _nearest_centroid_fit_predict(x_visual_train, y_train, x_visual_test)

        x_state_train = np.stack([r["voltage"][2] for r in train])
        x_state_test = np.stack([r["voltage"][2] for r in test])
        state_pred = _nearest_centroid_fit_predict(x_state_train, y_train, x_state_test)

        # At the shared center frame, the 0/1/2 voltage snapshots form the short
        # neural-state history. Trial identity remains the unit of splitting.
        x_hist_train = np.stack([r["voltage"][:3].reshape(-1) for r in train])
        x_hist_test = np.stack([r["voltage"][:3].reshape(-1) for r in test])
        hist_pred = _nearest_centroid_fit_predict(x_hist_train, y_train, x_hist_test)

        # Static baseline is the training conditional mean next column at center.
        next_cols_train = np.array([r["positions"][3] for r in train], dtype=float)
        center_cols_train = np.array([r["positions"][2] for r in train], dtype=int)
        static_pred = float(next_cols_train[center_cols_train == 2].mean())
        estimated_speed = float(np.mean([
            np.abs(np.diff(r["positions"].astype(float))) for r in train
        ]))

        # B maps history-decoded direction to the next center-adjacent column.
        dynamic_pred = 2 + hist_pred
        # C minimizes a tiny trajectory cost over legal discrete actions:
        # J(a)=(a-a_hat)^2 + lambda*(|a|-s_hat)^2.  a_hat is the
        # train-calibrated neural-history displacement; s_hat is mean training
        # speed. The cost never sees held-out future positions.
        action_pred, action_costs = [], []
        for action_hat in hist_pred:
            candidates = np.array([-1.0, 1.0])
            costs = (candidates - float(action_hat)) ** 2 + ACTION_LAMBDA * (
                np.abs(candidates) - estimated_speed) ** 2
            action_pred.append(2 + int(candidates[int(np.argmin(costs))]))
            action_costs.append(costs)
        action_pred = np.asarray(action_pred)

        seed_test_rows = []
        for i, r in enumerate(test):
            true_next = int(r["positions"][3])
            row = {
                "seed": seed, "split": "test", "trial_id": r["trial_id"], "direction": r["direction"],
                "visual_direction_prediction": int(visual_pred[i]),
                "neural_state_direction_prediction": int(state_pred[i]),
                "neural_history_direction_prediction": int(hist_pred[i]),
                "current_position": 2, "true_next_position": true_next,
                "static_prediction": static_pred, "dynamic_prediction": int(dynamic_pred[i]),
                "action_prediction": int(action_pred[i]),
                "static_abs_error": abs(static_pred - true_next),
                "dynamic_abs_error": abs(int(dynamic_pred[i]) - true_next),
                "action_abs_error": abs(int(action_pred[i]) - true_next),
                "action_cost_left": float(action_costs[i][0]),
                "action_cost_right": float(action_costs[i][1]),
                "estimated_train_speed": estimated_speed,
                "positions_json": json.dumps(r["positions"].tolist()),
                "frame_periods_s_json": json.dumps(r["periods"].round(6).tolist()),
                "voltage_history_mv_json": json.dumps(r["voltage"].round(4).tolist()),
                "spike_counts_by_frame_json": json.dumps(r["spikes"].tolist()),
                "voltage_trace_times_s_json": json.dumps([x.round(4).tolist() for x in r["trace_times"]]),
                "voltage_trace_mv_json": json.dumps([x.round(3).tolist() for x in r["voltage_trace"]]),
                "spike_events_by_frame_json": json.dumps(r["spike_events"]),
                "total_associative_spikes": int(r["spikes"].sum()),
            }
            scored.append(row)
            seed_test_rows.append(row)

        # Preserve the training trajectories and their neural recordings too;
        # prediction columns are blank because training rows are not evaluations.
        for r in train:
            scored.append({
                "seed": seed, "split": "train", "trial_id": r["trial_id"], "direction": r["direction"],
                "visual_direction_prediction": "", "neural_state_direction_prediction": "",
                "neural_history_direction_prediction": "", "current_position": 2,
                "true_next_position": int(r["positions"][3]), "static_prediction": "",
                "dynamic_prediction": "", "action_prediction": "", "static_abs_error": "",
                "dynamic_abs_error": "", "action_abs_error": "", "action_cost_left": "",
                "action_cost_right": "", "estimated_train_speed": "",
                "positions_json": json.dumps(r["positions"].tolist()),
                "frame_periods_s_json": json.dumps(r["periods"].round(6).tolist()),
                "voltage_history_mv_json": json.dumps(r["voltage"].round(4).tolist()),
                "spike_counts_by_frame_json": json.dumps(r["spikes"].tolist()),
                "voltage_trace_times_s_json": json.dumps([x.round(4).tolist() for x in r["trace_times"]]),
                "voltage_trace_mv_json": json.dumps([x.round(3).tolist() for x in r["voltage_trace"]]),
                "spike_events_by_frame_json": json.dumps(r["spike_events"]),
                "total_associative_spikes": int(r["spikes"].sum()),
            })

        # Descriptive cross-validated geometry at the common center frame.
        pos = x_state_test[y_test == 1]
        neg = x_state_test[y_test == -1]
        between = float(np.linalg.norm(pos.mean(axis=0) - neg.mean(axis=0)))
        within = float(np.mean([
            np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean(),
            np.linalg.norm(neg - neg.mean(axis=0), axis=1).mean(),
        ]))
        summaries[seed] = {
            "n_test": len(test),
            "visual_accuracy": float(np.mean(visual_pred == y_test)),
            "neural_state_accuracy": float(np.mean(state_pred == y_test)),
            "neural_history_accuracy": float(np.mean(hist_pred == y_test)),
            "between_center_state_distance_mv": between,
            "within_center_state_distance_mv": within,
            "mean_center_spikes_left": float(np.mean([r["spikes"][2].sum() for r in test if r["direction"] == -1])),
            "mean_center_spikes_right": float(np.mean([r["spikes"][2].sum() for r in test if r["direction"] == 1])),
            "static_mae": float(np.mean([row["static_abs_error"] for row in seed_test_rows])),
            "dynamic_mae": float(np.mean([row["dynamic_abs_error"] for row in seed_test_rows])),
            "action_mae": float(np.mean([row["action_abs_error"] for row in seed_test_rows])),
            "estimated_train_speed": estimated_speed,
        }
    return scored, summaries


def _write_csv(scored: list[dict]) -> None:
    path = OUT / "motion_trials.csv"
    fields = list(scored[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)


def _plot(records: list[dict], scored: list[dict], summaries: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for direction, color, label in [(-1, "#2878b5", "Right → Left"), (1, "#d95f02", "Left → Right")]:
        paths = [r for r in records if r["split"] == "test" and r["direction"] == direction]
        mean_voltage = np.stack([r["voltage"].mean(axis=1) for r in paths]).mean(axis=0)
        ax.plot(np.arange(5), mean_voltage, marker="o", color=color, label=label)
    ax.set_title("Mean associative voltage across held-out paths")
    ax.set_xlabel("Frame / column index (center at 2)")
    ax.set_ylabel("Mean membrane voltage (mV)")
    ax.grid(alpha=.25); ax.legend(frameon=False)

    methods = ["static_mae", "dynamic_mae", "action_mae"]
    labels = ["A Static", "B Neural history", "C Action cost"]
    means = [np.mean([summaries[s][key] for s in SEEDS]) for key in methods]
    lows = [min(summaries[s][key] for s in SEEDS) for key in methods]
    highs = [max(summaries[s][key] for s in SEEDS) for key in methods]
    x = np.arange(3)
    ax = axes[1]
    ax.bar(x, means, color=["#888888", "#4393c3", "#ef8a62"], width=.65)
    ax.errorbar(x, means, yerr=[np.array(means)-lows, np.array(highs)-means],
                fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("Held-out next-position absolute error (columns)")
    ax.set_title("Center-frame next-position prediction")
    ax.grid(axis="y", alpha=.25)
    fig.suptitle("Action pilot: history and one-step prediction", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _report(scored: list[dict], summaries: dict, elapsed_s: float,
            biological_s: float) -> None:
    def fmt(key: str) -> str:
        return ", ".join(f"{s}: {summaries[s][key]:.3f}" for s in SEEDS)
    avg = lambda key: float(np.mean([summaries[s][key] for s in SEEDS]))
    per_seed = {
        seed: (summaries[seed]["static_mae"], summaries[seed]["dynamic_mae"], summaries[seed]["action_mae"])
        for seed in SEEDS
    }
    text = f"""# Neural Dynamics and Action-Based Prediction — Pilot

## Protocol

Two fixed network seeds ({', '.join(map(str, SEEDS))}), with {N_TRAIN_PER_DIRECTION} training and {N_TEST_PER_DIRECTION} held-out trajectories per direction and seed. The object occupied one full vertical column of a binary 5×5 array and moved one column per frame from 0→4 or 4→0. Inter-frame durations were independently jittered from {FRAME_PERIOD_RANGE_S[0]*1000:.0f} to {FRAME_PERIOD_RANGE_S[1]*1000:.0f} ms. Whole trajectories, not individual frames, were split before fitting. STDP was disabled; the existing 25-input sparse visual projection and 18-cell LIF population were unchanged. Each trial began from a fresh, identically seeded network. The center observation is column 2; next-position prediction is evaluated there.

Visual arrays were the only sensory inputs. Direction and position labels were retained by the experiment controller only for splitting, fitting the readouts, and scoring. The Brian2 network received only binary visual spike events.

## Results

### 1. State at the same visual position

The center visual input is identical in both directions. The raw visual nearest-centroid control therefore scored **{avg('visual_accuracy'):.1%}** overall. On held-out whole trajectories, the 18-neuron voltage snapshot at the center scored **{avg('neural_state_accuracy'):.1%}** direction accuracy; concatenating the center snapshot with the preceding two neural snapshots scored **{avg('neural_history_accuracy'):.1%}**. Seed-level accuracies (42, 43) were {fmt('neural_state_accuracy')} for the current neural state and {fmt('neural_history_accuracy')} for neural history.

Across held-out center snapshots, the opposite-direction centroid distance was {avg('between_center_state_distance_mv'):.3f} in the feature space of recorded membrane voltages (mV coordinates); mean within-direction distance to each centroid was {avg('within_center_state_distance_mv'):.3f}. Mean center-frame associative spike counts were {avg('mean_center_spikes_left'):.2f} for right-to-left and {avg('mean_center_spikes_right'):.2f} for left-to-right. These are dynamical response differences in this seeded setup, not evidence that the network understands motion.

### 2. Next-position prediction at the center

| Method | Definition | Held-out MAE (columns) |
|---|---|---:|
| A. Static | Training mean next column conditioned only on current column 2 | {avg('static_mae'):.3f} |
| B. Neural history | Nearest-centroid direction readout from the preceding three neural snapshots, then one-column step | {avg('dynamic_mae'):.3f} |
| C. Action-inspired cost | Select among ±1 using `J(a)=(a−â)²+0.5(|a|−ŝ)²`; â is B's train-fitted history estimate and ŝ is mean absolute training step | {avg('action_mae'):.3f} |

Seed-level MAEs for A/B/C were {per_seed[SEEDS[0]][0]:.3f}/{per_seed[SEEDS[0]][1]:.3f}/{per_seed[SEEDS[0]][2]:.3f} (seed {SEEDS[0]}) and {per_seed[SEEDS[1]][0]:.3f}/{per_seed[SEEDS[1]][1]:.3f}/{per_seed[SEEDS[1]][2]:.3f} (seed {SEEDS[1]}). A predicts the training conditional mean (column 2) because opposite directions share that observation. The fixed action cost selects the same step as B: the learned speed prior is one column/frame, so both legal candidates satisfy it. C adds no independent predictive mechanism; matching B is expected and does not establish an action principle. The objective is only an action-inspired trajectory cost, not classical action minimization.

## Interpretation and limits

- The current image alone is direction-ambiguous by construction. Any above-chance result from voltage/history features indicates information about preceding inputs remains in the measured transient state over these short frame intervals.
- Results are conditional on two fixed projection seeds and this repeated five-column path. Jittered timing gives held-out sequences, but starting location is direction-correlated; this pilot does not show general motion abstraction or generalization to varied starts/speeds.
- The state-history classifier uses whole-trial train/test separation and train-only scaling/centroids. It does not alter neural dynamics or STDP.
- C is deliberately a tiny demonstration. Its speed prior is estimated from training trajectories and its action estimate reuses B, so it cannot independently establish a benefit. A more independent action-based comparison remains untested.
- Neural voltage separation is descriptive, not a semantic interpretation. One-step errors and small held-out counts are illustrative, not a benchmark claim.

## Smallest justified next experiment

Keep this network and learning rule fixed. Add a few independently seeded projection instances and randomly varied start columns (both directions through a shared interior position), preserve whole-trajectory train/test splits, and compare current-image, neural-state, and neural-history readouts. Only if history remains predictive should a separately trained transition model and a preregistered trajectory-consistency penalty be compared on held-out starts. Do not add recurrence based on this pilot.

## Runtime

Completed {sum(row['split'] == 'test' for row in scored)} held-out center evaluations across two seeds, preserving {len(scored)} train/test trajectory records. Simulated biological time: {biological_s:.3f} s across all trajectories. Wall-clock generation, analysis, and plotting: {elapsed_s:.2f} s. No parameter search or new dependencies were used.

Artifacts: `motion_trials.csv` contains train/test trajectory rows, full 1 ms sampled voltage traces, spike events, and held-out predictions/costs. `trajectory_comparison.png` plots population voltage traces and held-out errors.
"""
    (OUT / "ACTION_PILOT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    records = _make_data()
    biological_s = float(sum(r["periods"].sum() for r in records))
    scored, summaries = _classify_and_score(records)
    _write_csv(scored)
    _plot(records, scored, summaries)
    _report(scored, summaries, time.perf_counter() - started, biological_s)
    print((OUT / "ACTION_PILOT_REPORT.md").resolve())
    heldout = sum(row["split"] == "test" for row in scored)
    print(f"wall_seconds={time.perf_counter() - started:.2f}; trials={len(records)}; heldout={heldout}")
    for seed in SEEDS:
        print(seed, summaries[seed])


if __name__ == "__main__":
    main()
