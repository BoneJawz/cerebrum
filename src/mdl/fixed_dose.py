"""Fixed-dose, no-feedback validation of the three-way STDP association task."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .network import STDPAssociator
from .stimuli import make_task

SEEDS = (42, 43, 44, 45, 46)
EXPOSURES = (0, 5, 20, 50)
CONDITIONS = ("no_learning", "correct", "mismatched")
JITTER_REPLICATES = 20
JITTER_LIMIT_S = 0.001
RECALL_GAIN_MV = 60.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train(seed: int, exposures: int, condition: str):
    task = make_task(seed, exposures)
    network = STDPAssociator(seed=seed, learning=condition != "no_learning", synaptic_gain=18.0)
    taught_targets = {
        cue: ((target + 1) % task.n_items if condition == "mismatched" else target)
        for cue, target in enumerate(task.permutation)
    }
    before = network.weights().copy()
    for episode in task.episodes:
        target = episode.target
        if condition == "mismatched":
            target = (target + 1) % task.n_items
        network.present(episode.cue, target)
    trained_weights = network.weights().copy()
    network.set_plasticity(False)
    return network, task, taught_targets, before, trained_weights


def _recall(network: STDPAssociator, cue: int, jitter_s: float) -> dict:
    """One reset-state, teacher-free recall; scores use actual measured voltage."""
    network.set_plasticity(False)
    network.set_gain(RECALL_GAIN_MV)
    network.reset_transient()
    v_start = len(network.voltage.t)
    spike_start = len(network.monitor.t)
    weights_before = network.weights().copy()
    spike_counts = network.present(cue, target=None, duration=0.060, jitter=jitter_s)
    voltages = np.asarray(network.voltage.v[:, v_start:], dtype=float) / 1e-3
    times_ms = np.asarray(network.voltage.t[v_start:], dtype=float) / 1e-3
    if voltages.size:
        depolarization = np.maximum(voltages + 70.0, 0.0)
        peak_by_output = depolarization.max(axis=1)
        peak_time_by_output = times_ms[np.argmax(depolarization, axis=1)]
    else:
        peak_by_output = np.zeros(3)
        peak_time_by_output = np.full(3, -1.0)
    spike_times_ms = np.asarray(network.monitor.t[spike_start:], dtype=float) / 1e-3
    peak_prediction = int(np.argmax(peak_by_output))
    spike_prediction = int(np.argmax(spike_counts))
    spike_tie = bool(np.count_nonzero(spike_counts == np.max(spike_counts)) > 1)
    return {
        "peak_a_mv": float(peak_by_output[0]), "peak_b_mv": float(peak_by_output[1]), "peak_c_mv": float(peak_by_output[2]),
        "peak_time_a_ms": float(peak_time_by_output[0]), "peak_time_b_ms": float(peak_time_by_output[1]), "peak_time_c_ms": float(peak_time_by_output[2]),
        "spikes_a": int(spike_counts[0]), "spikes_b": int(spike_counts[1]), "spikes_c": int(spike_counts[2]),
        "peak_prediction": peak_prediction, "spike_prediction": spike_prediction,
        "spike_tie": int(spike_tie), "all_silent": int(np.sum(spike_counts) == 0),
        "first_spike_ms": float(spike_times_ms[0]) if len(spike_times_ms) else -1.0,
        "weight_change_during_recall": float(np.abs(network.weights() - weights_before).sum()),
    }


def run(outdir: str | Path = "results/fixed_dose_validation") -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261101)
    trials: list[dict] = []
    seed_summaries: list[dict] = []
    weight_summaries: list[dict] = []
    start_wall = time.perf_counter()

    for seed in SEEDS:
        for exposures in EXPOSURES:
            for condition in CONDITIONS:
                network, task, taught_targets, initial_weights, learned_weights = _train(seed, exposures, condition)
                for input_id in range(3):
                    weight_summaries.append({
                        "seed": seed, "exposures_per_association": exposures, "condition": condition,
                        "input_id": input_id, "intended_output": task.permutation[input_id],
                        "taught_output": taught_targets.get(input_id, task.permutation[input_id]),
                        "initial_weights": ";".join(f"{x:.8f}" for x in initial_weights[input_id]),
                        "final_weights": ";".join(f"{x:.8f}" for x in learned_weights[input_id]),
                    })

                correct_rows = []
                for cue, intended_target in enumerate(task.permutation):
                    for replicate in range(JITTER_REPLICATES):
                        jitter = float(rng.uniform(-JITTER_LIMIT_S, JITTER_LIMIT_S))
                        response = _recall(network, cue, jitter)
                        taught_target = taught_targets.get(cue, intended_target)
                        row = {
                            "seed": seed, "exposures_per_association": exposures, "condition": condition,
                            "cue": cue, "intended_target": intended_target, "taught_target": taught_target,
                            "replicate": replicate, "audio_jitter_ms": jitter * 1000.0,
                            **response,
                            "peak_correct_intended": int(response["peak_prediction"] == intended_target),
                            "peak_correct_taught": int(response["peak_prediction"] == taught_target),
                            "spike_correct_intended": int(response["spike_prediction"] == intended_target and not response["spike_tie"]),
                            "spike_correct_taught": int(response["spike_prediction"] == taught_target and not response["spike_tie"]),
                            "peak_margin_intended_mv": float(response[f"peak_{'abc'[intended_target]}_mv"] - max(response[f"peak_{'abc'[j]}_mv"] for j in range(3) if j != intended_target)),
                        }
                        trials.append(row)
                        correct_rows.append(row)

                seed_summaries.append({
                    "seed": seed, "exposures_per_association": exposures, "condition": condition,
                    "peak_accuracy_intended": float(np.mean([r["peak_correct_intended"] for r in correct_rows])),
                    "peak_accuracy_taught": float(np.mean([r["peak_correct_taught"] for r in correct_rows])),
                    "spike_accuracy_intended": float(np.mean([r["spike_correct_intended"] for r in correct_rows])),
                    "silent_fraction": float(np.mean([r["all_silent"] for r in correct_rows])),
                    "spike_tie_fraction": float(np.mean([r["spike_tie"] for r in correct_rows])),
                    "mean_peak_margin_mv": float(np.mean([r["peak_margin_intended_mv"] for r in correct_rows])),
                    "max_recall_weight_change": float(max(r["weight_change_during_recall"] for r in correct_rows)),
                })

    # Aggregate seed-level values, keeping the trained network as the unit of replication.
    aggregate = []
    for exposures in EXPOSURES:
        for condition in CONDITIONS:
            rows = [r for r in seed_summaries if r["exposures_per_association"] == exposures and r["condition"] == condition]
            for metric in ("peak_accuracy_intended", "peak_accuracy_taught", "spike_accuracy_intended", "silent_fraction", "spike_tie_fraction", "mean_peak_margin_mv"):
                vals = np.asarray([r[metric] for r in rows], dtype=float)
                aggregate.append({"exposures_per_association": exposures, "condition": condition, "metric": metric,
                                 "mean_across_seeds": float(vals.mean()), "sd_across_seeds": float(vals.std(ddof=1)),
                                 "min_seed": float(vals.min()), "max_seed": float(vals.max())})

    _write_csv(out / "heldout_recall_trials.csv", trials)
    _write_csv(out / "seed_level_results.csv", seed_summaries)
    _write_csv(out / "seed_aggregate_results.csv", aggregate)
    _write_csv(out / "weight_snapshots.csv", weight_summaries)
    config = {
        "seeds": SEEDS, "conditions": CONDITIONS, "exposures_per_association": EXPOSURES,
        "heldout_recall_replicates_per_cue": JITTER_REPLICATES, "heldout_audio_jitter_ms": [-1.0, 1.0],
        "recall_gain_mv": RECALL_GAIN_MV, "recall_duration_ms": 60.0,
        "primary_readout": "peak membrane depolarization above -70 mV",
        "secondary_readout": "spike count; ties are unresolved",
        "teaching_mapping": "seed-randomized permutation; mismatch rotates every target by +1 mod 3",
        "recall_transient_state": "reset to -70 mV before each trial; weights retained; plasticity gated off",
        "test_jitter_rng_seed": 20261101,
        "wall_seconds": time.perf_counter() - start_wall,
    }
    (out / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _make_figures(out, seed_summaries)
    _write_report(out, seed_summaries, config)
    return config


def _make_figures(out: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition in CONDITIONS:
            vals = [np.mean([r["peak_accuracy_intended"] for r in rows if r["condition"] == condition and r["exposures_per_association"] == exp]) for exp in EXPOSURES]
            ax.plot(EXPOSURES, vals, marker="o", label=condition)
        ax.axhline(1/3, color="black", linestyle="--", label="three-way chance")
        ax.set(xlabel="Fixed teaching exposures per association", ylabel="Held-out peak-voltage accuracy", ylim=(-.03, 1.03))
        ax.legend(); fig.tight_layout(); fig.savefig(out / "fixed_dose_learning_curve.png", dpi=150); plt.close(fig)
    except ImportError:
        return


def _write_report(out: Path, rows: list[dict], config: dict) -> None:
    lines = [
        "# Fixed-dose no-feedback validation",
        "",
        "This experiment replaces stop-when-it-looks-right training with predeclared teaching doses and held-out recall. Peak membrane depolarization is the primary readout; spike count is secondary. Recall uses a fixed 60 mV gain, ±1 ms timing jitter, a 60 ms window, no teacher input, and plasticity disabled. Transient voltage resets before each cue while learned weights persist.",
        "",
        f"Runtime: {config['wall_seconds']:.1f} wall seconds. Each seed-condition-exposure cell is one trained network; its 60 held-out trials are summarized within seed, not treated as independent networks.",
        "",
        "## Peak-voltage accuracy by fixed teaching dose",
        "",
        "| Exposures per association | No learning | Correct teaching | Mismatched teaching (intended mapping) | Mismatched teaching (taught mapping) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for exp in EXPOSURES:
        values = []
        for cond, metric in (("no_learning", "peak_accuracy_intended"), ("correct", "peak_accuracy_intended"), ("mismatched", "peak_accuracy_intended"), ("mismatched", "peak_accuracy_taught")):
            seed_rows = [r[metric] for r in rows if r["condition"] == cond and r["exposures_per_association"] == exp]
            values.append(f"{np.mean(seed_rows):.1%}")
        lines.append(f"| {exp} | " + " | ".join(values) + " |")
    lines += ["", "## Seed-level evidence", "", "See `seed_level_results.csv` for every trained network. `heldout_recall_trials.csv` records each presentation, jitter, prediction, margin, silent/tie state, and recall-time weight-change check. `weight_snapshots.csv` records initial and final connection rows for all three conditions.", "", "Correct teaching is evidence for experience-dependent learning only when its held-out intended-mapping accuracy exceeds both controls. Mismatched taught-mapping accuracy is reported separately to show whether preferences followed the alternative mapping. Repeated jitters share one trained network per seed and are not independent replications.", "", "The dashboard's adaptive stop rule can still be useful for exploration, but this fixed-dose protocol supplies the unbiased assessment: no answers or corrective feedback are used during its held-out recalls. No threshold was chosen based on test performance.", ""]
    (out / "FIXED_DOSE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
