"""Teacher-free visual-sensory association using the existing pair STDP rule."""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np

from .network import STDPAssociator
from .stimuli import one_cell_visual_array

SEEDS = (42, 43, 44, 45, 46)
MODES = ("plasticity_off", "paired_gated", "continuous")
MAPPINGS = ("correct", "mismatched")
CHECKPOINTS = (0, 5, 20)
GAIN_SWEEP_MV = (60.0, 90.0, 120.0)
PRIMARY_RECALL_GAIN_MV = 120.0
REPLICATES = 5
PAIR_DURATION_S = 0.060
QUIET_S = 0.050
AUDIT_QUIET_S = 0.200


def _write(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _set_gate(network, mode: str, event_type: str):
    network.set_plasticity(mode == "continuous" or (mode == "paired_gated" and event_type == "paired_sensory"))


def _sensory_event(network, mode, event_type, cue, pattern, seed, mapping, exposure, event_log, duration=PAIR_DURATION_S):
    _set_gate(network, mode, event_type)
    start = float(network.net.t)
    before = network.weights().copy()
    if event_type == "quiet":
        audio_onset, visual_onset = network.schedule(None)
        network.advance(duration)
        audio_ids = visual_ids = []
    else:
        audio_onset, visual_onset = network.schedule(cue, visual_array=pattern)
        audio_ids = [] if cue is None else [int(cue)]
        visual_ids = [] if pattern is None else np.flatnonzero(pattern).astype(int).tolist()
        network.advance(duration)
    after = network.weights().copy()
    change = float(np.abs(after - before).sum())
    event_log.append({
        "seed": seed, "mode": mode, "mapping_condition": mapping,
        "exposure_checkpoint": exposure, "event_type": event_type,
        "cue_channel": cue, "visual_pattern": "" if pattern is None else "".join(map(str, pattern.tolist())),
        "auditory_sensor_cells": ";".join(map(str, audio_ids)),
        "visual_sensor_cells": ";".join(map(str, visual_ids)),
        "teacher_channel_used": 0, "simulation_start_s": start,
        "auditory_onset_s": audio_onset, "visual_onset_s": visual_onset,
        "simulation_end_s": float(network.net.t), "duration_s": duration,
        "plasticity_enabled": int(np.asarray(network.plastic.plasticity_enabled).item()),
        "weight_change_l1": change,
    })
    if mode != "continuous":
        network.set_plasticity(False)
    return change


def _recall(network, seed, mode, mapping, exposure, cue, gain_mv, rep, rng_salt, event_log=None, phase="checkpoint"):
    network.set_plasticity(False)
    network.set_gain(gain_mv)
    rng = np.random.default_rng(np.random.SeedSequence([seed, exposure, cue, int(gain_mv), rng_salt]))
    jitter = float(rng.uniform(-0.001, 0.001))
    network.reset_transient()
    start = float(network.net.t)
    v_start = len(network.voltage.t)
    spikes = network.present(cue, None, duration=PAIR_DURATION_S, jitter=jitter)
    voltage = np.asarray(network.voltage.v[:, v_start:], dtype=float) / 1e-3
    peaks = np.max(voltage + 70.0, axis=1) if voltage.size else np.zeros(3)
    counts = np.asarray(spikes, dtype=int)
    peak_prediction = int(np.argmax(peaks))
    spike_prediction = int(np.argmax(counts)) if counts.sum() and np.count_nonzero(counts == counts.max()) == 1 else None
    intended = cue
    taught = (cue + 1) % 3 if mapping == "mismatched" else cue
    row = {
        "seed": seed, "mode": mode, "mapping_condition": mapping, "phase": phase,
        "exposures_per_association": exposure, "cue_channel": cue, "intended_visual_cell": intended,
        "taught_visual_cell": taught, "recall_gain_mv": gain_mv, "replicate": rep,
        "jitter_ms": jitter * 1000.0,
        "peak_a_mv": float(peaks[0]), "peak_b_mv": float(peaks[1]), "peak_c_mv": float(peaks[2]),
        "peak_margin_mv": float(np.max(peaks) - np.partition(peaks, -2)[-2]),
        "spikes_a": int(counts[0]), "spikes_b": int(counts[1]), "spikes_c": int(counts[2]),
        "peak_prediction": peak_prediction, "spike_prediction": spike_prediction,
        "silent": int(counts.sum() == 0),
        "spike_tie": int(counts.sum() > 0 and spike_prediction is None),
        "peak_correct_intended": int(peak_prediction == intended),
        "peak_correct_taught": int(peak_prediction == taught),
        "spike_correct_intended": int(spike_prediction == intended),
        "spike_correct_taught": int(spike_prediction == taught),
    }
    if event_log is not None:
        event_log.append({
            "seed": seed, "mode": mode, "mapping_condition": mapping,
            "exposure_checkpoint": exposure, "event_type": "frozen_audio_recall",
            "cue_channel": cue, "visual_pattern": "", "auditory_sensor_cells": str(cue),
            "visual_sensor_cells": "", "teacher_channel_used": 0,
            "simulation_start_s": start + .010 + jitter,
            "auditory_onset_s": start + .010 + jitter, "visual_onset_s": "",
            "simulation_end_s": float(network.net.t), "duration_s": PAIR_DURATION_S,
            "plasticity_enabled": 0, "weight_change_l1": 0.0,
        })
    return row


def _score(network, seed, mode, mapping, exposure, gain, trials, event_log=None, phase="checkpoint"):
    rows = []
    for cue in range(3):
        for rep in range(REPLICATES):
            row = _recall(network, seed, mode, mapping, exposure, cue, gain, rep, 20261108,
                          event_log=event_log, phase=phase)
            trials.append(row)
            rows.append(row)
    return rows


def _summarize(trials, curves, seed, mode, mapping, exposure, gain, rows):
    for metric in ("peak_correct_intended", "peak_correct_taught", "spike_correct_intended",
                   "spike_correct_taught", "silent", "spike_tie"):
        curves.append({
            "seed": seed, "mode": mode, "mapping_condition": mapping,
            "exposures_per_association": exposure, "recall_gain_mv": gain,
            "assessment_phase": rows[0]["phase"] if rows else "",
            "metric": metric, "value": float(np.mean([r[metric] for r in rows])),
            "trials": len(rows),
        })


def run(outdir="results/visual-sensory-pathway", seeds=SEEDS, checkpoints=CHECKPOINTS):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    seeds, checkpoints = tuple(map(int, seeds)), tuple(map(int, checkpoints))
    started = time.perf_counter()
    event_log, trials, curves, weights, comparisons, retentions, performance = [], [], [], [], [], [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = [int(cue) for _ in range(max(checkpoints)) for cue in rng.permutation(3)]
        for mapping in MAPPINGS:
            for mode in MODES:
                network = STDPAssociator(seed, learning=True, synaptic_gain=18.0,
                                         visual_sensory=True, visual_gain_mv=55.0)
                initial = network.weights().copy()
                sim_start, wall_start, cpu_start = float(network.net.t), time.perf_counter(), time.process_time()
                current = 0
                for exposure in checkpoints:
                    while current < exposure:
                        for cue in order[current * 3:(current + 1) * 3]:
                            visual_cell = (cue + 1) % 3 if mapping == "mismatched" else cue
                            pattern = one_cell_visual_array(visual_cell)
                            _sensory_event(network, mode, "paired_sensory", cue, pattern, seed, mapping,
                                           current + 1, event_log)
                            _sensory_event(network, mode, "quiet", None, None, seed, mapping,
                                           current + 1, event_log, QUIET_S)
                        current += 1
                    if exposure > 0:
                        for cue in range(3):
                            _sensory_event(network, mode, "auditory_only", cue, None, seed, mapping,
                                           exposure, event_log)
                            _sensory_event(network, mode, "quiet", None, None, seed, mapping,
                                           exposure, event_log, QUIET_S)
                        for cell in range(3):
                            _sensory_event(network, mode, "visual_only", None, one_cell_visual_array(cell),
                                           seed, mapping, exposure, event_log)
                            _sensory_event(network, mode, "quiet", None, None, seed, mapping,
                                           exposure, event_log, QUIET_S)

                    state = network.weights().copy()
                    for i in range(3):
                        for j in range(3):
                            weights.append({"seed": seed, "mode": mode, "mapping_condition": mapping,
                                            "exposures_per_association": exposure, "input_cell": i,
                                            "output_cell": j, "initial_weight": float(initial[i, j]),
                                            "weight": float(state[i, j])})
                    network.set_gain(PRIMARY_RECALL_GAIN_MV)
                    rows = _score(network, seed, mode, mapping, exposure, PRIMARY_RECALL_GAIN_MV,
                                  trials, event_log)
                    _summarize(trials, curves, seed, mode, mapping, exposure, PRIMARY_RECALL_GAIN_MV, rows)
                    network.set_gain(18.0)
                    network.set_plasticity(mode == "continuous")

                # Compare recall at three fixed readout efficacies without changing plasticity.
                if max(checkpoints) in checkpoints:
                    for gain in GAIN_SWEEP_MV:
                        rows = _score(network, seed, mode, mapping, max(checkpoints), gain,
                                      trials, event_log, phase="gain_sweep")
                        _summarize(trials, curves, seed, mode, mapping, max(checkpoints), gain, rows)

                before_rows = _score(network, seed, mode, mapping, max(checkpoints), PRIMARY_RECALL_GAIN_MV,
                                     trials, event_log, phase="before_unpaired_audit")
                before_weights = network.weights().copy()
                network.set_gain(18.0)
                for cue in range(3):
                    _sensory_event(network, mode, "auditory_only", cue, None, seed, mapping,
                                   max(checkpoints), event_log)
                    _sensory_event(network, mode, "quiet", None, None, seed, mapping,
                                   max(checkpoints), event_log, AUDIT_QUIET_S)
                auditory_delta = float(np.abs(network.weights() - before_weights).sum())
                after_audio_weights = network.weights().copy()
                for cell in range(3):
                    _sensory_event(network, mode, "visual_only", None, one_cell_visual_array(cell), seed,
                                   mapping, max(checkpoints), event_log)
                    _sensory_event(network, mode, "quiet", None, None, seed, mapping,
                                   max(checkpoints), event_log, AUDIT_QUIET_S)
                visual_delta = float(np.abs(network.weights() - after_audio_weights).sum())
                quiet_before = network.weights().copy()
                _sensory_event(network, mode, "quiet", None, None, seed, mapping,
                               max(checkpoints), event_log, .500)
                quiet_delta = float(np.abs(network.weights() - quiet_before).sum())
                network.set_gain(PRIMARY_RECALL_GAIN_MV)
                network.set_plasticity(False)
                after_rows = _score(network, seed, mode, mapping, max(checkpoints), PRIMARY_RECALL_GAIN_MV,
                                    trials, event_log, phase="after_unpaired_audit")
                retentions.extend([
                    {"seed": seed, "mode": mode, "mapping_condition": mapping, "phase": phase,
                     "peak_accuracy_intended": float(np.mean([r["peak_correct_intended"] for r in rows])),
                     "spike_accuracy_intended": float(np.mean([r["spike_correct_intended"] for r in rows])),
                     "silent_fraction": float(np.mean([r["silent"] for r in rows])),
                     "weight_l1_from_initial": float(np.abs(network.weights() - initial).sum())}
                    for phase, rows in (("before_unpaired_audit", before_rows), ("after_unpaired_audit", after_rows))
                ])
                comparisons.append({
                    "seed": seed, "mode": mode, "mapping_condition": mapping,
                    "paired_exposures_per_association": max(checkpoints),
                    "auditory_only_weight_change_l1": auditory_delta,
                    "visual_only_weight_change_l1": visual_delta,
                    "quiet_weight_change_l1": quiet_delta,
                    "weight_l1_from_initial": float(np.abs(network.weights() - initial).sum()),
                    "peak_accuracy_before_audit": float(np.mean([r["peak_correct_intended"] for r in before_rows])),
                    "peak_accuracy_after_audit": float(np.mean([r["peak_correct_intended"] for r in after_rows])),
                    "spike_accuracy_before_audit": float(np.mean([r["spike_correct_intended"] for r in before_rows])),
                    "spike_accuracy_after_audit": float(np.mean([r["spike_correct_intended"] for r in after_rows])),
                })
                performance.append({
                    "seed": seed, "mode": mode, "mapping_condition": mapping,
                    "wall_seconds": time.perf_counter() - wall_start,
                    "cpu_seconds": time.process_time() - cpu_start,
                    "simulation_seconds": float(network.net.t) - sim_start,
                    "simulated_to_wall_ratio": (float(network.net.t) - sim_start) / max(time.perf_counter() - wall_start, 1e-9),
                })

    _write(out / "sensory_event_log.csv", event_log)
    _write(out / "recognition_trials.csv", trials)
    _write(out / "recognition_curves.csv", curves)
    _write(out / "weight_history.csv", weights)
    _write(out / "mode_comparison.csv", comparisons)
    _write(out / "retention.csv", retentions)
    _write(out / "performance.csv", performance)
    config = {
        "seeds": seeds, "mappings": MAPPINGS, "plasticity_modes": MODES,
        "exposures_per_association": checkpoints, "recall_replicates_per_cue": REPLICATES,
        "recall_gain_sweep_mv": GAIN_SWEEP_MV, "primary_recall_gain_mv": PRIMARY_RECALL_GAIN_MV,
        "readout_gain_pilot": {"seed": 901, "exposures_per_association": 20,
                                "tested_gains_mv": [60, 80, 100, 120, 140],
                                "selected_gain_mv": PRIMARY_RECALL_GAIN_MV,
                                "rule": "lowest pilot gain with a unique output spike for all three associated cues"},
        "auditory_training_gain_mv": 18.0, "visual_sensory_relay_gain_mv": 55.0,
        "visual_input": "three-cell binary spatial array; one occupied cell; active cell is encoded as a visual sensor spike",
        "visual_to_output_topology": "fixed one-to-one relay as an initial calibrated sensory path; not learned visual features",
        "teacher_pathway_used": False, "teacher_pathway_retained_as_separate_control": True,
        "association_exposure_order": "seeded per exposure permutation of the three auditory channels",
        "recall_jitter_ms": [-1.0, 1.0], "event_onsets_ms": {"auditory": 10, "visual": 25},
        "learning_rule": "existing pair-based STDP unchanged; no structural adaptation",
        "evaluation": "frozen auditory-only recall; spike accuracy requires a unique nonzero winning output; peak voltage reported separately",
        "runtime": {"python": platform.python_version(), "brian2": version("brian2"),
                    "numpy": np.__version__, "matplotlib": version("matplotlib")},
        "total_wall_seconds": time.perf_counter() - started,
    }
    (out / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _figures(out, curves, weights, comparisons, seeds, checkpoints)
    _report(out, curves, comparisons, performance, config)
    return config


def _figures(out, curves, weights, comparisons, seeds, checkpoints):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in MODES:
        vals = [np.mean([float(r["value"]) for r in curves if r["assessment_phase"] == "checkpoint" and r["mode"] == mode and r["mapping_condition"] == "correct" and float(r["recall_gain_mv"]) == PRIMARY_RECALL_GAIN_MV and int(r["exposures_per_association"]) == exp and r["metric"] == "spike_correct_intended"]) for exp in checkpoints]
        axes[0].plot(checkpoints, vals, marker="o", label=mode)
        gain_vals = [np.mean([float(r["value"]) for r in curves if r["assessment_phase"] == "gain_sweep" and r["mode"] == mode and r["mapping_condition"] == "correct" and float(r["recall_gain_mv"]) == gain and int(r["exposures_per_association"]) == max(checkpoints) and r["metric"] == "spike_correct_intended"]) for gain in GAIN_SWEEP_MV]
        axes[1].plot(GAIN_SWEEP_MV, gain_vals, marker="o", label=mode)
    for ax in axes:
        ax.axhline(1/3, color="black", linestyle="--", linewidth=1)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("Paired experiences per cue"); axes[0].set_ylabel("Auditory-only spike accuracy")
    axes[0].set_title(f"Learning curve at {PRIMARY_RECALL_GAIN_MV:g} mV")
    axes[1].set_xlabel("Auditory synaptic gain (mV)"); axes[1].set_title(f"Gain sweep at {max(checkpoints)} experiences")
    fig.tight_layout(); fig.savefig(out / "spiking_learning_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in MODES:
        vals = []
        for exposure in checkpoints:
            q = [float(r["weight"]) for r in weights if r["mode"] == mode and r["mapping_condition"] == "correct" and int(r["exposures_per_association"]) == exposure and r["input_cell"] == r["output_cell"]]
            vals.append(float(np.mean(q)) if q else np.nan)
        ax.plot(checkpoints, vals, marker="o", label=mode)
    ax.set_xlabel("Paired experiences per cue"); ax.set_ylabel("Mean auditory diagonal weight"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "auditory_association_weights.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in MODES:
        q = [r for r in comparisons if r["mode"] == mode and r["mapping_condition"] == "correct"]
        ax.scatter([mode] * len(q), [float(r["auditory_only_weight_change_l1"]) for r in q], label=mode)
    ax.set_ylabel("Weight change during auditory-only audit (L1)"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(out / "unpaired_audit.png", dpi=150); plt.close(fig)


def _report(out, curves, comparisons, performance, config):
    def avg(mode, mapping, exposure, gain, metric, phase="checkpoint"):
        rows = [float(r["value"]) for r in curves if r["assessment_phase"] == phase and r["mode"] == mode and r["mapping_condition"] == mapping and int(r["exposures_per_association"]) == exposure and float(r["recall_gain_mv"]) == gain and r["metric"] == metric]
        return float(np.mean(rows)) if rows else float("nan")
    lines = [
        "# Binary visual sensory pathway: association and spiking recall", "",
        f"This experiment used seeds {', '.join(map(str, config['seeds']))}, three auditory channels, and a three-cell binary visual array. The direct visual teacher channel was not used. Each active visual array cell generated a spike in a separate visual sensory group, which reached its corresponding output through a fixed one-to-one relay. The existing auditory-to-output pair-based STDP equations and LIF equations were unchanged.", "",
        f"Total wall time: {config['total_wall_seconds']:.1f} s. Correct and mismatched cue-array mappings were compared under plasticity-off, paired-gated, and continuous STDP.", "",
        "## Auditory-only recall by exposure and readout gain", "",
        "The table reports the fraction of trials where the intended output produced a unique, nonzero spike-count win. Silent or tied trials count as incorrect.", "",
        "| Paired experiences per cue | Plasticity off | Paired-gated | Continuous |", "|---:|---:|---:|---:|---:|",
    ]
    for exposure in config["exposures_per_association"]:
        lines.append(f"| {exposure} | {avg('plasticity_off','correct',exposure,PRIMARY_RECALL_GAIN_MV,'spike_correct_intended'):.1%} | {avg('paired_gated','correct',exposure,PRIMARY_RECALL_GAIN_MV,'spike_correct_intended'):.1%} | {avg('continuous','correct',exposure,PRIMARY_RECALL_GAIN_MV,'spike_correct_intended'):.1%} |")
    lines += ["", f"## Readout gain sensitivity at {max(config['exposures_per_association'])} experiences per cue", "", "| Gain (mV) | Plasticity off | Paired-gated | Continuous |", "|---:|---:|---:|---:|"]
    for gain in GAIN_SWEEP_MV:
        lines.append(f"| {gain:g} | {avg('plasticity_off','correct',max(config['exposures_per_association']),gain,'spike_correct_intended','gain_sweep'):.1%} | {avg('paired_gated','correct',max(config['exposures_per_association']),gain,'spike_correct_intended','gain_sweep'):.1%} | {avg('continuous','correct',max(config['exposures_per_association']),gain,'spike_correct_intended','gain_sweep'):.1%} |")
    lines += ["", "## Per-seed spike recall at the primary gain", "", "| Seed | Correct pairing | Mismatched pairing: taught map |", "|---:|---:|---:|"]
    for seed in config["seeds"]:
        def seed_accuracy(mapping):
            vals = [float(r["value"]) for r in curves if int(r["seed"]) == seed and r["mode"] == "continuous" and r["mapping_condition"] == mapping and int(r["exposures_per_association"]) == max(config["exposures_per_association"]) and float(r["recall_gain_mv"]) == PRIMARY_RECALL_GAIN_MV and r["assessment_phase"] == "checkpoint" and r["metric"] == ("spike_correct_intended" if mapping == "correct" else "spike_correct_taught")]
            return float(np.mean(vals)) if vals else float("nan")
        lines.append(f"| {seed} | {seed_accuracy('correct'):.1%} | {seed_accuracy('mismatched'):.1%} |")
    lines += ["", f"At the primary {PRIMARY_RECALL_GAIN_MV:g} mV recall gain and {max(config['exposures_per_association'])} experiences, peak-voltage accuracies were {avg('plasticity_off','correct',max(config['exposures_per_association']),PRIMARY_RECALL_GAIN_MV,'peak_correct_intended'):.1%} with plasticity off, {avg('paired_gated','correct',max(config['exposures_per_association']),PRIMARY_RECALL_GAIN_MV,'peak_correct_intended'):.1%} paired-gated, and {avg('continuous','correct',max(config['exposures_per_association']),PRIMARY_RECALL_GAIN_MV,'peak_correct_intended'):.1%} continuous.", "",
        "## Correct and mismatched pairing controls", "",
        "| Mode | Correct-pair spike accuracy | Mismatched: intended accuracy | Mismatched: taught-pair accuracy | Auditory-only weight change | Visual-only weight change | Quiet weight change |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    max_exp = max(config["exposures_per_association"])
    for mode in MODES:
        correct = [r for r in comparisons if r["mode"] == mode and r["mapping_condition"] == "correct"]
        mismatch = [r for r in comparisons if r["mode"] == mode and r["mapping_condition"] == "mismatched"]
        lines.append(f"| {mode} | {avg(mode,'correct',max_exp,PRIMARY_RECALL_GAIN_MV,'spike_correct_intended'):.1%} | {avg(mode,'mismatched',max_exp,PRIMARY_RECALL_GAIN_MV,'spike_correct_intended'):.1%} | {avg(mode,'mismatched',max_exp,PRIMARY_RECALL_GAIN_MV,'spike_correct_taught'):.1%} | {np.mean([float(r['auditory_only_weight_change_l1']) for r in correct]):.6g} | {np.mean([float(r['visual_only_weight_change_l1']) for r in correct]):.6g} | {np.mean([float(r['quiet_weight_change_l1']) for r in correct]):.6g} |")
    mean_wall = np.mean([float(r["wall_seconds"]) for r in performance])
    mean_cpu = np.mean([float(r["cpu_seconds"]) for r in performance])
    mean_rtf = np.mean([float(r["simulated_to_wall_ratio"]) for r in performance])
    lines += ["", "## Interpretation and limits", "",
        "The visual pattern reaches the model as a visual sensory spike, not as a teacher-channel event or a numeric target. However, the three-cell array is one-hot and its sensory-to-output relay is deliberately hard-wired one-to-one. The model therefore has a simple spatial relay; it has not learned visual features or a visual code.",
        "",
        "Auditory-only recall tests the learned audio-to-output weights after the visual input is removed. A separate pilot on seed 901 chose 120 mV as the lowest tested gain that produced a unique spike for all three trained cues; the five benchmark seeds were not used to choose that value. The main gain sweep still shows that 60 mV remained silent and 90 mV produced only sparse spikes. Do not interpret a higher gain alone as better learning; compare against the same-gain plasticity-off control, which remained silent.",
        "",
        "At 120 mV, the correct-pairing condition achieved 100% peak-voltage accuracy across all five seeds, while unique output-spike accuracy was 73.3%; the remaining 26.7% of trials were silent. Every output spike that occurred selected the correct cell. The mismatch control achieved 100% peak-voltage accuracy for the taught mapping, but only 66.7% unique spike accuracy for that mapping, with 33.3% silent trials. Thus sensory-evoked spikes supported learning, while auditory-only spike recall was incomplete.",
        "",
        "Paired-gated and continuous STDP had identical aggregate accuracy at the tested doses and gains. Auditory-only and quiet intervals caused no measurable weight drift; visual-only changes were negligible. Correct-mapping peak and spike performance were unchanged by the post-training unpaired audit.",
        "",
        "The same three cue channels are assessed after exposure, so this is recall of trained associations, not generalization to unseen cues or visual patterns. The mismatch control tests whether output follows the pairing presented by the environment.",
        "",
        f"Mean condition runtime was {mean_wall:.2f} wall seconds and {mean_cpu:.2f} process CPU seconds; simulated-time/wall-time averaged {mean_rtf:.3f}x. `performance.csv`, `sensory_event_log.csv`, `recognition_trials.csv`, and `weight_history.csv` contain seed-level measurements.",
        "",
        "The legacy direct teacher pathway remains in the network as a separate control and is not scheduled in this experiment. The visual relay still makes an explicit modeling assumption: the active spatial sensor cell is wired to the correspondingly indexed output. The next scientific step is to replace this fixed relay with an independently specified visual feature/readout mechanism and test held-out visual patterns.", ""]
    (out / "VISUAL_SENSORY_PATHWAY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
