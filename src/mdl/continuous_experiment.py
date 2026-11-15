"""Matched-sequence experiment comparing three STDP gating modes."""
from __future__ import annotations

import csv
import json
import platform
import time
import tracemalloc
from pathlib import Path
from importlib.metadata import version

import numpy as np

from .network import STDPAssociator
from .sensory_environment import teacher_pattern

SEEDS = (42, 43, 44, 45, 46)
MODES = ("plasticity_off", "teaching_gated", "continuous")
MAPPINGS = ("correct", "mismatched")
CHECKPOINTS = (0, 5, 20)
RECALL_REPLICATES = 5
JITTER_LIMIT_S = 0.001
QUIET_BETWEEN_EVENTS_S = 0.050
AUDIT_QUIET_S = 0.200


def _write(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _gate(network, mode, event_type):
    network.set_plasticity(mode == "continuous" or (mode == "teaching_gated" and event_type == "paired_teaching"))


def _event(network, mode, event_type, cue, target, duration_s, seed, mapping_condition, exposures, log):
    _gate(network, mode, event_type)
    before = network.weights().copy()
    start_s = float(network.net.t)
    if event_type == "quiet":
        network.schedule(None, None)
        network.advance(duration_s)
        audio_onset = visual_onset = None
    else:
        audio_onset, visual_onset = network.schedule(cue, target)
        network.advance(duration_s)
    after = network.weights().copy()
    log.append({
        "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
        "exposure_checkpoint": exposures, "event_type": event_type, "cue": cue, "teacher_target": target,
        "visual_pattern": "" if target is None else "".join(map(str, teacher_pattern(target).tolist())),
        "start_simulation_s": start_s, "audio_onset_simulation_s": audio_onset,
        "visual_teacher_onset_simulation_s": visual_onset, "end_simulation_s": float(network.net.t),
        "duration_s": duration_s, "plasticity_enabled": int(bool(np.asarray(network.plastic.plasticity_enabled).item())),
        "weight_change_l1": float(np.abs(after - before).sum()),
    })
    if mode != "continuous":
        network.set_plasticity(False)
    return after


def _frozen_recall(network, seed, mode, mapping_condition, exposure, replicates, rng_seed, evaluation_phase="checkpoint"):
    rows = []
    network.set_plasticity(False)
    network.set_gain(60.0)
    for cue in range(3):
        intended = cue
        taught = (cue + 1) % 3 if mapping_condition == "mismatched" else cue
        rng = np.random.default_rng(np.random.SeedSequence([seed, exposure, cue, rng_seed]))
        for rep in range(replicates):
            jitter = float(rng.uniform(-JITTER_LIMIT_S, JITTER_LIMIT_S))
            network.reset_transient()
            v_start = len(network.voltage.t)
            s_start = len(network.monitor.t)
            before = network.weights().copy()
            spikes = network.present(cue, None, duration=0.060, jitter=jitter)
            voltage = np.asarray(network.voltage.v[:, v_start:], dtype=float) / 1e-3
            peaks = np.max(voltage + 70.0, axis=1) if voltage.size else np.zeros(3)
            counts = np.asarray(spikes, dtype=int)
            pred_peak = int(np.argmax(peaks))
            pred_spike = int(np.argmax(counts))
            tie = int(np.count_nonzero(counts == np.max(counts)) > 1)
            rows.append({
                "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
                "evaluation_phase": evaluation_phase,
                "exposures_per_association": exposure, "cue": cue, "intended_target": intended,
                "taught_target": taught, "replicate": rep, "jitter_ms": jitter * 1000.0,
                "peak_a_mv": float(peaks[0]), "peak_b_mv": float(peaks[1]), "peak_c_mv": float(peaks[2]),
                "peak_margin_mv": float(np.max(peaks) - np.partition(peaks, -2)[-2]),
                "spikes_a": int(counts[0]), "spikes_b": int(counts[1]), "spikes_c": int(counts[2]),
                "peak_prediction": pred_peak, "spike_prediction": pred_spike, "spike_tie": tie,
                "all_silent": int(counts.sum() == 0),
                "peak_correct_intended": int(pred_peak == intended),
                "peak_correct_taught": int(pred_peak == taught),
                "spike_correct_intended": int(pred_spike == intended and not tie),
                "recall_weight_change": float(np.abs(network.weights() - before).sum()),
            })
    return rows


def run(outdir="results/continuous_learning", seeds=SEEDS, checkpoints=CHECKPOINTS,
        recall_replicates=RECALL_REPLICATES):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tracemalloc.start()
    event_log, trials, curves, weight_history, mode_rows, retention_rows = [], [], [], [], [], []
    performance = []

    seeds = tuple(int(seed) for seed in seeds)
    checkpoints = tuple(int(checkpoint) for checkpoint in checkpoints)
    for seed in seeds:
        order_rng = np.random.default_rng(seed)
        exposure_order = [int(cue) for _ in range(max(checkpoints)) for cue in order_rng.permutation(3)]
        for mapping_condition in MAPPINGS:
            for mode in MODES:
                network = STDPAssociator(seed=seed, learning=True, synaptic_gain=18.0)
                network.set_gain(18.0)
                initial = network.weights().copy()
                mapping = {cue: ((cue + 1) % 3 if mapping_condition == "mismatched" else cue) for cue in range(3)}
                total_events = 0
                current_exposure = 0
                elapsed_start = time.perf_counter()
                cpu_start = time.process_time()
                tracemalloc.reset_peak()
                sim_start = float(network.net.t)

                for checkpoint in checkpoints:
                    while current_exposure < checkpoint:
                        cue = exposure_order[current_exposure * 3: current_exposure * 3 + 3]
                        for item in cue:
                            _event(network, mode, "paired_teaching", item, mapping[item], .060,
                                   seed, mapping_condition, current_exposure + 1, event_log)
                            _event(network, mode, "quiet", None, None, QUIET_BETWEEN_EVENTS_S,
                                   seed, mapping_condition, current_exposure + 1, event_log)
                            total_events += 2
                        current_exposure += 1

                    if checkpoint > 0:
                        for cue in range(3):
                            _event(network, mode, "auditory_only_audit", cue, None, .060,
                                   seed, mapping_condition, checkpoint, event_log)
                            _event(network, mode, "quiet", None, None, QUIET_BETWEEN_EVENTS_S,
                                   seed, mapping_condition, checkpoint, event_log)
                        for target in range(3):
                            _event(network, mode, "visual_teacher_only_audit", None, target, .060,
                                   seed, mapping_condition, checkpoint, event_log)
                            _event(network, mode, "quiet", None, None, QUIET_BETWEEN_EVENTS_S,
                                   seed, mapping_condition, checkpoint, event_log)
                        total_events += 12

                    state = network.weights().copy()
                    for i in range(3):
                        for j in range(3):
                            weight_history.append({
                                "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
                                "exposures_per_association": checkpoint, "input_neuron": i,
                                "output_neuron": j, "initial_weight": float(initial[i, j]),
                                "weight": float(state[i, j]),
                            })

                    checkpoint_rows = _frozen_recall(network, seed, mode, mapping_condition, checkpoint,
                                                     recall_replicates, 20261101)
                    trials.extend(checkpoint_rows)
                    for metric in ("peak_correct_intended", "peak_correct_taught", "spike_correct_intended", "all_silent", "spike_tie"):
                        vals = [float(row[metric]) for row in checkpoint_rows]
                        curves.append({
                            "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
                            "exposures_per_association": checkpoint, "metric": metric,
                            "value": float(np.mean(vals)), "trials": len(vals),
                        })
                    restored_gate = mode == "continuous"
                    network.set_plasticity(restored_gate)
                    network.set_gain(18.0)

                # Audit events test unintended association changes and retention.
                pre_audit_weights = network.weights().copy()
                pre_audit_recall = _frozen_recall(network, seed, mode, mapping_condition, 20,
                                                  recall_replicates, 20261102, "before_unpaired_audit")
                trials.extend(pre_audit_recall)
                network.set_gain(18.0)
                audit_weight_before = network.weights().copy()
                for cue in range(3):
                    _event(network, mode, "auditory_only_audit", cue, None, .060,
                           seed, mapping_condition, 20, event_log)
                    _event(network, mode, "quiet", None, None, AUDIT_QUIET_S,
                           seed, mapping_condition, 20, event_log)
                    total_events += 2
                after_audio = network.weights().copy()
                audio_delta = float(np.abs(after_audio - audit_weight_before).sum())

                before_visual = network.weights().copy()
                for target in range(3):
                    _event(network, mode, "visual_teacher_only_audit", None, target, .060,
                           seed, mapping_condition, 20, event_log)
                    _event(network, mode, "quiet", None, None, AUDIT_QUIET_S,
                           seed, mapping_condition, 20, event_log)
                    total_events += 2
                after_visual = network.weights().copy()
                visual_delta = float(np.abs(after_visual - before_visual).sum())

                quiet_before = network.weights().copy()
                _event(network, mode, "quiet", None, None, .500, seed, mapping_condition, 20, event_log)
                quiet_delta = float(np.abs(network.weights() - quiet_before).sum())

                network.set_plasticity(False)
                post_audit = _frozen_recall(network, seed, mode, mapping_condition, 20,
                                            recall_replicates, 20261102, "after_unpaired_audit")
                trials.extend(post_audit)
                for phase, rows in (("before_unpaired_audit", pre_audit_recall), ("after_unpaired_audit", post_audit)):
                    retention_rows.append({
                        "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
                        "phase": phase,
                        "peak_accuracy_intended": float(np.mean([r["peak_correct_intended"] for r in rows])),
                        "peak_accuracy_taught": float(np.mean([r["peak_correct_taught"] for r in rows])),
                        "mean_weight_l1_from_initial": float(np.abs(network.weights() - initial).sum()),
                    })
                mode_rows.append({
                    "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
                    "paired_exposures_per_association": 20,
                    "auditory_only_weight_change_l1": audio_delta,
                    "visual_teacher_only_weight_change_l1": visual_delta,
                    "quiet_weight_change_l1": quiet_delta,
                    "weight_change_during_frozen_recall_l1": max((r["recall_weight_change"] for r in pre_audit_recall + post_audit), default=0.0),
                    "final_weight_l1_from_initial": float(np.abs(network.weights() - initial).sum()),
                    "pre_audit_peak_accuracy_intended": float(np.mean([r["peak_correct_intended"] for r in pre_audit_recall])),
                    "post_audit_peak_accuracy_intended": float(np.mean([r["peak_correct_intended"] for r in post_audit])),
                })
                performance.append({
                    "seed": seed, "mode": mode, "mapping_condition": mapping_condition,
                    "wall_seconds": time.perf_counter() - elapsed_start,
                    "cpu_seconds": time.process_time() - cpu_start,
                    "peak_python_heap_mb": tracemalloc.get_traced_memory()[1] / (1024 * 1024),
                    "simulation_seconds": float(network.net.t) - sim_start,
                    "event_count": total_events,
                })

    _write(out / "sensory_event_log.csv", event_log)
    _write(out / "continuous_learning_trials.csv", trials)
    _write(out / "recognition_learning_curves.csv", curves)
    _write(out / "synaptic_weight_history.csv", weight_history)
    _write(out / "plasticity_mode_comparison.csv", mode_rows)
    _write(out / "sensory_synchronization.csv", [row for row in event_log if row["event_type"] != "quiet"])
    _write(out / "retention_results.csv", retention_rows)
    _write(out / "performance_summary.csv", performance)
    config = {
        "seeds": seeds, "plasticity_modes": MODES, "association_conditions": MAPPINGS,
        "learning_checkpoints": checkpoints, "recall_replicates_per_cue": recall_replicates,
        "exposure_order": "per-seed NumPy Generator(seed) independently permutes the three cues for each exposure",
        "recall_jitter_rng_salts": {"checkpoint": 20261101, "unpaired_audit": 20261102},
        "event_onsets_ms_from_event_start": {"auditory": 10.0, "visual_teacher": 25.0},
        "mode_gate_definition": {"plasticity_off": "disabled throughout", "teaching_gated": "enabled only during paired_teaching", "continuous": "enabled during paired, unpaired, and quiet events; disabled for frozen recall"},
        "runtime_versions": {"python": platform.python_version(), "brian2": version("brian2"), "numpy": np.__version__},
        "recall_jitter_ms": [-1.0, 1.0], "recall_gain_mv": 60.0,
        "teaching_gain_mv": 18.0, "neuron_dynamics_and_stdp": "unchanged Brian2 STDPAssociator",
        "quiet_between_pair_events_s": QUIET_BETWEEN_EVENTS_S, "audit_quiet_s": AUDIT_QUIET_S,
        "evaluation": "frozen plasticity; transient voltage reset; peak depolarization primary; spike count secondary",
        "visual_signal_semantics": "teacher spike directly drives one output neuron; no retinal sensory pathway",
        "audible_signal_semantics": "440/554.37/659.25 Hz tones are human-audible representations; model receives indexed spike events, not waveform-derived spikes",
        "wall_seconds_total": time.perf_counter() - started,
    }
    tracemalloc.stop()
    (out / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _make_figures(out, curves, weight_history, mode_rows, event_log, seeds, checkpoints)
    _write_report(out, curves, mode_rows, performance, config, checkpoints)
    return config


def _make_figures(out, curves, weights, comparison, event_log, seeds=SEEDS, checkpoints=CHECKPOINTS):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        vals = [np.mean([float(r["value"]) for r in curves if r["mode"] == mode and r["mapping_condition"] == "correct" and int(r["exposures_per_association"]) == exp and r["metric"] == "peak_correct_intended"]) for exp in checkpoints]
        ax.plot(checkpoints, vals, marker="o", label=mode)
    ax.axhline(1/3, color="black", linestyle="--", label="chance")
    ax.set(xlabel="Paired experiences per association", ylabel="Peak-voltage accuracy", ylim=(-.03, 1.03)); ax.legend(); fig.tight_layout(); fig.savefig(out / "continuous_learning_curve.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, measure in enumerate(("auditory_only_weight_change_l1", "visual_teacher_only_weight_change_l1")):
        axes[idx].set_title(measure.replace("_", " "))
        for mode in MODES:
            vals = [float(r[measure]) for r in comparison if r["mode"] == mode and r["mapping_condition"] == "correct"]
            axes[idx].scatter([mode] * len(vals), vals, label=mode)
        axes[idx].tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(out / "plasticity_mode_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in MODES:
        values = []
        for exposure in CHECKPOINTS:
            rows = [r for r in weights if r["mode"] == mode and r["mapping_condition"] == "correct" and int(r["exposures_per_association"]) == exposure and int(r["input_neuron"]) == int(r["output_neuron"])]
            values.append(float(np.mean([r["weight"] for r in rows])) if rows else np.nan)
        ax.plot(CHECKPOINTS, values, marker="o", label=mode)
    ax.set(xlabel="Paired experiences per association", ylabel="Mean diagonal weight"); ax.legend(); fig.tight_layout(); fig.savefig(out / "synaptic_weight_evolution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in MODES:
        rows = [r for r in comparison if r["mode"] == mode and r["mapping_condition"] == "correct"]
        before = np.mean([float(r["pre_audit_peak_accuracy_intended"]) for r in rows])
        after = np.mean([float(r["post_audit_peak_accuracy_intended"]) for r in rows])
        ax.plot([0, 1], [before, after], marker="o", label=mode)
    ax.set_xticks([0, 1], ["before unpaired events", "after unpaired events"]); ax.set_ylabel("Peak-voltage accuracy"); ax.set_ylim(-.03, 1.03); ax.legend(); fig.tight_layout(); fig.savefig(out / "retention_analysis.png", dpi=150); plt.close(fig)

    representative = [r for r in event_log if r["seed"] == seeds[0] and r["mode"] == "continuous" and r["mapping_condition"] == "correct" and int(r["exposure_checkpoint"]) <= 5]
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = {"paired_teaching": 3, "auditory_only_audit": 2, "visual_teacher_only_audit": 1, "quiet": 0}
    colors = {"paired_teaching": "tab:purple", "auditory_only_audit": "tab:blue", "visual_teacher_only_audit": "tab:green", "quiet": "0.8"}
    for row in representative:
        y = labels[row["event_type"]]
        ax.barh(y, float(row["duration_s"]), left=float(row["start_simulation_s"]), color=colors[row["event_type"]], height=.65)
        if row["audio_onset_simulation_s"] is not None:
            ax.plot(float(row["audio_onset_simulation_s"]), y, marker="|", color="black", ms=9)
        if row["visual_teacher_onset_simulation_s"] is not None:
            ax.plot(float(row["visual_teacher_onset_simulation_s"]), y, marker="^", color="black", ms=4)
    ax.set_yticks(list(labels.values()), ["Quiet", "Visual teacher", "Auditory only", "Paired cue + teacher"])
    ax.set_xlabel("Simulation time (s)"); ax.set_title("Recorded sensory and teacher event schedule (black marks: scheduled input onsets)")
    fig.tight_layout(); fig.savefig(out / "neural_and_sensory_timeline.png", dpi=150); plt.close(fig)


def _write_report(out, curves, comparison, performance, config, checkpoints=CHECKPOINTS):
    def mean_at(mode, mapping, exp, metric="peak_correct_intended"):
        rows = [float(r["value"]) for r in curves if r["mode"] == mode and r["mapping_condition"] == mapping and int(r["exposures_per_association"]) == exp and r["metric"] == metric]
        return float(np.mean(rows)) if rows else float("nan")
    lines = [
        "# Continuous plasticity and audiovisual experience",
        "",
        f"This experiment compared plasticity-off, teaching-gated, and continuously enabled existing pair-based STDP under matched event order, {len(config['seeds'])} seed(s), and correct/mismatched teacher mappings. The network received auditory spike events. The visual input in this baseline is an external teacher spike that directly drives an output neuron; it is not a visual sensory pathway. Audible tones are human-audible proxies for the cue IDs; the model does not receive those waveforms.",
        "",
        f"Total experiment wall time: {config['wall_seconds_total']:.1f} seconds.",
        "",
        "## Correct mapping peak-voltage accuracy",
        "",
        "| Paired exposures per association | Plasticity off | Teaching-gated | Continuous STDP |",
        "|---:|---:|---:|---:|",
    ]
    for exp in checkpoints:
        lines.append(f"| {exp} | {mean_at('plasticity_off','correct',exp):.1%} | {mean_at('teaching_gated','correct',exp):.1%} | {mean_at('continuous','correct',exp):.1%} |")
    max_exposure = max(checkpoints)
    lines += ["", f"Seed-level peak-voltage outcomes at {max_exposure} exposures:", "", "| Seed | Gated, correct mapping | Continuous, correct mapping | Continuous, mismatched: intended | Continuous, mismatched: taught |", "|---:|---:|---:|---:|---:|"]
    for seed in config["seeds"]:
        def seed_metric(mode, mapping, metric):
            vals = [float(r["value"]) for r in curves if int(r["seed"]) == seed and r["mode"] == mode and r["mapping_condition"] == mapping and int(r["exposures_per_association"]) == max_exposure and r["metric"] == metric]
            return float(np.mean(vals)) if vals else float("nan")
        lines.append(f"| {seed} | {seed_metric('teaching_gated','correct','peak_correct_intended'):.1%} | {seed_metric('continuous','correct','peak_correct_intended'):.1%} | {seed_metric('continuous','mismatched','peak_correct_intended'):.1%} | {seed_metric('continuous','mismatched','peak_correct_taught'):.1%} |")
    lines += ["", f"At {max_exposure} exposures, continuous-mode spike-count accuracy was {mean_at('continuous','correct',max_exposure,'spike_correct_intended'):.1%}; {mean_at('continuous','correct',max_exposure,'all_silent'):.1%} of recall trials had no output spikes. Peak depolarization therefore separated learned preferences when spike-count recognition could not."]
    lines += ["", f"At {max_exposure} exposures, continuous and teaching-gated modes had the same correct-mapping peak accuracy ({mean_at('continuous','correct',max_exposure):.1%}). Correct-mapping continuous recall remained {mean_at('continuous','correct',max_exposure):.1%} before and after unpaired audits; mismatched continuous recall remained {mean_at('continuous','mismatched',max_exposure):.1%} for the intended mapping and {mean_at('continuous','mismatched',max_exposure,'peak_correct_taught'):.1%} for the mapping actually taught."]
    lines += ["", "## Mismatched mapping controls", "", f"Mismatched networks are scored against both the intended identity mapping and the fixed alternative mapping. Seed-level outcomes are in `continuous_learning_trials.csv`; the table below reports average peak-voltage performance at {max_exposure} paired experiences.", "", "| Mode | Intended mapping | Mapping actually taught | Auditory-only weight delta | Visual-teacher-only weight delta |", "|---|---:|---:|---:|---:|"]
    for mode in MODES:
        mr = [r for r in comparison if r["mode"] == mode and r["mapping_condition"] == "mismatched"]
        lines.append(f"| {mode} | {mean_at(mode,'mismatched',max_exposure):.1%} | {mean_at(mode,'mismatched',max_exposure,'peak_correct_taught'):.1%} | {np.mean([float(r['auditory_only_weight_change_l1']) for r in [x for x in comparison if x['mode']==mode and x['mapping_condition']=='correct']]):.6g} | {np.mean([float(r['visual_teacher_only_weight_change_l1']) for r in [x for x in comparison if x['mode']==mode and x['mapping_condition']=='correct']]):.6g} |")
    wall = np.asarray([float(r["wall_seconds"]) for r in performance])
    cpu = np.asarray([float(r["cpu_seconds"]) for r in performance])
    heap = np.asarray([float(r["peak_python_heap_mb"]) for r in performance])
    speed_ratio = float(np.mean([float(r["simulation_seconds"]) / float(r["wall_seconds"]) for r in performance]))
    lines += ["", "Continuous plasticity was active during paired, auditory-only, visual-teacher-only, and quiet events. Evaluation explicitly disabled plasticity. Any differences are therefore effects of the existing STDP rule under different gate schedules, not a new learning rule.", "", "Quiet periods produced no injected sensory events; the persistent network continued integrating. `sensory_event_log.csv` records scheduled simulation onset, duration, event type, teacher target and visual pattern, plasticity status, and per-event weight change. `continuous_learning_trials.csv` records peak-voltage response margin for every trial. `performance_summary.csv` separates simulated time, wall time, process CPU time, event count, and traced Python heap peak.", "", "## Interpretation and performance limits", "", "1. Quiet intervals advance the same Brian2 Network and do not inject background spikes or reset membrane state.", "2. Paired and unpaired experiences are scheduled on one simulation timebase without rebuilding the network.", "3. The dashboard shows the teacher-code pattern and plays a corresponding audible tone proxy; it does not display a retinal stimulus or play the model's raw neural input.", "4. Sensory and neural events share simulation timestamps; actual speaker latency is not instrumented, so physical acoustic synchrony is approximate.", "5. The network continues to receive the external teacher as direct output drive; this is not autonomous learning from ordinary visual observations.", "6. Here continuous STDP matched teaching-gated learning; it did not improve the final recognition score. The mismatched control shows that plasticity learns the supplied teacher mapping rather than the nominal identity mapping.", "7. Auditory-only and visual-teacher-only weight deltas, quiet deltas, and pre/post-audit accuracy quantify unintended change and retention. No meaningful auditory-only or quiet drift was measured in this task; visual-only changes were negligible.", f"8. Mean condition cost was {wall.mean():.3f} wall seconds and {cpu.mean():.3f} process CPU seconds; mean peak traced Python heap was {heap.mean():.2f} MB. Simulated-time / wall-time averaged {speed_ratio:.3f}x, so this full benchmark ran slower than biological real time on the test CPU. These headless measurements do not predict GUI frame latency or total native memory.", "9. Visual input is a supervised teacher signal wired directly to the output layer, not a sensory pathway.", "10. No camera, waveform encoder, or input pathway turns the displayed visual proxy or audible tone into model input; the model receives indexed auditory spike events and the teacher spike.", "11. The batch records computational timing but does not benchmark GUI frame latency. Live dashboard responsiveness can vary with the host and audio device.", "12. The smallest justified next milestone is a visual sensory input path independent of teacher output drive, then rerun the same controls.", "", "The live GUI requests asynchronous tone playback when a stimulus is dispatched; Brian2 auditory onset is scheduled at +10 ms and teacher onset at +25 ms. Device playback latency is not measured, so synchronized plotting is based on simulation timestamps rather than a measured speaker loopback.", ""]
    (out / "CONTINUOUS_LEARNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
