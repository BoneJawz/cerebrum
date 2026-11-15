"""Trace why local-STDP output recall is silent; no parameters are tuned."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, Synapses, mV, ms

from mdl import DistributedVisualAssociator
from mdl.plasticity import AUDITORY_ON_POST, AUDITORY_ON_PRE, AUDITORY_SYNAPSE_MODEL


SEEDS = (42, 43)
PAIRED_EXPOSURES = 50
OUTPUT_EXPOSURES = 5
OUTPUT_GAIN_MV = 60.0
TEACHER_MV = 55.0
OUTPUT_EVENT_S = 0.040
OUTPUT_QUIET_S = 0.060
RECALL_S = 0.060
QUIET_S = 0.200
OUTPUTS = 3
OUTDIR = Path("results/internal_output_diagnostic")


def patterns() -> tuple[np.ndarray, ...]:
    vertical = np.zeros((5, 5), dtype=np.uint8)
    vertical[:, 2] = 1
    horizontal = np.zeros((5, 5), dtype=np.uint8)
    horizontal[2, :] = 1
    return vertical, horizontal, np.eye(5, dtype=np.uint8)


def add_output(model: DistributedVisualAssociator, seed: int):
    c = model.config
    cells = NeuronGroup(
        OUTPUTS,
        f"dv/dt = ({c.rest_mv}*mV-v)/({c.membrane_tau_ms}*ms) : volt",
        threshold=f"v > {c.threshold_mv}*mV",
        reset=f"v = {c.rest_mv}*mV", refractory=c.refractory_ms * ms,
        method="exact", clock=model.clock, name="diagnostic_output_population")
    cells.v = c.rest_mv * mV
    syn = Synapses(
        model.associative, cells, model=AUDITORY_SYNAPSE_MODEL,
        on_pre=AUDITORY_ON_PRE, on_post=AUDITORY_ON_POST,
        clock=model.clock, namespace={
            "pre_increment": c.stdp_pre_increment,
            "post_increment": c.stdp_post_increment,
            "weight_min": c.weight_min,
            "weight_max": c.weight_max,
        }, name="diagnostic_associative_to_output_stdp")
    syn.connect()
    syn.gain = OUTPUT_GAIN_MV * mV
    syn.tau = c.stdp_tau_ms
    syn.plasticity_enabled = 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2401]))
    syn.w = rng.uniform(c.auditory_weight_min, c.auditory_weight_max, len(syn.w))
    spikes = SpikeMonitor(cells, name="diagnostic_output_spikes")
    voltage = StateMonitor(cells, "v", record=True, name="diagnostic_output_voltage")
    model.net.add(cells, syn, spikes, voltage)
    return cells, syn, spikes, voltage


def teacher_pulse(model: DistributedVisualAssociator, cells, target: int) -> None:
    # This supervised operation is separate from the visual/audio input channels.
    cells.v[target] = (model.config.rest_mv + TEACHER_MV) * mV
    model.net.run(0.1 * ms)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float("nan")


def winner(scores: np.ndarray) -> tuple[int | None, str]:
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) == 0:
        return None, "silent"
    top = float(np.max(scores[finite]))
    top_ids = finite[np.isclose(scores[finite], top, rtol=0.0, atol=1e-12)]
    return (int(top_ids[0]), "selected") if len(top_ids) == 1 else (None, "tie")


def cosines_to_cues(vectors: list[np.ndarray]) -> list[list[float | None]]:
    result = []
    for a in vectors:
        row = []
        for b in vectors:
            value = cosine(a, b)
            row.append(float(value) if np.isfinite(value) else None)
        result.append(row)
    return result


def compact_activity(row: dict, response: dict, start_s: float,
                     output_ids=(), output_times=(), output_voltage=None) -> dict:
    assoc_ids = np.asarray(response["spike_ids"], dtype=int)
    assoc_times = np.asarray(response["spike_times_s"], dtype=float) - start_s
    stride = max(1, round(10.0 / row["timestep_ms"]))
    profile = np.asarray(response["voltage_mv"], dtype=float)[:, ::stride]
    if output_voltage is not None:
        out_profile = np.asarray(output_voltage, dtype=float)[:, ::stride]
        out_peaks = np.max(output_voltage, axis=1) if output_voltage.size else np.full(3, -70.0)
    else:
        out_profile, out_peaks = np.empty((0, 0)), np.array([])
    row.update({
        "assoc_spike_count": int(len(assoc_ids)),
        "assoc_active_ids": sorted(set(assoc_ids.tolist())),
        "assoc_spike_times_ms": np.round(assoc_times * 1000.0, 4).tolist(),
        "assoc_peak_voltage_mv": np.round(response["peak_depolarization_mv"], 6).tolist(),
        "assoc_voltage_profile_mv_10ms": np.round(profile, 5).tolist(),
        "output_spike_ids": np.asarray(output_ids, dtype=int).tolist(),
        "output_spike_times_ms": np.round((np.asarray(output_times) - start_s) * 1000, 4).tolist(),
        "output_peak_voltage_mv": np.round(out_peaks, 6).tolist(),
        "output_voltage_profile_mv_10ms": np.round(out_profile, 5).tolist(),
    })
    return row


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"))
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def visual_references(model: DistributedVisualAssociator,
                      visual: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    model.set_plasticity(False)
    refs = []
    for pattern in visual:
        result = model.run_event(visual=pattern, duration_s=0.060)
        refs.append(result["peak_depolarization_mv"].copy())
        model.rest(QUIET_S)
    return refs


def activity_row(seed: int, upstream: str, output_mode: str, phase: str,
                 cue: int, target: int, experience: int, response: dict,
                 start_s: float, timestep_ms: float, **extra) -> dict:
    row = compact_activity({
        "seed": seed, "upstream_condition": upstream, "output_condition": output_mode,
        "phase": phase, "cue": cue, "target": target,
        "experience": experience, "timestep_ms": timestep_ms,
    }, response, start_s, extra.pop("output_ids", ()), extra.pop("output_times", ()),
       extra.pop("output_voltage", None))
    row.update(extra)
    return row


def upstream_evaluation(model, seed: int, stage: str, refs, activity_rows: list[dict]):
    model.set_plasticity(False)
    responses, spikes, voltage, scores = [], [], [], []
    for cue in range(3):
        start = float(model.net.t / ms) / 1000.0
        response = model.run_event(cue=cue, duration_s=RECALL_S)
        responses.append(response)
        spikes.append(response["spike_counts"].astype(float))
        voltage.append(response["peak_depolarization_mv"].copy())
        sims = np.asarray([cosine(response["peak_depolarization_mv"], ref) for ref in refs])
        predicted, status = winner(sims)
        scores.append({"cue": cue, "predicted": predicted, "status": status,
                       "correct": bool(predicted == cue), "cosine_scores": sims})
        activity_rows.append(activity_row(
            seed, stage, "not_attached", "auditory_only_upstream_eval", cue, cue, 0,
            response, start, model.config.timestep_ms,
            external_cosine_scores=[float(x) if np.isfinite(x) else None for x in sims],
            external_cosine_prediction=predicted, external_cosine_status=status,
            external_cosine_correct=bool(predicted == cue)))
        model.rest(QUIET_S)
    spike_similarities = cosines_to_cues(spikes)
    voltage_similarities = cosines_to_cues(voltage)
    for cue in range(3):
        activity_rows[-3 + cue]["auditory_cue_spike_cosine_matrix_row"] = spike_similarities[cue]
        activity_rows[-3 + cue]["auditory_cue_voltage_cosine_matrix_row"] = voltage_similarities[cue]
    return responses, spikes, voltage, scores


def teach_output(model, cells, syn, output_spikes, output_voltage, *, seed: int,
                 upstream: str, mode: str, visual, activity_rows: list[dict],
                 output_weight_deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    syn.plasticity_enabled = int(mode == "paired_stdp")
    order_rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    active_by_cue = [set() for _ in range(3)]
    initial = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
    delta_by_cue = np.zeros((3, 18, 3), dtype=float)
    for exposure in range(1, OUTPUT_EXPOSURES + 1):
        for cue_value in order_rng.permutation(3):
            cue = int(cue_value)
            target = cue
            # Paired-gated rule: on during each sensory/teacher event, off during rest.
            syn.plasticity_enabled = int(mode == "paired_stdp")
            start = float(model.net.t / ms) / 1000.0
            assoc_spike0 = len(model.associative_spikes.t)
            output_spike0 = len(output_spikes.t)
            output_voltage0 = len(output_voltage.t)
            before = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
            response = model.run_event(cue=cue, visual=visual[target],
                                       duration_s=OUTPUT_EVENT_S)
            teacher_pulse(model, cells, target)
            out_ids = np.asarray(output_spikes.i[output_spike0:], dtype=int)
            out_times = np.asarray(output_spikes.t[output_spike0:] / ms, dtype=float) / 1000.0
            out_v = np.asarray(output_voltage.v[:, output_voltage0:] / mV, dtype=float)
            after = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
            delta_by_cue[cue] += after - before
            active_by_cue[cue].update(np.asarray(response["spike_ids"], dtype=int).tolist())
            pre_times_ms = ((response["spike_times_s"] - start) * 1000.0).tolist()
            post_times_ms = ((out_times - start) * 1000.0).tolist()
            pair_dt = [round(post - pre, 4) for pre in pre_times_ms for post in post_times_ms]
            activity_rows.append(activity_row(
                seed, upstream, mode, "audiovisual_output_teaching", cue, target,
                exposure, response, start, model.config.timestep_ms,
                output_ids=out_ids, output_times=out_times, output_voltage=out_v,
                output_plasticity_enabled=(mode == "paired_stdp"),
                presyn_postsyn_dt_ms=pair_dt,
            ))
            model.rest(OUTPUT_QUIET_S)
            syn.plasticity_enabled = 0
    final = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
    return initial, {"final": final, "delta_by_cue": delta_by_cue,
                     "active_by_cue": active_by_cue}


def reset_output_for_control(model, cells, syn, initial):
    model.rest(QUIET_S)
    cells.v = model.config.rest_mv * mV
    syn.w = initial.reshape(-1)
    syn.apre = 0
    syn.apost = 0
    syn.plasticity_enabled = 0


def output_recall(model, cells, syn, output_spikes, output_voltage, *, seed: int,
                  upstream: str, mode: str, refs, visual, taught_active,
                  activity_rows: list[dict], voltage_rows: list[dict]):
    syn.plasticity_enabled = 0
    responses, spike_vectors, voltage_vectors = [], [], []
    for cue in range(3):
        target = cue
        start = float(model.net.t / ms) / 1000.0
        out_spike0, out_voltage0 = len(output_spikes.t), len(output_voltage.t)
        response = model.run_event(cue=cue, duration_s=RECALL_S)
        out_ids = np.asarray(output_spikes.i[out_spike0:], dtype=int)
        out_times = np.asarray(output_spikes.t[out_spike0:] / ms, dtype=float) / 1000.0
        out_v = np.asarray(output_voltage.v[:, out_voltage0:] / mV, dtype=float)
        peaks = np.max(out_v, axis=1) if out_v.size else np.full(3, model.config.rest_mv)
        counts = np.bincount(out_ids, minlength=3)
        if counts.sum() == 0:
            status, predicted = "silent", None
        else:
            tops = np.flatnonzero(counts == np.max(counts))
            status, predicted = ("selected", int(tops[0])) if len(tops) == 1 else ("tie", None)
        sims = np.asarray([cosine(response["peak_depolarization_mv"], ref) for ref in refs])
        cosine_prediction, cosine_status = winner(sims)
        pre_ids = np.asarray(response["spike_ids"], dtype=int)
        syn_i, syn_j = np.asarray(syn.i[:], dtype=int), np.asarray(syn.j[:], dtype=int)
        syn_w = np.asarray(syn.w[:], dtype=float)
        excitation = np.zeros(3)
        for pre_id in pre_ids:
            mask = syn_i == pre_id
            np.add.at(excitation, syn_j[mask], syn_w[mask] * OUTPUT_GAIN_MV)
        recall_active = set(pre_ids.tolist())
        taught = set(taught_active[cue])
        overlap = len(taught & recall_active)
        union = len(taught | recall_active)
        spike_vectors.append(response["spike_counts"].astype(float))
        voltage_vectors.append(response["peak_depolarization_mv"].copy())
        responses.append(response)

        activity_rows.append(activity_row(
            seed, upstream, mode, "auditory_only_output_recall", cue, target, 0,
            response, start, model.config.timestep_ms, output_ids=out_ids,
            output_times=out_times, output_voltage=out_v,
            output_spike_counts=counts.tolist(), output_status=status,
            output_prediction=predicted, output_correct=bool(predicted == target),
            external_cosine_scores=[float(x) if np.isfinite(x) else None for x in sims],
            external_cosine_prediction=cosine_prediction,
            external_cosine_status=cosine_status,
            teaching_recall_overlap_count=overlap,
            teaching_recall_jaccard=(overlap / union if union else None),
        ))
        voltage_rows.append({
            "seed": seed, "upstream_condition": upstream, "output_condition": mode,
            "cue": cue, "target_output": target,
            "output_spike_counts": counts.tolist(), "output_status": status,
            "output_prediction": predicted, "correct": bool(predicted == target),
            "peak_voltage_mv": peaks.tolist(),
            "distance_below_threshold_mv": (model.config.threshold_mv - peaks).tolist(),
            "intended_vs_best_competitor_peak_margin_mv": float(peaks[target] - np.max(np.delete(peaks, target))),
            "assoc_presyn_spike_counts": response["spike_counts"].tolist(),
            "incoming_synaptic_excitation_mv": excitation.tolist(),
            "external_cosine_status": cosine_status,
            "external_cosine_prediction": cosine_prediction,
            "external_cosine_correct": bool(cosine_prediction == target),
        })
        model.rest(QUIET_S)
    similarities_spike = cosines_to_cues(spike_vectors)
    similarities_voltage = cosines_to_cues(voltage_vectors)
    for cue in range(3):
        activity_rows[-3 + cue]["auditory_cue_spike_cosine_matrix_row"] = similarities_spike[cue]
        activity_rows[-3 + cue]["auditory_cue_voltage_cosine_matrix_row"] = similarities_voltage[cue]
    return responses


def write_weight_rows(seed: int, upstream: str, mode: str, layer: str,
                      before: np.ndarray, after: np.ndarray, delta_by_cue,
                      recall_spikes: list[np.ndarray], weight_rows: list[dict]) -> None:
    if layer == "associative_to_output":
        for pre in range(18):
            for post in range(3):
                row = {"seed": seed, "upstream_condition": upstream,
                       "output_condition": mode, "layer": layer,
                       "pre_index": pre, "post_index": post,
                       "initial_weight": float(before[pre, post]),
                       "final_weight": float(after[pre, post]),
                       "net_delta": float(after[pre, post] - before[pre, post]),
                       "pre_spikes_during_recall_by_cue": [int(x[pre]) for x in recall_spikes]}
                if delta_by_cue is not None:
                    for cue in range(3):
                        row[f"delta_during_taught_cue_{cue}"] = float(delta_by_cue[cue, pre, post])
                weight_rows.append(row)
    else:
        for channel in range(3):
            for post in range(18):
                weight_rows.append({
                    "seed": seed, "upstream_condition": upstream,
                    "output_condition": "shared", "layer": layer,
                    "pre_index": channel, "post_index": post,
                    "initial_weight": float(before[channel, post]),
                    "final_weight": float(after[channel, post]),
                    "net_delta": float(after[channel, post] - before[channel, post]),
                    **{f"delta_during_taught_cue_{cue}": float(delta_by_cue[cue, channel, post])
                       for cue in range(3)},
                })


def run() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    activity_rows, weight_rows, voltage_rows = [], [], []
    upstream_summary = []
    started = time.perf_counter()
    visual = patterns()
    for seed in SEEDS:
        for upstream in ("fresh_frozen", "paired_50_per_cue"):
            model = DistributedVisualAssociator(seed=seed)
            refs = visual_references(model, visual)
            upstream_initial = model.weights()
            upstream_delta_by_cue = np.zeros((3, 3, 18), dtype=float)
            if upstream == "paired_50_per_cue":
                order_rng = np.random.default_rng(np.random.SeedSequence([seed, 801]))
                active_training = [set() for _ in range(3)]
                model.set_plasticity(False)
                for experience in range(1, PAIRED_EXPOSURES + 1):
                    for cue_value in order_rng.permutation(3):
                        cue = int(cue_value)
                        start = float(model.net.t / ms) / 1000.0
                        before = model.weights()
                        model.set_plasticity(True)
                        response = model.run_event(cue=cue, visual=visual[cue], duration_s=0.060)
                        after = model.weights()
                        upstream_delta_by_cue[cue] += after - before
                        active_training[cue].update(response["spike_ids"].tolist())
                        activity_rows.append(activity_row(
                            seed, upstream, "not_attached", "audiovisual_upstream_training",
                            cue, cue, experience, response, start,
                            model.config.timestep_ms,
                        ))
                        model.rest(0.050)
                        model.set_plasticity(False)
            else:
                active_training = [set() for _ in range(3)]
            upstream_final = model.weights()
            pre_output_eval = upstream_evaluation(
                model, seed, upstream, refs, activity_rows)
            upstream_recalls, upstream_spikes, upstream_voltage, scores = pre_output_eval
            upstream_ok = (upstream == "fresh_frozen" or
                           all(item["status"] == "selected" and item["correct"] for item in scores))
            upstream_summary.append({
                "seed": seed, "upstream_condition": upstream,
                "auditory_to_associative_weight_l1_change": float(np.abs(upstream_final - upstream_initial).sum()),
                "recall_spike_counts_by_cue": [int(v.sum()) for v in upstream_spikes],
                "active_associative_neurons_by_cue": [int(np.count_nonzero(v)) for v in upstream_spikes],
                "visual_reference_cosine_prediction_by_cue": [x["predicted"] for x in scores],
                "visual_reference_cosine_status_by_cue": [x["status"] for x in scores],
                "visual_reference_cosine_correct_by_cue": [x["correct"] for x in scores],
                "upstream_pre_output_valid": upstream_ok,
            })
            write_weight_rows(seed, upstream, "shared", "auditory_to_associative",
                              upstream_initial, upstream_final, upstream_delta_by_cue,
                              [], weight_rows)

            if upstream == "paired_50_per_cue" and not upstream_ok:
                continue
            model.set_plasticity(False)
            cells, syn, out_spikes, out_voltage = add_output(model, seed)
            initial_output = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
            order = ("paired_stdp", "plasticity_disabled") if seed == 42 else (
                "plasticity_disabled", "paired_stdp")
            per_mode = {}
            for mode in order:
                if mode != order[0]:
                    reset_output_for_control(model, cells, syn, initial_output)
                before, trained = teach_output(
                    model, cells, syn, out_spikes, out_voltage,
                    seed=seed, upstream=upstream, mode=mode, visual=visual,
                    activity_rows=activity_rows, output_weight_deltas=np.zeros((3, 18, 3)))
                recalls = output_recall(
                    model, cells, syn, out_spikes, out_voltage,
                    seed=seed, upstream=upstream, mode=mode, refs=refs,
                    visual=visual, taught_active=trained["active_by_cue"],
                    activity_rows=activity_rows, voltage_rows=voltage_rows)
                recall_spikes = [r["spike_counts"] for r in recalls]
                write_weight_rows(seed, upstream, mode, "associative_to_output",
                                  initial_output, trained["final"], trained["delta_by_cue"],
                                  recall_spikes, weight_rows)
                per_mode[mode] = {
                    "output_weight_l1_change": float(np.abs(trained["final"] - initial_output).sum()),
                    "output_recall_spike_counts": [row["output_spike_counts"] for row in voltage_rows[-3:]],
                    "output_recall_status": [row["output_status"] for row in voltage_rows[-3:]],
                    "output_recall_correct": [bool(row["correct"]) for row in voltage_rows[-3:]],
                }
            for row in upstream_summary:
                if row["seed"] == seed and row["upstream_condition"] == upstream:
                    row["output_conditions"] = per_mode

    append_csv(OUTDIR / "activity_summary.csv", activity_rows)
    append_csv(OUTDIR / "weight_analysis.csv", weight_rows)
    append_csv(OUTDIR / "output_voltage_summary.csv", voltage_rows)
    result = {"runtime_seconds": time.perf_counter() - started,
              "seeds": list(SEEDS), "paired_upstream_exposures_per_cue": PAIRED_EXPOSURES,
              "output_exposures_per_cue": OUTPUT_EXPOSURES,
              "upstream_summary": upstream_summary}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
