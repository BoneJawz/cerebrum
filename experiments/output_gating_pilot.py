"""Compare ordinary and teacher-target-gated output STDP on fixed upstreams."""
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
EVENT_MS = 40.0
TEACHER_WINDOW_MS = 0.1  # one simulator time step, beginning at 40.0 ms
REST_S = 0.060
RECALL_S = 0.060
QUIET_S = 0.200
OUTDIR = Path("results/output_gating_pilot")

# Traces continue to represent sensory-driven timing in both conditions. Only
# plasticity_enabled gates weight changes in the target-gated condition.
GATED_MODEL = """
w : 1
gain : volt (shared)
plasticity_enabled : 1 (shared)
trace_enabled : 1 (shared)
target_output : integer (shared)
dapre/dt = -apre/(tau*ms) : 1 (event-driven)
dapost/dt = -apost/(tau*ms) : 1 (event-driven)
tau : 1 (shared)
"""
GATED_ON_PRE = """
v_post += w*gain
apre += trace_enabled*pre_increment
w = clip(w + plasticity_enabled*int(j == target_output)*apost, weight_min, weight_max)
"""
GATED_ON_POST = """
apost += trace_enabled*post_increment
w = clip(w + plasticity_enabled*int(j == target_output)*apre, weight_min, weight_max)
"""


def visual_patterns() -> tuple[np.ndarray, ...]:
    vertical = np.zeros((5, 5), dtype=np.uint8)
    vertical[:, 2] = 1
    horizontal = np.zeros((5, 5), dtype=np.uint8)
    horizontal[2, :] = 1
    diagonal = np.eye(5, dtype=np.uint8)
    return vertical, horizontal, diagonal


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def attach_output(model: DistributedVisualAssociator, seed: int, gated: bool):
    config = model.config
    cells = NeuronGroup(
        3,
        f"dv/dt = ({config.rest_mv}*mV-v)/({config.membrane_tau_ms}*ms) : volt",
        threshold=f"v > {config.threshold_mv}*mV",
        reset=f"v = {config.rest_mv}*mV", refractory=config.refractory_ms * ms,
        method="exact", clock=model.clock, name="gating_pilot_output")
    cells.v = config.rest_mv * mV
    if gated:
        syn = Synapses(
            model.associative, cells, model=GATED_MODEL, on_pre=GATED_ON_PRE,
            on_post=GATED_ON_POST, clock=model.clock, namespace={
                "pre_increment": config.stdp_pre_increment,
                "post_increment": config.stdp_post_increment,
                "weight_min": config.weight_min,
                "weight_max": config.weight_max,
            }, name="target_gated_associative_to_output")
    else:
        syn = Synapses(
            model.associative, cells, model=AUDITORY_SYNAPSE_MODEL,
            on_pre=AUDITORY_ON_PRE, on_post=AUDITORY_ON_POST,
            clock=model.clock, namespace={
                "pre_increment": config.stdp_pre_increment,
                "post_increment": config.stdp_post_increment,
                "weight_min": config.weight_min,
                "weight_max": config.weight_max,
            }, name="ungated_associative_to_output")
    syn.connect()
    syn.gain = OUTPUT_GAIN_MV * mV
    syn.tau = config.stdp_tau_ms
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2401]))
    syn.w = rng.uniform(config.auditory_weight_min, config.auditory_weight_max,
                        len(syn.w))
    syn.plasticity_enabled = 0
    if gated:
        syn.trace_enabled = 1
    spikes = SpikeMonitor(cells, name="gating_pilot_output_spikes")
    voltage = StateMonitor(cells, "v", record=True, name="gating_pilot_output_voltage")
    model.net.add(cells, syn, spikes, voltage)
    return cells, syn, spikes, voltage


def train_upstream(model: DistributedVisualAssociator,
                   patterns: tuple[np.ndarray, ...], seed: int) -> tuple[list[np.ndarray], dict]:
    model.set_plasticity(False)
    visual_refs = []
    for image in patterns:
        response = model.run_event(visual=image, duration_s=0.060)
        visual_refs.append(response["peak_depolarization_mv"].copy())
        model.rest(QUIET_S)

    order_rng = np.random.default_rng(np.random.SeedSequence([seed, 801]))
    for _experience in range(1, PAIRED_EXPOSURES + 1):
        for cue_value in order_rng.permutation(3):
            cue = int(cue_value)
            model.set_plasticity(True)
            model.run_event(cue=cue, visual=patterns[cue], duration_s=0.060)
            model.rest(0.050)
            model.set_plasticity(False)

    # Verify the same trained representation used by the earlier diagnosis.
    predictions, counts = [], []
    for cue in range(3):
        response = model.run_event(cue=cue, duration_s=RECALL_S)
        similarities = np.asarray([
            cosine(response["peak_depolarization_mv"], ref) for ref in visual_refs
        ])
        predictions.append(int(np.argmax(similarities)))
        counts.append(int(response["spike_counts"].sum()))
        model.rest(QUIET_S)
    model.set_plasticity(False)
    return visual_refs, {"cosine_predictions": predictions,
                         "associative_spike_counts": counts,
                         "all_cosine_matches_correct": predictions == [0, 1, 2]}


def gate_for_teacher(syn, target: int, enabled: bool, *, target_specific: bool) -> None:
    # Brian2 shared gate applies only when this connection terminates on target.
    # Its presynaptic transmission statement remains unconditional.
    if target_specific:
        syn.target_output = target
    syn.plasticity_enabled = int(enabled)


def teach(model, cells, syn, spikes, *, condition: str, seed: int,
          patterns: tuple[np.ndarray, ...], deltas_by_cue: np.ndarray) -> None:
    gated = condition == "target_gated"
    order_rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    for _experience in range(1, OUTPUT_EXPOSURES + 1):
        for cue_value in order_rng.permutation(3):
            cue = int(cue_value)
            # Baseline keeps the original gate open through sensory and teacher
            # events. Gated mode keeps sensory transmission and traces active,
            # but holds all weight updates until the teacher time window.
            gate_for_teacher(syn, cue, enabled=(not gated), target_specific=gated)
            before = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
            model.run_event(cue=cue, visual=patterns[cue], duration_s=EVENT_MS / 1000)
            if gated:
                gate_for_teacher(syn, cue, enabled=True, target_specific=True)
            cells.v[cue] = (model.config.rest_mv + TEACHER_MV) * mV
            model.net.run(TEACHER_WINDOW_MS * ms)
            after = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
            deltas_by_cue[cue] += after - before
            gate_for_teacher(syn, cue, enabled=False, target_specific=gated)
            model.rest(REST_S)
    # Ensure the first recall event is explicitly non-plastic.
    gate_for_teacher(syn, 0, enabled=False, target_specific=gated)


def recall(model, syn, spikes, voltage, *, seed: int, condition: str) -> list[dict]:
    rows = []
    syn.plasticity_enabled = 0
    if hasattr(syn, "trace_enabled"):
        syn.trace_enabled = 0
    for cue in range(3):
        spike0, voltage0 = len(spikes.t), len(voltage.t)
        model.run_event(cue=cue, duration_s=RECALL_S)
        output_ids = np.asarray(spikes.i[spike0:], dtype=int)
        counts = np.bincount(output_ids, minlength=3)
        peaks = (np.max(np.asarray(voltage.v[:, voltage0:] / mV), axis=1)
                 if len(voltage.t) > voltage0 else np.full(3, model.config.rest_mv))
        if counts.sum() == 0:
            status, prediction = "silent", None
        else:
            winners = np.flatnonzero(counts == counts.max())
            status, prediction = (("selected", int(winners[0])) if len(winners) == 1
                                  else ("tie", None))
        rows.append({
            "seed": seed, "condition": condition, "cue": cue,
            "target": cue, "counts_abc": counts.tolist(), "status": status,
            "prediction": prediction,
            "correct": bool(prediction == cue), "peaks_mv": peaks.tolist(),
            "peak_margin_mv": round(float(peaks[cue] - np.max(np.delete(peaks, cue))), 6),
        })
        model.rest(QUIET_S)
    return rows


def run_condition(seed: int, condition: str,
                  patterns: tuple[np.ndarray, ...]) -> tuple[dict, list[dict], list[dict]]:
    model = DistributedVisualAssociator(seed=seed)
    _refs, upstream = train_upstream(model, patterns, seed)
    cells, syn, spikes, voltage = attach_output(
        model, seed, gated=(condition == "target_gated"))
    initial = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
    delta_by_cue = np.zeros((3, 18, 3), dtype=float)
    teach(model, cells, syn, spikes, condition=condition, seed=seed,
          patterns=patterns, deltas_by_cue=delta_by_cue)
    final = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
    trials = recall(model, syn, spikes, voltage, seed=seed, condition=condition)
    weight_rows = []
    for cue in range(3):
        delta = delta_by_cue[cue]
        target_summed = float(delta[:, cue].sum())
        competitor_summed = float(np.max(np.delete(delta.sum(axis=0), cue)))
        weight_rows.append({
            "seed": seed, "condition": condition, "taught_cue": cue,
            "weight_delta_sum_abc": np.round(delta.sum(axis=0), 8).tolist(),
            "target_delta_sum": target_summed,
            "strongest_competitor_delta_sum": competitor_summed,
            "target_minus_competitor": round(target_summed - competitor_summed, 8),
            "non_target_weight_change_l1": float(np.abs(np.delete(delta, cue, axis=1)).sum()),
        })
    stats = {
        "seed": seed, "condition": condition, **upstream,
        "output_weight_l1_change": float(np.abs(final - initial).sum()),
        "weight_delta_sum_by_output_abc": np.round((final - initial).sum(axis=0), 8).tolist(),
    }
    return stats, trials, weight_rows


def main() -> None:
    started = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    patterns = visual_patterns()
    summaries, trial_rows, weight_rows = [], [], []
    for seed in SEEDS:
        for condition in ("ungated", "target_gated"):
            summary, trials, weights = run_condition(seed, condition, patterns)
            summaries.append(summary)
            trial_rows.extend(trials)
            weight_rows.extend(weights)
    for path, rows in ((OUTDIR / "output_gating_trials.csv", trial_rows),
                       (OUTDIR / "output_gating_weights.csv", weight_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, separators=(",", ":"))
                              if isinstance(value, (list, dict)) else value
                              for key, value in row.items()} for row in rows)
    print(json.dumps({"runtime_seconds": time.perf_counter() - started,
                      "summaries": summaries, "trials": trial_rows,
                      "weights_by_cue": weight_rows}, indent=2))


if __name__ == "__main__":
    main()
