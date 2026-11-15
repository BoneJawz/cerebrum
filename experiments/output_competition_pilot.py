"""Test fixed mutual output inhibition with target-gated output STDP."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from brian2 import (NeuronGroup, SpikeMonitor, StateMonitor, Synapses, mV, ms,
                    second)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_gating_pilot import (  # noqa: E402
    GATED_MODEL, GATED_ON_POST, GATED_ON_PRE, OUTPUT_EXPOSURES,
    PAIRED_EXPOSURES, QUIET_S, REST_S, SEEDS, TEACHER_MV,
    attach_output, gate_for_teacher, teach, train_upstream, visual_patterns,
)
from mdl import DistributedVisualAssociator  # noqa: E402


OUTDIR = Path("results/output_competition")
INHIBITION_MV = 10.0  # fixed a priori: half the 20 mV rest-to-threshold gap
WINDOW_MS = 0.1
RECALL_S = 0.060
OUTPUT_GAIN_MV = 60.0

INHIBITION_MODEL = """
inhibition_mv : volt (shared)
"""
INHIBITION_ON_PRE = """
v_post -= inhibition_mv
inhibition_count_post += 1
last_inhibition_time_post = t
"""


def make_output(model: DistributedVisualAssociator, seed: int):
    c = model.config
    # The two additional state variables only record inhibitory events; the
    # membrane differential equation, threshold, reset, and refractory are
    # identical to the existing three-output LIF group.
    cells = NeuronGroup(
        3,
        f"dv/dt = ({c.rest_mv}*mV-v)/({c.membrane_tau_ms}*ms) : volt\n"
        "inhibition_count : integer\nlast_inhibition_time : second",
        threshold=f"v > {c.threshold_mv}*mV",
        reset=f"v = {c.rest_mv}*mV", refractory=c.refractory_ms * ms,
        method="exact", clock=model.clock, name="competition_output")
    cells.v = c.rest_mv * mV
    cells.inhibition_count = 0
    cells.last_inhibition_time = 0 * second
    syn = Synapses(
        model.associative, cells, model=GATED_MODEL,
        on_pre=GATED_ON_PRE, on_post=GATED_ON_POST,
        clock=model.clock, namespace={
            "pre_increment": c.stdp_pre_increment,
            "post_increment": c.stdp_post_increment,
            "weight_min": c.weight_min,
            "weight_max": c.weight_max,
        }, name="target_gated_associative_to_output")
    syn.connect()
    syn.gain = OUTPUT_GAIN_MV * mV
    syn.tau = c.stdp_tau_ms
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2401]))
    syn.w = rng.uniform(c.auditory_weight_min, c.auditory_weight_max, len(syn.w))
    syn.plasticity_enabled = 0
    syn.trace_enabled = 1

    spikes = SpikeMonitor(cells, name="competition_output_spikes")
    voltage = StateMonitor(
        cells, ["v", "inhibition_count", "last_inhibition_time"],
        record=True, name="competition_output_monitor")
    model.net.add(cells, syn, spikes, voltage)
    return cells, syn, spikes, voltage


def add_competition(model, cells):
    inhibition = Synapses(
        cells, cells, model=INHIBITION_MODEL, on_pre=INHIBITION_ON_PRE,
        clock=model.clock, name="fixed_mutual_output_inhibition")
    inhibition.connect(condition="i != j")
    inhibition.inhibition_mv = INHIBITION_MV * mV
    model.net.add(inhibition)
    return inhibition


def status(counts: np.ndarray, cue: int) -> tuple[str, int | None, bool]:
    if int(counts.sum()) == 0:
        return "silent", None, False
    winners = np.flatnonzero(counts == counts.max())
    if len(winners) != 1:
        return "tie", None, False
    prediction = int(winners[0])
    return ("correct" if prediction == cue else "incorrect"), prediction, True


def recall(model, spikes, monitor, *, seed: int, condition: str) -> list[dict]:
    rows = []
    for cue in range(3):
        spike0, state0 = len(spikes.t), len(monitor.t)
        start_s = float(model.net.t / second)
        model.run_event(cue=cue, duration_s=RECALL_S)
        end_s = float(model.net.t / second)
        ids = np.asarray(spikes.i[spike0:], dtype=int)
        times_s = np.asarray(spikes.t[spike0:] / second, dtype=float) - start_s
        counts = np.bincount(ids, minlength=3)
        first_ms = []
        for neuron in range(3):
            own = times_s[ids == neuron]
            first_ms.append(round(float(own.min() * 1000), 4) if own.size else None)
        groups: dict[float, list[int]] = {}
        for neuron, spike_time in zip(ids.tolist(), times_s.tolist()):
            key = round(spike_time / (model.config.timestep_ms / 1000))
            groups.setdefault(float(key), []).append(neuron)
        simultaneous_groups = [sorted(v) for v in groups.values() if len(set(v)) > 1]

        sample_times = np.asarray(monitor.t[state0:] / second, dtype=float)
        counts_trace = np.asarray(monitor.inhibition_count[:, state0:], dtype=int)
        last_trace = np.asarray(monitor.last_inhibition_time[:, state0:] / second,
                                dtype=float)
        voltage_mv = np.asarray(monitor.v[:, state0:] / mV, dtype=float)
        inhibition_times = [[] for _ in range(3)]
        peaks = np.max(voltage_mv, axis=1) if voltage_mv.shape[1] else np.full(3, -70.0)
        for neuron in range(3):
            diffs = np.diff(np.r_[counts_trace[neuron, 0], counts_trace[neuron]])
            changed = np.flatnonzero(diffs > 0)
            for sample_idx in changed:
                t_inhib = float(last_trace[neuron, sample_idx])
                if start_s <= t_inhib < end_s:
                    inhibition_times[neuron].append(t_inhib - start_s)

        label, prediction, _ = status(counts, cue)
        rows.append({
            "seed": seed, "condition": condition, "cue": cue,
            "counts_abc": counts.tolist(), "response": label,
            "prediction": prediction, "correct": label == "correct",
            "first_spike_ms_abc": first_ms,
            "simultaneous_spike_groups_abc": simultaneous_groups,
            "inhibition_arrivals_ms_by_output": [
                [round(t * 1000, 4) for t in ts] for ts in inhibition_times],
            "peak_voltage_mv_abc": np.round(peaks, 6).tolist(),
        })
        model.rest(QUIET_S)
    return rows


def run_condition(seed: int, condition: str, images) -> tuple[dict, list[dict]]:
    model = DistributedVisualAssociator(seed=seed)
    _refs, upstream = train_upstream(model, images, seed)
    cells, syn, spikes, monitor = make_output(model, seed)
    initial_weights = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
    if condition == "competition":
        add_competition(model, cells)
    deltas = np.zeros((3, 18, 3), dtype=float)
    teach(model, cells, syn, spikes, condition="target_gated", seed=seed,
          patterns=images, deltas_by_cue=deltas)
    final_weights = np.asarray(syn.w[:], dtype=float).reshape(18, 3).copy()
    rows = recall(model, spikes, monitor, seed=seed, condition=condition)
    per_cue_non_target_l1 = [
        float(np.abs(np.delete(deltas[cue], cue, axis=1)).sum()) for cue in range(3)
    ]
    return ({
        "seed": seed, "condition": condition, **upstream,
        "initial_weight_l1": float(initial_weights.sum()),
        "final_weight_l1": float(final_weights.sum()),
        "non_target_weight_change_l1_by_cue": per_cue_non_target_l1,
    }, rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"))
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main() -> None:
    started = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    images = visual_patterns()
    summaries, all_trials = [], []
    for seed in SEEDS:
        for condition in ("baseline", "competition"):
            summary, trials = run_condition(seed, condition, images)
            summaries.append(summary)
            all_trials.extend(trials)
    write_csv(OUTDIR / "output_competition_trials.csv", all_trials)
    print(json.dumps({"runtime_seconds": time.perf_counter() - started,
                      "inhibition_mv": INHIBITION_MV,
                      "summaries": summaries, "trials": all_trials}, indent=2))


if __name__ == "__main__":
    main()
