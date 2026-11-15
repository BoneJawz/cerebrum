"""Small supervised local-STDP pilot for an internal spiking output population."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, Synapses, mV, ms

from mdl import DistributedVisualAssociator
from mdl.plasticity import AUDITORY_ON_POST, AUDITORY_ON_PRE, AUDITORY_SYNAPSE_MODEL


SEEDS = (42, 43)
EXPOSURES_PER_CUE = 5
OUTPUTS = 3
OUTPUT_GAIN_MV = 60.0
TEACHER_DEPOLARIZATION_MV = 55.0
TEACHER_DELAY_S = 0.040
EVENT_DURATION_S = 0.040
BETWEEN_EXPERIENCES_S = 0.060
RECALL_DURATION_S = 0.060
RECALL_REST_S = 0.080
IDENTITY = (0, 1, 2)
REMAPPED = (1, 2, 0)


def patterns() -> tuple[np.ndarray, ...]:
    vertical = np.zeros((5, 5), dtype=np.uint8)
    vertical[:, 2] = 1
    horizontal = np.zeros((5, 5), dtype=np.uint8)
    horizontal[2, :] = 1
    diagonal = np.eye(5, dtype=np.uint8)
    return vertical, horizontal, diagonal


def add_internal_output(model: DistributedVisualAssociator, seed: int):
    """Attach exactly three output LIFs and one dense 18-to-3 local-STDP projection."""
    c = model.config
    output = NeuronGroup(
        OUTPUTS,
        f"dv/dt = ({c.rest_mv}*mV-v)/({c.membrane_tau_ms}*ms) : volt",
        threshold=f"v > {c.threshold_mv}*mV",
        reset=f"v = {c.rest_mv}*mV", refractory=c.refractory_ms * ms,
        method="exact", clock=model.clock, name="pilot_output_population")
    output.v = c.rest_mv * mV
    projection = Synapses(
        model.associative, output, model=AUDITORY_SYNAPSE_MODEL,
        on_pre=AUDITORY_ON_PRE, on_post=AUDITORY_ON_POST,
        clock=model.clock, namespace={
            "pre_increment": c.stdp_pre_increment,
            "post_increment": c.stdp_post_increment,
            "weight_min": c.weight_min,
            "weight_max": c.weight_max,
        }, name="associative_to_output_local_stdp")
    projection.connect()
    projection.gain = OUTPUT_GAIN_MV * mV
    projection.tau = c.stdp_tau_ms
    projection.plasticity_enabled = 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2401]))
    projection.w = rng.uniform(c.auditory_weight_min, c.auditory_weight_max, len(projection.w))
    spikes = SpikeMonitor(output, name="pilot_output_spikes")
    voltage = StateMonitor(output, "v", record=True, name="pilot_output_voltage")
    model.net.add(output, projection, spikes, voltage)
    return output, projection, spikes, voltage


def teacher_pulse(model: DistributedVisualAssociator, output, target: int) -> None:
    """Deliver a separate supervised depolarizing pulse, then let the output spike."""
    output.v[target] = (model.config.rest_mv + TEACHER_DEPOLARIZATION_MV) * mV
    model.net.run(0.1 * ms)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float("nan")


def unique_winner(scores: np.ndarray) -> tuple[int | None, str]:
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) == 0:
        return None, "silent"
    best = float(np.max(scores[finite]))
    winners = finite[np.isclose(scores[finite], best, rtol=0.0, atol=1e-12)]
    return (int(winners[0]), "selected") if len(winners) == 1 else (None, "tie")


def classify_spikes(counts: np.ndarray) -> tuple[int | None, str]:
    if int(np.sum(counts)) == 0:
        return None, "silent"
    winners = np.flatnonzero(counts == np.max(counts))
    return (int(winners[0]), "selected") if len(winners) == 1 else (None, "tie")


def evaluate_condition(seed: int, condition: str, mapping: tuple[int, int, int],
                       learn_output: bool, patterns_by_id: tuple[np.ndarray, ...]) -> dict:
    model = DistributedVisualAssociator(seed=seed)
    # Freeze the pre-existing auditory-to-associative pathway to isolate learning
    # in the newly added associative-to-output synapses for every condition.
    model.set_plasticity(False)
    output, projection, output_spikes, _ = add_internal_output(model, seed)

    visual_refs = []
    for pattern in patterns_by_id:
        response = model.run_event(visual=pattern, duration_s=0.060)
        visual_refs.append(response["peak_depolarization_mv"].copy())
        model.rest(RECALL_REST_S)

    weights_before = np.asarray(projection.w[:], dtype=float).copy()
    projection.plasticity_enabled = int(learn_output)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    for _ in range(EXPOSURES_PER_CUE):
        for cue in rng.permutation(3):
            target = mapping[int(cue)]
            model.run_event(cue=int(cue), visual=patterns_by_id[target],
                            duration_s=EVENT_DURATION_S)
            teacher_pulse(model, output, target)
            model.rest(BETWEEN_EXPERIENCES_S)

    weights_after = np.asarray(projection.w[:], dtype=float).copy()
    projection.plasticity_enabled = 0
    rows = []
    for cue in range(3):
        target = mapping[cue]
        before_count = np.asarray(output_spikes.count[:], dtype=int).copy()
        response = model.run_event(cue=cue, duration_s=RECALL_DURATION_S)
        output_counts = np.asarray(output_spikes.count[:], dtype=int) - before_count
        predicted, status = classify_spikes(output_counts)
        similarities = np.asarray([
            cosine(response["peak_depolarization_mv"], reference)
            for reference in visual_refs
        ])
        cosine_prediction, cosine_status = unique_winner(similarities)
        rows.append({
            "cue_channel": cue,
            "expected_output": target,
            "output_spike_counts": output_counts.tolist(),
            "output_status": status,
            "output_prediction": predicted,
            "output_correct": bool(predicted == target),
            "cosine_scores": [None if not np.isfinite(x) else float(x) for x in similarities],
            "cosine_status": cosine_status,
            "cosine_prediction": cosine_prediction,
            "cosine_correct": bool(cosine_prediction == target),
        })
        model.rest(RECALL_REST_S)

    return {
        "seed": seed,
        "condition": condition,
        "mapping": list(mapping),
        "output_learning_enabled_during_teaching": learn_output,
        "experiences_per_cue": EXPOSURES_PER_CUE,
        "total_teaching_experiences": 3 * EXPOSURES_PER_CUE,
        "output_weight_l1_change": float(np.abs(weights_after - weights_before).sum()),
        "output_weight_min": float(weights_after.min()),
        "output_weight_max": float(weights_after.max()),
        "recall": rows,
        "output_accuracy": sum(row["output_correct"] for row in rows) / len(rows),
        "output_silent_trials": sum(row["output_status"] == "silent" for row in rows),
        "output_tied_trials": sum(row["output_status"] == "tie" for row in rows),
        "cosine_accuracy": sum(row["cosine_correct"] for row in rows) / len(rows),
        "cosine_silent_trials": sum(row["cosine_status"] == "silent" for row in rows),
        "cosine_tied_trials": sum(row["cosine_status"] == "tie" for row in rows),
    }


def main() -> None:
    start = time.perf_counter()
    visual_patterns = patterns()
    conditions = (
        ("paired_output_stdp_identity", IDENTITY, True),
        ("plasticity_disabled_identity", IDENTITY, False),
        ("paired_output_stdp_remapped", REMAPPED, True),
    )
    rows = [
        evaluate_condition(seed, condition, mapping, learn, visual_patterns)
        for seed in SEEDS
        for condition, mapping, learn in conditions
    ]
    report = {
        "runtime_seconds": time.perf_counter() - start,
        "seeds": list(SEEDS),
        "output_neurons": OUTPUTS,
        "experiences_per_cue": EXPOSURES_PER_CUE,
        "results": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
