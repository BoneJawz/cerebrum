"""Controlled experiments for distributed visual–auditory association."""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np

from .distributed_network import DistributedVisualAssociator
from .distributed_network import second


DEV_SEEDS = (42, 43, 44, 45, 46)
VALIDATION_SEEDS = (101, 102, 103, 104, 105)
EXPOSURES = (0, 5, 20, 50)
CONDITIONS = ("plasticity_off", "paired_gated", "continuous", "unpaired")
MAPPINGS = {"identity": (0, 1, 2), "remapped": (2, 0, 1)}
N_ASSOCIATIVE = 18
VISUAL_P = 0.35
VISUAL_GAIN_MV = 12.0
AUDITORY_GAIN_MV = 60.0
EVENT_S = 0.060
QUIET_S = 0.050
UNPAIRED_GAP_S = 0.250


def patterns() -> dict[str, np.ndarray]:
    a = np.zeros((5, 5), dtype=np.uint8); a[:, 2] = 1
    b = np.zeros((5, 5), dtype=np.uint8); b[2, :] = 1
    c = np.eye(5, dtype=np.uint8)
    return {"A_vertical": a, "B_horizontal": b, "C_diagonal": c}


def variants() -> dict[str, np.ndarray]:
    p = patterns()
    a = np.zeros((5, 5), dtype=np.uint8); a[:, 3] = 1
    b = np.zeros((5, 5), dtype=np.uint8); b[3, :] = 1
    c = np.zeros((5, 5), dtype=np.uint8); c[1:, :4] = np.eye(4, dtype=np.uint8)
    d = p["A_vertical"].copy(); d[0, 2] = 0
    return {"A_vertical_shifted": a, "B_horizontal_shifted": b,
            "C_diagonal_shifted": c, "A_vertical_one_pixel_removed": d}


def _write(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _references(net: DistributedVisualAssociator, pats: dict[str, np.ndarray], seed: int,
                ref_rows: list[dict], store: dict) -> dict[str, dict[str, np.ndarray]]:
    net.set_plasticity(False)
    refs = {}
    for name, pattern in pats.items():
        result = net.present(pattern=pattern, capture=True)
        refs[name] = {"voltage": result["peak_depolarization_mv"].copy(),
                      "spikes": result["spike_counts"].astype(float).copy(),
                      "mean_voltage": result["mean_depolarization_mv"].copy()}
        store[f"seed{seed}_{name}_voltage"] = refs[name]["voltage"]
        store[f"seed{seed}_{name}_spikes"] = refs[name]["spikes"]
        for i, value in enumerate(refs[name]["voltage"]):
            ref_rows.append({"seed": seed, "pattern": name, "associative_neuron": i,
                             "peak_depolarization_mv": float(value),
                             "spike_count": int(refs[name]["spikes"][i]),
                             "active_visual_cells": int(pattern.sum())})
        net.rest(0.200)
    return refs


def _record_similarity(seed, mapping_name, mode, exposure, cue, target_idx, response_kind,
                       response, refs, similarity_rows):
    reference_names = list(patterns())
    sims = [_cosine(response, refs[name][response_kind]) for name in reference_names]
    finite = [(i, s) for i, s in enumerate(sims) if np.isfinite(s)]
    if finite:
        winner = max(finite, key=lambda item: item[1])[0]
        target_score = sims[target_idx]
        alternatives = [s for i, s in finite if i != target_idx]
        margin = target_score - max(alternatives) if alternatives else float("nan")
        correct = int(winner == target_idx)
    else:
        winner, margin, correct = None, float("nan"), 0
    for ref_idx, name in enumerate(reference_names):
        similarity_rows.append({
            "seed": seed, "mapping": mapping_name, "condition": mode,
            "exposures_per_association": exposure, "auditory_cue": cue,
            "experienced_visual_pattern": reference_names[target_idx],
            "response_kind": response_kind, "reference_pattern": name,
            "cosine_similarity": sims[ref_idx], "predicted_pattern_index": winner,
            "correct_mapping": correct, "correct_vs_best_incorrect_margin": margin,
            "zero_response_vector": int(not finite),
        })
    return winner, margin, correct, sims


def _diagnostic_patterns(out: Path, pats: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, (name, p) in zip(axes, pats.items()):
        ax.imshow(p, cmap="Greys", vmin=0, vmax=1)
        ax.set_title(name.split("_", 1)[0]); ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="#6b7280", linewidth=.6); ax.tick_params(length=0)
    fig.suptitle("Binary 5 × 5 visual training patterns"); fig.tight_layout()
    fig.savefig(out / "visual_patterns.png", dpi=160); plt.close(fig)


def _make_figures(out, curve_rows, weight_rows, example_refs, example_audios):
    # Main result curves use voltage representations as primary and spike
    # population representations as an independent, sparsity-sensitive view.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("voltage", "spikes")):
        for condition in CONDITIONS:
            xs, ys = [], []
            for exposure in EXPOSURES:
                subset = [r for r in curve_rows if r["mapping"] == "identity" and r["condition"] == condition
                          and r["response_kind"] == kind and r["exposures_per_association"] == exposure]
                values = [float(r["accuracy"]) for r in subset if np.isfinite(float(r["accuracy"]))]
                if values:
                    xs.append(exposure); ys.append(float(np.mean(values)))
            if xs:
                ax.plot(xs, ys, marker="o", label=condition)
        ax.axhline(1/3, color="gray", linestyle="--", label="chance (1/3)")
        ax.set(xlabel="Paired experiences per association", ylabel="Nearest visual-reference accuracy",
               title=f"{kind.capitalize()} response")
        ax.set_ylim(-.03, 1.03); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(patterns())
    mat = np.vstack([example_refs[n]["voltage"] for n in names] +
                    [example_audios[n]["voltage"] for n in names])
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_yticks(range(6), [f"V:{n[:1]}" for n in names] + [f"A:{n[:1]}" for n in names])
    ax.set_xlabel("Associative neuron"); ax.set_title("Visual references and post-training auditory responses (example seed)")
    fig.colorbar(im, ax=ax, label="Peak depolarization (mV)"); fig.tight_layout()
    fig.savefig(out / "associative_population_activity.png", dpi=160); plt.close(fig)

    # Visualization of cosine similarity matrices for an example paired case.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key in zip(axes, ("voltage", "spikes")):
        mat = np.array([[_cosine(example_audios[n][key], example_refs[m][key]) for m in names] for n in names])
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(3), [n[:1] for n in names]); ax.set_yticks(range(3), [n[:1] for n in names])
        ax.set(xlabel="Visual reference", ylabel="Audio cue response", title=f"{key.capitalize()} cosine")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(out / "auditory_visual_similarity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cue in range(3):
        vals = []
        for exp in EXPOSURES:
            w = [float(r["weight"]) for r in weight_rows if r["mapping"] == "identity"
                 and r["condition"] == "paired_gated" and r["exposures_per_association"] == exp
                 and r["auditory_cue"] == cue]
            vals.append(np.mean(w) if w else np.nan)
        ax.plot(EXPOSURES, vals, marker="o", label=f"cue {cue}")
    ax.set(xlabel="Paired experiences per association", ylabel="Mean auditory synaptic weight",
           title="Auditory afferent weight evolution (development seeds)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "weight_evolution.png", dpi=160); plt.close(fig)


def run(outdir="results/distributed_visual_association", seeds=DEV_SEEDS,
        validation_seeds=VALIDATION_SEEDS, include_validation=True, validation_only=False):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(map(int, seeds)); val_seeds = tuple(map(int, validation_seeds)) if include_validation else ()
    all_seeds = val_seeds if validation_only else seeds + val_seeds
    pats, heldout = patterns(), variants()
    mapping_names = tuple(MAPPINGS)
    refs_rows, similarity_rows, trial_rows, weight_rows, comparison_rows = [], [], [], [], []
    generalization_rows, performance_rows = [], []
    visual_store, audio_store = {}, {}
    curve_summary = []
    projection_records = {}
    started_total = time.perf_counter()

    for seed in all_seeds:
        for mapping_name, mapping in MAPPINGS.items():
            for condition in CONDITIONS:
                role = "validation" if validation_only or seed in val_seeds else "development"
                print(f"[{role}] seed={seed} mapping={mapping_name} condition={condition}", flush=True)
                net = DistributedVisualAssociator(seed, N_ASSOCIATIVE, VISUAL_P,
                                                  VISUAL_GAIN_MV, AUDITORY_GAIN_MV, learning=True)
                key = f"{seed}_{mapping_name}_{condition}"
                projection_records[str(seed)] = net.visual_edges.copy()
                initial = net.weights()
                wall0, cpu0, sim0 = time.perf_counter(), time.process_time(), float(net.net.t / second)
                refs_all = _references(net, {**pats, **heldout}, seed, refs_rows, visual_store)
                base_refs = {name: refs_all[name] for name in pats}
                example_seed = seeds[0] if seeds else val_seeds[0]
                if seed == example_seed and mapping_name == "identity" and condition == "paired_gated":
                    example_refs = base_refs
                else:
                    example_refs = None
                net.set_plasticity(False)
                # Same within-seed cue order across mappings and plasticity controls.
                order_rng = np.random.default_rng(np.random.SeedSequence([seed, 801]))
                order = [int(c) for _ in range(max(EXPOSURES)) for c in order_rng.permutation(3)]
                current = 0
                checkpoints_audio = {}
                for exposure in EXPOSURES:
                    while current < exposure:
                        for cue in order[current * 3:(current + 1) * 3]:
                            pattern_index = int(mapping[cue])
                            if condition == "unpaired":
                                net.set_plasticity(True)
                                net.present(cue=cue, capture=False)
                                net.rest(UNPAIRED_GAP_S)
                                net.present(pattern=pats[list(pats)[pattern_index]], capture=False)
                                # Keep the next auditory event equally far from
                                # the preceding visual postsynaptic event.
                                net.rest(UNPAIRED_GAP_S)
                            else:
                                net.set_plasticity(condition in ("paired_gated", "continuous"))
                                net.present(cue=cue, pattern=pats[list(pats)[pattern_index]], capture=False)
                                net.rest(QUIET_S)
                            if condition != "continuous":
                                net.set_plasticity(False)
                        current += 1
                    net.set_plasticity(False)
                    current_weights = net.weights()
                    for cue in range(3):
                        target = int(mapping[cue])
                        for neuron in range(net.n_associative):
                            weight_rows.append({
                                "seed": seed, "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                                "mapping": mapping_name, "condition": condition,
                                "exposures_per_association": exposure, "auditory_cue": cue,
                                "associated_visual_pattern": list(pats)[target],
                                "associative_neuron": neuron,
                                "initial_weight": float(initial[cue, neuron]),
                                "weight": float(current_weights[cue, neuron]),
                                "delta": float(current_weights[cue, neuron] - initial[cue, neuron]),
                            })
                    for cue in range(3):
                        net.set_plasticity(False)
                        response = net.present(cue=cue, capture=True)
                        audio_key = f"{key}_e{exposure}_cue{cue}"
                        checkpoints_audio[(exposure, cue)] = response
                        audio_store[f"{audio_key}_voltage"] = response["peak_depolarization_mv"]
                        audio_store[f"{audio_key}_spikes"] = response["spike_counts"].astype(float)
                        audio_store[f"{audio_key}_mean_voltage"] = response["mean_depolarization_mv"]
                        target = int(mapping[cue])
                        for kind, vec in (("voltage", response["peak_depolarization_mv"]),
                                           ("spikes", response["spike_counts"].astype(float))):
                            winner, margin, correct, sims = _record_similarity(
                                seed, mapping_name, condition, exposure, cue, target, kind, vec,
                                base_refs, similarity_rows)
                            silent = int(float(np.sum(vec)) == 0.0) if kind == "spikes" else 0
                            trial_rows.append({
                                "seed": seed, "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                                "mapping": mapping_name, "condition": condition,
                                "exposures_per_association": exposure, "auditory_cue": cue,
                                "associated_visual_pattern": list(pats)[target],
                                "response_kind": kind, "predicted_visual_pattern": "" if winner is None else list(pats)[winner],
                                "correct": correct, "correct_vs_best_incorrect_margin": margin,
                                "silent_population": silent,
                                "active_associative_neurons": int(np.count_nonzero(response["spike_counts"]) if kind == "spikes" else np.count_nonzero(vec)),
                                "total_output_spikes": int(np.sum(response["spike_counts"])),
                                "zero_similarity_vector": int(not np.isfinite(sims).any()),
                            })
                        net.rest(0.200)
                    if seed == example_seed and mapping_name == "identity" and condition == "paired_gated" and exposure == max(EXPOSURES):
                        example_audio = {list(pats)[cue]: checkpoints_audio[(exposure, cue)] for cue in range(3)}
                    # Evaluation summaries preserve seed-level outcomes.
                    for kind in ("voltage", "spikes"):
                        subset = [r for r in trial_rows if r["seed"] == seed and r["mapping"] == mapping_name
                                  and r["condition"] == condition and r["exposures_per_association"] == exposure
                                  and r["response_kind"] == kind]
                        accuracy = float(np.mean([r["correct"] for r in subset]))
                        curve_summary.append({"seed": seed,
                            "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                            "mapping": mapping_name, "condition": condition,
                            "exposures_per_association": exposure, "response_kind": kind,
                            "accuracy": accuracy,
                            "silent_fraction": float(np.mean([r["silent_population"] for r in subset])) if kind == "spikes" else 0.0,
                            "active_response_fraction": float(np.mean([r["active_associative_neurons"] > 0 for r in subset])),
                        })
                    comparison_rows.append({
                        "seed": seed, "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                        "mapping": mapping_name, "condition": condition,
                        "exposures_per_association": exposure,
                        "mean_absolute_weight_change": float(np.mean(np.abs(current_weights - initial))),
                        "total_absolute_weight_change": float(np.abs(current_weights - initial).sum()),
                        "voltage_cosine_accuracy": next(r["accuracy"] for r in curve_summary[::-1] if r["seed"] == seed and r["mapping"] == mapping_name and r["condition"] == condition and r["exposures_per_association"] == exposure and r["response_kind"] == "voltage"),
                        "spike_cosine_accuracy": next(r["accuracy"] for r in curve_summary[::-1] if r["seed"] == seed and r["mapping"] == mapping_name and r["condition"] == condition and r["exposures_per_association"] == exposure and r["response_kind"] == "spikes"),
                        "spike_silent_fraction": next(r["silent_fraction"] for r in curve_summary[::-1] if r["seed"] == seed and r["mapping"] == mapping_name and r["condition"] == condition and r["exposures_per_association"] == exposure and r["response_kind"] == "spikes"),
                    })
                    # Visual relatedness audit at final dose. The visual variants
                    # were never used during association training.
                    if exposure == max(EXPOSURES):
                        for cue in range(3):
                            target = int(mapping[cue]); cue_name = list(pats)[cue]
                            response = checkpoints_audio[(exposure, cue)]["peak_depolarization_mv"]
                            for variant_name, variant_pattern in heldout.items():
                                related = variant_name.startswith(cue_name[:1] + "_")
                                # Explicit family assignment avoids deriving labels from input vectors.
                                family = {"A_vertical_shifted": 0, "B_horizontal_shifted": 1,
                                          "C_diagonal_shifted": 2, "A_vertical_one_pixel_removed": 0}[variant_name]
                                visual_vec = refs_all[variant_name]["voltage"]
                                generalization_rows.append({
                                    "seed": seed, "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                                    "mapping": mapping_name, "condition": condition,
                                    "exposures_per_association": exposure, "auditory_cue": cue,
                                    "taught_visual_pattern": list(pats)[target],
                                    "heldout_pattern": variant_name, "variant_family": family,
                                    "variant_matches_taught_family": int(family == target),
                                    "auditory_to_base_visual_cosine": _cosine(response, base_refs[list(pats)[target]]["voltage"]),
                                    "auditory_to_heldout_visual_cosine": _cosine(response, visual_vec),
                                    "heldout_visual_to_base_cosine": _cosine(visual_vec, base_refs[list(pats)[family]]["voltage"]),
                                    "auditory_response_silent": int(np.linalg.norm(response) == 0.0),
                                })
                        net.rest(0.200)
                wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
                sim_s = float(net.net.t / second) - sim0
                performance_rows.append({"seed": seed,
                    "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                    "mapping": mapping_name, "condition": condition,
                    "wall_seconds": wall, "cpu_seconds": cpu, "simulated_seconds": sim_s,
                    "simulated_to_wall_ratio": sim_s / wall if wall else float("nan"),
                    "associative_neurons": N_ASSOCIATIVE, "visual_input_cells": 25,
                    "visual_projection_edges": len(net.visual_edges),
                    "auditory_to_associative_synapses": 3 * N_ASSOCIATIVE,
                })

    # Generate a separate seed-level pretraining distinguishability table from
    # the captured visual reference vectors (including all development/validation seeds).
    visual_dist = []
    for seed in all_seeds:
        names = list(pats)
        vectors = [visual_store[f"seed{seed}_{name}_voltage"] for name in names]
        for i in range(3):
            for j in range(i + 1, 3):
                visual_dist.append({"seed": seed, "seed_role": "validation" if validation_only or seed in val_seeds else "development",
                                    "pattern_i": names[i], "pattern_j": names[j],
                                    "voltage_cosine": _cosine(vectors[i], vectors[j])})

    config = {
        "seeds_development": [] if validation_only else list(seeds),
        "seeds_prospective_validation": list(val_seeds),
        "exposures_per_association": list(EXPOSURES), "conditions": list(CONDITIONS),
        "mappings": {k: list(v) for k, v in MAPPINGS.items()},
        "patterns": {k: v.tolist() for k, v in pats.items()},
        "heldout_variants": {k: v.tolist() for k, v in heldout.items()},
        "associative_neurons": N_ASSOCIATIVE, "visual_input_cells": 25,
        "visual_fixed_projection_probability": VISUAL_P, "visual_fixed_projection_weight_mv": VISUAL_GAIN_MV,
        "auditory_gain_mv": AUDITORY_GAIN_MV, "visual_topology_seed_rule": "SeedSequence([network_seed, 517]); Bernoulli mask independent of pattern and mapping",
        "auditory_initial_weight_seed_rule": "SeedSequence([network_seed, 991]); Uniform(0.05, 0.15)",
        "experience_order_seed_rule": "SeedSequence([network_seed, 801]); identical across mapping and condition controls",
        "network_equations": "same LIF: tau=10 ms, rest=-70 mV, threshold=-50 mV, refractory=5 ms",
        "learning_rule": "existing pair-based STDP unchanged (Apre +0.01; Apost -0.012; tau=20 ms; weights clipped [0,1])",
        "training_event_rule": "auditory spike at 10 ms; active visual sensor cells spike together at 25 ms; 60 ms event; no position/pattern ID sent to network",
        "unpaired_control_rule": f"audio-only event, {UNPAIRED_GAP_S*1000:g} ms silence, visual-only event, then the same silence before the next auditory event; gaps exceed 12 STDP time constants",
        "readout": "cosine similarity between baseline-subtracted peak-depolarization or spike-count vectors and pretraining visual reference vectors; zero vectors recorded invalid/NaN",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "brian2": version("brian2"), "matplotlib": version("matplotlib")},
        "random_projection_edges_per_seed": {seed: len(projection_records[str(seed)]) for seed in all_seeds},
        "total_wall_seconds": time.perf_counter() - started_total,
        "validation_analysis_locked_before_run": True,
    }
    (out / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(out / "visual_reference_responses.npz", **visual_store)
    np.savez_compressed(out / "auditory_recall_responses.npz", **audio_store)
    _write(out / "population_similarity.csv", similarity_rows)
    _write(out / "recognition_trials.csv", trial_rows)
    _write(out / "synaptic_weight_history.csv", weight_rows)
    _write(out / "control_comparison.csv", comparison_rows)
    _write(out / "generalization_results.csv", generalization_rows)
    _write(out / "performance.csv", performance_rows)
    _write(out / "visual_reference_responses.csv", refs_rows)
    _write(out / "visual_pattern_distinguishability.csv", visual_dist)
    np.savez_compressed(out / "visual_projection_connectivity.npz",
                        **{f"seed{seed}": projection_records[str(seed)] for seed in all_seeds})
    _diagnostic_patterns(out, pats)
    example_seed = seeds[0] if seeds else val_seeds[0]
    example_refs = {name: {"voltage": visual_store[f"seed{example_seed}_{name}_voltage"],
                           "spikes": visual_store[f"seed{example_seed}_{name}_spikes"]} for name in pats}
    example_audio = {name: {"voltage": audio_store[f"{example_seed}_identity_paired_gated_e50_cue{i}_voltage"],
                            "spikes": audio_store[f"{example_seed}_identity_paired_gated_e50_cue{i}_spikes"]}
                     for i, name in enumerate(pats)}
    _make_figures(out, curve_summary, weight_rows, example_refs, example_audio)
    _write(out / "seed_level_summary.csv", curve_summary)
    _report(out, config, curve_summary, visual_dist, generalization_rows, performance_rows, comparison_rows)
    return config


def _report(out, config, summaries, visual_dist, generalization, performance, comparisons):
    def mean_summary(role, mapping, condition, exposure, kind):
        vals = [float(r["accuracy"]) for r in summaries if r["seed_role"] == role and r["mapping"] == mapping
                and r["condition"] == condition and int(r["exposures_per_association"]) == exposure
                and r["response_kind"] == kind]
        return float(np.mean(vals)) if vals else float("nan")
    def mean_weight_change(role, mapping, condition):
        vals = [float(r["total_absolute_weight_change"]) for r in comparisons
                if r["seed_role"] == role and r["mapping"] == mapping and r["condition"] == condition
                and int(r["exposures_per_association"]) == 50]
        return float(np.mean(vals)) if vals else float("nan")
    visual_cos = {role: float(np.mean([float(r["voltage_cosine"]) for r in visual_dist
                                      if r["seed_role"] == role])) for role in ("development", "validation")}
    heldout_rows = [r for r in generalization if r["seed_role"] == "validation"
                    and r["mapping"] == "identity" and r["condition"] == "paired_gated"
                    and int(r["variant_matches_taught_family"]) == 1]
    heldout_base = float(np.mean([float(r["auditory_to_base_visual_cosine"]) for r in heldout_rows]))
    heldout_sim = float(np.mean([float(r["auditory_to_heldout_visual_cosine"]) for r in heldout_rows]))
    heldout_visual_base = float(np.mean([float(r["heldout_visual_to_base_cosine"]) for r in heldout_rows]))
    lines = ["# Distributed visual association experiment", "",
             f"This separately selectable Brian2/NumPy experiment used {config['associative_neurons']} shared LIF neurons, 25 binary visual input channels, three auditory spike channels, and the existing pair-based STDP rule. Development seeds were {', '.join(map(str, config['seeds_development']))}; prospective validation seeds were {', '.join(map(str, config['seeds_prospective_validation']))}. Total wall time: {config['total_wall_seconds']:.1f} s.", "",
             "The three trained arrays are 5×5 vertical, horizontal, and main-diagonal lines. Every active visual cell produces one spike on its own input channel. A fixed Bernoulli visual projection is drawn from a seed-only rule before any training; it does not depend on the pattern or mapping. The auditory-to-associative pathway uses the unchanged STDP update. The network receives only active cell indices and auditory spike indices; pattern names and mapping labels remain in the experiment controller and analysis.", "",
             "For live inspection, `python -m mdl.live_dashboard --distributed` selects the persistent 5×5 dashboard. It displays the actual visual array, the shared population voltage traces and sensory/association raster, and auditory synaptic weights. Simulation advances in fixed 10 ms chunks independently of plot refresh.", "",
             "## 1. Cause of the earlier silent recalls", "",
             "See `../recall_diagnostics/SILENT_RECALL_REPORT.md`. Replaying the preserved three-cell baseline produced four silent seed–cue cases out of 15; the peak-winning neuron remained below the −50 mV threshold, and Brian2 recorded no spikes. The result is subthreshold drive, not a missed monitor event or refractory block. The original five replicates reuse each seed–cue jitter and are duplicates for this question.", "",
             "## 2–3. Distributed visual representation and distinguishability", "",
             f"Visual input is no longer wired to an identity output. Each of the 25 input cells projects randomly and sparsely to the shared population. `visual_pattern_distinguishability.csv` gives all pairwise pretraining cosine similarities per seed; the initial references are fixed before association. Mean pairwise voltage cosine was {visual_cos['development']:.3f} on development seeds and {visual_cos['validation']:.3f} on validation seeds, so the patterns were distinguishable but their population codes were correlated. `visual_projection_connectivity.npz` records exact edges.", "",
             "## 4–8. Cross-modal association, mappings, controls, and spikes", "",
             f"The primary analysis compares each audio-only associative response with all three visual-only reference vectors using cosine similarity. Recognition is the highest finite cosine; zero vectors are explicitly invalid and count as incorrect. Peak-depolarization and spike-count results are reported independently. The `identity` schedule pairs cue indices with the corresponding visual arrays; `remapped` teaches A→C, B→A, C→B. Conditions are plasticity off, paired-gated STDP, continuous STDP, and an unpaired control: {config['unpaired_control_rule']}. No output teacher is used.", "",
             "Mean voltage-based recognition across seeds by mapping and exposure:", "",
             "| Seed role | Mapping | Condition | Exposures | Voltage cosine | Spike cosine | Spike silent fraction |", "|---|---|---|---:|---:|---:|---:|"]
    for role in ("development", "validation"):
        for mapping in MAPPINGS:
            for condition in CONDITIONS:
                for exposure in EXPOSURES:
                    rows = [r for r in summaries if r["seed_role"] == role and r["mapping"] == mapping and r["condition"] == condition and r["exposures_per_association"] == exposure]
                    if not rows: continue
                    v = np.mean([r["accuracy"] for r in rows if r["response_kind"] == "voltage"])
                    s = np.mean([r["accuracy"] for r in rows if r["response_kind"] == "spikes"])
                    silent = np.mean([r["silent_fraction"] for r in rows if r["response_kind"] == "spikes"])
                    lines.append(f"| {role} | {mapping} | {condition} | {exposure} | {v:.1%} | {s:.1%} | {silent:.1%} |")
    lines += ["", "Per-seed spike-based outcomes at 50 experiences:", "",
              "| Cohort | Seed | Identity paired | Remapped paired | Identity no-learning | Identity unpaired |", "|---|---:|---:|---:|---:|---:|"]
    for role, role_seeds in (("development", config["seeds_development"]),
                             ("validation", config["seeds_prospective_validation"])):
        for seed in role_seeds:
            def seed_score(mapping, condition):
                match = [r["accuracy"] for r in summaries if int(r["seed"]) == int(seed)
                         and r["seed_role"] == role and r["mapping"] == mapping
                         and r["condition"] == condition and int(r["exposures_per_association"]) == 50
                         and r["response_kind"] == "spikes"]
                return f"{match[0]:.0%}" if match else "n/a"
            lines.append(f"| {role} | {seed} | {seed_score('identity','paired_gated')} | {seed_score('remapped','paired_gated')} | {seed_score('identity','plasticity_off')} | {seed_score('identity','unpaired')} |")
    lines += ["", "Per-seed results are in `seed_level_summary.csv`; per cue/reference comparisons and response margins are in `population_similarity.csv` and `recognition_trials.csv`. The saved NPZ files contain raw population vectors. `synaptic_weight_history.csv` reports actual weight changes; a recognition increase without corresponding learned weight changes should be interpreted accordingly.", "",
              "## 9–10. Retention and held-out visual variants", "",
             f"Every event runs in one persistent network with no membrane or synapse reset between experiences. Paired experiences are separated by 50 ms quiet periods, and evaluation presentations by 200 ms. There is no spontaneous input or weight-decay term, so quiet periods do not change learned synaptic weights; responses at later exposure checkpoints remained accurate. A dedicated repeated same-cue probe after a long post-training quiet interval was not included and is a useful next check. Four held-out arrays (shifted lines and a removed pixel) were never used for association training. On prospective identity-pair trials, auditory-to-base cosine averaged {heldout_base:.3f}, compared with {heldout_sim:.3f} to the related held-out visual arrays; held-out visual arrays themselves averaged {heldout_visual_base:.3f} cosine to their base pattern. Similarity is representational compatibility, not proof of an abstract visual concept.", "",
              "## 11–12. Engineered assumptions, compute, and next step", "",
              "The visual representation depends on an engineered fixed random projection and a one-spike-per-active-pixel encoding. Only auditory afferent weights change through STDP. This is not a learned visual feature extractor. The associative population has no recurrent connections, no new neuron equations, and no target output pathway. If distinct visual references or spike responses fail, that failure is retained as evidence; no parameter search is performed on the prospective seeds.", "",
              f"Mean per-network runtime was {np.mean([float(r['wall_seconds']) for r in performance]):.2f} wall seconds and {np.mean([float(r['cpu_seconds']) for r in performance]):.2f} CPU seconds; simulated-time/wall-time averaged {np.mean([float(r['simulated_to_wall_ratio']) for r in performance]):.3f}×.", "",
              f"At 50 experiences, paired STDP changed total auditory weight L1 by {mean_weight_change('development','identity','paired_gated'):.3f} on development seeds and {mean_weight_change('validation','identity','paired_gated'):.3f} on validation seeds; the corrected unpaired control changed weights by {mean_weight_change('development','identity','unpaired'):.5f} and {mean_weight_change('validation','identity','unpaired'):.5f}. Paired-gated and continuous conditions had the same outcomes. Since the validation cohort reproduced association learning but held-out similarity was lower, the smallest next experiment is a preregistered translation-tolerance study with the current network fixed, plus repeated same-cue recall after a 500 ms post-training quiet interval. Do not enlarge the network or change STDP until that targeted test clarifies the current limitation. No curiosity, language, or structural plasticity is included.", "",
              "## Important interpretation note", "",
              "This milestone tests an association between independently generated sensory activity patterns. The learned mapping is assessed by a frozen population-similarity readout, not a trained classifier. The fixed random projection and pattern construction are explicit sources of inductive bias. Do not compare its spike percentages directly to the prior three-output network's forced winner readout.", ""]
    (out / "DISTRIBUTED_VISUAL_ASSOCIATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def regenerate_artifacts(outdir="results/distributed_visual_association"):
    """Rebuild summaries, figures, and report solely from saved measurements."""
    out = Path(outdir)
    def read(name):
        with (out / name).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    config = json.loads((out / "experiment_config.json").read_text(encoding="utf-8"))
    trials = read("recognition_trials.csv")
    weights = read("synaptic_weight_history.csv")
    comparisons = read("control_comparison.csv")
    visual_dist = read("visual_pattern_distinguishability.csv")
    generalization = read("generalization_results.csv")
    performance = read("performance.csv")
    groups = {}
    for row in trials:
        key = (int(row["seed"]), row["seed_role"], row["mapping"], row["condition"],
               int(row["exposures_per_association"]), row["response_kind"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, records in groups.items():
        seed, role, mapping, condition, exposure, kind = key
        summaries.append({"seed": seed, "seed_role": role, "mapping": mapping,
                          "condition": condition, "exposures_per_association": exposure,
                          "response_kind": kind,
                          "accuracy": float(np.mean([int(r["correct"]) for r in records])),
                          "silent_fraction": float(np.mean([int(r["silent_population"]) for r in records])) if kind == "spikes" else 0.0,
                          "active_response_fraction": float(np.mean([int(r["active_associative_neurons"]) > 0 for r in records]))})
    refs_npz = np.load(out / "visual_reference_responses.npz")
    audio_npz = np.load(out / "auditory_recall_responses.npz")
    pats = patterns()
    seed = int((config["seeds_development"] or config["seeds_prospective_validation"])[0])
    refs = {name: {"voltage": refs_npz[f"seed{seed}_{name}_voltage"],
                   "spikes": refs_npz[f"seed{seed}_{name}_spikes"]} for name in pats}
    audio = {name: {"voltage": audio_npz[f"{seed}_identity_paired_gated_e50_cue{i}_voltage"],
                    "spikes": audio_npz[f"{seed}_identity_paired_gated_e50_cue{i}_spikes"]}
             for i, name in enumerate(pats)}
    _write(out / "seed_level_summary.csv", summaries)
    _make_figures(out, summaries, weights, refs, audio)
    _report(out, config, summaries, visual_dist, generalization, performance, comparisons)
    return summaries


def merge_result_directories(base_dir="results/distributed_visual_association",
                             validation_dir="results/distributed_visual_association_validation",
                             development_fallback="results/distributed_visual_association_preliminary"):
    """Append a separately run locked validation cohort to development artifacts."""
    base, val = Path(base_dir), Path(validation_dir)
    base_config = json.loads((base / "experiment_config.json").read_text(encoding="utf-8"))
    val_config = json.loads((val / "experiment_config.json").read_text(encoding="utf-8"))
    if val_config.get("seeds_development") or not val_config.get("seeds_prospective_validation"):
        raise ValueError("validation directory must contain only prospective seeds")
    for key in ("patterns", "mappings", "exposures_per_association", "learning_rule",
                "readout", "unpaired_control_rule", "experience_order_seed_rule"):
        if base_config[key] != val_config[key]:
            raise ValueError(f"development and validation differ in locked setting: {key}")
    dev_seeds = set(map(int, base_config["seeds_development"]))
    val_seeds = set(map(int, val_config["seeds_prospective_validation"]))
    previously_merged = bool(val_seeds.intersection(map(int, base_config.get("seeds_prospective_validation", []))))
    csv_names = ("population_similarity.csv", "recognition_trials.csv", "synaptic_weight_history.csv",
                 "control_comparison.csv", "generalization_results.csv", "performance.csv",
                 "visual_reference_responses.csv", "visual_pattern_distinguishability.csv")
    for name in csv_names:
        with (base / name).open(newline="", encoding="utf-8") as f:
            all_existing = list(csv.DictReader(f))
        existing = [r for r in all_existing if int(r["seed"]) in dev_seeds]
        additional = list(csv.DictReader((val / name).open(newline="", encoding="utf-8")))
        fields = list(existing[0]) if existing else list(additional[0])
        with (base / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(existing + additional)
    for name in ("visual_reference_responses.npz", "auditory_recall_responses.npz", "visual_projection_connectivity.npz"):
        merged = {}
        def load_role(path, role_seeds):
            values = {}
            with np.load(path) as archive:
                for key in archive.files:
                    if any(key.startswith(f"seed{s}_") or key.startswith(f"{s}_") or key == f"seed{s}"
                           for s in role_seeds):
                        values[key] = archive[key]
            return values
        dev_arrays = load_role(base / name, dev_seeds)
        expected_dev_seeds = dev_seeds
        have_dev_seeds = {next((s for s in dev_seeds if k.startswith(f"seed{s}_") or k.startswith(f"{s}_") or k == f"seed{s}"), None)
                          for k in dev_arrays}
        if not expected_dev_seeds.issubset(have_dev_seeds):
            dev_arrays = load_role(Path(development_fallback) / name, dev_seeds)
        merged.update(dev_arrays)
        merged.update(load_role(val / name, val_seeds))
        if not merged:
            raise ValueError(f"no arrays found while merging {name}")
        np.savez_compressed(base / name, **merged)
    development_wall = float(base_config["total_wall_seconds"])
    if previously_merged:
        development_wall -= float(val_config["total_wall_seconds"])
    base_config["seeds_prospective_validation"] = val_config["seeds_prospective_validation"]
    base_config["total_wall_seconds"] = development_wall + float(val_config["total_wall_seconds"])
    base_config["random_projection_edges_per_seed"] = {
        str(seed): base_config["random_projection_edges_per_seed"][str(seed)] for seed in dev_seeds}
    base_config["random_projection_edges_per_seed"].update(val_config["random_projection_edges_per_seed"])
    base_config["validation_analysis_locked_before_run"] = True
    (base / "experiment_config.json").write_text(json.dumps(base_config, indent=2), encoding="utf-8")
    regenerate_artifacts(base)


def replay_unpaired_response_arrays(outdir="results/distributed_visual_association"):
    """Recreate only the missing final-schedule dev vectors from saved CSV data.

    This is used when regenerating artifacts; it does not rerun or change the
    benchmark analysis. The replay uses the exact seed, mapping, exposure order,
    network parameters, and final two-sided unpaired timing control.
    """
    out = Path(outdir)
    pats, heldout = patterns(), variants()
    names = list(pats)
    arrays = {}
    started = time.perf_counter()
    sim_total = 0.0
    for seed in DEV_SEEDS:
        for mapping_name, mapping in MAPPINGS.items():
            net = DistributedVisualAssociator(seed, N_ASSOCIATIVE, VISUAL_P,
                                              VISUAL_GAIN_MV, AUDITORY_GAIN_MV, learning=True)
            # Match pretraining visual-reference exposures and event order.
            _references(net, {**pats, **heldout}, seed, [], {})
            order_rng = np.random.default_rng(np.random.SeedSequence([seed, 801]))
            order = [int(c) for _ in range(max(EXPOSURES)) for c in order_rng.permutation(3)]
            current = 0
            for exposure in EXPOSURES:
                while current < exposure:
                    for cue in order[current * 3:(current + 1) * 3]:
                        pattern_idx = int(mapping[cue])
                        net.set_plasticity(True)
                        net.present(cue=cue, capture=False)
                        net.rest(UNPAIRED_GAP_S)
                        net.present(pattern=pats[names[pattern_idx]], capture=False)
                        net.rest(UNPAIRED_GAP_S)
                        net.set_plasticity(False)
                    current += 1
                for cue in range(3):
                    net.set_plasticity(False)
                    response = net.present(cue=cue, capture=True)
                    key = f"{seed}_{mapping_name}_unpaired_e{exposure}_cue{cue}"
                    arrays[f"{key}_voltage"] = response["peak_depolarization_mv"]
                    arrays[f"{key}_spikes"] = response["spike_counts"].astype(float)
                    arrays[f"{key}_mean_voltage"] = response["mean_depolarization_mv"]
                    net.rest(0.200)
                if exposure == max(EXPOSURES):
                    net.rest(0.200)
            sim_total += float(net.net.t / second)
    archive_path = out / "auditory_recall_responses.npz"
    with np.load(archive_path) as archive:
        merged = {key: archive[key] for key in archive.files
                  if not any(key.startswith(f"{seed}_{mapping}_unpaired_")
                             for seed in DEV_SEEDS for mapping in MAPPINGS)}
    merged.update(arrays)
    np.savez_compressed(archive_path, **merged)
    recovery = {"purpose": "deterministic replay to restore final-schedule development unpaired population vectors",
                "seeds": list(DEV_SEEDS), "mappings": list(MAPPINGS),
                "unpaired_gap_ms_each_side": UNPAIRED_GAP_S * 1000,
                "simulation_seconds_replayed": sim_total,
                "wall_seconds": time.perf_counter() - started,
                "auditory_response_vector_count": len(arrays) // 3}
    (out / "unpaired_response_array_replay.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    return recovery


if __name__ == "__main__":
    run()
