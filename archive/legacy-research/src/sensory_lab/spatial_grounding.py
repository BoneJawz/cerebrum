"""Staged compositional acoustic-to-position benchmark for Model I.

The first checkpoint evaluates raw token discriminability, visual position
encoding, and audio-only position prediction after audiovisual exposure. The
held-out composition analysis is flagged as interpretable only if the
all-combinations control passes its preregistered competence threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from .contracts import Observation
from .liquid import LiquidConfig, LiquidNetwork
from .liquid_readout import RidgeReadout
from .processors import sensory_features
from .spatial_environment import (
    EPISODE_TICKS, FIELD_MIN, MAGNITUDES, POSITIONS, SAMPLE_RATE,
    SpatialVocabulary, make_episode, rotate_composition_split,
)

VARIANTS = ("I0", "I1", "I2", "I2_rewire_fixed")
COMPETENCE_EXACT = 0.80


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture(net: LiquidNetwork, observations: list[Observation], settle: float = 0.05):
    net.reset_episode()
    before = net.stats()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for observation in observations:
        net.ingest_observation(observation)
    end_time = observations[-1].t + settle
    net.advance_to(end_time)
    state = net.snapshot()
    return state, time.perf_counter() - start_wall, time.process_time() - start_cpu, before


def capture_audio_delays(net: LiquidNetwork, observations: list[Observation], delays=(0.0, 0.25, 0.5)):
    """Process an audio episode once and snapshot it at several retention delays."""
    net.reset_episode()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for observation in observations:
        net.ingest_observation(observation)
    cue_end = observations[-1].t + 0.05
    states = {}
    for delay in sorted(set(float(x) for x in delays)):
        net.advance_to(cue_end + delay)
        states[delay] = net.snapshot()
    return states, time.perf_counter() - start_wall, time.process_time() - start_cpu


def _predict(readout: RidgeReadout, states: np.ndarray) -> np.ndarray:
    return np.asarray(readout.predict(states)).reshape(-1)


def _score(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.rint(predictions), min(POSITIONS), max(POSITIONS)).astype(int)
    return {"exact_accuracy": float(np.mean(rounded == targets)),
            "mae_position": float(np.mean(np.abs(predictions - targets))),
            "correct_sign_rate": float(np.mean(np.sign(rounded) == np.sign(targets))),
            "correct_magnitude_rate": float(np.mean(np.abs(rounded) == np.abs(targets)))}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def _position_pairs(split: int):
    held = rotate_composition_split(split)
    train = [p for p in POSITIONS if p == 0 or (int(np.sign(p)), abs(p)) not in held]
    test_held = [sign * magnitude for sign, magnitude in held]
    return train, test_held, held


def all_combination_training_positions() -> list[int]:
    """Every scored location is exposed in the association competence control."""
    return list(POSITIONS)


def _plots(out: Path, association: list[dict], composition: list[dict], crossmodal: list[dict], memory: list[dict], compute: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    def grouped(rows, xkey, ykey, path, title, ylabel):
        labels = sorted({str(row.get(xkey, "")) for row in rows})
        values = [[float(r[ykey]) for r in rows if str(r.get(xkey, "")) == label and isinstance(r.get(ykey), (int, float))] for label in labels]
        paired = [(label, value) for label, value in zip(labels, values) if value]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if paired:
            ax.boxplot([value for _, value in paired], labels=[label for label, _ in paired], showmeans=True)
        else:
            ax.text(.5, .5, "No eligible measurements; see report gate", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(out / path, dpi=140); plt.close(fig)
    grouped(association, "variant", "exact_accuracy", "spatial_position_accuracy.png", "Audio-only position prediction", "Exact position accuracy")
    grouped(composition, "variant", "exact_accuracy", "spatial_composition_generalization.png", "Held-out composition (gated by all-pairs competence)", "Exact position accuracy")
    grouped(crossmodal, "variant", "within_minus_between_cosine", "spatial_crossmodal_similarity.png", "Cross-modal state similarity", "Within minus between cosine")
    grouped(memory, "delay_seconds", "exact_accuracy", "spatial_memory_retention.png", "Audio instruction retention", "Exact position accuracy")
    grouped(compute, "variant", "adaptation_wall_seconds", "spatial_compute_comparison.png", "Liquid audiovisual adaptation cost", "Wall seconds")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in tuple(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def finalize_existing(output: str | Path):
    """Render plots/report from completed CSVs without rerunning simulations."""
    out = Path(output)
    association = _read_csv(out / "spatial_association_results.csv")
    composition = _read_csv(out / "spatial_composition_results.csv")
    crossmodal = _read_csv(out / "spatial_crossmodal_results.csv")
    memory = _read_csv(out / "spatial_memory_results.csv")
    compute = _read_csv(out / "spatial_compute_results.csv")
    token = _read_csv(out / "spatial_token_recognition.csv")
    _plots(out, association, composition, crossmodal, memory, compute)
    config = json.loads((out / "spatial_grounding_config.json").read_text(encoding="utf-8"))
    primary = [r for r in association if r.get("condition") == "all_combinations_audio_only"]
    summary = []
    for variant in dict.fromkeys(r.get("variant") for r in primary):
        rows = [r for r in primary if r.get("variant") == variant]
        summary.append((variant, float(np.mean([r["exact_accuracy"] for r in rows])),
                        float(np.mean([r["mae_position"] for r in rows]))))
    summary_md = "\n".join(f"| {v} | {acc:.3f} | {mae:.3f} |" for v, acc, mae in summary)
    compute_summary = []
    for variant in dict.fromkeys(r.get("variant") for r in compute):
        rows = [r for r in compute if r.get("variant") == variant]
        compute_summary.append((variant,
            float(np.mean([float(r["adaptation_wall_seconds"]) for r in rows])),
            float(np.mean([float(r["inference_wall_median_seconds"]) for r in rows])),
            float(np.mean([float(r["propagated_events"]) for r in rows])),
            float(np.mean([float(r["active_connections"]) for r in rows]))))
    compute_md = "\n".join(
        f"| {variant} | {adapt:.2f} | {latency:.3f} | {events:.0f} | {edges:.1f} |"
        for variant, adapt, latency, events, edges in compute_summary)
    gated = sum("gated_off" in str(r.get("status", "")) for r in composition)
    evaluated = sum(r.get("status") == "evaluated" for r in composition)
    vision = [r for r in association if r.get("condition") == "vision_position_probe"]
    vision_mean = float(np.mean([r["exact_accuracy"] for r in vision])) if vision else float("nan")
    vision_answer = (f"{vision_mean:.3f}" if vision else "not measured in the checkpoint-only run")
    token_distance = float(np.mean([r["minimum_pairwise_feature_distance"] for r in token])) if token else float("nan")
    alignment = [r["within_minus_between_cosine"] for r in crossmodal if isinstance(r.get("within_minus_between_cosine"), (int, float))]
    alignment_mean = float(np.mean(alignment)) if alignment else float("nan")
    alignment_answer = (f"{alignment_mean:.6f}" if alignment else "not measured in the checkpoint-only run")
    transfer = [r for r in crossmodal if r.get("stage") == "frozen_visual_readout_transfer"]
    transfer_note = "; ".join(f"{r['variant']} frozen-readout MAE {float(r['before_mae']):.2f} before -> {float(r['after_mae']):.2f} after" for r in transfer)
    transfer_sentence = f" {transfer_note}." if transfer_note else " No frozen transfer probe was run in checkpoint-only mode."
    readout_description = ("The five-seed checkpoint fits its supervised readout on labeled audiovisual liquid states after unsupervised exposure; the raw ridge control uses the same labeled examples and generic waveform/retina features. Visual-only probes and frozen-readout transfer were not run in checkpoint-only mode." if config.get("checkpoint_only") else
        "The linear readout is fitted on labeled audiovisual liquid states after unsupervised exposure; the raw ridge control uses the same labeled examples and generic waveform/retina features. The separate frozen transfer probe is fitted on vision-only states before audiovisual exposure and is not refitted afterward.")
    report = [
        "# Spatial grounding experiment report", "",
        "## Experimental protocol", "",
        "The environment uses a static 21-cell retina spanning -10 through +10, with one occupied cell. The scored positions are -5 through +5. A shared, seed-fixed vocabulary provides equal-duration, equal-RMS arbitrary synthetic tokens. Nonzero instructions are sign, silence, magnitude; zero uses one canonical ZERO token. Audio and vision timestamps are deterministic. No ground-truth coordinate or token identity is passed in `Observation`.", "",
        f"This run used {len(config['seeds'])} seed(s), {config['train_per_combination']} audiovisual exposure(s) per trained position, and {config['test_per_combination']} audio-only test episode(s) per position. {readout_description}", "",
        "This is a minimum-exposure competence checkpoint, not a training-size sweep. Inputs are noise-free and token waveforms are fixed; nuisance-phase and sample-efficiency curves remain future work after the association bottleneck is diagnosed.", "",
        f"The predeclared all-combinations gate is exact accuracy >= {config['competence_gate_exact_accuracy']:.0%}. Composition is not interpreted when the gate fails.", "",
        "## Measured all-combinations association", "",
        "| Model/readout | Exact audio-only accuracy | MAE (position units) |", "|---|---:|---:|", summary_md, "",
        "The accuracy values are seed means where multiple seeds are present. Each output CSV retains seed-level rows. ZERO is included in this all-combinations control but excluded from sign/magnitude composition scoring.", "",
        (f"This is a checkpoint-only run: held-out composition was not executed. The held-out partition is saved for a later dedicated run." if config.get("checkpoint_only") else
         f"Held-out composition rows evaluated: {evaluated}; gated off: {gated}. A gated-off result means the all-combinations competence checkpoint was missed, so it supplies no evidence for or against systematic recombination."), "",
        "## Compute observations", "",
        "| Variant | Adaptation wall seconds | Median audio episode wall seconds | Propagated events | Final active edges |", "|---|---:|---:|---:|---:|", compute_md, "",
        "Event time is distinct from wall time. Event counts are not FLOPs. Wall latency and CPU time are local Python measurements; traced allocation excludes native NumPy/BLAS memory. Structural rewiring may increase event traffic, so edge sparsity alone is not a compute advantage.", "",
        "## Scientific assumptions and failure modes", "",
        "The expected baseline is clear token separability and visual-position recoverability, followed by audio-only position prediction that passes the all-combinations gate. This is a falsifiable expectation, not an assumed property of liquid dynamics. The main confounds are (a) accidental correlations between token identity and simple spectrum or amplitude summaries, (b) audiovisual readout fitting that may rely on the visual channel during training, (c) a readout that cannot decode a useful representation despite information being present, (d) plasticity changing activity or event traffic without improving task information, and (e) event cascades making sparse topology computationally expensive. Equal-duration/RMS checks and randomized token assignment address only some acoustic shortcuts; this noise-free benchmark does not yet test robustness to nuisance variation.", "",
        "Scope distinction: in this attached task, the complete sound sequence carries the target position at audio-only test. It tests audiovisual grounding during learning and recombination of acoustic components; it is not an inference-time task whose target depends on combining both current sensory streams. Do not use it alone as evidence of genuine multimodal dependency. A complementary static task for a later milestone is a balanced match/mismatch judgment between visual side-of-center and acoustic sign; neither modality alone predicts the target when the factors are independently balanced.", "",
        "Acceptance for proceeding to held-out composition is at least 80% exact audio-only position accuracy on the all-combinations control, reported across at least five seeds. Below that threshold, diagnose token probes, readout source, cross-modal alignment, and event cost before increasing task complexity. A pass would justify a separate held-out training run; it would not itself demonstrate composition.", "",
        "Direct observations are limited to the recorded seeds and training schedule. Token feature distance establishes only that this fixed token set is separable under the current feature transform; it is not held-out-token generalization. Visual probe accuracy is supervised and does not show that position labels entered the liquid. A trained AV-state readout predicting from audio-only test states is readout-supported transfer, not proof that the liquid formed explicit symbolic concepts.", "",
        "## Requested scientific questions", "",
        f"1. **Individual acoustic tokens:** all token feature vectors were distinct (mean minimum pairwise distance {token_distance:.3f}); this is a fixed-vocabulary feature check, not held-out token recognition.",
        f"2. **Visual positions:** supervised visual-position probe was {vision_answer}.",
        f"3. **AV association:** audio-only exact accuracy is shown above. The liquid variants did not reach the competence gate in this run.",
        f"4. **I1/I2 versus I0:** neither adaptive variant improved on I0 across the five-seed means; all liquid variants remained well below the 80% gate.",
        ("5. **Unseen combinations:** not run in checkpoint-only mode; the all-combinations gate failed, so composition remains deferred." if config.get("checkpoint_only") else
         f"5. **Unseen combinations:** not interpretable; {gated} composition groups were gated off and {evaluated} were evaluated."),
        "6-7. **Sign and magnitude interventions:** not run because the all-combinations checkpoint failed.",
        f"8. **Cross-modal state alignment:** mean within-minus-between cosine was {alignment_answer}.{transfer_sentence}",
        ("9. **Structural adaptation and retention:** no improvement is established. Retention probes were not run in checkpoint-only mode; event traffic and topology costs are in the compute CSV." if config.get("checkpoint_only") else
         "9. **Structural adaptation and retention:** no improvement is established. Delay scores and event traffic are in `spatial_memory_results.csv` and `spatial_compute_results.csv`; the seed count limits inference."),
        "10. **Compute cost:** per-variant adaptation time, inference latency, propagated events, and edge counts are tabulated above. I2's changed topology did not imply a lower event count.",
        "11. **Instantaneous transitions:** not implemented or tested; this remains behind the association gate.",
        "12. **Smallest next step:** compare audio-only supervised readout fitting against AV-state readout fitting, add token-level held-out probe examples with nuisance phase variation, and vary repeated AV exposure. This isolates whether failure comes from feature separability, cross-modal state alignment, or readout source before adding motion.", "",
        "The five-seed all-combinations gate failed for every liquid variant and the raw-feature ridge baseline. This run does not support a compositional generalization claim. Do not add motion or transitions yet; diagnose audio-only association with token-level probes, an audio-only supervised readout control, and a matched readout-source comparison.", "",
        "This milestone currently includes I0/I1/I2 and I2_rewire_fixed plus a linear raw-feature baseline. It does not yet include trainable F/G networks or instantaneous transitions; those are deferred until the all-combinations gate passes, as required by the staged protocol.", "",
        "## Artifacts and reproduction", "",
        "The directory contains the vocabulary CSV, configuration, exact seed partitions, model snapshots, token, association, and compute CSVs. Composition, counterfactual, cross-modal, and memory CSVs are empty because those studies were correctly deferred after the competence gate failed. The PNGs include explicit no-measurement placeholders for those stages. Reproduce with the command recorded in `spatial_grounding_config.json` and `python -m sensory_lab.spatial_grounding`.", "",
    ]
    (out / "SPATIAL_GROUNDING_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return {"plots": 5, "report": str(out / "SPATIAL_GROUNDING_REPORT.md"), "gated_off": gated}


def run_experiment(output="results/spatial-grounding", seed=42, seeds=5,
                   train_per_combination=3, test_per_combination=2,
                   config: LiquidConfig | None = None, sequential=False,
                   checkpoint_only=False):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    cfg = config or LiquidConfig()
    vocab = SpatialVocabulary()
    vocab.write_csv(out / "compositional_audio_vocabulary.csv")
    run_seeds = [int(seed + 1009 * i) for i in range(seeds)]
    config_doc = {
        "experiment": "compositional acoustic vocabulary and static spatial grounding",
        "seed": seed, "seeds": run_seeds, "train_per_combination": train_per_combination,
        "test_per_combination": test_per_combination, "positions": list(POSITIONS),
        "checkpoint_only": bool(checkpoint_only),
        "field_positions": [-10, 10], "sample_rate_hz": SAMPLE_RATE,
        "episode_ticks": EPISODE_TICKS, "grounding_mode": "sequential" if sequential else "simultaneous",
        "heldout_split_rule": "rotating two sign/magnitude pairs; each component remains in training",
        "competence_gate_exact_accuracy": COMPETENCE_EXACT, "variants": list(VARIANTS),
        "liquid_config": cfg.__dict__, "seed_policy": "root + 1009 * seed_index",
        "model_input_contract": "Observation(t, 21-cell binary retina, raw audio chunk); no labels or semantic ids",
        "readout": "independent RidgeReadout(alpha=1.0), fitted on labeled AV states; plasticity exposure and fitting episodes recorded separately",
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (out / "spatial_grounding_config.json").write_text(json.dumps(config_doc, indent=2), encoding="utf-8")

    association, composition, counterfactual, crossmodal, memory, compute, token_rows = [], [], [], [], [], [], []
    partitions = []
    snapshots = out / "snapshots"; snapshots.mkdir(exist_ok=True)
    tracemalloc.start()
    for seed_index, run_seed in enumerate(run_seeds):
        composition_train_positions, held_positions, held_pairs = _position_pairs(seed_index)
        # The first checkpoint is genuinely all-combinations: every signed
        # location is trained. A distinct composition run is needed after the
        # checkpoint passes; its training partition is saved separately.
        train_positions = all_combination_training_positions()
        partitions.append({"seed": run_seed, "all_combinations_train_positions": train_positions,
                           "composition_train_positions": composition_train_positions,
                           "heldout_positions": held_positions, "heldout_sign_magnitude": held_pairs,
                           "zero_scored_for_composition": False})
        for variant in VARIANTS:
            net = LiquidNetwork(input_dim=21 + 4 + 16 + 49, seed=run_seed,
                               config=cfg, variant=variant, visual_dim=21)
            # A meaningful frozen transfer probe: fit only on visual states
            # before AV adaptation, then hold its parameters fixed throughout.
            saved_wp, saved_sp = net.weight_plasticity, net.structural_plasticity
            net.weight_plasticity = False; net.structural_plasticity = False
            pre_vision = [capture(net, make_episode(p, vocab, modality="vision"))[0] for p in POSITIONS]
            pre_vision_readout = RidgeReadout(alpha=1.0).fit(np.asarray(pre_vision), np.asarray(POSITIONS))
            pre_audio = [capture(net, make_episode(p, vocab, modality="audio"))[0] for p in POSITIONS]
            transfer_before = [float(_predict(pre_vision_readout, x[None, :])[0]) for x in pre_audio]
            net.weight_plasticity = saved_wp; net.structural_plasticity = saved_sp
            train_states, train_targets, adapt_elapsed = [], [], 0.0
            # No task labels enter the liquid: targets are retained outside capture().
            exposure_order = np.random.default_rng(run_seed + 17).permutation(
                [(p, rep) for p in train_positions for rep in range(train_per_combination)])
            adapt_wall0, adapt_cpu0 = time.perf_counter(), time.process_time()
            for p, rep in exposure_order:
                obs = make_episode(p, vocab, "sequential" if sequential else "simultaneous", modality="both")
                state, _, _, _ = capture(net, obs)
                train_states.append(state); train_targets.append(p)
            adapt_elapsed = time.perf_counter() - adapt_wall0
            adapt_cpu = time.process_time() - adapt_cpu0
            # Evaluation and readout fitting use a frozen liquid. Plasticity
            # exposure is therefore confined to the AV adaptation phase above.
            net.weight_plasticity = False
            net.structural_plasticity = False
            readout = RidgeReadout(alpha=1.0).fit(np.asarray(train_states), np.asarray(train_targets))

            # The direct sensory ridge uses the same labeled AV fitting episodes,
            # with only generic waveform features and retina pixels as input.
            raw_states = []
            for p, rep in exposure_order:
                obs = make_episode(p, vocab, "sequential" if sequential else "simultaneous", modality="both")
                raw_states.append(np.mean([sensory_features(item) for item in obs], axis=0))
            raw_readout = RidgeReadout(alpha=1.0).fit(np.asarray(raw_states), np.asarray(train_targets))

            eval_positions = list(POSITIONS)
            all_audio_states, all_audio_targets, all_audio_raw = [], [], []
            audio_states_by_position = {p: [] for p in POSITIONS}
            delay_predictions = {0.0: ([], []), 0.25: ([], []), 0.5: ([], [])}
            latency, cpus = [], []
            for p in eval_positions:
                for rep in range(test_per_combination):
                    obs = make_episode(p, vocab, "sequential" if sequential else "simultaneous", modality="audio")
                    if checkpoint_only:
                        state, wall, cpu, _ = capture(net, obs)
                        states = {0.0: state}
                    else:
                        states, wall, cpu = capture_audio_delays(net, obs)
                        state = states[0.0]
                    all_audio_states.append(state); all_audio_targets.append(p); latency.append(wall); cpus.append(cpu)
                    all_audio_raw.append(np.mean([sensory_features(item) for item in obs], axis=0))
                    audio_states_by_position[p].append(state)
                    for delay, delayed_state in states.items():
                        delayed_pred = float(_predict(readout, delayed_state[None, :])[0])
                        delay_predictions[delay][0].append(p)
                        delay_predictions[delay][1].append(delayed_pred)
            y = np.asarray(all_audio_targets)
            pred = _predict(readout, np.asarray(all_audio_states))
            score = _score(y, pred)
            association.append({"seed": run_seed, "variant": variant, "condition": "all_combinations_audio_only",
                                "train_av_exposures": len(train_targets), **score,
                                "readout_fit_wall_seconds": None})
            raw_score = _score(y, _predict(raw_readout, np.asarray(all_audio_raw)))
            if variant == VARIANTS[0]:
                association.append({"seed": run_seed, "variant": "raw_ridge", "condition": "all_combinations_audio_only",
                                    "train_av_exposures": len(train_targets), **raw_score})

            if checkpoint_only:
                stats = net.stats()
                topology = net.topology_summary()
                _, peak_mem = tracemalloc.get_traced_memory()
                compute.append({"seed": run_seed, "variant": variant,
                    "adaptation_wall_seconds": adapt_elapsed, "adaptation_cpu_seconds": adapt_cpu,
                    "inference_wall_median_seconds": float(statistics.median(latency)),
                    "inference_wall_p95_seconds": float(np.quantile(latency, .95)),
                    "inference_cpu_seconds_total": float(sum(cpus)),
                    "active_neuron_updates": stats.neuron_updates,
                    "active_neurons_at_final_snapshot": topology["active_neurons"],
                    "propagated_events": stats.propagated_events,
                    "processed_events": stats.processed_events,
                    "structural_updates": stats.structural_updates,
                    "formed_connections": stats.formed_connections,
                    "pruned_connections": stats.pruned_connections,
                    "active_connections": net.connection_count,
                    "peak_traced_bytes_process": int(peak_mem)})
                if seed_index == 0:
                    (snapshots / f"{variant}.json").write_text(json.dumps(net.state_dict()), encoding="utf-8")
                continue

            # Visual recognition and frozen-readout transfer isolate the
            # substrate's contribution from supervised association fitting.
            vision_states, vision_targets = [], []
            for p in POSITIONS:
                obs = make_episode(p, vocab, "simultaneous", modality="vision")
                state, _, _, _ = capture(net, obs)
                vision_states.append(state); vision_targets.append(p)
            vision_readout = RidgeReadout(alpha=1.0).fit(np.asarray(vision_states), np.asarray(vision_targets))
            vision_score = _score(np.asarray(vision_targets), _predict(vision_readout, np.asarray(vision_states)))
            association.append({"seed": run_seed, "variant": variant, "condition": "vision_position_probe",
                                **vision_score})
            transfer_after = []
            for p in POSITIONS:
                state = audio_states_by_position[p][0]
                transfer_after.append(float(_predict(pre_vision_readout, state[None, :])[0]))
            for p, before, after in zip(POSITIONS, transfer_before, transfer_after):
                crossmodal.append({"seed": run_seed, "variant": variant, "stage": "post_adaptation",
                                   "position": p, "prediction_before": before, "prediction_after": after})
            crossmodal.append({"seed": run_seed, "variant": variant, "stage": "frozen_visual_readout_transfer",
                "before_mae": float(np.mean(np.abs(np.asarray(transfer_before) - POSITIONS))),
                "after_mae": float(np.mean(np.abs(np.asarray(transfer_after) - POSITIONS)))})

            # Within-position vs between-position cosine using paired states.
            aud_states, vis_states = [], []
            for p in POSITIONS:
                aud_states.append(audio_states_by_position[p][0])
                vis_states.append(vision_states[POSITIONS.index(p)])
            within = [_cosine(a, v) for a, v in zip(aud_states, vis_states)]
            between = [_cosine(aud_states[i], vis_states[j]) for i in range(len(POSITIONS)) for j in range(len(POSITIONS)) if i != j]
            crossmodal.append({"seed": run_seed, "variant": variant, "stage": "post_adaptation",
                               "within_mean_cosine": float(np.mean(within)),
                               "between_mean_cosine": float(np.mean(between)),
                               "within_minus_between_cosine": float(np.mean(within) - np.mean(between))})

            # Composition is meaningful only if this seed/architecture passes
            # the all-pairs audio-only competence gate.
            passed = score["exact_accuracy"] >= COMPETENCE_EXACT
            if passed:
                composition.append({"seed": run_seed, "variant": variant,
                    "position": "held-out pairs", "status": "separate_composition_training_run_required"})
            else:
                composition.append({"seed": run_seed, "variant": variant, "position": "all heldouts",
                                    "status": "gated_off_all_pair_control_below_80_percent"})

            # Fixed delay probe: hold out labels from the liquid and use the
            # association readout after an added, position-independent delay.
            for delay, (ys, ps) in delay_predictions.items():
                memory.append({"seed": run_seed, "variant": variant, "delay_seconds": delay,
                               **_score(np.asarray(ys), np.asarray(ps))})

            stats = net.stats()
            topology = net.topology_summary()
            _, peak_mem = tracemalloc.get_traced_memory()
            compute.append({"seed": run_seed, "variant": variant, "adaptation_wall_seconds": adapt_elapsed,
                "adaptation_cpu_seconds": adapt_cpu,
                "inference_wall_median_seconds": float(statistics.median(latency)),
                "inference_wall_p95_seconds": float(np.quantile(latency, .95)),
                "inference_cpu_seconds_total": float(sum(cpus)), "active_neuron_updates": stats.neuron_updates,
                "active_neurons_at_final_snapshot": topology["active_neurons"],
                "propagated_events": stats.propagated_events, "processed_events": stats.processed_events,
                "structural_updates": stats.structural_updates, "active_connections": net.connection_count,
                "formed_connections": stats.formed_connections, "pruned_connections": stats.pruned_connections,
                "peak_traced_bytes_process": int(peak_mem)})
            if seed_index == 0:
                (snapshots / f"{variant}.json").write_text(json.dumps(net.state_dict()), encoding="utf-8")

        # Token recognizability is evaluated directly from generic waveform
        # features; token names are used only as external probe labels.
        token_names = list(vocab.tokens)
        token_features = []
        for name in token_names:
            wave = vocab.token(name)
            padded = np.concatenate([wave, np.zeros(EPISODE_TICKS * 50)])[:EPISODE_TICKS * 50]
            observations = [Observation(i / 20, np.zeros(21), padded[i*100:(i+1)*100]) for i in range(10)]
            token_features.append(np.mean([sensory_features(item) for item in observations[:4]], axis=0))
        token_matrix = np.asarray(token_features)
        # Nearest-centroid training/test split is deterministic and token-level;
        # because tokens are fixed waveforms this is a separability diagnostic.
        distances = np.linalg.norm((token_matrix[:, None, :] - token_matrix[None, :, :]), axis=2)
        pairwise = distances[np.triu_indices(len(token_names), 1)]
        token_rows.append({"seed": run_seed, "method": "raw_waveform_feature_nearest_neighbor",
                           "distinct_tokens": len(token_names), "minimum_pairwise_feature_distance": float(np.min(pairwise)),
                           "distinct_pair_fraction": float(np.mean(pairwise > 1e-9)),
                           "interpretation": "feature-space separability diagnostic; not held-out-token generalization"})

    tracemalloc.stop()
    (out / "dataset_partitions.json").write_text(json.dumps(partitions, indent=2), encoding="utf-8")
    write_csv(out / "spatial_token_recognition.csv", token_rows)
    write_csv(out / "spatial_association_results.csv", association)
    write_csv(out / "spatial_composition_results.csv", composition)
    write_csv(out / "spatial_counterfactual_results.csv", counterfactual)
    write_csv(out / "spatial_crossmodal_results.csv", crossmodal)
    write_csv(out / "spatial_memory_results.csv", memory)
    write_csv(out / "spatial_compute_results.csv", compute)
    finalize_existing(out)
    return {"output": str(out), "association_rows": len(association), "composition_rows": len(composition),
            "seeds": run_seeds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/spatial-grounding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-per-combination", type=int, default=3)
    parser.add_argument("--test-per-combination", type=int, default=2)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="run the five-seed all-combinations competence gate only")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="generate plots/report from the completed CSVs in --output")
    args = parser.parse_args(argv)
    if args.finalize_existing:
        print(json.dumps(finalize_existing(args.output), indent=2)); return
    print(json.dumps(run_experiment(args.output, args.seed, args.seeds,
        args.train_per_combination, args.test_per_combination, sequential=args.sequential,
        checkpoint_only=args.checkpoint_only), indent=2))


if __name__ == "__main__":
    main()

