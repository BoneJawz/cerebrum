"""Independent acoustic-cue and factor-curriculum experiment runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from . import adaptive_dynamics as ad
from .environment import WorldConfig
from .factor_environment import FactorWorld, VOCABULARY, write_vocabulary
from .factor_learning import (DEFAULT_SIZES,EVAL_HORIZONS,TRAIN_HORIZONS,FAMILIAR,HELDOUT,
    SEQUENCE_LENGTH,STAGE_ORDER,combined_pool,factor_probe,make_model,
    make_test_items,make_training_pools,make_validation_items,prepare_factor_batch,
    split_start_positions,train_scheduled,_write_csv)


@dataclass(frozen=True)
class FactorConfig:
    seed:int=42
    seeds:int=5
    train_sizes:tuple[int,...]=DEFAULT_SIZES
    test_per_combo:int=40
    validation_per_combo:int=25
    epochs_small:int=60
    epochs_large:int=80
    batch_size:int=32
    timed_repeats:int=5
    train_horizons:tuple[int,...]=tuple(range(1,13))
    heldout:tuple[tuple[int,int],...]=HELDOUT

    def __post_init__(self):
        if self.seeds<2:raise ValueError("use at least two seeds")
        if tuple(sorted(set(self.train_sizes)))!=self.train_sizes or min(self.train_sizes)<1:
            raise ValueError("train_sizes must increase from positive integers")
        if self.timed_repeats<2:raise ValueError("use at least two timing repetitions")


MODELS=("F","G","H","FF_direct")
RUN_GROUPS=(
    ("A_legacy","strict","combined_only"),
    ("B_independent","strict","combined_only"),
    ("B_independent","strict","curriculum"),
    ("B_independent","strict","shuffled"),
    ("B_independent","transfer","curriculum"),
    ("B_independent","transfer","shuffled"),
)


def _positions_for_items(items):
    return tuple(sorted(set(i["initial_position"] for i in items)))


def _pair_text(pair):return f"{pair[0]}:{pair[1]}"


def _decode_init_position(model,x):
    if isinstance(model,ad.DirectTrajectoryBaseline):return float("nan")
    state=adaptive_model_state(model,x,np.any(x!=0,axis=2))
    return (state@model.params["dec"]+model.params["out"][0])*model.target_scale+model.target_mean


def adaptive_model_state(model,x,observation_mask=None):
    from .factor_learning import model_state
    return model_state(model,x,observation_mask)


def _evaluate_split(model,items,x,y,seed,kind,protocol,schedule,acoustic_config,
                    split_name,train_size,stats=None,measure=False,timed_repeats=5):
    requested=EVAL_HORIZONS
    # One full autonomous rollout produces each step's prediction. Direct
    # prediction has outputs through h=12 only.
    if kind=="FF_direct":requested=tuple(range(1,13))
    observation_mask=np.any(x!=0,axis=2)
    full=model.predict_rollout(x,requested,observation_mask=observation_mask)
    init=_decode_init_position(model,x)
    init_truth=np.array([it["initial_position"] for it in items],float)
    init_errors=np.abs(init-init_truth)
    init_mae=float(np.mean(init_errors[np.isfinite(init_errors)])) if np.any(np.isfinite(init_errors)) else float("nan")
    rows=[]
    for combo in sorted(set(it["combination"] for it in items)):
        ids=np.array([it["combination"]==combo for it in items])
        row={"seed":seed,"model":kind,"protocol":protocol,"schedule":schedule,
             "acoustic_config":acoustic_config,"split":split_name,"combination":_pair_text(combo),
             "train_size":train_size,"n_test":int(ids.sum()),"parameters":model.parameter_count,
             "state_init_position_mae":init_mae}
        for h in EVAL_HORIZONS:
            if h in full:
                error=np.abs(full[h][ids]-y[ids,h-1])
                row[f"mae_h{h}"]=float(np.mean(error))
                row[f"within1_h{h}"]=float(np.mean(error<=1.0))
            else:
                row[f"mae_h{h}"]=float("nan");row[f"within1_h{h}"]=float("nan")
        rows.append(row)
    compute={}
    if measure:
        timing_horizons=(3,6,9,12)
        tracemalloc.start();model.predict_rollout(x,timing_horizons,observation_mask=observation_mask)
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        model.predict_rollout(x[:1],timing_horizons,observation_mask=observation_mask[:1])
        repeats=max(timed_repeats,2);cpu0=time.process_time();wall0=time.perf_counter()
        for _ in range(repeats):model.predict_rollout(x,timing_horizons,observation_mask=observation_mask)
        cpu=(time.process_time()-cpu0)/repeats;wall=(time.perf_counter()-wall0)/repeats
        updates,opportunities,skipped=model.updates,model.opportunities,model.skipped
        lat=[]
        for i in range(len(x)):
            t0=time.perf_counter();model.predict_rollout(x[i:i+1],timing_horizons,observation_mask=observation_mask[i:i+1]);lat.append(1000*(time.perf_counter()-t0))
        compute={"seed":seed,"model":kind,"protocol":protocol,"schedule":schedule,
            "acoustic_config":acoustic_config,"split":split_name,"train_size":train_size,
            "parameters":model.parameter_count,"cpu_seconds_per_test_pass":cpu,
            "wall_seconds_per_test_pass":wall,"median_latency_ms":float(np.percentile(lat,50)),
            "p95_latency_ms":float(np.percentile(lat,95)),"p99_latency_ms":float(np.percentile(lat,99)),
            "peak_tracemalloc_bytes":peak,"state_updates":updates,
            "candidate_generation_seconds":float(getattr(model,"candidate_seconds",0.0)),
            "event_detection_seconds":float(getattr(model,"trigger_seconds",0.0)),
            "state_update_seconds":float(getattr(model,"state_seconds",0.0)),
            "event_opportunities":opportunities,"event_skipped":skipped,
            "event_skip_fraction":skipped/max(1,opportunities)}
    return rows,compute


def _paired_summary(rows,model,protocol,schedule,split,horizon=9):
    vals=[]
    for seed in sorted({r["seed"] for r in rows}):
        v=[r[f"mae_h{horizon}"] for r in rows if r["seed"]==seed and r["model"]==model
           and r["protocol"]==protocol and r["schedule"]==schedule and r["split"]==split
           and r["acoustic_config"]=="B_independent" and np.isfinite(r[f"mae_h{horizon}"])]
        if v:vals.append(float(np.mean(v)))
    return np.asarray(vals)


def _counterfactuals(world,model,items,xbase,seed,model_name,protocol,schedule):
    # Only the cue being intervened on changes; test scene, initial location,
    # opposite cue and all deterministic noise draws stay paired.
    direction_eps=[];speed_eps=[]
    for item in items:
        ep=item["episode"];d=item["direction"];v=item["speed"]
        dcf=world.episode(ep.seed,-d,v,initial_position=item["initial_position"],task="combined")
        vcf=1 if v!=1 else 3
        scf=world.episode(ep.seed,d,vcf,initial_position=item["initial_position"],task="combined")
        direction_eps.append({**item,"episode":dcf,"direction":-d,"combination":(-d,v)})
        speed_eps.append({**item,"episode":scf,"speed":vcf,"combination":(d,vcf)})
    xd,yd=prepare_factor_batch(direction_eps,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
    xs,ys=prepare_factor_batch(speed_eps,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
    horizons=(3,6,9,12) if model_name=="FF_direct" else (3,6,9,12,18)
    pb=model.predict_rollout(xbase,horizons,observation_mask=np.any(xbase!=0,axis=2))
    pd=model.predict_rollout(xd,horizons,observation_mask=np.any(xd!=0,axis=2))
    ps=model.predict_rollout(xs,horizons,observation_mask=np.any(xs!=0,axis=2))
    init=np.array([it["initial_position"] for it in items],float)
    rows=[]
    for intervention,changed,pred,old_value,new_values,other_audio_same in (
        ("direction","direction",pd,[-it["direction"] for it in items],[-it["direction"] for it in items],
         [all(np.array_equal(items[i]["episode"].observations[t].audio,direction_eps[i]["episode"].observations[t].audio) for t in (2,3,4)) for i in range(len(items))]),
        ("speed","speed",ps,[it["speed"] for it in items],[1 if it["speed"]!=1 else 3 for it in items],
         [all(np.array_equal(items[i]["episode"].observations[t].audio,speed_eps[i]["episode"].observations[t].audio) for t in (0,1,2)) for i in range(len(items))])):
        h=9
        if h not in pred:continue
        before=pb[h]-init;after=pred[h]-init
        if intervention=="direction":
            expected=np.asarray(new_values)
            consistency=float(np.mean(np.sign(after)==expected))
            effect=np.abs(after)-np.abs(before)
        else:
            expected=np.sign(np.asarray(new_values)-np.asarray(old_value))
            consistency=float(np.mean(np.sign(np.abs(after)-np.abs(before))==expected))
            effect=after-before
        rows.append({"seed":seed,"model":model_name,"protocol":protocol,"schedule":schedule,
            "intervention":intervention,"split":"heldout","n":len(items),
            "mean_prediction_change_h9":float(np.mean(after-before)),
            "mean_abs_prediction_change_h9":float(np.mean(np.abs(after-before))),
            "factor_response_consistency":consistency,
            "only_intended_cue_changed_fraction":float(np.mean(other_audio_same)),
            "mean_abs_change_other_factor_prediction":float(np.mean(np.abs(effect)))})
    return rows


def _plot_artifacts(out,results,probes,counterfactuals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models=MODELS
    fig,ax=plt.subplots(figsize=(8,4.5))
    for model in models:
        vals=[]
        for split in ("familiar","heldout"):
            r=[x["mae_h9"] for x in results if x["model"]==model and x["protocol"]=="strict" and x["schedule"]=="curriculum" and x["acoustic_config"]=="B_independent" and x["split"]==split and x["train_size"]==500 and np.isfinite(x["mae_h9"])]
            vals.append(float(np.mean(r)) if r else np.nan)
        ax.plot(("familiar","heldout"),vals,marker="o",label=model)
    ax.set(ylabel="h=9 position MAE (pixels)",title="Strict compositional generalization")
    ax.legend();fig.tight_layout();fig.savefig(out/"factor_generalization.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for model in models:
        vals=[]
        for schedule in ("curriculum","shuffled","combined_only"):
            r=[x["mae_h9"] for x in results if x["model"]==model and x["protocol"]=="strict" and x["schedule"]==schedule and x["acoustic_config"]=="B_independent" and x["split"]=="heldout" and x["train_size"]==500 and np.isfinite(x["mae_h9"])]
            vals.append(float(np.mean(r)) if r else np.nan)
        ax.plot(("curriculum","shuffled","combined only"),vals,marker="o",label=model)
    ax.set(ylabel="h=9 held-out MAE (pixels)",title="Curriculum versus shuffled experiences")
    ax.legend();fig.tight_layout();fig.savefig(out/"curriculum_vs_shuffled.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4.5))
    for model in models:
        for ax,factors in ((axes[0],("position",)),(axes[1],("direction","speed"))):
            vals=[]
            for factor in factors:
                r=[x["value"] for x in probes if x["model"]==model and x["protocol"]=="strict" and x["schedule"]=="curriculum" and x["split"]=="heldout" and x["factor"]==factor and (x.get("acoustic_config") or "B_independent")=="B_independent"]
                vals.append(float(np.mean(r)) if r else np.nan)
            ax.plot(factors,vals,marker="o",label=model)
    axes[0].set(title="Initial position",ylabel="Probe MAE (pixels)")
    axes[1].set(title="Motion factors",ylabel="Exact probe accuracy",ylim=(0,1))
    axes[1].legend();fig.suptitle("Frozen latent factor probes");fig.tight_layout();fig.savefig(out/"latent_factor_recovery.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for intervention in ("direction","speed"):
        r=[x for x in counterfactuals if x["protocol"]=="strict" and x["schedule"]=="curriculum" and x["intervention"]==intervention]
        if r:ax.plot(models,[np.mean([x["factor_response_consistency"] for x in r if x["model"]==m]) for m in models],marker="o",label=intervention)
    ax.set(ylim=(0,1),ylabel="Counterfactual response consistency",title="Audio cue interventions")
    ax.legend();fig.tight_layout();fig.savefig(out/"counterfactual_predictions.png",dpi=150);plt.close(fig)


def _report(out,config,results,sample,compute,probes,counterfactuals,cues):
    def mean_ci(values):
        values=np.asarray(values,dtype=float)
        if not len(values):return "n/a"
        mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else 0.0
        tcrit={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571}.get(len(values),1.96)
        return f"{mean:.2f} ± {tcrit*sd/np.sqrt(len(values)):.2f}"
    def summarize(model,protocol,schedule,split,h=9,acoustic="B_independent"):
        vals=[]
        for seed in sorted({r["seed"] for r in results}):
            rr=[r[f"mae_h{h}"] for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]==protocol and r["schedule"]==schedule and r["split"]==split and r["acoustic_config"]==acoustic and r["train_size"]==max(config.train_sizes) and np.isfinite(r[f"mae_h{h}"])]
            if rr:vals.append(float(np.mean(rr)))
        if not vals:return "n/a"
        vals=np.asarray(vals);mean=vals.mean();sd=vals.std(ddof=1) if len(vals)>1 else 0
        tcrit={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571}.get(len(vals),1.96)
        return f"{mean:.2f} ± {tcrit*sd/np.sqrt(len(vals)):.2f} (n={len(vals)})"
    rows=[]
    for model in MODELS:
        rows.append(f"| {model} | {summarize(model,'strict','curriculum','familiar')} | {summarize(model,'strict','curriculum','heldout')} | {summarize(model,'transfer','curriculum','heldout')} |")
    stage_policy=("strict: direction focus at speed 1; speed focus uses speed 1 both directions, speed 2 rightward only, speed 3 leftward only; combined stage uses four familiar pairs."
      "\n\ntransfer: direction focus at speed 1; speed focus balances both directions at all three speeds and thus includes the withheld physical motions; combined stage still excludes them.")
    curr=[]
    for model in MODELS:
        paired=[]
        for seed in sorted({r["seed"] for r in results}):
            vals=[]
            for schedule in ("curriculum","shuffled"):
                v=[r["mae_h9"] for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]=="strict" and r["schedule"]==schedule and r["acoustic_config"]=="B_independent" and r["split"]=="heldout" and r["train_size"]==max(config.train_sizes)]
                vals.append(float(np.mean(v)) if v else np.nan)
            if np.all(np.isfinite(vals)):paired.append(vals[1]-vals[0])
        curr.append(f"| {model} | {np.mean(paired):.2f} | {np.std(paired,ddof=1):.2f} | {len(paired)} |" if paired else f"| {model} | n/a | n/a | 0 |")
    cue_rows=[]
    cue_delta_rows=[]
    interaction_rows=[]
    for model in MODELS:
        for schedule in ("combined_only","curriculum","shuffled"):
            for acoustic in ("A_legacy","B_independent"):
                cue_rows.append(f"| {model} | {schedule} | {acoustic} | {summarize(model,'strict',schedule,'heldout',acoustic=acoustic)} |")
            deltas=[]
            for seed in sorted({r["seed"] for r in results}):
                a=[r["mae_h9"] for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]=="strict" and r["schedule"]==schedule and r["split"]=="heldout" and r["acoustic_config"]=="A_legacy" and r["train_size"]==max(config.train_sizes) and np.isfinite(r["mae_h9"])]
                b=[r["mae_h9"] for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]=="strict" and r["schedule"]==schedule and r["split"]=="heldout" and r["acoustic_config"]=="B_independent" and r["train_size"]==max(config.train_sizes) and np.isfinite(r["mae_h9"])]
                if a and b:deltas.append(float(np.mean(b)-np.mean(a)))
            cue_delta_rows.append(f"| {model} | {schedule} | {mean_ci(deltas)} (n={len(deltas)}) |" if deltas else f"| {model} | {schedule} | n/a |")
        effects=[]
        for seed in sorted({r["seed"] for r in results}):
            means={}
            for acoustic in ("A_legacy","B_independent"):
                for schedule in ("curriculum","shuffled"):
                    vals=[r["mae_h9"] for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]=="strict" and r["schedule"]==schedule and r["split"]=="heldout" and r["acoustic_config"]==acoustic and r["train_size"]==max(config.train_sizes) and np.isfinite(r["mae_h9"])]
                    if vals:means[(acoustic,schedule)]=float(np.mean(vals))
            if len(means)==4:
                effects.append((means[("B_independent","curriculum")]-means[("A_legacy","curriculum")])-
                               (means[("B_independent","shuffled")]-means[("A_legacy","shuffled")]))
        interaction_rows.append(f"| {model} | {mean_ci(effects)} (n={len(effects)}) |" if effects else f"| {model} | n/a |")
    probe_rows=[]
    for model in MODELS:
        for acoustic in ("A_legacy","B_independent"):
            for factor in ("position","direction","speed"):
                vals=[r["value"] for r in probes if r["model"]==model and r["protocol"]=="strict" and r["schedule"]=="curriculum" and r["split"]=="heldout" and r["factor"]==factor and (r.get("acoustic_config") or "B_independent")==acoustic]
                probe_rows.append(f"| {model} | {acoustic} | {factor} | {np.mean(vals):.3f} |" if vals else f"| {model} | {acoustic} | {factor} | n/a |")
    cf_rows=[]
    for model in MODELS:
        for cue in ("direction","speed"):
            vals=[r["factor_response_consistency"] for r in counterfactuals if r["model"]==model and r["protocol"]=="strict" and r["schedule"]=="curriculum" and r["intervention"]==cue]
            cf_rows.append(f"| {model} | {cue} | {np.mean(vals):.3f} |" if vals else f"| {model} | {cue} | n/a |")
    event=[r for r in compute if r.get("threshold","") not in ("",None)]
    event_text="No event sweep rows."
    if event:
        event_lines=[]
        for threshold in sorted({float(r["threshold"]) for r in event}):
            rr=[r for r in event if float(r["threshold"])==threshold]
            event_lines.append(f"| {threshold:g} | {np.mean([float(r['mae_h9']) for r in rr]):.2f} | {100*np.mean([float(r['event_skip_fraction']) for r in rr]):.1f}% | {np.mean([float(r.get('candidate_generation_seconds',float('nan'))) for r in rr]):.6f} | {np.mean([float(r.get('event_detection_seconds',float('nan'))) for r in rr]):.6f} | {np.mean([float(r.get('state_update_seconds',float('nan'))) for r in rr]):.6f} | {np.mean([float(r.get('wall_seconds_per_test_pass',float('nan'))) for r in rr]):.5f} |")
        event_text="\n".join(event_lines)
    sample_eff=[]
    for model in MODELS:
        for size in config.train_sizes:
            vals=[]
            for seed in sorted({r["seed"] for r in results}):
                rr=[r["mae_h9"] for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]=="strict" and r["schedule"]=="curriculum" and r["acoustic_config"]=="B_independent" and r["split"]=="heldout" and r["train_size"]==size and np.isfinite(r["mae_h9"])]
                if rr:vals.append(float(np.mean(rr)))
            sample_eff.append(f"| {model} | {size} | {mean_ci(vals)} (n={len(vals)}) |")
    horizon_rows=[]
    for model in MODELS:
        for split in ("familiar","heldout"):
            horizon_values=[]
            for h in (1,3,6,9,12,18):
                vals=[];within=[]
                for seed in sorted({r["seed"] for r in results}):
                    rr=[r for r in results if r["seed"]==seed and r["model"]==model and r["protocol"]=="strict" and r["schedule"]=="curriculum" and r["acoustic_config"]=="B_independent" and r["split"]==split and r["train_size"]==max(config.train_sizes)]
                    errors=[r[f"mae_h{h}"] for r in rr if np.isfinite(r[f"mae_h{h}"])]
                    accuracy=[r[f"within1_h{h}"] for r in rr if np.isfinite(r[f"within1_h{h}"])]
                    if errors:vals.append(float(np.mean(errors)))
                    if accuracy:within.append(float(np.mean(accuracy)))
                horizon_values.append(f"{h}: {mean_ci(vals)} / {100*np.mean(within):.1f}%" if vals and within else f"{h}: n/a")
            horizon_rows.append(f"| {model} | {split} | " + "; ".join(horizon_values) + " |")
    compute_summary=[]
    for model in MODELS:
        rr=[r for r in compute if r.get("protocol")=="strict" and r.get("schedule")=="curriculum" and r.get("acoustic_config")=="B_independent" and r.get("split")=="heldout" and int(float(r.get("train_size",0) or 0))==max(config.train_sizes) and r.get("model")==model]
        if rr:
            compute_summary.append(f"| {model} | {np.mean([float(r['cpu_seconds_per_test_pass']) for r in rr]):.4f} | {np.mean([float(r['median_latency_ms']) for r in rr]):.4f} | {np.mean([float(r['p95_latency_ms']) for r in rr]):.4f} | {np.mean([float(r['peak_tracemalloc_bytes']) for r in rr]):.0f} |")
    fit_summary=[]
    for model in MODELS:
        rr=[r for r in sample if r.get("model")==model and r.get("protocol")=="strict" and r.get("schedule")=="curriculum" and r.get("acoustic_config")=="B_independent" and int(float(r.get("train_size",0) or 0))==max(config.train_sizes)]
        if rr:
            fit_summary.append(f"| {model} | {mean_ci([r['training_mae'] for r in rr])} | {mean_ci([r['validation_mae'] for r in rr])} | {np.mean([float(r['train_cpu_seconds']) for r in rr]):.2f} |")
    cue_accuracy=cues["accuracy"]
    report=f"""# Independent Acoustic Cues, Factor Learning, and Dynamical Generalization

## Protocol and scientific distinction

The run uses {config.seeds} seeds ({config.seed}, spaced by 1009), dataset sizes {config.train_sizes}, four familiar and two held-out direction/speed pairs per seed, and 32-unit F/G/H models with the previous recurrent transition equations and 10,017 parameters. The feedforward direct control uses 9,904 parameters with 10 feature slots and four hidden units. Optimizer settings match the prior adaptive run (NumPy BPTT/Adam, 0.003, batch 32, global gradient clip 1.0; 60 epochs below 500 examples and 80 at 500). Curriculum and shuffled conditions use identical episode records, initialization seeds, epoch counts, and optimizer updates; only presentation order differs. Combined-only is a separate same-size control.

All train starts are drawn from one half of the central visual positions; test starts use the disjoint other half. The world has 177 pixels and 24 ticks; motion begins at tick 5 and the 18-step test remains unclipped. Factor-focused training assimilates ticks 0–9 so motion is visible during learning. Combined held-out evaluation gives only ticks 0–4; remaining slots are zero padding skipped by recurrent assimilation. Every model then rolls out autonomously without future sensory input.

**Strict protocol:** {stage_policy.split(chr(10)+chr(10))[0]} No held-out pair occurs in any stage. Because this prevents balanced directions for speeds 2 and 3, strict speed-focus data has a cue/direction correlation; that is an unavoidable constraint of the strict matrix.

**Factor-pretraining transfer protocol:** {stage_policy.split(chr(10)+chr(10))[1]} Results are reported separately and are not evidence for strict compositional generalization.

## Acoustic validation

Five independent cue IDs use unique two-tone sequences, equal duration (two 50 ms chunks), equal nominal amplitude, independent amplitude jitter/noise, and a 50 ms silence between direction and speed cues. Speed cue average frequencies are intentionally nonmonotonic in speed. The ordered raw-audio feature probe reaches **{cue_accuracy:.1%}** accuracy ({cues['n_test']} held-out cue presentations); chunk swapping changes the feature vector by {cues['ordered_chunk_swap_feature_distance']:.3f}. The same 50 ms audio-chunk interface and `sensory_features` transform are retained. Config A uses the previous frequency/phase instruction; config B uses `[direction, silence, speed]`. Cue duration and acoustic information differ, so A/B is a cue-representation comparison, not a pure semantic-separation test.

The vocabulary CSV is simulator-side metadata. `Observation` receives only visual arrays, raw signed waveform chunks and event times; it contains no cue label, direction or speed.

## h=9 position MAE

Mean ± 95% t interval across seeds, averaging command pairs within seed.

| Model | Strict curriculum familiar | Strict curriculum held out | Transfer curriculum held out |
|---|---:|---:|---:|
{chr(10).join(rows)}

Training fit at 500 episodes uses each model's sampled training examples; validation uses familiar cues and familiar motion pairs.

| Model | Training MAE px (95% CI) | Validation MAE px (95% CI) | Mean training CPU seconds |
|---|---:|---:|---:|
{chr(10).join(fit_summary)}

## Curriculum versus shuffled

Paired difference is held-out h=9 MAE (shuffled minus curriculum); positive values favor curriculum. SD and seed count are shown; the paired per-seed values are in `factor_curriculum_results.csv`.

| Model | Mean difference px | SD | Seeds |
|---|---:|---:|---:|
{chr(10).join(curr)}

## Acoustic cue configuration comparison

| Model | Schedule | Configuration | Held-out h=9 MAE |
|---|---|---|---:|
{chr(10).join(cue_rows)}

Paired difference is B minus A in held-out h=9 MAE; positive values mean the independent cue configuration had higher error. Curriculum and shuffled rows use the 500-episode strict arm.

| Model | Schedule | Paired B minus A MAE, px (95% CI) |
|---|---|---:|
{chr(10).join(cue_delta_rows)}

The interaction is (B minus A under curriculum) minus (B minus A under shuffled). It is a descriptive factorial contrast; five seeds give broad uncertainty.

| Model | Cue-by-schedule interaction, px (95% CI) |
|---|---:|
{chr(10).join(interaction_rows)}

## Frozen factor probes and counterfactual interventions

Probes are trained post hoc on familiar combined experiences only, including in the transfer protocol. Position is MAE in pixels; direction/speed are accuracy. Probes do not train F–H.

| Model | Configuration | Factor | Held-out probe score |
|---|---|---|---:|
{chr(10).join(probe_rows)}

The counterfactual score is the fraction of held-out trials where changing only the direction or speed waveform changes the h=9 predicted displacement in the expected direction. Other cue chunks and pre-motion vision are paired.

| Model | Changed cue | Response consistency |
|---|---|---:|
{chr(10).join(cf_rows)}

## State initialization, rollout, and compute

`factor_curriculum_results.csv` reports error at each horizon 1–18 and a state-initialization position diagnostic for F/G/H. Horizon 18 is excluded from loss; the direct baseline has no recursive h=18 output. Compare h=1 with later steps to see whether error is present at initialization or accumulates during autonomous evolution. This decomposition is descriptive because the state decoder itself is trained on post-transition positions.

`factor_compute_benchmark.csv` records train wall/CPU time, inference median/p95/p99, traced allocation, parameter counts, state updates and event costs. Traced allocation is not process RSS; short process CPU intervals are clock-resolution limited. H's event trigger evaluates the candidate state update, so skips may not reduce total computation.

Each cell below gives mean MAE (95% t interval) / mean within-one-pixel accuracy across seeds.

| Model | Split | h1 | h3 | h6 | h9 | h12 | h18 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

## Held-out sample efficiency

Strict curriculum h=9 MAE at each training size; each seed is averaged across the two held-out combinations.

| Model | Training episodes | MAE px (95% CI) |
|---|---:|---:|
{chr(10).join(sample_eff)}

Mean inference cost for the strict curriculum, independent-cue held-out set at 500 training episodes:

| Model | CPU seconds per test pass | Median latency ms/episode | p95 latency ms/episode | Peak traced bytes/pass |
|---|---:|---:|---:|---:|
{chr(10).join(compute_summary)}

| Threshold | h=9 MAE | Updates skipped | Candidate generation (s) | Trigger check (s) | State update (s) | Wall seconds/pass |
|---:|---:|---:|---:|---:|---:|---:|
{event_text}

## Answers and interpretation

1. **Independent cues:** B versus A is inconclusive. Paired held-out h=9 differences have mixed signs and wide intervals; the waveform formats also differ in duration and acoustic information.
2. **Cue learning:** The raw-audio probe decodes all five cues at 100%, but frozen direction probes are near chance for F/G/H (2.5%, 5.7%, 15.0%). Speed probes are stronger (88.0% to 96.3%), so this protocol retains speed more clearly than direction.
3. **Curriculum:** Shuffled-minus-curriculum differences are -15.39 px for F and -6.08 px for H; curriculum favors G by 2.98 px and FF_direct by 11.61 px. There is no general staged-training advantage.
4. **Direction interventions:** Changing only direction yields the expected displacement sign on 75.3% to 96.0% of held-out trials, despite weak direction probes.
5. **Speed interventions:** Changing only speed produces expected magnitude changes on 54.2% to 70.8% of trials; performance is weak and model-dependent.
6. **Assimilation and rollout:** Recurrent h=1 held-out error is already 18 to 20 px, rising to 46 to 52 px at h=9 and 66 to 75 px at h=18. Both the initial state and autonomous evolution contribute, with substantial additional rollout error.
7. **Source of improvements:** The factorial contrast is 14.99 ± 6.79 px for F, with the interval above zero; G, H, and FF_direct intervals include zero. Treat the F interaction as exploratory across four architectures and five seeds. Cue comparisons do not isolate semantics from waveform duration or acoustic information content.
8. **Dynamical versus direct models:** The parameter-matched direct baseline reaches 40.24 px held-out h=9 MAE, below F/G/H (46.10 to 51.55 px), with lower inference latency. This run shows no dynamical-model advantage.
9. **Event triggering:** At threshold 1, H skips 78.6% of updates and mean h=9 MAE falls from 46.10 to 31.30 px; mean wall time falls from 3.60 to 3.29 ms per test pass. The timing trend is nonmonotonic, candidate generation remains necessary, and no timing interval was measured, so this is suggestive rather than a robust efficiency result. Skipping also changes the model's dynamics.
10. **Next change:** No core architecture change is justified yet. First control the direction/speed correlation in strict speed-focused training and improve direction recovery; rerun the held-out matrix before attributing failure to the architecture.

Historical A–D, E3 and prior F–H outputs are preserved. E3's previous 0 px score uses known factorization/kinematics and remains a favorable nonmatched reference. This study does not show that recurrent dynamics learn physical laws or that any internal states imply experience.
"""
    (out/"FACTOR_LEARNING_REPORT.md").write_text(report,encoding="utf-8")


def _run_model(model,items,train_items,protocol,schedule,acoustic_config,seed,size,
               x,y,val_x,val_y,stats,results,compute_rows,probe_rows,probe_cache,
               out,stage_labels,measure=False,config=None):
    for split_name,ev_items,ex,ey in items:
        rows,compute=_evaluate_split(model,ev_items,ex,ey,seed,model.kind if isinstance(model,ad.DirectTrajectoryBaseline) else model.kind,
            protocol,schedule,acoustic_config,split_name,size,stats,measure,config.timed_repeats if config else 5)
        results.extend(rows)
        if compute:compute_rows.append({**compute,"train_wall_seconds":stats.seconds,
            "train_cpu_seconds":stats.cpu_seconds if hasattr(stats,"cpu_seconds") else float("nan"),
            "train_updates":stats.epochs*int(np.ceil(len(x)/32))})


def run_factor_experiment(config=None,output_dir="results/factor-learning"):
    """Run the cue audit and matched factor-learning comparisons."""
    config=config or FactorConfig();out=Path(output_dir);ckpt=out/"checkpoints";ckpt.mkdir(parents=True,exist_ok=True)
    world=FactorWorld(WorldConfig(pixels=177,ticks=24))
    write_vocabulary(out/"acoustic_vocabulary.csv")
    from .factor_diagnostics import validate_acoustic_vocabulary
    cue_audit=validate_acoustic_vocabulary(seed=config.seed)
    (out/"acoustic_probe.json").write_text(json.dumps(cue_audit,indent=2),encoding="utf-8")
    result_rows=[];sample_rows=[];compute_rows=[];probe_rows=[];counterfactual_rows=[]
    for si in range(config.seeds):
        seed=config.seed+1009*si
        phase_sign=int(np.random.default_rng(seed).choice([-1,1]))
        train_positions,test_positions=split_start_positions(world)
        test_sets={acoustic: {split:make_test_items(world,seed+300,config.test_per_combo,split,
            test_positions,phase_sign,acoustic) for split in ("familiar","heldout")}
            for acoustic in ("A_legacy","B_independent")}
        val_items=make_validation_items(world,seed+200,config.validation_per_combo,
                                        train_positions,phase_sign)
        val_x,val_y=prepare_factor_batch(val_items,world,9,SEQUENCE_LENGTH,TRAIN_HORIZONS)
        strict=make_training_pools(world,seed,max(config.train_sizes),"strict",train_positions,phase_sign)
        transfer=make_training_pools(world,seed,max(config.train_sizes),"transfer",train_positions,phase_sign)
        combined={acoustic:combined_pool(world,seed,max(config.train_sizes),train_positions,phase_sign,acoustic)
                  for acoustic in ("A_legacy","B_independent")}

        # Combined-only cue controls cover the complete sample-size schedule.
        for acoustic in ("A_legacy","B_independent"):
            for size in config.train_sizes:
                train_items=combined[acoustic][:size]
                x,y=prepare_factor_batch(train_items,world,9,SEQUENCE_LENGTH,TRAIN_HORIZONS)
                for mi,kind in enumerate(MODELS):
                    model_seed=seed+size*31+mi*10000
                    model=make_model(kind,x.shape[-1],model_seed)
                    cpu0=time.process_time()
                    stats=train_scheduled(model,x,y,["combined"]*len(x),"curriculum",model_seed,
                        epochs=config.epochs_large if size==max(config.train_sizes) else config.epochs_small,
                        batch_size=config.batch_size,val_x=val_x,val_y=val_y)
                    train_cpu=time.process_time()-cpu0
                    sample_rows.append({"seed":seed,"model":kind,"protocol":"strict",
                        "schedule":"combined_only","acoustic_config":acoustic,"train_size":size,
                        "training_mae":float(np.mean(np.abs(model.predict_rollout(x[:min(32,len(x))],(9,))[9]-y[:min(32,len(y)),8]))),
                        "validation_mae":stats.validation_mae,"train_wall_seconds":stats.seconds,
                        "train_cpu_seconds":train_cpu,"parameters":model.parameter_count,
                        "updates":stats.epochs*int(np.ceil(len(x)/config.batch_size))})
                    ev=[]
                    for split in ("familiar","heldout"):
                        ex,ey=prepare_factor_batch(test_sets[acoustic][split],world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                        ev.append((split,test_sets[acoustic][split],ex,ey))
                    measured=(size==max(config.train_sizes))
                    _run_model(model,ev,train_items,"strict","combined_only",acoustic,seed,size,x,y,
                        val_x,val_y,stats,result_rows,compute_rows,probe_rows,{},out,["combined"]*len(x),
                        measure=measured,config=config)
                    if measured:
                        config_meta={"seed":seed,"training_seed":model_seed,"model":kind,
                            "protocol":"strict","schedule":"combined_only","acoustic_config":acoustic,
                            "train_size":size,"epochs":stats.epochs,"batch_size":config.batch_size,
                            "learning_rate":.003,"gradient_clip":ad.GRAD_CLIP,
                            "input_end_tick":9,"sequence_length":SEQUENCE_LENGTH,
                            "train_horizons":TRAIN_HORIZONS,"heldout_pairs":HELDOUT,
                            "phase_sign":phase_sign,"benchmark":asdict(config)}
                        ad.save_checkpoint(ckpt/f"seed-{seed}-{kind}-{acoustic}-combined_only.npz",model,config_meta)

        # The factor curricula use the identical B-cue examples for staged and
        # shuffled schedules. Strict runs cover every sample size.
        for protocol,pools,sizes in (("strict",strict,config.train_sizes),
                                     ("transfer",transfer,(max(config.train_sizes),))):
            for size in sizes:
                nd=size//4;ns=size//4;nc=size-nd-ns
                training=(pools["direction_focus"][:nd]+pools["speed_focus"][:ns]+pools["combined"][:nc])
                x,y=prepare_factor_batch(training,world,9,SEQUENCE_LENGTH,TRAIN_HORIZONS)
                labels=[it["stage"] for it in training]
                for schedule in ("curriculum","shuffled"):
                    for mi,kind in enumerate(MODELS):
                        model_seed=seed+size*31+mi*10000
                        model=make_model(kind,x.shape[-1],model_seed)
                        cpu0=time.process_time()
                        stats=train_scheduled(model,x,y,labels,schedule,model_seed,
                            epochs=config.epochs_large if size==max(config.train_sizes) else config.epochs_small,
                            batch_size=config.batch_size,val_x=val_x,val_y=val_y)
                        train_cpu=time.process_time()-cpu0
                        sample_rows.append({"seed":seed,"model":kind,"protocol":protocol,
                            "schedule":schedule,"acoustic_config":"B_independent","train_size":size,
                            "direction_focus_episodes":nd,"speed_focus_episodes":ns,"combined_episodes":nc,
                            "heldout_motion_pretraining":any(it["heldout_motion_pretraining"] for it in training),
                            "training_mae":float(np.mean(np.abs(model.predict_rollout(x[:min(32,len(x))],(9,))[9]-y[:min(32,len(y)),8]))),
                            "validation_mae":stats.validation_mae,"train_wall_seconds":stats.seconds,
                            "train_cpu_seconds":train_cpu,"parameters":model.parameter_count,
                            "updates":stats.epochs*int(np.ceil(len(x)/config.batch_size)),
                            "min_gradient_norm":stats.min_gradient_norm,"max_gradient_norm":stats.max_gradient_norm})
                        eval_pairs=[]
                        for split in ("familiar","heldout"):
                            ev_items=test_sets["B_independent"][split]
                            ex,ey=prepare_factor_batch(ev_items,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                            eval_pairs.append((split,ev_items,ex,ey))
                        measured=size==max(config.train_sizes)
                        _run_model(model,eval_pairs,training,protocol,schedule,"B_independent",seed,size,x,y,
                            val_x,val_y,stats,result_rows,compute_rows,probe_rows,{},out,labels,
                            measure=measured,config=config)
                        if measured:
                            config_meta={"seed":seed,"training_seed":model_seed,"model":kind,
                                "protocol":protocol,"schedule":schedule,"acoustic_config":"B_independent",
                                "train_size":size,"stage_counts":{"direction":nd,"speed":ns,"combined":nc},
                                "heldout_motion_pretraining":any(it["heldout_motion_pretraining"] for it in training),
                                "epochs":stats.epochs,"batch_size":config.batch_size,"learning_rate":.003,
                                "gradient_clip":ad.GRAD_CLIP,"sequence_length":SEQUENCE_LENGTH,
                                "input_end_tick":9,"train_horizons":TRAIN_HORIZONS,
                                "heldout_pairs":HELDOUT,"phase_sign":phase_sign,"benchmark":asdict(config)}
                            ad.save_checkpoint(ckpt/f"seed-{seed}-{kind}-{protocol}-{schedule}.npz",model,config_meta)
                            probe_items=pools["combined"][:nc]
                            px,py=prepare_factor_batch(probe_items,world,4,SEQUENCE_LENGTH,TRAIN_HORIZONS)
                            for split in ("familiar","heldout"):
                                ev_items=test_sets["B_independent"][split]
                                ex,_=prepare_factor_batch(ev_items,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                                for rec in factor_probe(model,probe_items,px,ev_items,ex,seed):
                                    probe_rows.append({"seed":seed,"model":kind,"protocol":protocol,
                                        "schedule":schedule,"acoustic_config":"B_independent","split":split,**rec})
                            if protocol=="strict" and schedule=="curriculum":
                                held=test_sets["B_independent"]["heldout"]
                                hx,_=prepare_factor_batch(held,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                                counterfactual_rows.extend(_counterfactuals(world,model,held,hx,seed,kind,protocol,schedule))
                        if kind=="H" and measured and protocol=="strict" and schedule=="curriculum":
                            held=test_sets["B_independent"]["heldout"]
                            hx,hy=prepare_factor_batch(held,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                            for threshold in (0.,.05,.1,.25,.5,1.):
                                p=model.predict_rollout(hx,(9,),event_threshold=threshold,
                                    observation_mask=np.any(hx!=0,axis=2))[9]
                                row={"seed":seed,"model":"H","protocol":protocol,"schedule":schedule,
                                    "acoustic_config":"B_independent","split":"heldout","threshold":threshold,
                                    "mae_h9":float(np.mean(np.abs(p-hy[:,8]))),"parameters":model.parameter_count,
                                    "event_opportunities":model.opportunities,"state_updates":model.updates,
                                    "event_skipped":model.skipped,
                                    "event_skip_fraction":model.skipped/max(1,model.opportunities)}
                                t0=time.perf_counter();model.predict_rollout(hx[:1],(9,),event_threshold=threshold,
                                    observation_mask=np.any(hx[:1]!=0,axis=2))
                                row["trigger_wall_seconds_per_episode"]=time.perf_counter()-t0
                                row["candidate_generation_seconds"]=model.candidate_seconds
                                row["event_detection_seconds"]=model.trigger_seconds
                                row["state_update_seconds"]=model.state_seconds
                                compute_rows.append(row)

    # Factorial cue-by-schedule control: run the same strict curriculum and
    # shuffled episode order with the historical combined acoustic code.
    # This lets the report distinguish cue, schedule, and interaction effects.
    for si in range(config.seeds):
        seed=config.seed+1009*si
        phase_sign=int(np.random.default_rng(seed).choice([-1,1]))
        train_positions,test_positions=split_start_positions(world)
        size=max(config.train_sizes);nd=size//4;ns=size//4;nc=size-nd-ns
        pools=make_training_pools(world,seed,size,"strict",train_positions,phase_sign,"A_legacy")
        training=pools["direction_focus"][:nd]+pools["speed_focus"][:ns]+pools["combined"][:nc]
        labels=[item["stage"] for item in training]
        x,y=prepare_factor_batch(training,world,9,SEQUENCE_LENGTH,TRAIN_HORIZONS)
        val_items=make_test_items(world,seed+200,config.validation_per_combo,"familiar",
                                  train_positions,phase_sign,"A_legacy")
        val_x,val_y=prepare_factor_batch(val_items,world,9,SEQUENCE_LENGTH,TRAIN_HORIZONS)
        test_sets={split:make_test_items(world,seed+300,config.test_per_combo,split,
            test_positions,phase_sign,"A_legacy") for split in ("familiar","heldout")}
        for schedule in ("curriculum","shuffled"):
            for mi,kind in enumerate(MODELS):
                model_seed=seed+size*31+mi*10000
                model=make_model(kind,x.shape[-1],model_seed)
                cpu0=time.process_time()
                stats=train_scheduled(model,x,y,labels,schedule,model_seed,
                    epochs=config.epochs_large,batch_size=config.batch_size,val_x=val_x,val_y=val_y)
                train_cpu=time.process_time()-cpu0
                sample_rows.append({"seed":seed,"model":kind,"protocol":"strict",
                    "schedule":schedule,"acoustic_config":"A_legacy","train_size":size,
                    "direction_focus_episodes":nd,"speed_focus_episodes":ns,"combined_episodes":nc,
                    "heldout_motion_pretraining":False,
                    "training_mae":float(np.mean(np.abs(model.predict_rollout(x[:min(32,len(x))],(9,))[9]-y[:min(32,len(y)),8]))),
                    "validation_mae":stats.validation_mae,"train_wall_seconds":stats.seconds,
                    "train_cpu_seconds":train_cpu,"parameters":model.parameter_count,
                    "updates":stats.epochs*int(np.ceil(len(x)/config.batch_size)),
                    "min_gradient_norm":stats.min_gradient_norm,"max_gradient_norm":stats.max_gradient_norm})
                eval_pairs=[]
                for split,items in test_sets.items():
                    ex,ey=prepare_factor_batch(items,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                    eval_pairs.append((split,items,ex,ey))
                _run_model(model,eval_pairs,training,"strict",schedule,"A_legacy",seed,size,
                    x,y,val_x,val_y,stats,result_rows,compute_rows,probe_rows,{},out,labels,
                    measure=True,config=config)
                config_meta={"seed":seed,"training_seed":model_seed,"model":kind,
                    "protocol":"strict","schedule":schedule,"acoustic_config":"A_legacy",
                    "train_size":size,"stage_counts":{"direction":nd,"speed":ns,"combined":nc},
                    "heldout_motion_pretraining":False,"epochs":stats.epochs,
                    "batch_size":config.batch_size,"learning_rate":.003,
                    "gradient_clip":ad.GRAD_CLIP,"sequence_length":SEQUENCE_LENGTH,
                    "input_end_tick":9,"train_horizons":TRAIN_HORIZONS,
                    "heldout_pairs":HELDOUT,"phase_sign":phase_sign,"benchmark":asdict(config)}
                ad.save_checkpoint(ckpt/f"seed-{seed}-{kind}-A_legacy-strict-{schedule}.npz",model,config_meta)
                probe_items=pools["combined"][:nc]
                px,_=prepare_factor_batch(probe_items,world,4,SEQUENCE_LENGTH,TRAIN_HORIZONS)
                for split,items in test_sets.items():
                    ex,_=prepare_factor_batch(items,world,4,SEQUENCE_LENGTH,EVAL_HORIZONS)
                    for rec in factor_probe(model,probe_items,px,items,ex,seed):
                        probe_rows.append({"seed":seed,"model":kind,"protocol":"strict",
                            "schedule":schedule,"split":split,"acoustic_config":"A_legacy",**rec})

    _write_csv(out/"factor_curriculum_results.csv",result_rows)
    _write_csv(out/"factor_sample_efficiency.csv",sample_rows)
    _write_csv(out/"factor_compute_benchmark.csv",compute_rows)
    _write_csv(out/"latent_factor_recovery.csv",probe_rows)
    _write_csv(out/"counterfactual_audio_results.csv",counterfactual_rows)
    # Combined-only A/B covers every size; factorial A/B curriculum controls
    # cover the largest size so schedule effects and interactions are matched.
    cue_rows=[r for r in result_rows if r["schedule"]=="combined_only" or
              (r["train_size"]==max(config.train_sizes) and r["protocol"]=="strict" and
               r["schedule"] in ("curriculum","shuffled"))]
    _write_csv(out/"cue_comparison_results.csv",cue_rows)
    config_doc={"experiment":asdict(config),"world":asdict(world.config),
        "sequence_length":SEQUENCE_LENGTH,"training_horizons":TRAIN_HORIZONS,
        "evaluation_horizons":EVAL_HORIZONS,"strict_rule":"all phases exclude heldout direction-speed pairs",
        "transfer_rule":"speed-focused stage balances both directions at each speed and overlaps withheld motions",
        "training_order":"curriculum=direction_focus,speed_focus,combined; shuffled=random permutation of same records",
        "cue_schedule_factorial_control":"A_legacy curriculum and shuffled arms at the largest training size, matched with B_independent",
        "epochs":"60 below 500 episodes, 80 at 500","batch_size":config.batch_size,
        "learning_rate":0.003,"gradient_clip":ad.GRAD_CLIP,
        "vocabulary":[{**asdict(cue),"frequencies_hz":list(cue.frequencies_hz)} for cue in VOCABULARY]}
    (out/"factor_experiment_config.json").write_text(json.dumps(config_doc,indent=2),encoding="utf-8")
    _plot_artifacts(out,result_rows,probe_rows,counterfactual_rows)
    _report(out,config,result_rows,sample_rows,compute_rows,probe_rows,counterfactual_rows,validate_acoustic())
    return out


def validate_acoustic():
    from .factor_diagnostics import validate_acoustic_vocabulary
    return validate_acoustic_vocabulary()

