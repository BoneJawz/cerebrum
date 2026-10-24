"""Controlled observation-window and state-dynamics validation study.

The study deliberately reuses the established F/G/H equations and direct
baseline. Simulator labels are retained in experiment records only; model
inputs are fixed sensory features from Observation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

import numpy as np

from . import adaptive_dynamics as ad
from .environment import WorldConfig
from .factor_environment import FactorWorld, VOCABULARY, target_trajectory, write_vocabulary
from .factor_learning import make_model, split_start_positions
from .processors import sensory_features
from .contracts import Observation
from .factor_environment import VOCAB_BY_ID

WINDOWS={"A_instruction_only":4,"B_observed_motion":9,"C_transfer_eval":4}
HORIZONS=(1,3,6,9,12,18)
TRAIN_HORIZONS=tuple(range(1,13))
KINDS=("F","G","H","FF_direct")


def _csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def _episode_batch(world,seed,n,positions,all_pairs=True):
    rng=np.random.default_rng(seed);pairs=[(d,s) for d in (-1,1) for s in (1,2,3)]
    items=[]
    for i in range(n):
        d,s=pairs[i%6] if all_pairs else pairs[int(rng.integers(6))]
        p=int(rng.choice(positions));ep_seed=seed*100000+i
        ep=world.episode(ep_seed,d,s,p,task="combined",acoustic_config="B_independent")
        items.append({"episode":ep,"direction":d,"speed":s,"initial_position":p,"seed":ep_seed})
    rng.shuffle(items)
    return items


def _batch(items,cutoff,world,seq_len=10,horizons=TRAIN_HORIZONS):
    xs=[];ys=[];mask=[]
    for item in items:
        ep=item["episode"]
        obs=ep.observations[:cutoff+1]
        fs=[sensory_features(o) for o in obs]
        width=len(fs[0]);valid=len(fs)
        fs.extend([np.zeros(width) for _ in range(seq_len-valid)])
        xs.append(fs);mask.append([True]*valid+[False]*(seq_len-valid))
        ys.append(target_trajectory(ep,cutoff,horizons))
    return np.asarray(xs,float),np.asarray(ys,float),np.asarray(mask,bool)


def _normalize(model,x,mask,y):
    values=x[mask]
    model.input_mean=values.mean(0);model.input_scale=values.std(0)
    model.input_scale[model.input_scale<1e-8]=1.
    model.target_mean=float(y.mean());model.target_scale=float(y.std()) or 1.
    z=(x-model.input_mean)/model.input_scale;z[~mask]=0.
    return z,(y-model.target_mean)/model.target_scale


def _fit(model,x,y,mask,seed,epochs,batch_size=32,lr=.003):
    xn,yn=_normalize(model,x,mask,y);rng=np.random.default_rng(seed);losses=[]
    cpu0=time.process_time();wall0=time.perf_counter();tracemalloc.start()
    for _ in range(epochs):
        order=rng.permutation(len(x));total=0.
        for begin in range(0,len(order),batch_size):
            ids=order[begin:begin+batch_size]
            if isinstance(model,ad.DirectTrajectoryBaseline):
                loss,grads,_=__import__("sensory_lab.factor_learning",fromlist=["_direct_grad"])._direct_grad(model,xn[ids],yn[ids])
            else:
                loss,grads,_=ad._batch_grad(model,xn[ids],y[ids],mask[ids])
            # Same Adam implementation as the existing factor-learning study.
            from .factor_learning import _adam_step
            _adam_step(model,grads,lr);total+=loss*len(ids)
        losses.append(total/len(x))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return {"train_wall_seconds":time.perf_counter()-wall0,"train_cpu_seconds":time.process_time()-cpu0,
            "peak_python_bytes":peak,"losses":losses,"gradient_updates":epochs*math.ceil(len(x)/batch_size)}


def _trace(model,x,mask,max_h=18):
    """Return decoded state after assimilation (h0) and each autonomous step."""
    z=(x-model.input_mean)/model.input_scale;z[~mask]=0.
    if isinstance(model,ad.DirectTrajectoryBaseline):return None
    h,states=model._encode(z.transpose(1,0,2),mask.T)
    out={0:(h@model.params["dec"]+model.params["out"][0])*model.target_scale+model.target_mean}
    for step in range(1,max_h+1):
        for _ in range(model.substeps):h=model._transition(h)[0]
        out[step]=(h@model.params["dec"]+model.params["out"][0])*model.target_scale+model.target_mean
    return out


def _hidden_states(model,x,mask,steps=(0,1,3,6,12,18)):
    z=(x-model.input_mean)/model.input_scale;z[~mask]=0.
    h,states=model._encode(z.transpose(1,0,2),mask.T)
    result={0:h.copy()};wanted=set(steps)
    for step in range(1,max(steps)+1):
        for _ in range(model.substeps):h=model._transition(h)[0]
        if step in wanted:result[step]=h.copy()
    return result


def _probe_rows(model,train_x,train_mask,train_items,eval_x,eval_mask,eval_items,condition,seed):
    """Ridge probes fit only on training episodes; dual solve supports raw features."""
    tr_hidden=_hidden_states(model,train_x,train_mask);ev_hidden=_hidden_states(model,eval_x,eval_mask)
    raw_tr=train_x.reshape(len(train_x),-1);raw_ev=eval_x.reshape(len(eval_x),-1)
    sources={"raw_permitted_features":(raw_tr,raw_ev)}
    sources.update({f"hidden_after_{h}":(tr_hidden[h],ev_hidden[h]) for h in tr_hidden})
    labels={"position":(np.array([i["initial_position"] for i in train_items],float),np.array([i["initial_position"] for i in eval_items],float)),
        "direction":(np.array([i["direction"] for i in train_items],float),np.array([i["direction"] for i in eval_items],float)),
        "speed":(np.array([i["speed"] for i in train_items],float),np.array([i["speed"] for i in eval_items],float))}
    rows=[]
    for source,(a,b) in sources.items():
        mean=a.mean(0);scale=a.std(0);scale[scale<1e-8]=1.
        a=(a-mean)/scale;b=(b-mean)/scale
        gram=a@a.T+1e-2*np.eye(len(a))
        for factor,(yt,ye) in labels.items():
            ym=yt.mean();coef=np.linalg.solve(gram,yt-ym);pred=b@a.T@coef+ym
            if factor=="position":metric="mae_px";value=np.mean(np.abs(pred-ye))
            elif factor=="direction":metric="accuracy";value=np.mean(np.sign(pred)==ye)
            else:
                lv=np.array(sorted(set(yt)));pred=lv[np.argmin(np.abs(pred[:,None]-lv),axis=1)]
                metric="exact_accuracy";value=np.mean(pred==ye)
            rows.append({"condition":condition,"seed":seed,"model":model.kind,"source":source,
                "factor":factor,"metric":metric,"value":float(value),"n_probe_train":len(a),"n_eval":len(b),
                "training_combinations":"all six familiar combinations"})
    return rows


def _evaluate(model,x,mask,items,cutoff,world,seed,split,condition,latency_trials=5):
    pred=model.predict_rollout(x,HORIZONS,observation_mask=mask)
    truth=np.stack([target_trajectory(i["episode"],cutoff,HORIZONS) for i in items])
    origin=np.array([np.dot(i["episode"].observations[max(cutoff,5)].vision,
        np.arange(len(i["episode"].observations[max(cutoff,5)].vision))) /
        np.sum(i["episode"].observations[max(cutoff,5)].vision) for i in items])
    rows=[]
    for hi,h in enumerate(HORIZONS):
        if h not in pred:continue
        p=pred[h];t=truth[:,hi];disp=p-origin;true_disp=t-origin
        direction=np.array([i["direction"] for i in items])
        speed=np.array([i["speed"] for i in items])
        correct=(np.sign(disp)==direction)&(np.abs(disp)>=.5)
        for d in (-1,1):
            for s in (1,2,3):
                sel=(direction==d)&(speed==s)
                if not np.any(sel):continue
                rows.append({"condition":condition,"split":split,"model":model.kind,"horizon":h,
                    "direction":d,"speed":s,"n":int(sel.sum()),"mae_px":float(np.mean(np.abs(p[sel]-t[sel]))),
                    "within_1px":float(np.mean(np.abs(p[sel]-t[sel])<=1)),
                    "direction_correct_nontrivial":float(np.mean(correct[sel])),
                    "displacement_magnitude_mae_px":float(np.mean(np.abs(np.abs(disp[sel])-np.abs(true_disp[sel]))))})
    # Measurement repeats are deliberately separate from model scoring.
    samples=[];cpu0=time.process_time();model.predict_rollout(x[:1],(1,));infer_cpu=time.process_time()-cpu0
    for _ in range(latency_trials):
        start=time.perf_counter_ns();model.predict_rollout(x[:1],(1,));samples.append((time.perf_counter_ns()-start)/1e6)
    comp={"condition":condition,"split":split,"model":model.kind,"latency_median_ms":float(np.median(samples)),
          "latency_p95_ms":float(np.percentile(samples,95)),"latency_p99_ms":float(np.percentile(samples,99)),
          "parameter_count":model.parameter_count,"state_updates_per_episode":int(getattr(model,"updates",0)),
          "update_opportunities":int(getattr(model,"opportunities",0)),"skipped_updates":int(getattr(model,"skipped",0)),
          "inference_cpu_seconds_single_example":infer_cpu,
          "event_candidate_seconds":float(getattr(model,"candidate_seconds",0.0)),
          "event_trigger_seconds":float(getattr(model,"trigger_seconds",0.0)),
          "event_state_assignment_seconds":float(getattr(model,"state_seconds",0.0))}
    return rows,comp,pred,truth,origin


def _write_plot(path,plots,xlabel="Forecast horizon (motion frames)",ylabel="Position MAE (pixels)"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:return False
    fig,ax=plt.subplots(figsize=(8,4.5))
    for label,x,y in plots:ax.plot(x,y,marker="o",label=label)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig);return True


def _cue_feature(world,audio,seed):
    obs=Observation(0.,np.zeros(world.config.pixels,dtype=np.int8),np.asarray(audio,float))
    return sensory_features(obs)[world.config.pixels:]


def _cue_diagnostics(world,seed):
    rng=np.random.default_rng(seed);isolated=[]
    ids=[c.concept_id for c in VOCABULARY]
    train={k:[] for k in ids};test={k:[] for k in ids}
    for key in ids:
        cue=VOCAB_BY_ID[key]
        for j in range(40):
            chunks=[world._cue_chunk(cue,t,rng)+rng.normal(0,.015,world.chunk) for t in range(2)]
            chunk=np.concatenate(chunks)
            dest=train if j<24 else test
            dest[key].append(_cue_feature(world,chunk,seed))
    tx=np.stack([v for k in ids for v in train[k]]);ty=np.array([k for k in ids for _ in train[k]])
    ex=np.stack([v for k in ids for v in test[k]]);ey=np.array([k for k in ids for _ in test[k]])
    mu=tx.mean(0);sd=tx.std(0);sd[sd<1e-8]=1
    centers=np.stack([(tx[ty==k]-mu).mean(0)/sd for k in ids])
    pred=np.array(ids)[np.argmin(np.sum(((ex-mu)/sd)[:,None,:]-centers[None,:,:],axis=2)**2,axis=1)]
    isolated.append({"test":"isolated_cue_nearest_centroid","cue_set":"five independent cue classes","accuracy":float(np.mean(pred==ey)),"n_train":len(tx),"n_test":len(ex),"feature_source":"same sensory_features audio suffix"})
    combos=[(d,s) for d in (-1,1) for s in (1,2,3)]
    tr=[];tl=[];te=[];el=[]
    for d,s in combos:
        for j in range(32):
            ep=world.episode(seed*10000+len(tr)+len(te)+j,d,s,initial_position=world.config.pixels//2,
                task="combined",acoustic_config="B_independent")
            seq=np.stack([sensory_features(o)[world.config.pixels:] for o in ep.observations[:5]]).ravel()
            if j<20:tr.append(seq);tl.append((d,s))
            else:te.append(seq);el.append((d,s))
    tr=np.asarray(tr);te=np.asarray(te);mu=tr.mean(0);sd=tr.std(0);sd[sd<1e-8]=1
    tr=(tr-mu)/sd;te=(te-mu)/sd;labels=list(dict.fromkeys(tl))
    cent=np.stack([tr[np.array(tl,dtype=object)==pair].mean(0) for pair in labels]) if False else np.stack([np.mean([tr[i] for i,p in enumerate(tl) if p==pair],axis=0) for pair in labels])
    guess=[labels[i] for i in np.argmin(np.sum((te[:,None,:]-cent[None,:,:])**2,axis=2),axis=1)]
    isolated.append({"test":"sequential_pair_nearest_centroid","cue_set":"direction then silence then speed","accuracy":float(np.mean(np.all(np.asarray(guess)==np.asarray(el),axis=1))),"n_train":len(tr),"n_test":len(te),"feature_source":"five consecutive sensory_features audio suffixes"})
    return isolated


def run_controlled_study(output="results/controlled-factor-learning",seed=42,seeds=5,
                         sizes=(10,25,50,100,250,500),test_per_combo=20,epochs=40):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config={"seed":seed,"seeds":seeds,"sizes":list(sizes),"epochs":epochs,"test_per_combo":test_per_combo,
        "windows":WINDOWS,"sequence_length":10,"audio_hz":2000,"visual_hz":20,"audio_samples_per_tick":100,
        "cue_segments":{"direction":[0,2],"silence":[2,3],"speed":[3,5]},"horizons":list(HORIZONS),
        "training_horizons":list(TRAIN_HORIZONS),"strict_holdout":[[-1,2],[1,3]],
        "transfer_protocol":"balanced factor exposure; held-out combined cue sequence may have appeared in factor pretraining",
        "feature_extractor":"sensory_features: 177 binary vision cells + 4 audio summaries + 16 signed samples + 49 FFT magnitudes",
        "python":platform.python_version(),"numpy":np.__version__}
    (out/"experiment_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    (out/"seeds.json").write_text(json.dumps({"root_seed":seed,"run_seeds":[seed+i*1009 for i in range(seeds)]},indent=2),encoding="utf-8")
    write_vocabulary(out/"acoustic_vocabulary.csv")
    world=FactorWorld(WorldConfig(pixels=177,ticks=32,visual_hz=20,audio_hz=2000,noise_std=.015))
    train_positions,test_positions=split_start_positions(world)
    protocol=[];rollout=[];compute=[];state_rows=[];probe_rows=[];cue_rows=[];gen_rows=[];curr_rows=[];counter_rows=[]
    plot_data={}
    cue_rows.extend(_cue_diagnostics(world,seed))
    for run in range(seeds):
        run_seed=seed+run*1009
        test_items=[]
        for d in (-1,1):
            for s in (1,2,3):
                for j in range(test_per_combo):
                    p=test_positions[(j+run*7)%len(test_positions)]
                    es=run_seed*100000+70000+((d+1)//2)*3000+s*100+j
                    test_items.append({"episode":world.episode(es,d,s,p,task="combined",acoustic_config="B_independent"),
                        "direction":d,"speed":s,"initial_position":p,"seed":es})
        for size in sizes:
            train_items=_episode_batch(world,run_seed+size,size,train_positions)
            for condition,cutoff in (("A_instruction_only",4),("B_observed_motion",9)):
                tx,ty,tm=_batch(train_items,cutoff,world)
                for kind in KINDS:
                    model=make_model(kind,tx.shape[2],run_seed+size)
                    if kind=="H":model.event_threshold=.05
                    fit=_fit(model,tx,ty,tm,run_seed+size+17,epochs)
                    ex,ey,em=_batch(test_items,cutoff,world,horizons=HORIZONS)
                    mets,comp,preds,truth,orig=_evaluate(model,ex,em,test_items,cutoff,world,run_seed,"all_six",condition)
                    protocol.extend([{**r,"train_size":size,"seed":run_seed} for r in mets]);compute.append({**comp,"train_size":size,"seed":run_seed,
                        "train_wall_seconds":fit["train_wall_seconds"],"train_cpu_seconds":fit["train_cpu_seconds"],
                        "peak_python_bytes":fit["peak_python_bytes"],"gradient_updates":fit["gradient_updates"]})
                    for hi,h in enumerate(HORIZONS):
                        if h not in preds:continue
                        mae=float(np.mean(np.abs(preds[h]-truth[:,hi])))
                        rollout.append({"condition":condition,"seed":run_seed,"train_size":size,"model":kind,
                            "horizon":h,"mae_px":mae,"within_1px":float(np.mean(np.abs(preds[h]-truth[:,hi])<=1)),
                            "predicted_displacement_mean":float(np.mean(preds[h]-orig)),
                            "true_displacement_mean":float(np.mean(truth[:,hi]-orig))})
                    if size==max(sizes):
                        if condition=="A_instruction_only":
                            for h in (3,6,9,12):
                                if h not in preds:continue
                                lookup={(i["initial_position"],i["direction"],i["speed"]):j for j,i in enumerate(test_items)}
                                for (p0,d0,s0),idx0 in lookup.items():
                                    opposite=lookup.get((p0,-d0,s0))
                                    faster=lookup.get((p0,d0,min(3,s0+1)))
                                    if opposite is not None:
                                        delta=float((preds[h][opposite]-p0)-(preds[h][idx0]-p0))
                                        counter_rows.append({"condition":condition,"model":kind,"seed":run_seed,"horizon":h,
                                            "intervention":"direction_flip","initial_position":p0,"direction":d0,"speed":s0,
                                            "predicted_displacement_change":delta,"expected_displacement_change":float(-2*d0*s0*h),
                                            "absolute_response_error":abs(delta+2*d0*s0*h)})
                                    if faster is not None and faster!=idx0:
                                        delta=float(abs(preds[h][faster]-p0)-abs(preds[h][idx0]-p0))
                                        counter_rows.append({"condition":condition,"model":kind,"seed":run_seed,"horizon":h,
                                            "intervention":"speed_increment","initial_position":p0,"direction":d0,"speed":s0,
                                            "predicted_displacement_change":delta,"expected_displacement_change":float(d0*h),
                                            "absolute_response_error":abs(delta-d0*h)})
                        # State probes use familiar six-way experiences; linear ridge is intentionally diagnostic.
                        trace=_trace(model,ex,em,18)
                        if trace is not None:
                            for h,state_pos in trace.items():
                                state_target=(orig if h==0 else np.array([
                                    target_trajectory(i["episode"],cutoff,(h,))[0] for i in test_items]))
                                state_rows.append({"condition":condition,"seed":run_seed,"model":kind,"state_step":h,
                                    "mean_decoded_position":float(np.mean(state_pos)),"mean_position_error":float(np.mean(np.abs(state_pos-state_target))),
                                    "mean_displacement":float(np.mean(state_pos-orig)),"n":len(test_items)})
                        if kind!="FF_direct":
                            probe_rows.extend(_probe_rows(model,tx,tm,train_items,ex,em,test_items,
                                condition,run_seed))
                        # Training loss checkpoints provide optimization progress evidence.
                        for epno,loss in enumerate(fit["losses"],1):
                            curr_rows.append({"condition":condition,"schedule":"all_six_shuffled","seed":run_seed,
                                "train_size":size,"model":kind,"epoch":epno,"training_mse_normalized":loss,
                                "gradient_updates":math.ceil(size/32)*epno})
        # C transfer: B-trained model tested with A observations; matched comparison above remains A/B.
        for kind in KINDS:
            size=max(sizes);train_items=_episode_batch(world,run_seed+size,size,train_positions)
            bx,by,bm=_batch(train_items,9,world)
            model=make_model(kind,bx.shape[2],run_seed+size)
            if kind=="H":model.event_threshold=.05
            fit=_fit(model,bx,by,bm,run_seed+91,epochs)
            ex,ey,em=_batch(test_items,4,world,horizons=HORIZONS)
            mets,comp,preds,truth,orig=_evaluate(model,ex,em,test_items,4,world,run_seed,"all_six","C_Btrain_Aeval")
            protocol.extend([{**r,"train_size":size,"seed":run_seed} for r in mets]);compute.append({**comp,"train_size":size,"seed":run_seed,
                "train_wall_seconds":fit["train_wall_seconds"],"train_cpu_seconds":fit["train_cpu_seconds"],
                "peak_python_bytes":fit["peak_python_bytes"],"gradient_updates":fit["gradient_updates"]})
            for hi,h in enumerate(HORIZONS):
                if h in preds:
                    rollout.append({"condition":"C_Btrain_Aeval","seed":run_seed,"train_size":size,"model":kind,
                        "horizon":h,"mae_px":float(np.mean(np.abs(preds[h]-truth[:,hi]))),
                        "within_1px":float(np.mean(np.abs(preds[h]-truth[:,hi])<=1)),
                        "predicted_displacement_mean":float(np.mean(preds[h]-orig)),
                        "true_displacement_mean":float(np.mean(truth[:,hi]-orig))})
    _csv(out/"observation_protocol_results.csv",protocol)
    _csv(out/"rollout_diagnostics.csv",rollout)
    _csv(out/"compute_results.csv",compute)
    _csv(out/"state_assimilation_results.csv",state_rows)
    _csv(out/"factor_probe_results.csv",probe_rows)
    _csv(out/"cue_interpretation_results.csv",cue_rows)
    _csv(out/"counterfactual_displacement.csv",counter_rows)
    _csv(out/"curriculum_control_results.csv",curr_rows)
    _csv(out/"generalization_control_results.csv",gen_rows)
    # Plots summarize measured prediction rows from the largest training size and all seeds.
    for condition in ("A_instruction_only","B_observed_motion","C_Btrain_Aeval"):
        for kind in KINDS:
            rr=[r for r in rollout if r["condition"]==condition and r["model"]==kind and r["train_size"]==max(sizes)]
            if rr:plot_data.setdefault((condition,kind),([],[]));plot_data[(condition,kind)][0].extend([r["horizon"] for r in rr]);plot_data[(condition,kind)][1].extend([r["mae_px"] for r in rr])
    for filename,cond in (("observation_window_comparison.png",None),("rollout_error_comparison.png",None),
                          ("direction_speed_recovery.png",None),("counterfactual_displacement.png",None),
                          ("curriculum_learning_curves.png",None)):
        plots=[]
        if filename=="curriculum_learning_curves.png":
            for kind in KINDS:
                rr=[r for r in curr_rows if r["model"]==kind and r["condition"]=="A_instruction_only"]
                if rr:plots.append((kind,[r["epoch"] for r in rr],[r["training_mse_normalized"] for r in rr]))
        elif filename=="direction_speed_recovery.png":
            for kind in KINDS:
                vals=[]
                for h in HORIZONS:
                    z=[float(r["direction_correct_nontrivial"]) for r in protocol if r["condition"]=="A_instruction_only" and r["model"]==kind and int(r["train_size"])==max(sizes) and int(r["horizon"])==h]
                    if z:vals.append((h,float(np.mean(z))))
                if vals:plots.append((kind,[v[0] for v in vals],[v[1] for v in vals]))
        elif filename=="counterfactual_displacement.png":
            for kind in KINDS:
                vals=[]
                for h in (3,6,9,12):
                    z=[abs(float(r["predicted_displacement_change"])) for r in counter_rows if r["model"]==kind and int(r["horizon"])==h and r["intervention"]=="direction_flip"]
                    if z:vals.append((h,float(np.mean(z))))
                if vals:plots.append((kind,[v[0] for v in vals],[v[1] for v in vals]))
        else:
            for (c,k),(hx,hy) in plot_data.items():
                if filename=="observation_window_comparison.png" and k=="F":plots.append((c,hx,hy))
                elif filename!="observation_window_comparison.png":plots.append((f"{c} {k}",hx,hy))
        labels={"direction_speed_recovery":("Forecast horizon (motion frames)","Direction accuracy (nontrivial displacement)"),
            "counterfactual_displacement":("Forecast horizon (motion frames)","Response to reversing direction cue (px)"),
            "curriculum_learning_curves":("Training epoch","Normalized training MSE (all-six shuffled)")}
        _write_plot(out/filename,plots,*labels.get(filename,("Forecast horizon (motion frames)","Position MAE (pixels)")))
    # Scientific report keeps measured facts distinct from interpretation.
    def avg(condition,kind,horizon,field="mae_px"):
        vals=[float(r[field]) for r in rollout if r["condition"]==condition and r["model"]==kind and int(r["train_size"])==max(sizes) and int(r["horizon"])==horizon]
        return "not available" if not vals else f"{np.mean(vals):.2f} ± {np.std(vals,ddof=1):.2f}" if len(vals)>1 else f"{vals[0]:.2f}"
    summary=["| Condition | Model | h1 MAE | h9 MAE | h12 MAE | h18 MAE |","|---|---|---:|---:|---:|---:|"]
    for condition in ("A_instruction_only","B_observed_motion","C_Btrain_Aeval"):
        for kind in KINDS:
            summary.append(f"| {condition} | {kind} | {avg(condition,kind,1)} | {avg(condition,kind,9)} | {avg(condition,kind,12)} | {avg(condition,kind,18)} |")
    cue_summary=list(csv.DictReader((out/"cue_interpretation_results.csv").open(encoding="utf-8")))
    cue_text="; ".join(f"{r['test']}: {float(r['accuracy']):.1%} (n={r['n_test']})" for r in cue_summary)
    report=["# Controlled factor-learning and state-dynamics validation", "",
      "## Protocol audit", "",
      "Combined acoustic cues are direction ticks 0–1, silence tick 2, and speed ticks 3–4. At 20 Hz these finish at 0.25 s; the tick-4 observation includes the complete speed cue. The visual array at tick 5 is the initial motion origin. A uses observations 0–4 and predicts from frame 5; B uses 0–9 and predicts from frame 9. Horizons 1, 3, 6, 9, 12, and 18 target frames origin+h. The environment has 32 ticks to support B horizon 18.", "",
      "Inputs contain only the fixed sensory feature vector: 177 visual cells, four waveform summary values, 16 signed waveform samples, and 49 FFT magnitudes. Cue names and simulator labels are kept in experiment metadata and never passed to models. Each acoustic chunk is 100 raw samples at 2 kHz; visual frames arrive at 20 Hz. The same independent vocabulary and waveform generation are used in every condition.", "",
      "## Measured results", "",
      f"Study configuration: {seeds} seeds, sizes {', '.join(map(str,sizes))}, {epochs} epochs per fit, six direction-speed combinations, {test_per_combo} held-out trajectories per combination. Paired seed means and standard deviations at the largest training size are shown below. Detailed per-factor metrics are in `observation_protocol_results.csv`; training/inference costs are in `compute_results.csv`.", "",
      *summary,"",
      f"Acoustic diagnostics: {cue_text}. Isolated cue classes use nearest-centroid classification over independently generated full two-chunk waveforms; combined instructions use the ordered five-tick waveform sequence.", "",
      "Results are generated from direct model predictions. A has targets relative to frame 5, B relative to frame 9, and C evaluates a B-trained model from frame 5. Compare matched A with A and B with B; C is explicitly a window-transfer condition. FF_direct has no autonomous hidden-state trace and does not forecast beyond its trained direct horizon.", "",
      "## Limits and interpretation", "",
      "This run validates all-six-combination in-distribution prediction under matched windows before interpreting compositional transfer. Its A/B difference combines added visual motion evidence and a shorter forecast distance, so it is not a pure sensory-modality effect. Cue scores are diagnostic nearest-centroid baselines, not the learned models' own concept reports. State trace values are decoded positions from hidden states, not direct evidence that a factor is represented. The current first phase does not execute strict-holdout versus factor-pretraining transfer or curriculum versus shuffled controls; those are deferred until the matched-window baseline is stable. The curriculum CSV contains only training-loss checkpoints from the matched all-six shuffled fit and must not be interpreted as a schedule comparison.", "",
      "## Next experiment", "",
      "Use these matched-window results to decide whether the remaining failure is already present on the six-way task. If matched A remains poor, first test normalization, input timing, and supervised optimization on a smaller fixed diagnostic set. If A and B work but C fails, isolate the observation-window transfer. Only then run strict compositional holdout and balanced factor-pretraining transfer, with curriculum and shuffled schedules replaying identical episode IDs and update counts.", ""]
    (out/"CONTROLLED_FACTOR_LEARNING_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description="Run controlled observation-window and state-dynamics validation")
    p.add_argument("--output",default="results/controlled-factor-learning");p.add_argument("--seed",type=int,default=42)
    p.add_argument("--seeds",type=int,default=5);p.add_argument("--sizes",type=int,nargs="+",default=[10,25,50,100,250,500])
    p.add_argument("--test-per-combo",type=int,default=20);p.add_argument("--epochs",type=int,default=40)
    a=p.parse_args(argv);print(run_controlled_study(a.output,a.seed,a.seeds,a.sizes,a.test_per_combo,a.epochs))


if __name__=="__main__":main()
