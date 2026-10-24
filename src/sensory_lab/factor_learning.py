"""Curriculum, transfer, cue-comparison and diagnostic runners.

All labels and codebook metadata remain outside Observation. This module uses
the existing F/G/H transition equations and NumPy optimizers from the adaptive
dynamics experiment.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
import time
import tracemalloc

import numpy as np

from . import adaptive_dynamics as ad
from .environment import MOTION_START_TICK, WorldConfig
from .factor_environment import FactorWorld, VOCABULARY, write_vocabulary, target_trajectory
from .processors import sensory_features


DEFAULT_SIZES=(10,25,50,100,250,500)
STAGE_ORDER=("direction_focus","speed_focus","combined")
SEQUENCE_LENGTH=10
TRAIN_HORIZONS=tuple(range(1,13))
EVAL_HORIZONS=tuple(range(1,19))
HELDOUT=((-1,2),(1,3))
FAMILIAR=((-1,1),(-1,3),(1,1),(1,2))


def _balanced_pairs(pairs,count,rng):
    out=[]
    while len(out)<count:
        cycle=list(pairs);rng.shuffle(cycle);out.extend(cycle)
    return out[:count]


def _stage_commands(stage,count,protocol,rng):
    if stage=="direction_focus":
        return [(-1,1),(1,1)]*(count//2)+([(-1,1)] if count%2 else [])
    if stage=="combined":
        return _balanced_pairs(FAMILIAR,count,rng)
    if protocol=="strict":
        # Avoid the withheld physical trajectories. This necessarily means
        # speed 2 and 3 are not balanced over direction in the strict arm.
        speed_pairs={1:[(-1,1),(1,1)],2:[(1,2)],3:[(-1,3)]}
        out=[]
        while len(out)<count:
            for speed in (1,2,3):
                out.append(speed_pairs[speed][int(rng.integers(len(speed_pairs[speed])))])
                if len(out)==count:break
        return out
    if protocol=="transfer":
        # Each speed is equally exposed under both directions. The withheld
        # motion pairs therefore occur in speed-focused pretraining.
        out=[]
        while len(out)<count:
            for speed in (1,2,3):
                out.append((-1 if rng.integers(2)==0 else 1,speed))
                if len(out)==count:break
        return out
    raise ValueError("protocol must be strict or transfer")


def _make_stage_pool(world,seed,count,stage,protocol,position_values,phase_sign,
                     acoustic_config="B_independent"):
    rng=np.random.default_rng(seed);commands=_stage_commands(stage,count,protocol,rng)
    out=[]
    for i,(direction,speed) in enumerate(commands):
        initial=int(rng.choice(position_values))
        ep=world.episode(seed=seed*100000+i,direction=direction,speed=speed,
            initial_position=initial,task=stage,acoustic_config=acoustic_config,
            phase_sign=phase_sign)
        out.append({"episode":ep,"stage":stage,"direction":direction,"speed":speed,
            "initial_position":initial,"combination":(direction,speed),
            "seed":seed*100000+i,
            "heldout_motion_pretraining":bool((direction,speed) in HELDOUT and stage!="combined")})
    return out


def make_training_pools(world,seed,max_size,protocol,train_positions,phase_sign,
                        acoustic_config="B_independent"):
    nd=max_size//4;ns=max_size//4;nc=max_size-nd-ns
    return {
        "direction_focus":_make_stage_pool(world,seed+11,nd,"direction_focus",protocol,train_positions,phase_sign,acoustic_config),
        "speed_focus":_make_stage_pool(world,seed+23,ns,"speed_focus",protocol,train_positions,phase_sign,acoustic_config),
        "combined":_make_stage_pool(world,seed+37,nc,"combined",protocol,train_positions,phase_sign,acoustic_config),
    }


def combined_pool(world,seed,max_size,positions,phase_sign,acoustic_config):
    rng=np.random.default_rng(seed+37);commands=_balanced_pairs(FAMILIAR,max_size,rng)
    out=[]
    for i,(d,v) in enumerate(commands):
        start=int(rng.choice(positions));ep=world.episode(seed=(seed+37)*100000+i,
            direction=d,speed=v,initial_position=start,task="combined",
            acoustic_config=acoustic_config,phase_sign=phase_sign)
        out.append({"episode":ep,"stage":"combined","direction":d,"speed":v,
            "initial_position":start,"combination":(d,v),"seed":(seed+37)*100000+i,
            "heldout_motion_pretraining":False})
    return out


def make_test_items(world,seed,n_per_combo,split,positions,phase_sign,acoustic_config="B_independent"):
    pairs=FAMILIAR if split=="familiar" else HELDOUT
    rng=np.random.default_rng(seed+53);commands=_balanced_pairs(pairs,n_per_combo*len(pairs),rng)
    out=[]
    for i,(d,v) in enumerate(commands):
        initial=int(rng.choice(positions));ep=world.episode(seed=(seed+53)*100000+i,
            direction=d,speed=v,initial_position=initial,task="combined",
            acoustic_config=acoustic_config,phase_sign=phase_sign)
        out.append({"episode":ep,"stage":"test","direction":d,"speed":v,
            "initial_position":initial,"combination":(d,v),"seed":(seed+53)*100000+i,
            "heldout_motion_pretraining":False})
    return out


def make_validation_items(world,seed,n_per_combo,positions,phase_sign):
    return make_test_items(world,seed,n_per_combo,"familiar",positions,phase_sign,"B_independent")


def split_start_positions(world):
    lo=world.config.pixels//3;hi=2*world.config.pixels//3
    candidates=list(range(lo,hi))
    train=candidates[::2];test=candidates[1::2]
    if not train or not test:raise ValueError("visual field has too few start positions")
    return train,test


def prepare_factor_batch(items,world,input_end_tick,seq_length=SEQUENCE_LENGTH,
                         horizons=TRAIN_HORIZONS):
    xs=[];ys=[]
    for item in items:
        ep=item["episode"]
        features=[sensory_features(o) for o in ep.observations[:input_end_tick+1]]
        if len(features)>seq_length:raise ValueError("observation prefix exceeds model sequence capacity")
        width=len(features[0]);features.extend([np.zeros(width)]*(seq_length-len(features)))
        xs.append(np.stack(features))
        ys.append(target_trajectory(ep,input_end_tick,horizons))
    return np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)


def _adam_step(model,grads,learning_rate=.003):
    if hasattr(model,"opt_step"):
        model.opt_step+=1;t=model.opt_step;ms=model.opt_m;vs=model.opt_v
    else:
        model.step+=1;t=model.step;ms=model.m;vs=model.v
    for key,g in grads.items():
        ms[key]=.9*ms[key]+.1*g;vs[key]=.999*vs[key]+.001*g*g
        m=ms[key]/(1-.9**t);v=vs[key]/(1-.999**t)
        model.params[key]-=learning_rate*m/(np.sqrt(v)+1e-8)


def _direct_grad(model,xb,yb):
    p=model.params;flat=xb.reshape(len(xb),-1)
    z=np.tanh(flat@p["wx"]+p["b"])
    err=z@p["out"]+p["bo"]-yb
    loss=float(np.mean(err*err));d=2*err/(len(xb)*ad.TRAIN_HORIZON)
    grads={"out":z.T@d,"bo":d.sum(0)}
    dz=(d@p["out"].T)*(1-z*z)
    grads["wx"]=flat.T@dz;grads["b"]=dz.sum(0)
    norm=float(np.sqrt(sum(np.sum(g*g) for g in grads.values())))
    if norm>ad.GRAD_CLIP:
        for k in grads:grads[k]*=ad.GRAD_CLIP/(norm+1e-12)
    return loss,grads,norm


def train_scheduled(model,x,y,stage_labels,schedule,seed,epochs=60,batch_size=32,
                    learning_rate=.003,val_x=None,val_y=None):
    """Same examples/update count/initialization, varying only episode order."""
    model.input_mean=x.mean((0,1));model.input_scale=x.std((0,1));model.input_scale[model.input_scale<1e-8]=1.
    model.target_mean=float(y.mean());model.target_scale=float(y.std()) or 1.
    xn=(x-model.input_mean)/model.input_scale;yn=(y-model.target_mean)/model.target_scale
    labels=np.asarray(stage_labels);rng=np.random.default_rng(seed);losses=[]
    max_norm=0.;min_norm=float("inf");start=time.perf_counter()
    stage_rank={name:i for i,name in enumerate(STAGE_ORDER)}
    curriculum_order=np.argsort([stage_rank.get(str(label),2) for label in labels],kind="stable")
    for epoch in range(epochs):
        order=curriculum_order if schedule=="curriculum" else rng.permutation(len(xn))
        epoch_loss=0.
        for begin in range(0,len(order),batch_size):
            ids=order[begin:begin+batch_size]
            if isinstance(model,ad.DirectTrajectoryBaseline):
                loss,grads,norm=_direct_grad(model,xn[ids],yn[ids])
            else:
                loss,grads,norm=ad._batch_grad(model,xn[ids],y[ids])
            max_norm=max(max_norm,norm);min_norm=min(min_norm,norm)
            _adam_step(model,grads,learning_rate)
            epoch_loss+=loss*len(ids)
        losses.append(epoch_loss/len(xn))
    val_x=x if val_x is None else val_x;val_y=y if val_y is None else val_y
    preds=model.predict_rollout(val_x,(3,6,9,12))
    val_mae=float(np.mean([np.mean(np.abs(preds[h]-val_y[:,h-1])) for h in (3,6,9,12)]))
    return ad.TrainStats(epochs,time.perf_counter()-start,val_mae,losses,max_norm,min_norm)


def make_model(kind,input_dim,seed):
    if kind=="FF_direct":
        # 9,904 parameters at the current 246-feature sensor width, within 1.2%
        # of 32-unit F/G/H (10,017 parameters).
        return ad.DirectTrajectoryBaseline(input_dim,seed=seed,hidden=4,sequence_length=SEQUENCE_LENGTH)
    return ad.TrainableStateModel(input_dim,hidden=ad.HIDDEN,seed=seed,kind=kind)


def model_state(model,x,observation_mask=None):
    raw=np.asarray(x)
    mask=(np.ones(raw.shape[:2],dtype=bool) if observation_mask is None
          else np.asarray(observation_mask,dtype=bool))
    z=(raw-model.input_mean)/model.input_scale;z[~mask]=0
    return model._encode(z.transpose(1,0,2),mask.T)[0]


def factor_probe(model,train_items,train_x,eval_items,eval_x,seed):
    # Probe training uses only familiar combined experiences, even in transfer
    # runs where factor-pretraining contains held-out physical trajectories.
    st=model_state(model,train_x,np.any(train_x!=0,axis=2))
    se=model_state(model,eval_x,np.any(eval_x!=0,axis=2))
    ytr={"position":np.array([i["initial_position"] for i in train_items],float),
         "direction":np.array([i["direction"] for i in train_items],float),
         "speed":np.array([i["speed"] for i in train_items],float)}
    yev={"position":np.array([i["initial_position"] for i in eval_items],float),
         "direction":np.array([i["direction"] for i in eval_items],float),
         "speed":np.array([i["speed"] for i in eval_items],float)}
    perm=np.random.default_rng(seed).permutation(len(st));st=st[perm];ytr={k:v[perm] for k,v in ytr.items()}
    mean=st.mean(0);scale=st.std(0);scale[scale<1e-8]=1.
    a=(st-mean)/scale;b=(se-mean)/scale;gram=a@a.T+1e-2*np.eye(len(a));rows=[]
    for name in ("position","direction","speed"):
        ym=float(ytr[name].mean());coef=np.linalg.solve(gram,ytr[name]-ym);pred=b@a.T@coef+ym
        if name=="position":metric="mae_px";value=float(np.mean(np.abs(pred-yev[name])))
        elif name=="direction":metric="accuracy";value=float(np.mean(np.sign(pred)==yev[name]))
        else:
            levels=np.array(sorted(set(ytr[name])));pred=levels[np.argmin(np.abs(pred[:,None]-levels),axis=1)]
            metric="exact_accuracy";value=float(np.mean(pred==yev[name]))
        rows.append({"factor":name,"metric":metric,"value":value,"n_probe_train":len(st),"n_eval":len(se)})
    return rows


def _write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="raise");writer.writeheader();writer.writerows(rows)

