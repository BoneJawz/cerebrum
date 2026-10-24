"""Trainable recurrent state baselines for the compositional motion task.

The implementation uses NumPy and hand-derived reverse-mode gradients, so the
first adaptive benchmark adds no autodiff framework dependency.  Training is
supervised trajectory prediction; direction/speed labels are never inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import dataclasses
import json
import os
from pathlib import Path
import time
import tracemalloc

import numpy as np

from . import research_benchmark as rb
from .benchmark import _ablate
from .processors import sensory_features


TRAIN_HORIZON = 12
EXTRAPOLATION_HORIZON = 18
HIDDEN = 32
DT = 0.05
TAU = 0.1
GRAD_CLIP = 1.0


@dataclass
class TrainStats:
    epochs: int
    seconds: float
    validation_mae: float
    losses: list[float]
    max_gradient_norm: float = 0.0
    min_gradient_norm: float = 0.0


class TrainableStateModel:
    """Small shared-decoder recurrent/continuous-time state model.

    ``kind`` is F (discrete), G (Euler-integrated), or H (G with optional
    inference event threshold).  An episode always starts with a zero state.
    """

    def __init__(self, input_dim: int, hidden: int = HIDDEN, seed: int = 0,
                 kind: str = "F", event_threshold: float | None = None,
                 dt: float = DT, tau: float = TAU):
        if kind not in {"F", "G", "H"}:
            raise ValueError("kind must be F, G, or H")
        self.kind, self.hidden = kind, hidden
        interval=1.0/20.0
        substeps=interval/dt if dt>0 else 0
        if tau <= 0 or dt <= 0 or dt / tau > 1.0 or not np.isclose(substeps,round(substeps)):
            raise ValueError("Euler dt must divide a 0.05 s motion interval and dt/tau must be <= 1")
        self.dt, self.tau = float(dt), float(tau)
        self.substeps = 1 if kind == "F" else int(round(substeps))
        self.event_threshold = event_threshold if kind == "H" else None
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(max(1, hidden))
        self.params = {
            "wx": rng.normal(0, 0.08 / np.sqrt(input_dim), (input_dim, hidden)),
            "wh": rng.normal(0, scale * 0.4, (hidden, hidden)),
            "b": np.zeros(hidden),
            "a": rng.normal(0, scale * 0.3, (hidden, hidden)),
            "bd": np.zeros(hidden),
            "dec": rng.normal(0, scale, hidden),
            "out": np.zeros(1),
        }
        self.opt_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.opt_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.opt_step = 0
        self.input_mean = np.zeros(input_dim)
        self.input_scale = np.ones(input_dim)
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.updates = 0
        self.opportunities = 0
        self.skipped = 0
        self.trigger_seconds = 0.0
        self.state_seconds = 0.0

    @property
    def parameter_count(self):
        return sum(v.size for v in self.params.values())

    def _transition(self, h):
        p = self.params
        q = np.tanh(h @ p["a"] + p["bd"])
        if self.kind in {"G", "H"}:
            alpha = self.dt / self.tau
            return h + alpha * (q - h), q, alpha
        return q, q, 1.0

    def _encode(self, x):
        p = self.params
        h = np.zeros((x.shape[1], self.hidden))
        states = []
        for xt in x:
            h = np.tanh(xt @ p["wx"] + h @ p["wh"] + p["b"])
            states.append(h)
        return h, states

    def predict_rollout(self, x, horizons=(3, 6, 9, 12, 18),
                        event_threshold=None, measure=False):
        """Process observed inputs, then evolve autonomously with shared weights."""
        x = (np.asarray(x) - self.input_mean) / self.input_scale
        h, _ = self._encode(x.transpose(1, 0, 2))
        wanted = set(horizons)
        max_h = max(wanted)
        predictions = {}
        threshold = self.event_threshold if event_threshold is None else event_threshold
        self.updates = self.opportunities = self.skipped = 0
        self.trigger_seconds = self.state_seconds = 0.0
        for step in range(1, max_h + 1):
            for _substep in range(self.substeps):
                start = time.perf_counter()
                candidate, _, _ = self._transition(h)
                if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > 1e6:
                    raise FloatingPointError("non-finite or divergent state during autonomous rollout")
                state_start = time.perf_counter()
                if threshold is None:
                    h = candidate
                    self.updates += len(h)
                else:
                    self.opportunities += len(h)
                    trigger = np.linalg.norm(candidate - h, axis=1) > threshold
                    h = np.where(trigger[:, None], candidate, h)
                    self.updates += int(np.sum(trigger))
                    self.skipped += int(len(h) - np.sum(trigger))
                    self.trigger_seconds += max(0.0, state_start - start)
                self.state_seconds += time.perf_counter() - state_start
            if step in wanted:
                predictions[step] = (h @ self.params["dec"] + self.params["out"][0]) * self.target_scale + self.target_mean
        return predictions


class DirectTrajectoryBaseline:
    """Parameter-matched tanh MLP with one direct output per training horizon."""
    kind = "FF_direct"

    def __init__(self, input_dim, seed=0, hidden=8):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.params = {"wx":rng.normal(0,1/np.sqrt(input_dim*5),(input_dim*5,hidden)),
                       "b":np.zeros(hidden),
                       "out":rng.normal(0,1/np.sqrt(hidden),(hidden,TRAIN_HORIZON)),
                       "bo":np.zeros(TRAIN_HORIZON)}
        self.m={k:np.zeros_like(v) for k,v in self.params.items()}
        self.v={k:np.zeros_like(v) for k,v in self.params.items()}
        self.step=0
        self.input_mean=np.zeros(input_dim); self.input_scale=np.ones(input_dim)
        self.target_mean=0.; self.target_scale=1.
        self.parameter_count=sum(v.size for v in self.params.values())
        self.updates=0; self.opportunities=0; self.skipped=0

    def _state(self,x):
        return np.tanh(x.reshape(len(x),-1) @ self.params["wx"] + self.params["b"])

    def _encode(self,time_major_x):
        state=self._state(time_major_x.transpose(1,0,2))
        return state,[state]

    def predict_rollout(self,x,horizons=(3,6,9,12)):
        z=(np.asarray(x)-self.input_mean)/self.input_scale
        out=self._state(z)@self.params["out"]+self.params["bo"]
        self.updates=len(x); self.opportunities=self.skipped=0
        return {h:out[:,h-1]*self.target_scale+self.target_mean for h in horizons if h<=TRAIN_HORIZON}

    def train(self,x,y,seed,epochs=80,batch_size=32,lr=.003,val_x=None,val_y=None):
        raw_x,raw_y=x,y
        self.input_mean=x.mean((0,1)); self.input_scale=x.std((0,1)); self.input_scale[self.input_scale<1e-8]=1.
        self.target_mean=float(y.mean()); self.target_scale=float(y.std()) or 1.
        x=(x-self.input_mean)/self.input_scale; y=(y-self.target_mean)/self.target_scale
        rng=np.random.default_rng(seed); losses=[]; start=time.perf_counter(); max_grad=0.; min_grad=float("inf")
        for _ in range(epochs):
            order=rng.permutation(len(x)); total=0.
            for begin in range(0,len(order),batch_size):
                ids=order[begin:begin+batch_size]; xb=x[ids].reshape(len(ids),-1); yb=y[ids]
                z=np.tanh(xb@self.params["wx"]+self.params["b"])
                err=z@self.params["out"]+self.params["bo"]-yb
                total+=float(np.mean(err**2))*len(ids)
                d=2*err/(len(ids)*TRAIN_HORIZON)
                grads={"out":z.T@d,"bo":d.sum(0)}
                dz=(d@self.params["out"].T)*(1-z*z)
                grads["wx"]=xb.T@dz; grads["b"]=dz.sum(0)
                norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
                max_grad=max(max_grad,float(norm)); min_grad=min(min_grad,float(norm))
                if norm>GRAD_CLIP:
                    grads={k:g*GRAD_CLIP/norm for k,g in grads.items()}
                self.step+=1
                for k,g in grads.items():
                    self.m[k]=.9*self.m[k]+.1*g; self.v[k]=.999*self.v[k]+.001*g*g
                    mh=self.m[k]/(1-.9**self.step); vh=self.v[k]/(1-.999**self.step)
                    self.params[k]-=lr*mh/(np.sqrt(vh)+1e-8)
            losses.append(total/len(x))
        vx=raw_x if val_x is None else val_x
        vy=raw_y if val_y is None else val_y
        pred=self.predict_rollout(vx,(3,6,9,12))
        return TrainStats(epochs,time.perf_counter()-start,float(np.mean([np.mean(np.abs(pred[h]-vy[:,h-1])) for h in (3,6,9,12)])),losses,max_grad,min_grad)


def prepare_examples(examples, modality, pixels):
    xs = []
    ys = []
    for item in examples:
        obs = [_ablate(o, modality) for o in item["observations"][:5]]
        xs.append(np.stack([sensory_features(o) for o in obs]))
        # Ground-truth targets for each future interval are labels only.
        ys.append([item["targets"][h] for h in range(1, TRAIN_HORIZON + 1)])
    return np.asarray(xs), np.asarray(ys, dtype=float)


def _batch_grad(model, xb, yb):
    """BPTT gradients through the observed encoder and autonomous rollout."""
    p, batch = model.params, len(xb)
    state, enc_states = model._encode(xb.transpose(1, 0, 2))
    enc_inputs = xb.transpose(1, 0, 2)
    cache = []
    trans_states = [state]
    # Integrate each 0.05 s motion interval using dt-sized Euler substeps.
    for _ in range(TRAIN_HORIZON):
        for _substep in range(model.substeps):
            nxt, q, alpha = model._transition(state)
            if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > 1e6:
                raise FloatingPointError("non-finite or divergent state during BPTT")
            cache.append((state, q, alpha))
            state = nxt
            trans_states.append(state)
    y_norm = (yb - model.target_mean) / model.target_scale
    pred = np.stack([trans_states[h*model.substeps] @ p["dec"] + p["out"][0]
                     for h in range(1, TRAIN_HORIZON + 1)], axis=1)
    err = pred - y_norm
    loss = float(np.mean(err ** 2))
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dh = np.zeros((batch, model.hidden))
    # Equal trajectory weight at every step.
    factor = 2.0 / (batch * TRAIN_HORIZON)
    for index in range(TRAIN_HORIZON*model.substeps-1,-1,-1):
        transition_step=index+1
        if transition_step % model.substeps == 0:
            horizon=transition_step//model.substeps
            e=err[:,horizon-1]*factor
            h=trans_states[transition_step]
            grads["dec"] += h.T @ e
            grads["out"][0] += np.sum(e)
            dh += e[:,None]*p["dec"][None,:]
        prev, q, alpha = cache[index]
        if model.kind in {"G", "H"}:
            dq = dh * alpha
            dh = dh * (1.0 - alpha)
        else:
            dq = dh
            dh = np.zeros_like(dh)
        dz = dq * (1.0 - q * q)
        grads["a"] += prev.T @ dz
        grads["bd"] += np.sum(dz, axis=0)
        dh += dz @ p["a"].T
    # Backpropagation through the five sensory-driven recurrent updates.
    for t in range(4, -1, -1):
        ht = enc_states[t]
        hp = np.zeros_like(ht) if t == 0 else enc_states[t - 1]
        dz = dh * (1.0 - ht * ht)
        grads["wx"] += enc_inputs[t].T @ dz
        grads["wh"] += hp.T @ dz
        grads["b"] += np.sum(dz, axis=0)
        dh = dz @ p["wh"].T
    norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > GRAD_CLIP:
        scale = GRAD_CLIP / (norm + 1e-12)
        for g in grads.values():
            g *= scale
    return loss, grads, norm


def train_model(model, train_x, train_y, val_x, val_y, seed, epochs=60,
                batch_size=32, learning_rate=0.003):
    model.input_mean = np.mean(train_x, axis=(0, 1))
    model.input_scale = np.std(train_x, axis=(0, 1))
    model.input_scale[model.input_scale < 1e-8] = 1.0
    model.target_mean = float(np.mean(train_y))
    model.target_scale = float(np.std(train_y)) or 1.0
    train_x = (train_x - model.input_mean) / model.input_scale
    rng = np.random.default_rng(seed)
    losses = []
    max_gradient_norm=0.0
    min_gradient_norm=float("inf")
    start = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(train_x))
        epoch_loss = 0.0
        for begin in range(0, len(order), batch_size):
            ids = order[begin:begin + batch_size]
            loss, grads, grad_norm = _batch_grad(model, train_x[ids], train_y[ids])
            max_gradient_norm=max(max_gradient_norm,grad_norm)
            min_gradient_norm=min(min_gradient_norm,grad_norm)
            model.opt_step += 1
            t = model.opt_step
            for key, grad in grads.items():
                model.opt_m[key] = 0.9 * model.opt_m[key] + 0.1 * grad
                model.opt_v[key] = 0.999 * model.opt_v[key] + 0.001 * grad * grad
                m = model.opt_m[key] / (1 - 0.9 ** t)
                v = model.opt_v[key] / (1 - 0.999 ** t)
                model.params[key] -= learning_rate * m / (np.sqrt(v) + 1e-8)
            epoch_loss += loss * len(ids)
        losses.append(epoch_loss / len(train_x))
    predictions = model.predict_rollout(val_x, horizons=(3, 6, 9, 12))
    val_errors = [np.mean(np.abs(predictions[h] - val_y[:, h - 1]))
                  for h in (3, 6, 9, 12)]
    if not np.all(np.isfinite(val_errors)):
        raise FloatingPointError("non-finite validation prediction")
    return TrainStats(epochs, time.perf_counter() - start, float(np.mean(val_errors)), losses,max_gradient_norm,min_gradient_norm)


def save_checkpoint(path, model, config):
    payload = {f"p_{k}": v for k, v in model.params.items()}
    payload.update(input_mean=model.input_mean, input_scale=model.input_scale,
                   target_mean=np.array(model.target_mean),
                   target_scale=np.array(model.target_scale))
    payload["config_json"] = np.array(json.dumps(config, sort_keys=True))
    np.savez_compressed(path, **payload)


def load_checkpoint(path, kind="F"):
    data = np.load(path, allow_pickle=False)
    first = data["p_wx"]
    if kind == "FF_direct":
        model=DirectTrajectoryBaseline(first.shape[0]//5,seed=0,hidden=first.shape[1])
        model.params={k:data[f"p_{k}"] for k in ("wx","b","out","bo")}
        model.input_mean,model.input_scale=data["input_mean"],data["input_scale"]
        model.target_mean,model.target_scale=float(data["target_mean"]),float(data["target_scale"])
        return model
    model = TrainableStateModel(first.shape[0], first.shape[1], kind=kind)
    model.params = {k: data[f"p_{k}"] for k in ("wx", "wh", "b", "a", "bd", "dec", "out")}
    model.input_mean, model.input_scale = data["input_mean"], data["input_scale"]
    model.target_mean, model.target_scale = float(data["target_mean"]), float(data["target_scale"])
    return model


def latent_probe_rows(model, train_x, train_items, eval_x, eval_items, seed):
    """Post-hoc probes on frozen states; probe labels do not update F/G/H."""
    def states(x):
        z = (x - model.input_mean) / model.input_scale
        return model._encode(z.transpose(1, 0, 2))[0]
    tr, ev = states(train_x), states(eval_x)
    targets = {
        "position": (np.array([it["initial_position"] for it in train_items], float),
                     np.array([it["initial_position"] for it in eval_items], float)),
        "direction": (np.array([it["combination"][0] for it in train_items], float),
                      np.array([it["combination"][0] for it in eval_items], float)),
        "speed": (np.array([it["combination"][1] for it in train_items], float),
                  np.array([it["combination"][1] for it in eval_items], float)),
    }
    order = np.random.default_rng(seed).permutation(len(tr)); tr = tr[order]
    mean, scale = tr.mean(0), tr.std(0); scale[scale < 1e-8] = 1.0
    a, b = (tr - mean) / scale, (ev - mean) / scale
    gram = a @ a.T + 1e-2 * np.eye(len(a))
    rows = []
    for name, (ytr, yev) in targets.items():
        ytr = ytr[order]
        y_mean=float(np.mean(ytr))
        coef = np.linalg.solve(gram, ytr-y_mean)
        pred = b @ a.T @ coef + y_mean
        if name == "direction":
            value, metric = float(np.mean(np.sign(pred) == yev)), "accuracy"
        elif name == "speed":
            levels = np.array(sorted(set(ytr)))
            pred = levels[np.argmin(np.abs(pred[:, None] - levels), axis=1)]
            value, metric = float(np.mean(pred == yev)), "exact_accuracy"
        else:
            value, metric = float(np.mean(np.abs(pred - yev))), "mae_px"
        rows.append({"probe":name,"metric":metric,"value":value,"n_train":len(tr),"n_eval":len(ev)})
    return rows


def latent_cluster_row(model, eval_x, eval_items):
    """Descriptive silhouette-like score for command-pair grouping of frozen states."""
    z=(eval_x-model.input_mean)/model.input_scale
    states=model._encode(z.transpose(1,0,2))[0]
    if len(states)<3:
        return {"mean_within_distance":float("nan"),"mean_nearest_between_distance":float("nan"),"mean_silhouette":float("nan")}
    states=states-states.mean(0)
    scale=states.std(0); scale[scale<1e-8]=1.
    states=states/scale
    delta=states[:,None,:]-states[None,:,:]
    dist=np.sqrt(np.sum(delta*delta,axis=2))
    labels=[tuple(it["combination"]) for it in eval_items]
    silhouettes=[]; within=[]; between=[]
    for i,label in enumerate(labels):
        same=np.array([value==label for value in labels]); same[i]=False
        if not np.any(same): continue
        a=float(np.mean(dist[i,same])); alternatives=[]
        for other in set(labels):
            if other==label: continue
            mask=np.array([value==other for value in labels])
            if np.any(mask): alternatives.append(float(np.mean(dist[i,mask])))
        if not alternatives: continue
        b=min(alternatives); within.append(a); between.append(b)
        silhouettes.append((b-a)/max(a,b,1e-12))
    return {"mean_within_distance":float(np.mean(within)),
            "mean_nearest_between_distance":float(np.mean(between)),
            "mean_silhouette":float(np.mean(silhouettes))}


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(results, outdir, learning_rows=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in results})
    horizons = [3, 6, 9, 12]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_size=max(r["train_size"] for r in results if r["split"]!="balanced_control")
    for model in models:
        rows = [r for r in results if r["model"] == model and r["split"] == "heldout" and r["train_size"] == max_size]
        if rows:
            ax.plot(horizons, [np.mean([r[f"mae_h{h}"] for r in rows]) for h in horizons], marker="o", label=model)
    ax.set(xlabel="Prediction horizon (motion intervals)", ylabel="Position MAE (pixels)", title="Adaptive model held-out rollout error")
    if ax.lines: ax.legend()
    fig.tight_layout(); fig.savefig(Path(outdir) / "adaptive_rollout_error.png", dpi=150); plt.close(fig)
    fig, ax=plt.subplots(figsize=(8,4.5))
    for model in models:
        rows=[r for r in results if r["model"]==model and r["split"]=="heldout" and r["modality"]=="combined"]
        sizes=sorted({r["train_size"] for r in rows})
        if sizes: ax.plot(sizes,[np.mean([r["mae_h9"] for r in rows if r["train_size"]==s]) for s in sizes],marker="o",label=model)
    ax.set(xscale="log",xlabel="Training episodes",ylabel="Held-out 9-step MAE (pixels)",title="Adaptive sample efficiency")
    if ax.lines: ax.legend()
    fig.tight_layout(); fig.savefig(Path(outdir)/"adaptive_learning_curves.png",dpi=150); plt.close(fig)
    if learning_rows:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for model in sorted({r["model"] for r in learning_rows}):
            rr=[r for r in learning_rows if r["model"]==model]
            epochs=sorted({r["epoch"] for r in rr})
            ax.plot(epochs,[np.mean([r["loss"] for r in rr if r["epoch"]==ep]) for ep in epochs],label=model)
        ax.set(xlabel="Training epoch",ylabel="Normalized trajectory MSE",title="Training loss")
        if ax.lines: ax.legend()
        fig.tight_layout(); fig.savefig(Path(outdir)/"adaptive_learning_curves.png",dpi=150); plt.close(fig)
    comp=Path(outdir)/"adaptive_compute_benchmark.csv"
    if comp.exists():
        import csv as _csv
        with comp.open(encoding="utf-8",newline="") as f: rows=list(_csv.DictReader(f))
        if rows:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for m in sorted({r["model"] for r in rows}):
                rr=[r for r in rows if r["model"]==m and r.get("threshold","")==""]
                if rr: ax.scatter([float(r["median_latency_ms"]) for r in rr],[float(r["mae_h9"]) for r in rr],label=m,alpha=.75)
            ax.set(xlabel="Median inference latency / episode (ms)",ylabel="Held-out 9-step MAE (pixels)",title="Prediction accuracy and compute")
            if ax.collections: ax.legend()
            fig.tight_layout(); fig.savefig(Path(outdir)/"adaptive_accuracy_vs_compute.png",dpi=150); plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4.5))
            for m in models:
                rr=[r for r in results if r["model"]==m and r["split"]=="heldout" and r["train_size"]==max_size]
                if rr: ax.plot((3,6,9,12),[np.mean([r[f"mae_h{h}"] for r in rr]) for h in (3,6,9,12)],marker="o",label=m)
            ax.set(xlabel="Prediction horizon",ylabel="Held-out MAE (pixels)",title="Recursive rollout error")
            if ax.lines: ax.legend()
            fig.tight_layout(); fig.savefig(Path(outdir)/"adaptive_generalization.png",dpi=150); plt.close(fig)


def run_adaptive(config=None, output_dir="results/adaptive"):
    """Run the new F-H experiment and write reproducible CSV/checkpoint artifacts."""
    config = config or rb.ResearchConfig()
    out = Path(output_dir); checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    adaptive_config = dataclasses.replace(config, horizons=tuple(range(1, TRAIN_HORIZON + 1)) + (EXTRAPOLATION_HORIZON,))
    results, sample_rows, compute_rows, probe_rows, learning_rows, position_rows, cluster_rows, timestep_rows = [], [], [], [], [], [], [], []
    models = ("F", "G", "H", "FF_direct")
    for seed_index in range(config.seeds):
        seed = config.seed + 1009 * seed_index
        split = rb.make_split(adaptive_config, seed)
        train_sizes = config.train_sizes
        for size in train_sizes:
            train_items = split.train_pool[:size]
            val_items = split.validation
            # Combined modality is primary; ablations are evaluated at the largest size.
            modalities = ("combined", "vision_only", "audio_only") if size == max(train_sizes) else ("combined",)
            for modality in modalities:
                train_x, train_y = prepare_examples(train_items, modality, split.world_config.pixels)
                val_x, val_y = prepare_examples(val_items, modality, split.world_config.pixels)
                for kind in models:
                    training_cpu_start=time.process_time()
                    if kind == "FF_direct":
                        model = DirectTrajectoryBaseline(train_x.shape[-1],seed=seed+size)
                        stats = model.train(train_x,train_y,seed+size,epochs=60 if size<500 else 80,val_x=val_x,val_y=val_y)
                    else:
                        model = TrainableStateModel(train_x.shape[-1], hidden=HIDDEN,
                                                    seed=seed + size, kind=kind)
                        stats = train_model(model, train_x, train_y, val_x, val_y,
                                            seed + size, epochs=60 if size < 500 else 80)
                    training_cpu=time.process_time()-training_cpu_start
                    training_n = min(32, len(train_x))
                    training_prediction = model.predict_rollout(train_x[:training_n], (9,))[9]
                    sample_rows.append({"seed":seed,"model":kind,"modality":modality,"train_size":size,
                                        "training_mae":float(np.mean(np.abs(training_prediction - train_y[:training_n, 8]))),
                                        "validation_mae":stats.validation_mae,"train_seconds":stats.seconds,
                                        "train_cpu_seconds":training_cpu,"max_gradient_norm":stats.max_gradient_norm,
                                        "min_gradient_norm":stats.min_gradient_norm,
                                        "parameters":model.parameter_count,"epochs":stats.epochs,
                                        "last_loss":stats.losses[-1]})
                    learning_rows.extend({"seed":seed,"model":kind,"modality":modality,
                                          "train_size":size,"epoch":ep,"loss":loss}
                                         for ep,loss in enumerate(stats.losses,1))
                    if modality != "combined":
                        continue
                    for split_name, items in (("observed", split.observed_test), ("heldout", split.heldout_test)):
                        x_eval, y_eval = prepare_examples(items, modality, split.world_config.pixels)
                        tracemalloc.start()
                        requested_horizons=(3,6,9,12) if kind == "FF_direct" else (3,6,9,12,18)
                        timing_horizons=(3,6,9,12)
                        full_predictions=model.predict_rollout(x_eval,horizons=requested_horizons)
                        _, peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                        repeats=max(5,config.timed_repeats)
                        model.predict_rollout(x_eval[:1],horizons=timing_horizons)
                        cpu_start=time.process_time(); wall_start=time.perf_counter()
                        for _ in range(repeats): predictions=model.predict_rollout(x_eval,horizons=timing_horizons)
                        wall=(time.perf_counter()-wall_start)/repeats; cpu=(time.process_time()-cpu_start)/repeats
                        # Per-episode samples support median and tail latency reporting.
                        episode_ms=[]
                        for ix in range(len(x_eval)):
                            t0=time.perf_counter(); model.predict_rollout(x_eval[ix:ix+1],timing_horizons)
                            episode_ms.append((time.perf_counter()-t0)*1000)
                        yh=full_predictions[9]
                        for initial in sorted({it["initial_position"] for it in items}):
                            ids_pos=np.array([it["initial_position"]==initial for it in items])
                            position_rows.append({"seed":seed,"model":kind,"split":split_name,
                                "train_size":size,"initial_position":initial,"n":int(ids_pos.sum()),
                                "mae_h9":float(np.mean(np.abs(yh[ids_pos]-y_eval[ids_pos,8])))})
                        compute_rows.append({"seed":seed,"model":kind,"threshold":"","split":split_name,
                            "mae_h9":float(np.mean(np.abs(yh-y_eval[:,8]))),"parameters":model.parameter_count,
                            "opportunities":model.opportunities,"executed_updates":model.updates,"skipped_updates":model.skipped,
                            "skip_fraction":model.skipped/max(1,model.opportunities),"cpu_seconds":cpu,"wall_seconds":wall,
                            "median_latency_ms":float(np.percentile(episode_ms,50)),"p95_latency_ms":float(np.percentile(episode_ms,95)),
                            "p99_latency_ms":float(np.percentile(episode_ms,99)),"peak_tracemalloc_bytes":peak,
                            "trigger_seconds":getattr(model,"trigger_seconds",0.0),"state_update_seconds":getattr(model,"state_seconds",0.0)})
                        for combo in sorted(set(i["combination"] for i in items)):
                            ids=np.array([i["combination"]==combo for i in items])
                            row={"seed":seed,"model":kind,"modality":modality,"split":split_name,
                                 "combination":f"{combo[0]}:{combo[1]}","train_size":size,
                                 "parameters":model.parameter_count,"train_seconds":stats.seconds,
                                 "cpu_seconds":cpu,"latency_ms":1000*wall/max(1,len(items)),
                                 "peak_tracemalloc_bytes":peak,"updates":model.updates,
                                 "event_opportunities":model.opportunities,"event_skipped":model.skipped}
                            for h in (3,6,9,12): row[f"mae_h{h}"]=float(np.mean(np.abs(full_predictions[h][ids]-y_eval[ids,h-1])))
                            if 18 in full_predictions:
                                actual18=np.array([i["targets"][18] for i,m in zip(items,ids) if m])
                                row["mae_h18"]=float(np.mean(np.abs(full_predictions[18][ids]-actual18)))
                            else:
                                row["mae_h18"]=float("nan")
                            results.append(row)
                    if size == max(train_sizes) and modality == "combined":
                        probe_train_x, _ = prepare_examples(train_items, "combined", split.world_config.pixels)
                        for probe_name, probe_items in (("observed", split.observed_test), ("heldout", split.heldout_test)):
                            probe_x, _ = prepare_examples(probe_items, "combined", split.world_config.pixels)
                            for probe in latent_probe_rows(model, probe_train_x, train_items, probe_x, probe_items, seed):
                                probe_rows.append({"seed":seed,"model":kind,"split":probe_name,**probe})
                            cluster_rows.append({"seed":seed,"model":kind,"split":probe_name,
                                **latent_cluster_row(model,probe_x,probe_items),
                                "n_eval":len(probe_items),"grouping":"direction_speed_pair"})
                        if kind == "G":
                            fine=TrainableStateModel(train_x.shape[-1],hidden=HIDDEN,seed=seed,kind="G",dt=0.025,tau=TAU)
                            fine.params={k:v.copy() for k,v in model.params.items()}
                            fine.input_mean=model.input_mean.copy();fine.input_scale=model.input_scale.copy()
                            fine.target_mean=model.target_mean;fine.target_scale=model.target_scale
                            coarse_pred=model.predict_rollout(x_eval,(9,18))
                            fine_pred=fine.predict_rollout(x_eval,(9,18))
                            timestep_rows.append({"seed":seed,"dt_coarse":DT,"dt_fine":0.025,"fine_substeps":fine.substeps,
                                "coarse_mae_h9":float(np.mean(np.abs(coarse_pred[9]-y_eval[:,8]))),
                                "fine_mae_h9":float(np.mean(np.abs(fine_pred[9]-y_eval[:,8]))),
                                "mean_abs_prediction_change_h9":float(np.mean(np.abs(fine_pred[9]-coarse_pred[9])))})
                    if size == max(train_sizes) and kind == "H" and modality == "combined":
                        x_event, y_event = prepare_examples(split.heldout_test, "combined", split.world_config.pixels)
                        target9 = y_event[:, 8]
                        for threshold in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
                            model.predict_rollout(x_event[:1],(9,),event_threshold=threshold)
                            cpu0, wall0 = time.process_time(), time.perf_counter()
                            event_pred = model.predict_rollout(x_event, (9,), event_threshold=threshold)[9]
                            cpu_s, wall_s = time.process_time()-cpu0, time.perf_counter()-wall0
                            opportunities, executed, skipped = model.opportunities, model.updates, model.skipped
                            trigger_s, update_s = model.trigger_seconds, model.state_seconds
                            per_episode = []
                            for ix in range(len(x_event)):
                                t0 = time.perf_counter()
                                model.predict_rollout(x_event[ix:ix+1], (9,), event_threshold=threshold)
                                per_episode.append(1000*(time.perf_counter()-t0))
                            compute_rows.append({"seed":seed,"model":"H","threshold":threshold,
                                "split":"heldout","mae_h9":float(np.mean(np.abs(event_pred-target9))),
                                "parameters":model.parameter_count,"peak_tracemalloc_bytes":"",
                                "opportunities":opportunities,"executed_updates":executed,"skipped_updates":skipped,
                                "skip_fraction":skipped/max(1,opportunities),"cpu_seconds":cpu_s,"wall_seconds":wall_s,
                                "median_latency_ms":float(np.percentile(per_episode,50)),
                                "p95_latency_ms":float(np.percentile(per_episode,95)),
                                "p99_latency_ms":float(np.percentile(per_episode,99)),
                                "trigger_seconds":trigger_s,"state_update_seconds":update_s})
                    if size == max(train_sizes) and modality == "combined":
                        all_pairs=[(d,v) for d in (-1,1) for v in config.velocities]
                        control_train=rb._generate_examples(split.world,config.generalization_train_episodes,
                            seed+91,all_pairs,split.phase_by_direction,adaptive_config.horizons,split.world_config.pixels)
                        control_val=rb._generate_examples(split.world,config.validation_per_combo*len(all_pairs),
                            seed+92,all_pairs,split.phase_by_direction,adaptive_config.horizons,split.world_config.pixels)
                        control_test=rb._generate_examples(split.world,config.test_per_combo*len(all_pairs),
                            seed+93,all_pairs,split.phase_by_direction,adaptive_config.horizons,split.world_config.pixels)
                        cx,cy=prepare_examples(control_train,"combined",split.world_config.pixels)
                        cvx,cvy=prepare_examples(control_val,"combined",split.world_config.pixels)
                        if kind == "FF_direct":
                            control_model=DirectTrajectoryBaseline(cx.shape[-1],seed=seed+91)
                            control_stats=control_model.train(cx,cy,seed+91,epochs=80,val_x=cvx,val_y=cvy)
                        else:
                            control_model=TrainableStateModel(cx.shape[-1],hidden=HIDDEN,seed=seed+91,kind=kind)
                            control_stats=train_model(control_model,cx,cy,cvx,cvy,seed+91,epochs=80)
                        save_checkpoint(checkpoints/f"seed-{seed}-{kind}-balanced-{config.generalization_train_episodes}.npz",
                            control_model,{"seed":seed,"training_seed":seed+91,"kind":kind,
                            "distribution":"all_direction_speed_pairs","train_size":config.generalization_train_episodes,
                            "hidden":control_model.hidden,"dt":getattr(control_model,"dt",None),"tau":getattr(control_model,"tau",None),
                            "epochs":control_stats.epochs,"learning_rate":0.003,"batch_size":32,
                            "gradient_clip":GRAD_CLIP,"horizons":list(range(1,13)),
                            "prediction_horizons":[3,6,9,12,18],"benchmark":dataclasses.asdict(config),
                            "world_pixels":split.world_config.pixels})
                        tx,ty=prepare_examples(control_test,"combined",split.world_config.pixels)
                        control_horizons=(3,6,9,12,18) if kind!="FF_direct" else (3,6,9,12)
                        cp=control_model.predict_rollout(tx,control_horizons)
                        for combo in sorted(set(it["combination"] for it in control_test)):
                            ids=np.array([it["combination"]==combo for it in control_test])
                            cr={"seed":seed,"model":kind,"modality":"combined","split":"balanced_control",
                                "combination":f"{combo[0]}:{combo[1]}","train_size":config.generalization_train_episodes,
                                "parameters":control_model.parameter_count,"train_seconds":control_stats.seconds}
                            for h in (3,6,9,12): cr[f"mae_h{h}"]=float(np.mean(np.abs(cp[h][ids]-ty[ids,h-1])))
                            if 18 in cp:
                                actual18=np.array([it["targets"][18] for it,m in zip(control_test,ids) if m])
                                cr["mae_h18"]=float(np.mean(np.abs(cp[18][ids]-actual18)))
                            else:
                                cr["mae_h18"]=float("nan")
                            results.append(cr)
                    if modality == "combined" or size == max(train_sizes):
                        ckpt=checkpoints/f"seed-{seed}-{kind}-size-{size}-{modality}.npz"
                        save_checkpoint(ckpt,model,{"seed":seed,"training_seed":seed+size,"kind":kind,
                            "train_size":size,"modality":modality,"dt":getattr(model,"dt",None),
                            "tau":getattr(model,"tau",None),"hidden":model.hidden,
                            "epochs":stats.epochs,"learning_rate":0.003,"batch_size":32,
                            "gradient_clip":GRAD_CLIP,"horizons":list(range(1,13)),
                            "prediction_horizons":[3,6,9,12,18],"benchmark":dataclasses.asdict(config),
                            "world_pixels":split.world_config.pixels})
    _write_csv(out/"adaptive_dynamics_results.csv",results)
    _write_csv(out/"adaptive_sample_efficiency.csv",sample_rows)
    _write_csv(out/"adaptive_compute_benchmark.csv",compute_rows)
    _write_csv(out/"adaptive_latent_probes.csv",probe_rows)
    _write_csv(out/"adaptive_latent_cluster_diagnostics.csv",cluster_rows)
    _write_csv(out/"adaptive_timestep_sensitivity.csv",timestep_rows)
    _write_csv(out/"adaptive_initial_position_results.csv",position_rows)
    paired_rows=[]
    for model in ("G","H","FF_direct"):
        for split_name in ("observed","heldout"):
            for seed in sorted({r["seed"] for r in results if r["split"]==split_name and r["train_size"]==max(train_sizes)}):
                def seed_mean(which):
                    rr=[r["mae_h9"] for r in results if r["seed"]==seed and r["model"]==which and r["split"]==split_name and r["train_size"]==max(train_sizes) and r["modality"]=="combined"]
                    return float(np.mean(rr)) if rr else float("nan")
                paired_rows.append({"seed":seed,"split":split_name,"comparison":f"F_minus_{model}",
                                    "difference_h9_px":seed_mean("F")-seed_mean(model)})
    _write_csv(out/"adaptive_paired_differences.csv",paired_rows)
    _write_csv(out/"adaptive_learning_curves.csv",learning_rows)
    _plot(results,out,learning_rows)
    from .adaptive_report import format_report
    report = format_report(results,sample_rows,probe_rows,compute_rows,config,
                          recurrent_parameters=TrainableStateModel(246).parameter_count,
                          direct_parameters=DirectTrajectoryBaseline(246).parameter_count,
                          cluster_rows=cluster_rows,timestep_rows=timestep_rows)
    (out/"ADAPTIVE_DYNAMICS_REPORT.md").write_text(report,encoding="utf-8")
    return out
