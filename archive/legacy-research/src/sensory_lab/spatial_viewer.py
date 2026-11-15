"""Interactive, truth-gated viewer for static spatial grounding experiments."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.parse import urlparse

import numpy as np

from .adaptive_dynamics import DirectTrajectoryBaseline, TrainableStateModel, train_model
from .contracts import Observation
from .liquid import LiquidConfig, LiquidNetwork
from .liquid_readout import RidgeReadout
from .processors import PROCESSORS, sensory_features
from .spatial_environment import (
    CHUNK_SIZE, SAMPLE_RATE, DigitSpatialVocabulary, make_digit_episode,
    visual_array,
)

VIEW_MIN = -99
VIEW_MAX = 99
VIEW_POSITIONS = tuple(range(VIEW_MIN, VIEW_MAX + 1))


MODEL_CATALOG = [
    {"id": "A", "group": "A-D", "name": "Feedforward temporal fusion", "learning": "fixed processor + supervised ridge readout"},
    {"id": "B", "group": "A-D", "name": "Recurrent fusion", "learning": "fixed processor + supervised ridge readout"},
    {"id": "C", "group": "A-D", "name": "Coupled continuous-time dynamics", "learning": "fixed processor + supervised ridge readout"},
    {"id": "D", "group": "A-D", "name": "Event-triggered coupled dynamics", "learning": "fixed processor + supervised ridge readout"},
    {"id": "E1", "group": "E", "name": "Raw-feature linear ridge", "learning": "supervised ridge on generic waveform features"},
    {"id": "FF_direct", "group": "E control", "name": "Direct trainable MLP control", "learning": "supervised Adam; direct static-position outputs"},
    {"id": "F", "group": "F-H", "name": "Trainable recurrent state", "learning": "supervised BPTT + Adam; static target across rollout"},
    {"id": "G", "group": "F-H", "name": "Trainable Euler-integrated state", "learning": "supervised BPTT + Adam; static target across rollout"},
    {"id": "H", "group": "F-H", "name": "Trainable state with event threshold", "learning": "supervised BPTT + Adam; event-gated inference"},
    {"id": "I0", "group": "I", "name": "Fixed liquid", "learning": "fixed graph exposure + supervised ridge readout"},
    {"id": "I1", "group": "I", "name": "Weight-adaptive liquid", "learning": "local weight plasticity + supervised ridge readout"},
    {"id": "I2", "group": "I", "name": "Structurally adaptive liquid", "learning": "local weight and topology plasticity + supervised ridge readout"},
    {"id": "I2_rewire_fixed", "group": "I", "name": "Fixed-weight rewiring control", "learning": "structural plasticity only + supervised ridge readout"},
]

REFERENCE_MODELS = [
    {"id": "E2", "name": "Privileged oracle", "reason": "requires ground-truth state at prediction; not an eligible predictor"},
    {"id": "E3", "name": "Factorized known-rule diagnostic", "reason": "depends on a hand-specified interpretation/rule, not learned spatial grounding"},
]


def _features_for(processor_type, observations, seed, pixels, hidden=64):
    processor = processor_type(pixels=pixels, hidden=hidden, seed=seed)
    processor.reset()
    feature = None
    for observation in observations:
        feature = processor.step(observation)
    return feature


def _liquid_state(net: LiquidNetwork, observations):
    net.reset_episode()
    for observation in observations:
        net.ingest_observation(observation)
    net.advance_to(observations[-1].t + 0.05)
    return net.snapshot()


def _mean_raw_features(observations):
    return np.mean([sensory_features(observation) for observation in observations], axis=0)


def _timed(call):
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    value = call()
    return value, (time.perf_counter_ns() - wall_start) / 1e6, (time.process_time_ns() - cpu_start) / 1e6


def _json_observations(observations):
    return [{"t": float(o.t), "vision": o.vision.astype(int).tolist(),
             "audio": o.audio.tolist()} for o in observations]


class SpatialViewerSession:
    """Owns a reproducible UI demonstration; test labels are reveal-only."""

    def __init__(self, seed=42, train_repetitions=1, test_per_position=1,
                 neural_epochs=4, liquid_config: LiquidConfig | None = None):
        if train_repetitions < 1 or test_per_position < 1 or neural_epochs < 1:
            raise ValueError("repetitions and epochs must be positive")
        self.seed = int(seed)
        self.train_repetitions = int(train_repetitions)
        self.test_per_position = int(test_per_position)
        self.neural_epochs = int(neural_epochs)
        # Keep the all-199-position browser demo responsive on a home CPU.
        # Research benchmark configurations are defined separately.
        self.liquid_config = liquid_config or LiquidConfig(
            neurons=24, initial_out_degree=3, max_out_degree=5,
            max_connections=128, max_sensory_targets=6)
        self.vocabulary = DigitSpatialVocabulary(seed=self.seed)
        self.field_min = VIEW_MIN
        self.field_max = VIEW_MAX
        self.field_size = self.field_max - self.field_min + 1
        self.train = []
        for position in VIEW_POSITIONS:
            for repetition in range(self.train_repetitions):
                self.train.append({
                    "position": position, "repetition": repetition,
                    "observations": make_digit_episode(position, self.vocabulary,
                        modality="both", field_min=self.field_min, field_max=self.field_max),
                })
        rng = np.random.default_rng(self.seed + 1)
        test_specs = [(p, n) for p in VIEW_POSITIONS for n in range(self.test_per_position)]
        order = rng.permutation(len(test_specs))
        self.test = []
        for idx in order:
            position, repetition = test_specs[int(idx)]
            self.test.append({
                "position": position, "repetition": repetition,
                "observations": make_digit_episode(position, self.vocabulary, modality="audio",
                    field_min=self.field_min, field_max=self.field_max),
            })
        self.models = None
        self.fit_seconds = None
        self.fit_rows = []
        self.predictions: dict[int, list[dict]] = {}
        self.revealed: set[int] = set()
        self._lock = threading.Lock()
        self._fitting = False

    def metadata(self):
        return {
            "seed": self.seed, "train_count": len(self.train),
            "test_count": len(self.test), "position_count": len(VIEW_POSITIONS),
            "position_min": self.field_min, "position_max": self.field_max,
            "field_cells": self.field_size,
            "sample_rate_hz": SAMPLE_RATE, "visual_rate_hz": 20,
            "chunk_size": CHUNK_SIZE,
            "episode_ticks": DigitSpatialVocabulary.INSTRUCTION_TICKS,
            "neural_epochs": self.neural_epochs,
            "liquid_neurons": self.liquid_config.neurons,
            "model_catalog": MODEL_CATALOG, "reference_models": REFERENCE_MODELS,
            "evaluation": "audio-only absolute position prediction after audiovisual training",
            "note": "Interactive single-seed demo; not the five-seed benchmark result.",
        }

    def training_example(self, index):
        if not 0 <= index < len(self.train):
            raise IndexError("training example index out of range")
        item = self.train[index]
        position = item["position"]
        tokens = self.vocabulary.token_sequence(position)
        return {"index": index, "position": position, "tokens": tokens,
                "repetition": item["repetition"] + 1,
                "observations": _json_observations(item["observations"])}

    def test_example(self, index):
        if not 0 <= index < len(self.test):
            raise IndexError("test example index out of range")
        item = self.test[index]
        # The label is stored privately and is not included in this payload.
        return {"index": index, "observations": _json_observations(item["observations"]),
                "input_modality": "audio only", "vision_withheld": True}

    def _fit_a_to_d(self):
        rows = []
        for code, processor_type in PROCESSORS.items():
            start = time.perf_counter()
            features = [_features_for(processor_type, item["observations"], self.seed,
                                      self.field_size)
                        for item in self.train]
            readout = RidgeReadout(alpha=1.0).fit(
                np.asarray(features), np.asarray([item["position"] for item in self.train]))
            rows.append({"id": code, "name": MODEL_CATALOG[ord(code) - ord("A")]["name"],
                         "learning_seconds": time.perf_counter() - start, "readout": readout,
                         "kind": processor_type, "readouts": 1})
        return rows

    def _fit_e1(self):
        start = time.perf_counter()
        x = np.asarray([_mean_raw_features(item["observations"]) for item in self.train])
        y = np.asarray([item["position"] for item in self.train])
        readout = RidgeReadout(alpha=1.0).fit(x, y)
        return {"id": "E1", "name": "Raw-feature linear ridge",
                "learning_seconds": time.perf_counter() - start, "readout": readout,
                "readouts": 1}

    def _fit_f_to_h(self):
        x = np.asarray([[sensory_features(o) for o in item["observations"]]
                        for item in self.train], dtype=np.float64)
        y_positions = np.asarray([item["position"] for item in self.train], dtype=np.float64)
        # This UI experiment asks the model to retain the grounded position
        # through its autonomous rollout. Labels remain training targets only.
        y = np.repeat(y_positions[:, None], 12, axis=1)
        rows = []
        for offset, kind in enumerate(("F", "G", "H")):
            model = TrainableStateModel(input_dim=x.shape[2], hidden=32,
                seed=self.seed + 101 + offset, kind=kind,
                event_threshold=0.04 if kind == "H" else None)
            start = time.perf_counter()
            train_model(model, x, y, x, y, seed=self.seed + 201 + offset,
                        epochs=self.neural_epochs, batch_size=min(16, len(x)))
            rows.append({"id": kind, "name": next(m["name"] for m in MODEL_CATALOG if m["id"] == kind),
                         "learning_seconds": time.perf_counter() - start,
                         "model": model, "readouts": 1})
        return rows

    def _fit_direct_control(self):
        x = np.asarray([[sensory_features(o) for o in item["observations"]]
                        for item in self.train], dtype=np.float64)
        positions = np.asarray([item["position"] for item in self.train], dtype=np.float64)
        y = np.repeat(positions[:, None], 12, axis=1)
        model = DirectTrajectoryBaseline(input_dim=x.shape[2], seed=self.seed + 87,
                                         hidden=8, sequence_length=x.shape[1])
        start = time.perf_counter()
        model.train(x, y, seed=self.seed + 187, epochs=self.neural_epochs,
                    batch_size=min(16, len(x)))
        return {"id": "FF_direct", "name": "Direct trainable MLP control",
                "learning_seconds": time.perf_counter() - start,
                "model": model, "readouts": 1}

    def _fit_i(self):
        rows = []
        input_dim = self.field_size + 4 + 16 + 49
        targets = np.asarray([item["position"] for item in self.train])
        for variant in ("I0", "I1", "I2", "I2_rewire_fixed"):
            net = LiquidNetwork(input_dim=input_dim, seed=self.seed,
                                config=self.liquid_config, variant=variant, visual_dim=self.field_size)
            states = []
            start = time.perf_counter()
            for item in self.train:
                states.append(_liquid_state(net, item["observations"]))
            adaptation_seconds = time.perf_counter() - start
            net.weight_plasticity = False
            net.structural_plasticity = False
            readout = RidgeReadout(alpha=1.0).fit(np.asarray(states), targets)
            rows.append({"id": variant, "name": next(m["name"] for m in MODEL_CATALOG if m["id"] == variant),
                         "learning_seconds": adaptation_seconds,
                         "net": net, "readout": readout, "readouts": 1})
        return rows

    def fit(self):
        with self._lock:
            if self.models is not None:
                return {"seconds": self.fit_seconds, "models": self.fit_rows, "already_fit": True}
            if self._fitting:
                raise RuntimeError("model fitting is already in progress")
            self._fitting = True
        start = time.perf_counter()
        try:
            fitted = []
            fitted.extend(self._fit_a_to_d())
            fitted.append(self._fit_e1())
            fitted.append(self._fit_direct_control())
            fitted.extend(self._fit_f_to_h())
            fitted.extend(self._fit_i())
            elapsed = time.perf_counter() - start
            self.models = {row["id"]: row for row in fitted}
            self.fit_rows = [{"id": row["id"], "name": row["name"],
                              "learning_seconds": float(row["learning_seconds"])} for row in fitted]
            self.fit_seconds = elapsed
            return {"seconds": elapsed, "models": self.fit_rows, "already_fit": False}
        finally:
            self._fitting = False

    def predict(self, index):
        if self.models is None:
            raise RuntimeError("train the models before asking for predictions")
        if not 0 <= index < len(self.test):
            raise IndexError("test example index out of range")
        observations = self.test[index]["observations"]
        predictions = []
        for code, processor_type in PROCESSORS.items():
            def infer_processor():
                feature = _features_for(processor_type, observations, self.seed,
                                        self.field_size)
                return float(self.models[code]["readout"].predict(feature[None, :])[0])
            position, wall_ms, cpu_ms = _timed(infer_processor)
            predictions.append({"id": code, "name": self.models[code]["name"],
                                "predicted_position": position, "latency_ms": wall_ms,
                                "cpu_ms": cpu_ms, "method": "fixed dynamics + AV-fitted readout"})
        def infer_e1():
            raw = _mean_raw_features(observations)
            return float(self.models["E1"]["readout"].predict(raw[None, :])[0])
        position, wall_ms, cpu_ms = _timed(infer_e1)
        predictions.append({"id": "E1", "name": self.models["E1"]["name"],
                            "predicted_position": position, "latency_ms": wall_ms,
                            "cpu_ms": cpu_ms, "method": "raw waveform features + AV-fitted readout"})
        for code in ("F", "G", "H"):
            model = self.models[code]["model"]
            def infer_state(model=model):
                seq = np.asarray([[sensory_features(o) for o in observations]], dtype=np.float64)
                return float(model.predict_rollout(seq, horizons=(12,))[12][0])
            position, wall_ms, cpu_ms = _timed(infer_state)
            predictions.append({"id": code, "name": self.models[code]["name"],
                                "predicted_position": position, "latency_ms": wall_ms,
                                "cpu_ms": cpu_ms, "method": "supervised state rollout; static target"})
        direct = self.models["FF_direct"]["model"]
        def infer_direct():
            seq = np.asarray([[sensory_features(o) for o in observations]], dtype=np.float64)
            return float(direct.predict_rollout(seq, horizons=(12,))[12][0])
        position, wall_ms, cpu_ms = _timed(infer_direct)
        predictions.append({"id": "FF_direct", "name": self.models["FF_direct"]["name"],
                            "predicted_position": position, "latency_ms": wall_ms,
                            "cpu_ms": cpu_ms, "method": "supervised direct MLP; static target"})
        for code in ("I0", "I1", "I2", "I2_rewire_fixed"):
            row = self.models[code]
            def infer_liquid(row=row):
                state = _liquid_state(row["net"], observations)
                return float(row["readout"].predict(state[None, :])[0])
            position, wall_ms, cpu_ms = _timed(infer_liquid)
            predictions.append({"id": code, "name": row["name"],
                                "predicted_position": position, "latency_ms": wall_ms,
                                "cpu_ms": cpu_ms, "method": "liquid state + AV-fitted readout"})
        self.predictions[index] = predictions
        return predictions

    def reveal(self, index):
        if index not in self.predictions:
            raise RuntimeError("ask the models for predictions before revealing the target")
        if not 0 <= index < len(self.test):
            raise IndexError("test example index out of range")
        item = self.test[index]
        position = item["position"]
        self.revealed.add(index)
        return {"position": position, "tokens": self._token_names(position),
                "visual": visual_array(position, self.field_min,
                                       self.field_max).astype(int).tolist(),
                "seed": self.seed}

    @staticmethod
    def _token_names(position):
        return DigitSpatialVocabulary().token_sequence(position)

    def scorecard(self):
        rows = []
        for index in self.revealed:
            predictions = self.predictions[index]
            truth = self.test[index]["position"]
            for row in predictions:
                error = abs(row["predicted_position"] - truth)
                rows.append({"model": row["id"], "position_error": error,
                             "within_one": error <= 1.0})
        return rows


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial Grounding · A-I Model Viewer</title>
<style>
:root{color-scheme:dark;--bg:#111820;--panel:#1a2630;--line:#30424e;--ink:#eef4f4;--muted:#a8bac2;--teal:#78ddc3;--gold:#f2c879;--red:#f08989}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1180px;margin:auto;padding:26px 18px 56px}h1{font-size:clamp(1.7rem,4vw,2.5rem);line-height:1.15}h2{font-size:1.25rem;margin:0 0 10px}.sub,.muted{color:var(--muted)}.sub{max-width:850px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:18px;margin:16px 0}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.spread{justify-content:space-between}.tabs{display:flex;gap:8px;margin:20px 0 10px}.tab,button{border:1px solid var(--line);border-radius:9px;background:#22323e;color:var(--ink);padding:9px 13px;font:inherit;cursor:pointer}.tab.active,button.primary{background:var(--teal);color:#09221c;border-color:var(--teal);font-weight:700}button:disabled{opacity:.48;cursor:wait}.badge{border:1px solid var(--line);border-radius:999px;padding:4px 10px;color:var(--muted);font-size:.88rem}.stage{display:none}.stage.active{display:block}.canvas{background:#0d141a;border:1px solid var(--line);border-radius:9px;padding:10px;margin:10px 0}.canvas canvas{display:block;width:100%;height:110px}.wave canvas{height:86px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin:12px 0}.card{background:#121c23;border:1px solid var(--line);padding:10px;border-radius:8px}.card b{display:block;color:var(--muted);font-size:.78rem;text-transform:uppercase}.notice{padding:11px 13px;background:#142b2a;border-left:3px solid var(--teal);border-radius:6px}.question{font-size:1.2rem;color:var(--gold);font-weight:700}.tablewrap{overflow:auto}.results{width:100%;border-collapse:collapse;margin-top:12px}.results th,.results td{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);white-space:nowrap}.results th{color:var(--muted);font-size:.84rem}.spinner{display:inline-block;width:15px;height:15px;border:2px solid #58716e;border-top-color:var(--teal);border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:8px}@keyframes spin{to{transform:rotate(360deg)}}.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}.model{border:1px solid var(--line);border-radius:8px;padding:9px;background:#121c23}.model strong{color:var(--teal)}footer{color:var(--muted);font-size:.88rem;margin-top:20px}@media(max-width:650px){main{padding:18px 11px}.panel{padding:13px}}
</style></head><body><main>
<h1>Spatial Grounding · A–I Model Viewer</h1>
<p class="sub">Inspect the paired audiovisual examples in order from −99 to +99, start model fitting, then ask every task-compatible architecture for an audio-only position prediction. Instructions use a sign token followed by the decimal digit tokens (for example, + → 8 → 7). The test retina is withheld so the visible target cannot leak into the prediction.</p>
<div class="row"><span id="seed" class="badge"></span><span id="counts" class="badge"></span><span id="field" class="badge"></span></div>
<div class="tabs"><button class="tab active" data-stage="training">Training examples</button><button class="tab" data-stage="testing">Audio-only predictions</button><button class="tab" data-stage="catalog">Model catalog</button></div>
<section id="training" class="stage active"><div class="panel">
<div class="row spread"><div><h2>Audiovisual training set</h2><div class="muted">The position and token meanings shown here are researcher-side ground truth. Models receive only the binary vision array and raw audio chunks; labels are used only by their permitted learning/readout rules.</div></div><div class="row"><button id="trPrev">Previous</button><span id="trCount" class="badge"></span><button id="trNext">Next</button></div></div>
<div class="canvas"><div class="muted">Vision · one active cell</div><canvas id="trVision" width="1050" height="110"></canvas></div><div class="canvas wave"><div class="muted">Audio · raw waveform</div><canvas id="trWave" width="1050" height="86"></canvas></div>
<div class="row"><button id="trPlay" class="primary">Animate and play example</button><span id="trMeta" class="muted"></span></div><div id="trTruth" class="cards"></div></div>
<div class="panel"><h2>Train the model set</h2><p class="muted">A–D keep their processor dynamics fixed and fit readouts. E1 fits a raw-feature ridge readout. F–H use supervised BPTT on a static position target across their rollout. I0/I1/I2 adapt the liquid during audiovisual exposure, then fit a separate ridge readout. This is a one-seed interactive demonstration, not the five-seed benchmark.</p><div class="row"><button id="fit" class="primary">Train A–I models</button><span id="fitStatus" class="muted">Waiting to train.</span></div><div id="fitRows" class="cards"></div></div></section>
<section id="testing" class="stage"><div class="panel"><div class="row spread"><div><h2>Audio-only test</h2><div class="muted">The visual channel is deliberately blank. The target position remains hidden until you reveal it.</div></div><div class="row"><button id="tePrev">Previous</button><span id="teCount" class="badge"></span><button id="teNext">Next</button></div></div>
<div class="canvas"><div class="muted">Vision · withheld during this prediction</div><canvas id="teVision" width="1050" height="72"></canvas></div><div class="canvas wave"><div class="muted">Audio · instruction waveform</div><canvas id="teWave" width="1050" height="86"></canvas></div><div class="row"><button id="tePlay">Play audio</button><span id="teMeta" class="muted"></span></div>
<p class="question">What position do you predict?</p><div class="row"><button id="ask" class="primary" disabled>Ask every trained model</button><span id="askStatus" class="muted">Train the model set first.</span></div><div id="predictions" class="tablewrap"></div><div class="row" style="margin-top:14px"><button id="reveal" disabled>Reveal true position</button></div><div id="truth" class="cards"></div><div id="truthCanvas" class="canvas" style="display:none"><div class="muted">Revealed visual target</div><canvas id="truthVision" width="1050" height="72"></canvas></div>
</div><div class="panel"><h2>Scorecard for revealed trials</h2><div id="score"></div></div></section>
<section id="catalog" class="stage"><div class="panel"><h2>Task-compatible predictors</h2><div id="modelCatalog" class="models"></div><h2 style="margin-top:22px">Reference-only models</h2><div id="referenceCatalog" class="models"></div></div><div class="notice">The current spatial benchmark measured Model I variants and a raw ridge control. A–D and F–H predictions here are produced by task-specific interactive fitting; they are not results from the saved five-seed benchmark. E2 is privileged and E3 assumes a hand-specified rule, so neither is presented as a fair learned predictor.</div></section>
<footer>Model input: raw waveform chunks and binary vision only. Training labels go to supervised readouts or learning targets, never into sensory observations. Test labels are returned only after a prediction request and the reveal action. Predictions are point estimates, not probabilities.</footer>
</main><script>
const $=id=>document.getElementById(id);let meta,trIndex=0,teIndex=0,tr,te,fitted=false,pred=null,revealed=new Map(),ctx;
async function get(path){const r=await fetch(path);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}async function post(path){const r=await fetch(path,{method:'POST'});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function retina(id,frames){const c=$(id),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);const cells=frames[0]?.vision||Array(meta.field_cells).fill(0),pad=20,cw=(w-pad*2)/cells.length;cells.forEach((v,i)=>{g.fillStyle=v?'#78ddc3':'#263640';g.fillRect(pad+i*cw,18,Math.max(1,cw-1),h-33)});g.fillStyle='#a8bac2';g.font='12px system-ui';g.fillText(String(meta.position_min),pad,h-4);g.fillText('0',w/2-4,h-4);g.fillText(`+${meta.position_max}`,w-pad-28,h-4)}
function waveform(id,frames){const c=$(id),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);const a=frames.flatMap(x=>x.audio),step=Math.max(1,Math.floor(a.length/w));g.strokeStyle='#78ddc3';g.beginPath();let started=false;for(let i=0;i<a.length;i+=step){const x=i/Math.max(1,a.length-1)*w,y=h/2-a[i]*(h*.42);if(!started){g.moveTo(x,y);started=true}else g.lineTo(x,y)}g.stroke();g.strokeStyle='#30424e';g.beginPath();g.moveTo(0,h/2);g.lineTo(w,h/2);g.stroke()}
function play(frames){if(!ctx)ctx=new(window.AudioContext||window.webkitAudioContext)();const a=frames.flatMap(x=>x.audio),n=Math.max(1,Math.round(a.length*ctx.sampleRate/meta.sample_rate_hz)),b=ctx.createBuffer(1,n,ctx.sampleRate),ch=b.getChannelData(0),ratio=meta.sample_rate_hz/ctx.sampleRate;for(let i=0;i<n;i++){const p=i*ratio,j=Math.min(a.length-1,Math.floor(p)),f=p-j;ch[i]=a[j]*(1-f)+a[Math.min(j+1,a.length-1)]*f}const s=ctx.createBufferSource();s.buffer=b;s.connect(ctx.destination);s.start()}
function card(title,value){return `<div class="card"><b>${title}</b>${value}</div>`}
async function loadTrain(){tr=await get(`/api/train/${trIndex}`);$('trCount').textContent=`${trIndex+1} / ${meta.train_count}`;retina('trVision',tr.observations);waveform('trWave',tr.observations);$('trMeta').textContent=`paired example · repetition ${tr.repetition}`;$('trTruth').innerHTML=card('True position',`${tr.position>0?'+':''}${tr.position}`)+card('Acoustic token sequence',tr.tokens.join(' → '));$('trPrev').disabled=trIndex===0;$('trNext').disabled=trIndex>=meta.train_count-1}
async function loadTest(){te=await get(`/api/test/${teIndex}`);pred=null;$('predictions').innerHTML='';$('truth').innerHTML='';$('truthCanvas').style.display='none';$('teCount').textContent=`${teIndex+1} / ${meta.test_count}`;retina('teVision',te.observations);waveform('teWave',te.observations);$('teMeta').textContent=`${te.observations.length} raw audio chunks · retina withheld`;$('tePrev').disabled=teIndex===0;$('teNext').disabled=teIndex>=meta.test_count-1;$('ask').disabled=!fitted;$('reveal').disabled=true;$('askStatus').textContent=fitted?'Ask for audio-only position predictions.':'Train the model set first.';if(revealed.has(teIndex)){showTruth(revealed.get(teIndex).truth);$('reveal').disabled=true}}
function renderPredictions(){if(!pred)return;const rows=pred.map(x=>`<tr><td><strong>${x.id}</strong> · ${x.name}</td><td>${x.predicted_position.toFixed(2)}</td><td>${x.method}</td>${revealed.has(teIndex)?`<td>${Math.abs(x.predicted_position-revealed.get(teIndex).truth.position).toFixed(2)}</td>`:''}</tr>`).join('');$('predictions').innerHTML=`<table class="results"><thead><tr><th>Model</th><th>Predicted position</th><th>Learning / readout</th>${revealed.has(teIndex)?'<th>Absolute error</th>':''}</tr></thead><tbody>${rows}</tbody></table>`}
function showTruth(t){$('truth').innerHTML=card('True position',`${t.position>0?'+':''}${t.position}`)+card('Correct acoustic concepts',t.tokens.join(' → '));retina('truthVision',[{vision:t.visual}]);$('truthCanvas').style.display='block'}
function renderScore(rows){if(!rows.length){$('score').innerHTML='<p class="muted">Reveal a test trial to begin scoring.</p>';return}const by={};for(const x of rows){(by[x.model]??=[]).push(x)}$('score').innerHTML=`<div class="tablewrap"><table class="results"><thead><tr><th>Model</th><th>Revealed trials</th><th>Mean absolute error</th><th>Within 1 position</th></tr></thead><tbody>${Object.entries(by).map(([k,v])=>`<tr><td>${k}</td><td>${v.length}</td><td>${(v.reduce((s,x)=>s+x.position_error,0)/v.length).toFixed(2)}</td><td>${(v.filter(x=>x.within_one).length/v.length).toFixed(2)}</td></tr>`).join('')}</tbody></table></div>`}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.stage').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.stage).classList.add('active')});
$('trPrev').onclick=async()=>{trIndex--;await loadTrain()};$('trNext').onclick=async()=>{trIndex++;await loadTrain()};$('trPlay').onclick=()=>play(tr.observations);$('tePlay').onclick=()=>play(te.observations);$('tePrev').onclick=async()=>{teIndex--;await loadTest()};$('teNext').onclick=async()=>{teIndex++;await loadTest()};
$('fit').onclick=async()=>{const b=$('fit');b.disabled=true;$('fitStatus').innerHTML='<span class="spinner"></span>Training readouts, recurrent state models, and liquid variants…';try{const r=await post('/api/fit');fitted=true;$('fitStatus').textContent=`Fit complete in ${r.seconds.toFixed(1)} s. ${r.models.length} predictors are ready; inspect per-model learning time below.`;$('fitRows').innerHTML=r.models.map(x=>card(`${x.id} · ${x.name}`,`${x.learning_seconds.toFixed(2)} s`)).join('');$('ask').disabled=false;$('askStatus').textContent='Models ready. Select an audio-only trial and ask.';await loadTest();document.querySelector('[data-stage="testing"]').click()}catch(e){$('fitStatus').textContent=e.message;b.disabled=false}};
$('ask').onclick=async()=>{const b=$('ask');b.disabled=true;$('askStatus').innerHTML='<span class="spinner"></span>Each model is processing the audio-only stream…';try{pred=await get(`/api/predict/${teIndex}`);renderPredictions();$('askStatus').textContent=`${pred.length} model predictions returned. The true position is still hidden.`;$('reveal').disabled=false}catch(e){$('askStatus').textContent=e.message;b.disabled=false}};
$('reveal').onclick=async()=>{try{const t=await get(`/api/reveal/${teIndex}`);revealed.set(teIndex,{truth:t});showTruth(t);renderPredictions();$('reveal').disabled=true;renderScore(await get('/api/score'))}catch(e){$('askStatus').textContent=e.message}};
(async()=>{try{meta=await get('/api/meta');$('seed').textContent=`seed ${meta.seed} · interactive demo`;$('counts').textContent=`${meta.train_count} AV examples · ${meta.test_count} audio-only tests`;$('field').textContent=`${meta.field_cells}-cell retina · ${meta.position_min}…+${meta.position_max}`;$('modelCatalog').innerHTML=meta.model_catalog.map(x=>`<div class="model"><strong>${x.id}</strong> · ${x.name}<div class="muted">${x.learning}</div></div>`).join('');$('referenceCatalog').innerHTML=meta.reference_models.map(x=>`<div class="model"><strong>${x.id}</strong> · ${x.name}<div class="muted">${x.reason}</div></div>`).join('');await loadTrain();await loadTest();renderScore([])}catch(e){document.body.innerHTML=`<main><h1>Viewer error</h1><p>${e.message}</p></main>`}})();
</script></body></html>'''


def make_handler(session: SpatialViewerSession):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data, content_type="application/json; charset=utf-8"):
            body = data.encode("utf-8") if isinstance(data, str) else json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/": return self._send(200, PAGE, "text/html; charset=utf-8")
                if path == "/api/meta": return self._send(200, session.metadata())
                if path == "/api/score": return self._send(200, session.scorecard())
                if path.startswith("/api/train/"):
                    return self._send(200, session.training_example(int(path.rsplit("/", 1)[1])))
                if path.startswith("/api/test/"):
                    return self._send(200, session.test_example(int(path.rsplit("/", 1)[1])))
                if path.startswith("/api/predict/"):
                    return self._send(200, session.predict(int(path.rsplit("/", 1)[1])))
                if path.startswith("/api/reveal/"):
                    return self._send(200, session.reveal(int(path.rsplit("/", 1)[1])))
                return self._send(404, {"error": "not found"})
            except (ValueError, IndexError, RuntimeError) as exc:
                return self._send(400, {"error": str(exc)})

        def do_POST(self):
            if urlparse(self.path).path != "/api/fit":
                return self._send(404, {"error": "not found"})
            try:
                return self._send(200, session.fit())
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as exc:
                return self._send(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print("[spatial-viewer] " + fmt % args)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open the spatial grounding A-I model viewer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-repetitions", type=int, default=1)
    parser.add_argument("--test-per-position", type=int, default=1)
    parser.add_argument("--neural-epochs", type=int, default=4)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    session = SpatialViewerSession(args.seed, args.train_repetitions,
                                   args.test_per_position, args.neural_epochs)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"Spatial A-I viewer ready at http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping spatial viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
