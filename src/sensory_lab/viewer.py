"""Local interactive viewer for raw V0.2 examples and readout predictions."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from urllib.parse import urlparse

import numpy as np

from .benchmark import BenchmarkConfig, RidgeReadout, _ablate
from .contracts import Observation
from .environment import LineWorld, WorldConfig
from .processors import PROCESSORS


def _center(frame):
    pixels = np.arange(len(frame.vision), dtype=float)
    return float(np.sum(pixels * frame.vision) / np.sum(frame.vision))


def _make_examples(world, count, seed, frequency_map, outcome_tick):
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(count):
        direction = int(rng.choice([-1, 1]))
        observations = world.episode(seed=seed * 10000 + i,
                                     direction=direction,
                                     tone_hz=frequency_map[direction])
        examples.append({
            "observations": observations,
            "target_position": _center(observations[outcome_tick]),
            "direction": direction,
        })
    return examples


def _observations_json(observations):
    return [{"t": float(o.t), "vision": o.vision.astype(int).tolist(),
             "audio": o.audio.tolist()} for o in observations]


class ViewerSession:
    """Owns one reproducible split; held-out truths are served only on reveal."""
    def __init__(self, config: BenchmarkConfig, world_config: WorldConfig | None = None):
        self.config = config
        self.world_config = world_config or WorldConfig()
        if config.outcome_tick >= self.world_config.ticks:
            raise ValueError("outcome_tick must be inside the episode")
        self.seed = config.seed
        codebook = np.random.default_rng(self.seed).permutation([140., 340.])
        self.frequency_map = {-1: float(codebook[0]), 1: float(codebook[1])}
        world = LineWorld(self.world_config)
        self.train = _make_examples(world, config.train_episodes, self.seed + 1,
                                    self.frequency_map, config.outcome_tick)
        self.test = _make_examples(world, config.test_episodes, self.seed + 2,
                                   self.frequency_map, config.outcome_tick)
        self.models = None
        self.fit_seconds = None

    def metadata(self):
        return {
            "seed": self.seed,
            "train_count": len(self.train),
            "test_count": len(self.test),
            "pixels": self.world_config.pixels,
            "visual_hz": self.world_config.visual_hz,
            "audio_hz": self.world_config.audio_hz,
            "prediction_tick": self.config.prediction_tick,
            "outcome_tick": self.config.outcome_tick,
            "task": "predict visual object position at the outcome frame",
        }

    def training_example(self, index):
        if not 0 <= index < len(self.train):
            raise IndexError("training example index out of range")
        item = self.train[index]
        return {
            "index": index,
            "observations": _observations_json(item["observations"]),
            "direction": item["direction"],
            "target_position": item["target_position"],
            "start_position": _center(item["observations"][0]),
            "tone_hz": self.frequency_map[item["direction"]],
        }

    def fit(self):
        if self.models is not None:
            return {"seconds": self.fit_seconds, "readouts": len(self.models)}
        model_data = [(x["observations"], x["target_position"]) for x in self.train]
        models = {}
        started = time.perf_counter()
        for code, kind in PROCESSORS.items():
            for modality in ("vision_only", "audio_only", "combined"):
                processor = kind(pixels=self.world_config.pixels,
                                 hidden=self.config.hidden, seed=self.seed)
                features, targets = [], []
                for observations, target in model_data:
                    processor.reset()
                    feature = None
                    for observation in observations[:self.config.prediction_tick + 1]:
                        feature = processor.step(_ablate(observation, modality))
                    features.append(feature)
                    targets.append(target)
                readout = RidgeReadout()
                readout.fit(np.asarray(features), np.asarray(targets))
                models[(code, modality)] = (kind, readout)
        self.fit_seconds = time.perf_counter() - started
        self.models = models
        return {"seconds": self.fit_seconds, "readouts": len(models)}

    def test_example(self, index):
        if not 0 <= index < len(self.test):
            raise IndexError("test example index out of range")
        item = self.test[index]
        shown = item["observations"][:self.config.prediction_tick + 1]
        return {"index": index, "observations": _observations_json(shown),
                "prediction_tick": self.config.prediction_tick}

    def predict(self, index):
        if self.models is None:
            raise RuntimeError("fit readouts before asking for predictions")
        if not 0 <= index < len(self.test):
            raise IndexError("test example index out of range")
        observations = self.test[index]["observations"][:self.config.prediction_tick + 1]
        results = []
        for code, kind in PROCESSORS.items():
            for modality in ("vision_only", "audio_only", "combined"):
                _, readout = self.models[(code, modality)]
                processor = kind(pixels=self.world_config.pixels,
                                 hidden=self.config.hidden, seed=self.seed)
                processor.reset()
                feature = None
                for observation in observations:
                    feature = processor.step(_ablate(observation, modality))
                position = float(readout.predict(np.asarray(feature)[None, :])[0])
                start = _center(observations[0])
                direction = 1 if position >= start else -1
                results.append({"architecture": code, "modality": modality,
                                "predicted_position": position,
                                "predicted_direction": direction})
        return results

    def reveal(self, index):
        if not 0 <= index < len(self.test):
            raise IndexError("test example index out of range")
        item = self.test[index]
        outcome = item["observations"][self.config.outcome_tick]
        return {"target_position": item["target_position"],
                "direction": item["direction"],
                "start_position": _center(item["observations"][0]),
                "outcome_tick": self.config.outcome_tick,
                "outcome_visual": outcome.vision.astype(int).tolist(),
                "tone_hz": self.frequency_map[item["direction"]]}


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multisensory Dynamics Lab · Trial Viewer</title>
<style>
:root{color-scheme:dark;--bg:#111820;--panel:#1a2630;--line:#30424e;--ink:#eef4f4;--muted:#a8bac2;--teal:#78ddc3;--gold:#f2c879;--red:#f08989}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1120px;margin:auto;padding:28px 20px 60px}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(1.7rem,4vw,2.5rem);line-height:1.15}h2{font-size:1.3rem}.sub,.muted{color:var(--muted)}.sub{max-width:760px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin:18px 0}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.spread{justify-content:space-between}.tabs{display:flex;gap:8px;margin:24px 0 10px}.tab,button{border:1px solid var(--line);border-radius:10px;background:#22323e;color:var(--ink);padding:10px 14px;font:inherit;cursor:pointer}.tab.active,button.primary{background:var(--teal);color:#09221c;border-color:var(--teal);font-weight:700}button:disabled{opacity:.5;cursor:wait}.badge{border:1px solid var(--line);border-radius:999px;padding:4px 10px;color:var(--muted);font-size:.88rem}.stage{display:none}.stage.active{display:block}.canvas-wrap{background:#0d141a;border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}.canvas-wrap canvas{display:block;width:100%;height:130px}.canvas-wrap.wave canvas{height:100px}.labels{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}.label{background:#121c23;border:1px solid var(--line);padding:10px 12px;border-radius:9px}.label b{display:block;color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}.notice{padding:12px 14px;background:#142b2a;border-left:3px solid var(--teal);border-radius:6px}.question{font-size:1.25rem;color:var(--gold);font-weight:700}.results{width:100%;border-collapse:collapse;margin-top:14px}.results th,.results td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}.results th{color:var(--muted);font-size:.84rem}.correct{color:var(--teal)}.wrong{color:var(--red)}.spinner{display:inline-block;width:15px;height:15px;border:2px solid #58716e;border-top-color:var(--teal);border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:8px}@keyframes spin{to{transform:rotate(360deg)}}.hidden{display:none!important}.progress{height:8px;background:#0d141a;border-radius:8px;overflow:hidden}.progress i{display:block;height:100%;background:var(--teal);width:0}footer{font-size:.88rem;color:var(--muted);margin-top:24px}@media(max-width:650px){main{padding:20px 12px}.panel{padding:15px}.results{font-size:.87rem}}
</style></head><body><main>
<h1>Multisensory Dynamics Lab</h1><p class="sub">Training demonstrations pair a sound cue with an object moving in the indicated direction. At test, processors see an initial visual position and the raw directional sound cue, then predict the object's later position.</p>
<div class="row"><span class="badge" id="seed"></span><span class="badge" id="split"></span><span class="badge" id="rule"></span></div>
<div class="tabs"><button class="tab active" data-stage="training">Training examples</button><button class="tab" data-stage="heldout">Held-out predictions</button></div>
<section id="training" class="stage active">
<div class="panel"><div class="row spread"><div><h2>Training demonstrations</h2><div class="muted">Each cue is paired with the matching visible movement. Ground truth is shown here for you. Readouts are fit from initial visual frames plus the cue; later frames show each demonstrated outcome.</div></div><div class="row"><button id="trainPrev">Previous</button><span id="trainCount" class="badge"></span><button id="trainNext">Next</button></div></div>
<div class="canvas-wrap"><div class="muted">Vision · one-dimensional binary retina</div><canvas id="trainVision" width="990" height="130"></canvas></div>
<div class="canvas-wrap wave"><div class="muted">Audio · raw signed waveform</div><canvas id="trainWave" width="990" height="100"></canvas></div>
<div class="row"><button id="trainPlay" class="primary">Animate + play this example</button><span id="trainAudioInfo" class="muted"></span></div><div id="trainTruth" class="labels"></div>
</div>
<div class="panel"><h2>Fit the readouts</h2><p>Processors A–D have fixed internal parameters. This step fits the supervised kernel-ridge readouts on the training set; it does not train the processor dynamics.</p><div class="row"><button id="fitButton" class="primary">Fit 12 readouts</button><span id="fitStatus" class="muted">Waiting for you to start fitting.</span></div></div>
</section>
<section id="heldout" class="stage"><div class="panel"><div class="row spread"><div><h2>Held-out trial</h2><div class="muted">This trial was not used to fit the readouts. Its true factors and target stay hidden until you reveal them.</div></div><div class="row"><button id="testPrev">Previous</button><span id="testCount" class="badge"></span><button id="testNext">Next</button></div></div>
<div class="canvas-wrap"><div class="muted">Vision · binary retinal frames</div><canvas id="testVision" width="990" height="130"></canvas></div>
<div class="canvas-wrap wave"><div class="muted">Audio · raw signed waveform</div><canvas id="testWave" width="990" height="100"></canvas></div>
<div class="row"><button id="testPlay">Animate + play this example</button><span id="testAudioInfo" class="muted"></span></div>
<p class="question" style="margin-top:20px">What do you predict?</p><div class="row"><button id="askButton" class="primary" disabled>Ask A–D under all three modality conditions</button><span id="askStatus" class="muted">Fit the readouts first.</span></div>
<div id="predictions"></div><div class="row" style="margin-top:16px"><button id="revealButton" disabled>Reveal true values</button></div><div id="reveal" class="labels"></div><div id="outcomeWrap" class="canvas-wrap hidden"><div class="muted">Actual later visual state</div><canvas id="outcomeVision" width="990" height="130"></canvas></div>
</div><div class="panel"><h2>Revealed trial scorecard</h2><div class="muted">Accuracy over held-out examples you have revealed so far. This is a running UI tally, not a replacement for the full benchmark.</div><div id="scorecard"></div></div></section>
<footer>Target: object position at the outcome frame. Test inputs stop before visible movement. Model inputs never include target labels or future frames. Predicted positions are readout values, not calibrated probabilities.</footer>
</main><script>
const $=id=>document.getElementById(id);let meta,trainIndex=0,testIndex=0,trainEx,testEx,fitDone=false,pred=null,revealed=new Map(),audioCtx;
const jget=async p=>{const r=await fetch(p);if(!r.ok)throw Error((await r.json()).error||r.statusText);return r.json()};
const jpost=async p=>{const r=await fetch(p,{method:'POST'});if(!r.ok)throw Error((await r.json()).error||r.statusText);return r.json()};
function drawVision(canvasId,frames,active=-1){const c=$(canvasId),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);const pad=22,rows=frames.length,cw=(w-2*pad)/meta.pixels,rh=Math.min(20,(h-30)/rows);frames.forEach((f,k)=>{const y=15+k*rh;f.vision.forEach((v,i)=>{ctx.fillStyle=v?'#78ddc3':'#263640';ctx.fillRect(pad+i*cw,y,Math.max(1,cw-2),Math.max(3,rh-2));});if(k===active){ctx.strokeStyle='#f2c879';ctx.lineWidth=2;ctx.strokeRect(pad-3,y-2,w-2*pad+6,rh);}});ctx.fillStyle='#a8bac2';ctx.font='12px system-ui';ctx.fillText('time →',pad,h-3);}
function drawWave(canvasId,frames){const c=$(canvasId),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#78ddc3';ctx.lineWidth=1;ctx.beginPath();let n=0;for(const f of frames)n+=f.audio.length;const stride=Math.max(1,Math.floor(n/w));let vals=[];for(const f of frames)for(let i=0;i<f.audio.length;i+=stride)vals.push(f.audio[i]);ctx.beginPath();vals.forEach((v,i)=>{const x=i/(Math.max(1,vals.length-1))*w,y=h/2-v*(h*.42);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.strokeStyle='#30424e';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();}
function trainingCard(x){const side=x.direction===1?'right (+1)':'left (−1)';return `<div class="label"><b>Starting position</b>${x.start_position.toFixed(1)} px</div><div class="label"><b>Audio cue means</b>move ${side}</div><div class="label"><b>Observed outcome</b>${x.target_position.toFixed(1)} px</div><div class="label"><b>Cue frequency</b>${x.tone_hz} Hz</div>`;}
function truthCard(x){const side=x.direction===1?'right (+1)':'left (−1)';return `<div class="label"><b>Audio cue means</b>move ${side}</div><div class="label"><b>Starting position</b>${x.start_position.toFixed(1)} px</div><div class="label"><b>True outcome position</b>${x.target_position.toFixed(1)} px</div><div class="label"><b>Cue frequency</b>${x.tone_hz} Hz</div>`;}
function playAudio(frames){if(!audioCtx)audioCtx=new (window.AudioContext||window.webkitAudioContext)();const all=frames.flatMap(x=>x.audio),n=Math.max(1,Math.round(all.length*audioCtx.sampleRate/meta.audio_hz)),buf=audioCtx.createBuffer(1,n,audioCtx.sampleRate),ch=buf.getChannelData(0),ratio=meta.audio_hz/audioCtx.sampleRate;for(let i=0;i<n;i++){const p=i*ratio,j=Math.min(all.length-1,Math.floor(p)),f=p-j;ch[i]=all[j]*(1-f)+all[Math.min(j+1,all.length-1)]*f;}const src=audioCtx.createBufferSource();src.buffer=buf;src.connect(audioCtx.destination);src.start();}
function animate(frames,canvasId){let k=0;drawVision(canvasId,frames,k);const timer=setInterval(()=>{k++;if(k>=frames.length){clearInterval(timer);drawVision(canvasId,frames);return;}drawVision(canvasId,frames,k);},1000/meta.visual_hz);playAudio(frames);}
async function loadTrain(){trainEx=await jget(`/api/train/${trainIndex}`);$('trainCount').textContent=`${trainIndex+1} / ${meta.train_count}`;drawVision('trainVision',trainEx.observations);drawWave('trainWave',trainEx.observations);$('trainTruth').innerHTML=trainingCard(trainEx);$('trainAudioInfo').textContent=`${trainEx.observations.length} frames · ${meta.audio_hz} Hz waveform samples`;$('trainPrev').disabled=trainIndex===0;$('trainNext').disabled=trainIndex===meta.train_count-1;}
function renderPredictions(){if(!pred)return;const rows=pred.map(x=>`<tr><td>${x.architecture}</td><td>${x.modality.replace('_',' ')}</td><td>${x.predicted_position.toFixed(2)} px</td><td>${x.predicted_direction>0?'right':'left'}</td></tr>`).join('');$('predictions').innerHTML=`<table class="results"><thead><tr><th>Processor</th><th>Input</th><th>Predicted outcome</th><th>Implied movement</th></tr></thead><tbody>${rows}</tbody></table>`;}
async function loadTest(){testEx=await jget(`/api/test/${testIndex}`);pred=null;$('predictions').innerHTML='';$('reveal').innerHTML='';$('outcomeWrap').classList.add('hidden');$('testCount').textContent=`${testIndex+1} / ${meta.test_count}`;drawVision('testVision',testEx.observations);drawWave('testWave',testEx.observations);$('testAudioInfo').textContent=`${testEx.observations.length} initial frames · actual outcome at frame ${meta.outcome_tick}`;$('testPrev').disabled=testIndex===0;$('testNext').disabled=testIndex===meta.test_count-1;$('askButton').disabled=!fitDone;$('revealButton').disabled=true;$('askStatus').textContent=fitDone?'Predict the later visual position from this starting scene and audio cue.':'Fit the readouts first.';if(revealed.has(testIndex)){$('reveal').innerHTML=truthCard(revealed.get(testIndex).truth);drawVision('outcomeVision',[{vision:revealed.get(testIndex).truth.outcome_visual}]);$('outcomeWrap').classList.remove('hidden');$('revealButton').disabled=true;}}
function updateScore(){if(!revealed.size){$('scorecard').innerHTML='<p class="muted">Reveal a trial to start the tally.</p>';return;}let html='<table class="results"><thead><tr><th>Processor</th><th>Input</th><th>Mean absolute position error</th><th>Within 1 px</th></tr></thead><tbody>';for(const arch of ['A','B','C','D'])for(const mod of ['vision_only','audio_only','combined']){let n=0,err=0,ok=0;for(const [i,r] of revealed){const p=r.predictions.find(x=>x.architecture===arch&&x.modality===mod);n++;err+=Math.abs(p.predicted_position-r.truth.target_position);if(Math.abs(p.predicted_position-r.truth.target_position)<=1)ok++;}html+=`<tr><td>${arch}</td><td>${mod.replace('_',' ')}</td><td>${(err/n).toFixed(2)} px</td><td>${(ok/n).toFixed(3)}</td></tr>`;}html+='</tbody></table>'; $('scorecard').innerHTML=html;}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.stage').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.stage).classList.add('active');});
$('trainPrev').onclick=async()=>{trainIndex--;await loadTrain()};$('trainNext').onclick=async()=>{trainIndex++;await loadTrain()};$('trainPlay').onclick=()=>animate(trainEx.observations,'trainVision');$('testPlay').onclick=()=>animate(testEx.observations,'testVision');
$('fitButton').onclick=async()=>{const b=$('fitButton');b.disabled=true;$('fitStatus').innerHTML='<span class="spinner"></span>Fitting the readouts on the training examples…';try{const r=await jpost('/api/fit');fitDone=true;$('fitStatus').textContent=`Fit complete in ${(r.seconds*1000).toFixed(1)} ms · ${r.readouts} readouts · processor dynamics stayed fixed.`;$('askButton').disabled=false;$('askStatus').textContent='Readouts ready. Choose a held-out trial and ask for predictions.';await loadTest();document.querySelector('[data-stage="heldout"]').click();}catch(e){$('fitStatus').textContent=e.message;b.disabled=false;}};
$('testPrev').onclick=async()=>{testIndex--;await loadTest()};$('testNext').onclick=async()=>{testIndex++;await loadTest()};
$('askButton').onclick=async()=>{const b=$('askButton');b.disabled=true;$('askStatus').innerHTML='<span class="spinner"></span>Processors are advancing through the sensory stream…';try{pred=await jget(`/api/predict/${testIndex}`);renderPredictions();$('askStatus').textContent='Predictions are in. The true target is still hidden.';$('revealButton').disabled=false;}catch(e){$('askStatus').textContent=e.message;b.disabled=false;}};
$('revealButton').onclick=async()=>{const truth=await jget(`/api/reveal/${testIndex}`);revealed.set(testIndex,{truth,predictions:pred});$('reveal').innerHTML=truthCard(truth);drawVision('outcomeVision',[{vision:truth.outcome_visual}]);$('outcomeWrap').classList.remove('hidden');$('revealButton').disabled=true;updateScore();};
(async()=>{try{meta=await jget('/api/meta');$('seed').textContent=`seed ${meta.seed}`;$('split').textContent=`${meta.train_count} demonstrations · ${meta.test_count} held-out`;$('rule').textContent=`predict future position at frame ${meta.outcome_tick}`;await loadTrain();await loadTest();}catch(e){document.body.innerHTML=`<main><h1>Viewer error</h1><p>${e.message}</p></main>`;}})();
</script></body></html>'''


def make_handler(session):
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
                if path == "/":
                    return self._send(200, PAGE, "text/html; charset=utf-8")
                if path == "/api/meta":
                    return self._send(200, session.metadata())
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
            except (ValueError, np.linalg.LinAlgError) as exc:
                return self._send(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print("[viewer] " + fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Open the local multisensory trial viewer")
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = BenchmarkConfig(args.train, args.test, args.seed, args.hidden, 1)
    session = ViewerSession(config)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"V0.2 viewer ready at http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
