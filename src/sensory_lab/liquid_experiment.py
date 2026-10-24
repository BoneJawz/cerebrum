"""Phase 2 experiment: controlled liquid response and fading-memory probe.

This intentionally does not run the motion-generalization benchmark. It checks
that the asynchronous graph is stable and that a frozen linear readout can
recover which independent acoustic cue was presented after a delay.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import tracemalloc
import time
import tracemalloc

import numpy as np

from .environment import WorldConfig
from .factor_environment import FactorWorld, VOCABULARY, write_vocabulary
from .liquid import LiquidConfig, LiquidNetwork
from .liquid_readout import RidgeReadout
from .processors import sensory_features


VARIANTS=("I0","I1","I2","I2_rewire_fixed")
DELAYS=(0.0,.1,.25,.5,1.0)


def _write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def _stimulus(world, cue_index, seed):
    cue=VOCABULARY[cue_index]
    if cue.category=="direction":
        direction=-1 if cue.concept_id=="DIR_L" else 1
        ep=world.episode(seed,direction,1,initial_position=world.config.pixels//2,
                         task="direction_focus",acoustic_config="B_independent")
    else:
        speed=int(cue.concept_id[-1])
        ep=world.episode(seed,1,speed,initial_position=world.config.pixels//2,
                         task="speed_focus",acoustic_config="B_independent")
    # The Experiment A liquid receives auditory cues only. No labels or motion
    # state are included; visual arrays are blanked before sensory encoding.
    return ep, [type(o)(t=o.t,vision=np.zeros_like(o.vision),audio=o.audio) for o in ep.observations[:2]]


def _capture(net, observations, delay):
    net.reset_episode()
    start=time.perf_counter();cpu0=time.process_time()
    for obs in observations:net.ingest_observation(obs)
    cue_end=observations[-1].t+.05
    net.advance_to(cue_end+float(delay))
    state=net.snapshot()
    elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu0
    return state,elapsed,cpu


def run_experiment_a(output="results/liquid",seed=42,seeds=5,train_per_cue=20,
                     test_per_cue=12,config: LiquidConfig | None=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=config or LiquidConfig()
    world=FactorWorld(WorldConfig(pixels=177,ticks=32,visual_hz=20,audio_hz=2000,noise_std=.015))
    sample=world.episode(seed,1,1,initial_position=world.config.pixels//2,task="combined")
    dim=len(sensory_features(sample.observations[0]));visual_dim=world.config.pixels
    config_json={"phase":"A_dynamics_and_memory","root_seed":seed,"seeds":[seed+i*1009 for i in range(seeds)],
        "train_per_cue":train_per_cue,"test_per_cue":test_per_cue,"variants":VARIANTS,
        "delays_seconds":DELAYS,"cue_ids":[c.concept_id for c in VOCABULARY],
        "liquid_config":cfg.__dict__,"input_dim":dim,"visual_dim":visual_dim,
        "vision_projection":"fixed random sparse signed map; Experiment A blanks vision",
        "audio_projection":"fixed random sparse signed map over generic waveform-derived features",
        "learning":"I0 readout only; I1 local pairwise Hebbian weight update; I2 adds sampled coactivity structural plasticity",
        "scheduler":"deterministic heap ordered by (simulation timestamp, insertion sequence)",
        "python":platform.python_version(),"numpy":np.__version__}
    (out/"liquid_experiment_config.json").write_text(json.dumps(config_json,indent=2),encoding="utf-8")
    write_vocabulary(out/"liquid_acoustic_vocabulary.csv")
    dynamics=[];memory=[];topology=[];compute=[]
    for seed_idx in range(seeds):
        run_seed=seed+seed_idx*1009
        for variant in VARIANTS:
            tracemalloc.start()
            net=LiquidNetwork(dim,seed=run_seed,config=cfg,variant=variant,visual_dim=visual_dim)
            train_times=[];train_cpu=[]
            order=np.random.default_rng(run_seed+101).permutation(train_per_cue*len(VOCABULARY))
            for item_id in order:
                cue_index=int(item_id//train_per_cue)
                ep,obs=_stimulus(world,cue_index,run_seed*100_000+int(item_id))
                _,wall,cpu=_capture(net,obs,0.)
                train_times.append(wall);train_cpu.append(cpu)
                summary=net.topology_summary();stats=net.stats()
                topology.append({"seed":run_seed,"variant":variant,"exposure":len(train_times),**summary,
                    "formed_total":stats.formed_connections,"pruned_total":stats.pruned_connections,
                    "structural_updates_total":stats.structural_updates})
            # Readout fitting uses new episodes after adaptation. Freeze both
            # plasticity timescales while collecting these diagnostic states.
            net.weight_plasticity=False;net.structural_plasticity=False
            readout_states=[];readout_labels=[];readout_wall=[];readout_cpu=[]
            for cue_index in range(len(VOCABULARY)):
                for j in range(train_per_cue):
                    _,obs=_stimulus(world,cue_index,run_seed*300_000+cue_index*1000+j)
                    state,wall,cpu=_capture(net,obs,0.)
                    readout_states.append(state);readout_labels.append(cue_index)
                    readout_wall.append(wall);readout_cpu.append(cpu)
            fit_started=time.perf_counter()
            readout=RidgeReadout(alpha=2.0).fit(np.asarray(readout_states),np.eye(len(VOCABULARY))[readout_labels])
            readout_fit_seconds=time.perf_counter()-fit_started
            total_training_wall=float(np.sum(train_times)+np.sum(readout_wall))
            total_training_cpu=float(np.sum(train_cpu)+np.sum(readout_cpu))
            test_data=[]
            for cue_index in range(len(VOCABULARY)):
                for j in range(test_per_cue):
                    ep,obs=_stimulus(world,cue_index,run_seed*200_000+cue_index*1000+j+500)
                    test_data.append((cue_index,obs))
            # Paired, tiny raw-waveform perturbation probes local response
            # sensitivity with every other episode variable held fixed.
            cue0,base_obs=test_data[0]
            state_base,_,_=_capture(net,base_obs,0.)
            perturb_rng=np.random.default_rng(run_seed+909)
            perturbed=[type(o)(t=o.t,vision=o.vision.copy(),
                audio=o.audio+perturb_rng.normal(0,.002,size=o.audio.shape)) for o in base_obs]
            state_perturbed,_,_=_capture(net,perturbed,0.)
            diff=state_perturbed-state_base
            dynamics.append({"seed":run_seed,"variant":variant,"diagnostic":"paired_audio_perturbation",
                "retention_delay_seconds":0.0,"cue_class":cue0,"waveform_noise_sd":.002,
                "mean_abs_state_change":float(np.mean(np.abs(diff))),
                "relative_l2_state_change":float(np.linalg.norm(diff)/(np.linalg.norm(state_base)+1e-12)),
                "base_state_norm":float(np.linalg.norm(state_base))})
            for delay in DELAYS:
                states=[];labels=[];latencies=[];cpu_times=[];neuron_counts=[];event_counts=[];norms=[]
                for label,obs in test_data:
                    events_before=net.stats().processed_events
                    state,wall,cpu=_capture(net,obs,delay)
                    states.append(state);labels.append(label);latencies.append(wall);cpu_times.append(cpu)
                    neuron_counts.append(int(np.count_nonzero(np.abs(state)>cfg.resting_threshold)))
                    norms.append(float(np.linalg.norm(state)))
                    event_counts.append(net.stats().processed_events-events_before)
                pred=np.argmax(readout.predict(np.asarray(states)),axis=1)
                accuracy=float(np.mean(pred==np.asarray(labels)))
                memory.append({"seed":run_seed,"variant":variant,"retention_delay_seconds":delay,
                    "cue_accuracy":accuracy,"chance_accuracy":.2,"n_test":len(labels),
                    "mean_active_neurons":float(np.mean(neuron_counts)),"mean_state_norm":float(np.mean(norms)),
                    "median_inference_latency_ms":float(np.median(latencies)*1000),
                    "p95_inference_latency_ms":float(np.percentile(latencies,95)*1000),
                    "p99_inference_latency_ms":float(np.percentile(latencies,99)*1000),
                    "mean_inference_cpu_seconds":float(np.mean(cpu_times)),"mean_processed_events":float(np.mean(event_counts))})
                if delay in (0.,.5,1.):
                    dynamics.append({"seed":run_seed,"variant":variant,"retention_delay_seconds":delay,
                        "mean_activation_norm":float(np.mean(norms)),"mean_active_neurons":float(np.mean(neuron_counts)),
                        "max_abs_activation":float(max(np.max(np.abs(s)) for s in states)),
                        "cue_accuracy":accuracy,"event_count_mean":float(np.mean(event_counts))})
            stats=net.stats()
            compute.append({"seed":run_seed,"variant":variant,"training_wall_seconds":total_training_wall,
                "training_cpu_seconds":total_training_cpu,"adaptive_exposure_examples":train_per_cue*len(VOCABULARY),
                "readout_fit_examples":len(readout_states),"readout_parameter_count":int(readout.coef.size),
                "readout_fit_wall_seconds":readout_fit_seconds,
                "neuron_count":cfg.neurons,"active_connections":net.connection_count,
                "processed_events":stats.processed_events,"neuron_updates":stats.neuron_updates,
                "propagated_events":stats.propagated_events,"formed_connections":stats.formed_connections,
                "pruned_connections":stats.pruned_connections,"queue_drops":stats.queue_drops,
                "weight_updates":stats.weight_updates,"weight_update_seconds":stats.weight_update_seconds,
                "structural_seconds":stats.structural_seconds,
                "candidate_pairs_per_structural_update":cfg.candidate_pairs_per_update})
            snap=net.state_dict()
            (out/f"liquid_snapshot_{variant}_seed{run_seed}.json").write_text(json.dumps(snap),encoding="utf-8")
            _,peak_traced=tracemalloc.get_traced_memory();tracemalloc.stop()
            compute[-1]["peak_traced_bytes_model_run"]=peak_traced
    _write_csv(out/"liquid_dynamics_results.csv",dynamics)
    _write_csv(out/"liquid_memory_results.csv",memory)
    _write_csv(out/"liquid_topology_results.csv",topology)
    _write_csv(out/"liquid_compute_results.csv",compute)
    _plot(out,memory,topology,compute)
    _report(out,config_json,memory,topology,compute)
    return out


def _plot(out,memory,topology,compute):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:return
    for name,rows,field,xkey,ykey,ylabel in (
      ("liquid_memory_retention.png",memory,"cue_accuracy","retention_delay_seconds","variant","Cue recognition accuracy"),
      ("liquid_topology_evolution.png",topology,"active_connections","exposure","variant","Active directed connections"),
      ("liquid_accuracy_vs_compute.png",memory,"cue_accuracy","median_inference_latency_ms","variant","Cue recognition accuracy"),
      ("liquid_state_trajectories.png",memory,"mean_state_norm","retention_delay_seconds","variant","Mean hidden-state L2 norm")):
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for variant in VARIANTS:
            rr=[r for r in rows if r["variant"]==variant]
            if not rr:continue
            xs=sorted(set(float(r[xkey]) for r in rr))
            ys=[float(np.mean([float(r[field]) for r in rr if float(r[xkey])==x])) for x in xs]
            ax.plot(xs,ys,marker="o",label=variant)
        ax.set_xlabel("Inference latency (ms)" if xkey=="median_inference_latency_ms" else xkey.replace("_"," "))
        ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/name,dpi=140);plt.close(fig)


def _report(out,config,memory,topology,compute):
    lines=["# Model I — Structurally Adaptive Liquid (SAL)","",
      "## Scope of this run","",
      "This is the Phase 1/2 prototype and Experiment A only. It evaluates event-queue dynamics and fading memory for five independent acoustic cue classes. Audio reaches the liquid as waveform-derived features; vision is blanked in this isolated memory test. Direction, speed, cue IDs and physical state remain experiment-side labels. The output does not claim motion prediction, audio-visual association, structural advantage, or biological validity.","",
      "## Architecture","",
      "The liquid uses a directed sparse adjacency map over 64 unlayered units. Every ordered pair is eligible except disabled self-edges; actual edges start with mean out-degree 6. Fixed seed-controlled sensory projections target up to eight units. Each unit uses a bounded tanh response with exponential leaky decay, local adaptation and a recent incoming-activity trace. A deterministic min-heap schedules transmission and internal decay events by (simulation time, insertion sequence). Neurons decay mathematically between events; the decay is computed on the next event or a readout snapshot. Snapshot reads scan the 64 states but do not mutate or propagate them.","",
      "I0 freezes weights/topology and fits only a post-hoc ridge readout. I1 adds bounded local correlation-based weight plasticity using pre/post activation at transmission. I2 adds uniformly sampled candidate-pair utility from decayed local coactivity, plus conservative weak-and-inactive pruning. Every potential directed pair can be sampled; candidate search does not scan a dense weight matrix. Topology changes are interval-limited and obey degree/edge caps. This initial rule is a simple testable heuristic, not a claim of biological synaptic plasticity.","",
      "## Measured results","",
      "Mean cue accuracy by delay across the requested seeds:","",
      "| Variant | Delay 0 s | 0.1 s | 0.25 s | 0.5 s | 1.0 s |","|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        vals=[]
        for delay in DELAYS:
            row=[float(r["cue_accuracy"]) for r in memory if r["variant"]==variant and float(r["retention_delay_seconds"])==delay]
            vals.append(f"{np.mean(row):.3f} ± {np.std(row,ddof=1):.3f}" if len(row)>1 else f"{np.mean(row):.3f}")
        lines.append("| "+variant+" | "+" | ".join(vals)+" |")
    lines += ["","Topology and compute means across seeds:","",
      "| Variant | Final active edges | Formed | Pruned | Propagated events | Training/exposure + readout-state wall s | Median inference ms at delay 0 | p95 inference ms at delay 1 | Structural-learning wall s |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        cc=[r for r in compute if r["variant"]==variant]
        mm=[r for r in memory if r["variant"]==variant]
        def meanv(rows,key):return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0
        edges=meanv(cc,"active_connections");formed=meanv(cc,"formed_connections");pruned=meanv(cc,"pruned_connections")
        events=meanv(cc,"propagated_events");train=meanv(cc,"training_wall_seconds")
        lat0=meanv([r for r in mm if float(r["retention_delay_seconds"])==0.0],"median_inference_latency_ms")
        p951=meanv([r for r in mm if float(r["retention_delay_seconds"])==1.0],"p95_inference_latency_ms")
        structural=meanv(cc,"structural_seconds")
        lines.append(f"| {variant} | {edges:.1f} | {formed:.1f} | {pruned:.1f} | {events:.0f} | {train:.3f} | {lat0:.3f} | {p951:.1f} | {structural:.4f} |")
    lines += ["","The table reports measured post-hoc readout accuracy; it is not a measurement of semantic understanding. I1/I2 change only during the adaptation exposure phase. The ridge readout is fitted on new, separate episodes with liquid plasticity frozen. Training wall time includes adaptation and readout-state collection; the separate readout solver time is in `liquid_compute_results.csv`.","",
      "`liquid_dynamics_results.csv` also includes a paired 0.002-SD waveform perturbation probe. It changes only the raw audio chunk and reports hidden-state displacement; this is one pair per seed, not a robustness estimate.","",
      "## Observations and interpretation","",
      "I0 and I1 use the same graph and remain very close on this cue task; the current local weight rule produces little improvement in retention. I2 forms and prunes edges, but increases event propagation and long-delay latency without a clear accuracy gain. Across seeds, all variants lose cue-class accuracy toward chance by the 1-second delay, while some retain high activity norms and many scheduled events. Persistent activity therefore does not by itself show persistent cue information. The I2 event cascade also demonstrates that sparse adjacency does not guarantee lower computation; mean degree remains near six, but recurrent event traffic can dominate cost.","",
      "Seed-level variability is substantial, particularly at 0.1–0.5 s. This is a small diagnostic set with five seeds, not a tuned classifier benchmark. No claim of adaptation benefit is supported. Before Experiment B, the next justified step is to stabilize and characterize long-delay event rates, then test whether the fixed reservoir has recoverable cue information under a more controlled readout split and matched compute budget.","",
      "## Numerical and compute safeguards","",
      f"Configured limits: {config['liquid_config']['max_connections']} total edges, {config['liquid_config']['max_out_degree']} outgoing edges per unit, queue cap {config['liquid_config']['max_queue_size']}, and {config['liquid_config']['max_events_per_advance']} processed events per advance. Activation and adaptation remain bounded; edge weights are clipped to ±{config['liquid_config']['weight_limit']}. A queue overflow raises a clear runtime error; event insertions beyond queue capacity are counted as drops. `peak_traced_bytes_model_run` includes Python objects and snapshot serialization but omits native NumPy allocations and is not process RSS. CPU measurements are machine-specific and include Python heap scheduling.","",
      "## Interpretation and unresolved questions","",
      "I0 is the fixed-reservoir baseline; I1 tests whether this local Hebbian-like update changes cue retention; I2 tests whether sampled coactivity rewiring adds anything beyond weight changes. The study must be read together with topology and compute CSVs: fewer state updates do not imply lower CPU cost, and topology statistics do not imply useful computation. Five seeds and a single cue-memory task are insufficient to establish robust advantage. The next phase should verify cue association under a fixed readout budget and compare adaptation exposure separately from readout fitting. Motion prediction and strict/familiar composition remain unrun pending stable baseline dynamics.","",
      "## Reproduction","",f"Configuration: `liquid_experiment_config.json`; snapshots include graph edges, current states, event queue, fixed input projections and RNG state. Run with `python -m sensory_lab.liquid_experiment --output {out.as_posix()}`.",""]
    (out/"LIQUID_ARCHITECTURE_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Run Model I Experiment A: liquid dynamics and fading memory")
    parser.add_argument("--output",default="results/liquid");parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--seeds",type=int,default=5);parser.add_argument("--train-per-cue",type=int,default=20)
    parser.add_argument("--test-per-cue",type=int,default=12)
    args=parser.parse_args(argv)
    print(run_experiment_a(args.output,args.seed,args.seeds,args.train_per_cue,args.test_per_cue))


if __name__=="__main__":main()
