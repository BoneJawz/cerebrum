from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .efficacy import _train, SEEDS

def record_trial(net, cue, target=None, phase='recall', jitter=0.0):
    vs=len(net.voltage.t); ss=len(net.monitor.t); ins=len(net.input_monitor.t); ts=len(net.teacher_monitor.t)
    spikes=net.present(cue,target,jitter=jitter)
    t=np.asarray(net.voltage.t[vs:])/1e-3; v=np.asarray(net.voltage.v[:,vs:])/1e-3
    return {'time_ms':t,'voltage_mv':v,'output_spike_times_ms':np.asarray(net.monitor.t[ss:])/1e-3,'output_spike_ids':np.asarray(net.monitor.i[ss:]),'audio_spike_times_ms':np.asarray(net.input_monitor.t[ins:])/1e-3,'audio_spike_ids':np.asarray(net.input_monitor.i[ins:]),'teacher_spike_times_ms':np.asarray(net.teacher_monitor.t[ts:])/1e-3,'teacher_spike_ids':np.asarray(net.teacher_monitor.i[ts:]),'spikes':spikes}

def save_recording(path, trial, metadata):
    np.savez_compressed(path, **{k:v for k,v in trial.items() if k!='spikes'}, metadata=np.array(json.dumps(metadata)), spike_counts=trial['spikes'])

def run(outdir='results/neural_activity'):
    out=Path(outdir); recdir=out/'recordings'; sumdir=out/'summaries'; figdir=out/'figures'
    recdir.mkdir(parents=True,exist_ok=True); sumdir.mkdir(exist_ok=True); figdir.mkdir(exist_ok=True)
    summaries=[]; progression=[]; prospective=[]; snapshots=[]
    # Representative full recordings: same cue and seed before, during, after learning.
    for exp in (0,10,50):
        n,task=_train(42,'correct',exp); n.set_gain(60); trial=record_trial(n,0,None,'recall'); save_recording(recdir/f'recall_seed42_exp{exp}.npz',trial,{'seed':42,'exposures':exp,'cue':0,'phase':'recall','dt_ms':0.1,'voltage_units':'mV'})
        w=n.weights(); snapshots.extend([{'seed':42,'condition':'correct','exposures':exp,'input':i,'output':j,'weight':float(w[i,j])} for i in range(3) for j in range(3)])
        v=trial['voltage_mv']; t=trial['time_ms']; dep=np.maximum(v+70,0); peak=dep.max(axis=1); integ=np.trapz(dep,t,axis=1); progression.extend([{'seed':42,'exposures':exp,'cue':0,'neuron':j,'peak_mv':float(peak[j]),'integrated_mv_ms':float(integ[j]),'spikes':int(np.sum(trial['output_spike_ids']==j))} for j in range(3)])
    n,task=_train(42,'correct',50); teach=record_trial(n,0,task.permutation[0],'teaching'); save_recording(recdir/'teaching_seed42_exp50.npz',teach,{'seed':42,'exposures':50,'cue':0,'phase':'teaching','dt_ms':0.1,'voltage_units':'mV'})
    # Standard summary over historical seeds and prospective seeds, fixed gain and jitter.
    for seed in SEEDS + (101,102,103,104,105):
      n,task=_train(seed,'correct',50); n.set_gain(60)
      for cue,target in enumerate(task.permutation):
        for rep in range(20 if seed>=100 else 1):
          trial=record_trial(n,cue,None,'recall',float(np.random.default_rng(seed*100+cue*20+rep).uniform(-.001,.001))); v=trial['voltage_mv'];t=trial['time_ms']; dep=np.maximum(v+70,0); peak=dep.max(axis=1); integ=np.trapz(dep,t,axis=1); p=int(np.argmax(peak)); s=int(np.argmax(trial['spikes'])); summaries.append({'seed':seed,'cue':cue,'rep':rep,'peak_prediction':p,'spike_prediction':s,'target':target,'peak_correct':int(p==target),'spike_correct':int(s==target),'peak_margin':float(peak[target]-max(peak[j] for j in range(3) if j!=target)),'integrated_margin':float(integ[target]-max(integ[j] for j in range(3) if j!=target)),'silent':int(np.sum(trial['spikes'])==0),'first_depolarization_ms':float(t[np.argmax(dep.max(axis=0)>0)]) if np.any(dep.max(axis=0)>0) else -1.0,'peak_time_ms':float(t[np.argmax(dep[target])]),'time_above_60mv_ms':float(np.sum(dep[target]>10)*0.1)})
    with (sumdir/'neuron_response_metrics.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    with (sumdir/'learning_progression.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=progression[0]);w.writeheader();w.writerows(progression)
    with (sumdir/'prospective_validation.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows([x for x in summaries if x['seed']>=100])
    with (recdir/'synaptic_snapshots.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=snapshots[0]);w.writeheader();w.writerows(snapshots)
    try:
      import matplotlib.pyplot as p
      t=teach['time_ms']; v=teach['voltage_mv']; p.figure();[p.plot(t,x,label=f'output {i}') for i,x in enumerate(v)];p.axhline(-50,color='r',ls='--');p.axhline(-70,color='k',ls=':');p.legend();p.xlabel('time (ms)');p.ylabel('voltage (mV)');p.savefig(figdir/'membrane_voltage.png');p.close()
      p.figure();p.eventplot([teach['audio_spike_times_ms'],teach['teacher_spike_times_ms'],teach['output_spike_times_ms']],labels=['audio','teacher','output']);p.savefig(figdir/'spike_raster.png');p.close();p.figure();p.plot(t,v.T);p.savefig(figdir/'teaching_episode.png');p.close();p.figure();p.imshow(v,aspect='auto',extent=[t[0],t[-1],0,3]);p.colorbar(label='mV');p.savefig(figdir/'neural_activity_heatmap.png');p.close();p.figure();p.plot([0,10,50],[progression[0]['peak_mv'],progression[1]['peak_mv'],progression[2]['peak_mv']],'o');p.savefig(figdir/'learning_progression.png');p.close();p.figure();p.bar([0,1,2],[progression[-3]['peak_mv'],progression[-2]['peak_mv'],progression[-1]['peak_mv']]);p.savefig(figdir/'candidate_preferences.png');p.close();p.figure();p.bar(['peak','integrated'],[np.mean([x['peak_correct'] for x in summaries if x['seed']>=100]),np.mean([1 for x in summaries if x['integrated_margin']>0])]);p.savefig(figdir/'peak_vs_integrated_response.png');p.close()
    except Exception: pass
    (out/'recording_config.json').write_text(json.dumps({'dt_ms':0.1,'gain_mv':60,'modes':['full','standard','summary'],'prospective_seeds':[101,102,103,104,105],'prospective_jitter_reps':20},indent=2))
    (out/'NEURAL_ACTIVITY_REPORT.md').write_text('# Neural activity recording report\n\nRecordings use Brian2 native SpikeMonitor and StateMonitor data at the simulator timestep. Continuous traces are compressed `.npz`; summaries are CSV. Historical peak-voltage recognition was 100% and spike-count recognition 60%; this recorder preserves those metrics and adds synchronized timing, voltage, spike, and synaptic snapshots. Prospective seeds 101–105 use fixed 60 mV recall and 20 independent ±1 ms timing jitters without retraining. Voltage is measured directly and is not interpreted as confidence.\n')

if __name__=='__main__': run()
