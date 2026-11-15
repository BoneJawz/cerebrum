from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
from .efficacy import _train, SEEDS

LEVELS=(0,1,5,10,20,50); N_JITTER=20

def _write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def softmax(x,temp=1.0):
    z=(np.asarray(x,float)-np.max(x))/temp; e=np.exp(np.clip(z,-700,700)); return e/e.sum()

def _measure(n,cue,target,jitter=0.0):
    n.set_gain(60); n.reset_transient(); start=len(n.monitor.t); vstart=len(n.voltage.t); spikes=n.present(cue,None,jitter=jitter)
    v=np.asarray(n.voltage.v[:,vstart:])/1e-3; t=np.asarray(n.voltage.t[vstart:])/1e-3
    dep=np.maximum(v+70.,0.); peak=dep.max(axis=1) if dep.size else np.zeros(3); integ=np.trapz(dep,t,axis=1) if len(t)>1 else np.zeros(3)
    ps=int(np.argmax(spikes)); pv=int(np.argmax(peak)); pi=int(np.argmax(integ)); probs=softmax(integ,1.0); ent=float(-np.sum(probs*np.log(np.maximum(probs,1e-12))))
    return {'spike_a':int(spikes[0]),'spike_b':int(spikes[1]),'spike_c':int(spikes[2]),'spike_prediction':ps,'peak_prediction':pv,'integrated_prediction':pi,'spike_correct':int(ps==target),'peak_correct':int(pv==target),'integrated_correct':int(pi==target),'silent':int(np.sum(spikes)==0),'tie':int(np.sum(spikes==spikes.max())>1),'peak_a':float(peak[0]),'peak_b':float(peak[1]),'peak_c':float(peak[2]),'integrated_a':float(integ[0]),'integrated_b':float(integ[1]),'integrated_c':float(integ[2]),'spike_margin':float(spikes[target]-max(spikes[j] for j in range(3) if j!=target)),'peak_margin':float(peak[target]-max(peak[j] for j in range(3) if j!=target)),'integrated_margin':float(integ[target]-max(integ[j] for j in range(3) if j!=target)),'max_choice_weight':float(probs.max()),'entropy':ent}

def run(outdir='results/graded_recognition'):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    trials=[]; norm=[]; margins=[]; progress=[]; controls=[]; jitter_rows=[]
    rng=np.random.default_rng(20261025)
    for seed in SEEDS:
      for cond in ('no_learning','correct','mismatched'):
        for exp in LEVELS:
          n,task=_train(seed,cond,exp)
          for cue,target in enumerate(task.permutation):
            r=_measure(n,cue,target); r.update({'seed':seed,'condition':cond,'exposures':exp,'cue':cue,'target':target}); trials.append(r)
            norm.append({'seed':seed,'condition':cond,'exposures':exp,'cue':cue,'p_a':float(softmax([r['integrated_a'],r['integrated_b'],r['integrated_c']])[0]),'p_b':float(softmax([r['integrated_a'],r['integrated_b'],r['integrated_c']])[1]),'p_c':float(softmax([r['integrated_a'],r['integrated_b'],r['integrated_c']])[2]),'max_weight':r['max_choice_weight'],'entropy':r['entropy']})
            margins.append({'seed':seed,'condition':cond,'exposures':exp,'cue':cue,'spike_margin':r['spike_margin'],'peak_margin':r['peak_margin'],'integrated_margin':r['integrated_margin'],'integrated_correct':r['integrated_correct']})
          z=[x for x in trials if x['seed']==seed and x['condition']==cond and x['exposures']==exp]; w=n.weights(); intended=np.mean([w[i,task.permutation[i]] for i in range(3)]); comp=np.mean([w[i,j] for i in range(3) for j in range(3) if j!=task.permutation[i]])
          progress.append({'seed':seed,'condition':cond,'exposures':exp,'spike_accuracy':np.mean([x['spike_correct'] for x in z]),'peak_accuracy':np.mean([x['peak_correct'] for x in z]),'integrated_accuracy':np.mean([x['integrated_correct'] for x in z]),'integrated_margin':np.mean([x['integrated_margin'] for x in z]),'intended_weight':float(intended),'competing_weight':float(comp)})
        # held-out nuisance timing, same fixed learned state and 60mV
        n,task=_train(seed,cond,50)
        for cue,target in enumerate(task.permutation):
          for rep in range(N_JITTER):
            j=float(rng.uniform(-0.001,0.001)); r=_measure(n,cue,target,j); jitter_rows.append({'seed':seed,'condition':cond,'cue':cue,'rep':rep,'jitter_ms':j*1000,'target':target,'spike_correct':r['spike_correct'],'peak_correct':r['peak_correct'],'integrated_correct':r['integrated_correct'],'integrated_margin':r['integrated_margin'],'silent':r['silent'],'tie':r['tie']})
      # controls summarized separately for intended and taught mappings
    for r in progress:
      if r['exposures']==50: controls.append(r)
    for name,rows in [('trial_responses.csv',trials),('spike_vs_voltage_accuracy.csv',trials),('normalized_choice_weights.csv',norm),('response_margins.csv',margins),('learning_progression.csv',progress),('pairing_controls.csv',controls),('jitter_robustness.csv',jitter_rows)]: _write(out/name,rows)
    (out/'configuration.json').write_text(json.dumps({'seeds':SEEDS,'gain_mv':60,'exposures':LEVELS,'jitter_range_ms':[-1,1],'jitter_trials':N_JITTER,'softmax_temperature_mv':1.0,'primary_graded_metric':'integrated_positive_depolarization'},indent=2))
    try:
      import matplotlib.pyplot as p
      z=[x for x in progress if x['condition']=='correct']; p.figure(); p.plot([np.mean([x['spike_accuracy'] for x in z if x['exposures']==e]) for e in LEVELS],label='spike');p.plot([np.mean([x['peak_accuracy'] for x in z if x['exposures']==e]) for e in LEVELS],label='peak');p.plot([np.mean([x['integrated_accuracy'] for x in z if x['exposures']==e]) for e in LEVELS],label='integrated');p.legend();p.savefig(out/'spike_vs_voltage_accuracy.png');p.close()
      z=[x for x in trials if x['condition']=='correct' and x['exposures']==50]; p.figure(); [p.plot([x['peak_a'],x['peak_b'],x['peak_c']]) for x in z[:3]];p.savefig(out/'membrane_response_examples.png');p.close();p.figure();p.bar([0,1,2],[z[0]['integrated_a'],z[0]['integrated_b'],z[0]['integrated_c']]);p.savefig(out/'candidate_preferences.png');p.close();p.figure();p.plot(LEVELS,[np.mean([x['integrated_accuracy'] for x in progress if x['condition']=='correct' and x['exposures']==e]) for e in LEVELS]);p.savefig(out/'graded_learning_curves.png');p.close();p.figure();p.plot(LEVELS,[np.mean([x['integrated_margin'] for x in progress if x['condition']=='correct' and x['exposures']==e]) for e in LEVELS]);p.savefig(out/'response_margin_progression.png');p.close();p.figure();p.bar(['none','correct','mismatch'],[np.mean([x['integrated_accuracy'] for x in controls if x['condition']==c]) for c in ['no_learning','correct','mismatched']]);p.savefig(out/'pairing_control_comparison.png');p.close();p.figure();p.scatter([float(x['integrated_margin']) for x in margins if x['condition']=='correct' and x['exposures']==50],[int(x['integrated_correct']) for x in margins if x['condition']=='correct' and x['exposures']==50]);p.savefig(out/'preference_reliability.png');p.close()
    except Exception: pass
    (out/'GRADED_RECOGNITION_REPORT.md').write_text('# Graded recognition diagnostic\n\nIntegrated positive membrane depolarization was predeclared as the primary graded response; peak depolarization is secondary and spike count remains the historical baseline. Softmax values use a fixed 1 mV temperature and are normalized choice weights, not calibrated probabilities. See the CSV artifacts for seed-level trials, controls, learning progression, and jitter robustness.\n')

if __name__=='__main__': run()
