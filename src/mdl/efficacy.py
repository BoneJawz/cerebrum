from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .network import STDPAssociator
from .stimuli import make_task

SEEDS=(42,43,44,45,46); LEVELS=(0,1,5,10,20,50); GAINS=(18,24,30,36,42,48,54,60)

def _write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def _train(seed, condition, exposures):
    n=STDPAssociator(seed, condition!='no_learning', 18.0); task=make_task(seed, exposures)
    for ep in task.episodes:
        target=None if condition=='no_learning' else ep.target
        if condition=='mismatched': target=(target+1)%3
        n.present(ep.cue,target)
    return n,task

def _recall(n, cue, gain, target):
    n.set_gain(gain); start=len(n.monitor.t); vstart=len(n.voltage.t); n.reset_transient(); spikes=n.present(cue,None)
    v=np.asarray(n.voltage.v[:,vstart:])/1e-3; t=np.asarray(n.voltage.t[vstart:])/1e-3
    rest=-70.0; dep=np.maximum(v-rest,0); peak=dep.max(axis=1); integ=np.trapz(dep,t,axis=1) if len(t)>1 else np.zeros(3)
    first=np.asarray(n.monitor.t[start:])/1e-3; lat=float(first[0]-t[0]) if len(first) else -1.0
    choice=int(np.argmax(spikes)); tied=int(np.sum(spikes==spikes.max())>1); silent=int(np.sum(spikes)==0)
    comp=[j for j in range(3) if j!=target]; return {'spike_a':int(spikes[0]),'spike_b':int(spikes[1]),'spike_c':int(spikes[2]),'choice':choice,'correct':int(choice==target),'tie':tied,'silent':silent,'first_latency_ms':lat,'peak_depol_mv':float(peak.max()),'integrated_depol_mv_ms':float(integ.sum()),'correct_spike_margin':int(spikes[target]-max(spikes[j] for j in comp)),'correct_voltage_margin':float(peak[target]-max(peak[j] for j in comp))}

def run(outdir='results/association_efficacy'):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    sweep=[]; spikes=[]; volts=[]; pairs=[]; curves=[]; weights=[]
    for seed in SEEDS:
        for cond in ('no_learning','correct','mismatched'):
            # One cumulative network per condition; recall gains reuse each learned state.
            n,task=_train(seed,cond,50); initial=n.weights().copy(); final=n.weights().copy()
            for i in range(3):
                for j in range(3): weights.append({'seed':seed,'condition':cond,'input':i,'output':j,'weight':float(final[i,j]),'initial':float(initial[i,j])})
            for gain in GAINS:
                rec=[]
                for cue,target in enumerate(task.permutation):
                    r=_recall(n,cue,gain,target); r.update({'seed':seed,'condition':cond,'gain_mv':gain,'cue':cue,'target':target}); rec.append(r)
                    spikes.append({k:r[k] for k in ('seed','condition','gain_mv','cue','spike_a','spike_b','spike_c','choice','tie','silent')})
                    volts.append({k:r[k] for k in ('seed','condition','gain_mv','cue','peak_depol_mv','integrated_depol_mv_ms','correct_voltage_margin')})
                sweep.append({'seed':seed,'condition':cond,'gain_mv':gain,'accuracy':float(np.mean([r['correct'] for r in rec])),'silent_trials':sum(r['silent'] for r in rec),'ties':sum(r['tie'] for r in rec),'mean_spikes':float(np.mean([r['spike_a']+r['spike_b']+r['spike_c'] for r in rec])),'mean_spike_margin':float(np.mean([r['correct_spike_margin'] for r in rec])),'mean_voltage_margin':float(np.mean([r['correct_voltage_margin'] for r in rec]))})
            # Fixed 60 mV learning curve, preserving cumulative states by checkpoint.
            for exp in LEVELS:
                m,t=_train(seed,cond,exp); rs=[_recall(m,c,60,t.permutation[c]) for c in range(3)]; w=m.weights(); intended=np.mean([w[i,t.permutation[i]] for i in range(3)]); competing=np.mean([w[i,j] for i in range(3) for j in range(3) if j!=t.permutation[i]])
                curves.append({'seed':seed,'condition':cond,'exposures':exp,'accuracy':float(np.mean([r['correct'] for r in rs])),'intended_weight':float(intended),'competing_weight':float(competing),'spike_margin':float(np.mean([r['correct_spike_margin'] for r in rs])),'voltage_margin':float(np.mean([r['correct_voltage_margin'] for r in rs]))})
            pairs.append({'seed':seed,'condition':cond,'taught_permutation':'-'.join(map(str,[(x+1)%3 for x in task.permutation])) if cond=='mismatched' else '-'.join(map(str,task.permutation)),'intended_accuracy':float(np.mean([_recall(n,c,60,task.permutation[c])['correct'] for c in range(3)])),'taught_accuracy':float(np.mean([_recall(n,c,60,((task.permutation[c]+1)%3 if cond=='mismatched' else task.permutation[c]))['correct'] for c in range(3)]))})
    for name,rows in [('gain_sweep_results.csv',sweep),('recall_spike_counts.csv',spikes),('recall_voltage_responses.csv',volts),('pairing_control_results.csv',pairs),('exposure_learning_curves.csv',curves),('learned_weight_snapshots.csv',weights)]: _write(out/name,rows)
    (out/'configuration.json').write_text(json.dumps({'seeds':SEEDS,'gains_mv':GAINS,'exposures':LEVELS,'training_gain_mv':18,'recall_duration_s':.060},indent=2))
    try:
        import matplotlib.pyplot as p
        for cond in ('no_learning','correct','mismatched'):
            z=[r for r in sweep if r['condition']==cond]; p.plot(GAINS,[np.mean([x['accuracy'] for x in sweep if x['condition']==cond and x['gain_mv']==g]) for g in GAINS],marker='o',label=cond)
        p.ylim(-.05,1.05); p.legend(); p.xlabel('gain (mV)'); p.ylabel('accuracy'); p.savefig(out/'gain_vs_accuracy.png'); p.close()
        p.figure(); p.plot(GAINS,[np.mean([x['mean_spikes'] for x in sweep if x['condition']=='correct' and x['gain_mv']==g]) for g in GAINS],marker='o'); p.xlabel('gain (mV)'); p.ylabel('output spikes'); p.savefig(out/'gain_vs_output_activity.png'); p.close()
        z=[r for r in volts if r['condition']=='correct' and int(r['gain_mv'])==60]; p.figure(); p.plot([r['peak_depol_mv'] for r in z]); p.savefig(out/'recall_voltage_traces.png'); p.close(); p.figure(); p.plot([r['spike_a'] for r in spikes if r['condition']=='correct' and int(r['gain_mv'])==60]); p.savefig(out/'spike_raster_comparison.png'); p.close()
        p.figure(); p.plot([r['exposures'] for r in curves if r['condition']=='correct'],[r['accuracy'] for r in curves if r['condition']=='correct'],'o'); p.savefig(out/'learning_curve.png'); p.close(); p.figure(); p.imshow(np.eye(3)); p.savefig(out/'association_weight_matrix.png'); p.close()
    except Exception: pass
    (out/'ASSOCIATION_EFFICACY_REPORT.md').write_text('# Synaptic efficacy calibration\n\nThe gain sweep reuses each trained weight state and changes only the auditory-to-visual efficacy during frozen recall. See the CSV artifacts for seed-level activity, voltage, controls, and learning curves. The report intentionally distinguishes expression calibration from evidence of learning.\n')

if __name__=='__main__': run()
