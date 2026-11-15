from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .efficacy import _train, _recall, SEEDS

def _write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(outdir='results/association_path_analysis'):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    strengths=[]; drives=[]; responses=[]; failed=[]; controls=[]
    for seed in SEEDS:
        for condition in ('no_learning','correct','mismatched'):
            for exp in (0,50):
                n,task=_train(seed,condition,exp); w=n.weights().copy()
                for cue,target in enumerate(task.permutation):
                    taught=(target+1)%3 if condition=='mismatched' else target
                    row=w[cue]; rank=int(1+np.sum(row>row[target])); intended=target
                    strengths.append({'seed':seed,'condition':condition,'exposures':exp,'cue':cue,'target':target,'w_a':float(row[0]),'w_b':float(row[1]),'w_c':float(row[2]),'intended_strength':float(row[intended]),'strongest_competitor':float(max(row[j] for j in range(3) if j!=intended)),'margin':float(row[intended]-max(row[j] for j in range(3) if j!=intended)),'intended_rank':rank,'taught_target':taught})
                    r=_recall(n,cue,60.0,target); peak=np.zeros(3); # response values are measured by _recall's margins; repeat for full peak vector
                    n.set_gain(60); vs=len(n.voltage.t); n.reset_transient(); sp=n.present(cue,None); vv=np.asarray(n.voltage.v[:,vs:])/1e-3; peak=np.max(vv,axis=1) if vv.size else np.full(3,-70.)
                    drive=row*60.0
                    choice=int(np.argmax(sp)); drive_rank=int(1+np.sum(drive>drive[target])); voltage_choice=int(np.argmax(peak)); spike_choice=choice
                    if rank!=1: category='incorrect_learning'
                    elif drive_rank!=1: category='effective_drive_competition'
                    elif r['silent'] or r['tie']: category='subthreshold_or_tie'
                    elif choice!=target: category='readout_failure'
                    else: category='successful_association'
                    rec={'seed':seed,'condition':condition,'exposures':exp,'cue':cue,'target':target,'choice':choice,'spike_a':int(sp[0]),'spike_b':int(sp[1]),'spike_c':int(sp[2]),'peak_a_mv':float(peak[0]),'peak_b_mv':float(peak[1]),'peak_c_mv':float(peak[2]),'silent':r['silent'],'tie':r['tie'],'weight_rank':rank,'drive_rank':drive_rank,'voltage_choice':voltage_choice,'category':category}
                    responses.append(rec)
                    drives.append({'seed':seed,'condition':condition,'exposures':exp,'cue':cue,'input_spikes':1,'event_time_ms':10.0,'drive_a_mv':float(drive[0]),'drive_b_mv':float(drive[1]),'drive_c_mv':float(drive[2]),'weight_rank':rank,'drive_rank':drive_rank})
                    if exp==50 and condition=='correct' and category!='successful_association': failed.append(rec)
            if condition=='mismatched': controls.append({'seed':seed,'condition':condition,'evaluated_against':'original','n':3,'accuracy':float(np.mean([x['choice']==x['target'] for x in responses if x['seed']==seed and x['condition']==condition and x['exposures']==50]))})
    for name,rows in [('association_strengths.csv',strengths),('effective_synaptic_drive.csv',drives),('output_responses.csv',responses),('failed_recognition_trials.csv',failed),('control_comparison.csv',controls)]: _write(out/name,rows)
    (out/'configuration.json').write_text(json.dumps({'seeds':SEEDS,'recall_gain_mv':60,'exposures':(0,50),'conditions':('no_learning','correct','mismatched')},indent=2))
    try:
        import matplotlib.pyplot as p
        z=[x for x in strengths if x['condition']=='correct' and x['exposures']==50]; p.figure(); p.imshow(np.array([[x['w_a'] for x in z],[x['w_b'] for x in z]]),aspect='auto'); p.colorbar(); p.savefig(out/'learned_association_matrices.png'); p.close()
        z=[x for x in responses if x['condition']=='correct' and x['exposures']==50]; p.figure(); p.scatter([x['weight_rank'] for x in z],[x['choice']==x['target'] for x in z]); p.xlabel('weight rank'); p.ylabel('selected correctly'); p.savefig(out/'connection_vs_response.png'); p.close()
        p.figure(); p.plot([x['peak_a_mv'] for x in z],label='A'); p.plot([x['peak_b_mv'] for x in z],label='B'); p.plot([x['peak_c_mv'] for x in z],label='C'); p.legend(); p.savefig(out/'recall_failure_examples.png'); p.close()
        cats={}; [cats.update({x['category']:cats.get(x['category'],0)+1}) for x in failed]; p.figure(); p.bar(list(cats),list(cats.values())); p.xticks(rotation=30); p.tight_layout(); p.savefig(out/'failure_classification.png'); p.close()
    except Exception: pass
    (out/'ASSOCIATION_PATH_ANALYSIS_REPORT.md').write_text('# Individual association path analysis\n\nThis report traces each cue through the learned auditory-to-visual weights, effective 60 mV synaptic drive, membrane peak, spike count, and argmax selection. It does not alter the STDP rule, neuron equation, gain, or network size. See the CSV files for complete seed/cue records and the figures for aggregate diagnostics.\n')

if __name__=='__main__': run()
