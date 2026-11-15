from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .network import STDPAssociator
from .stimuli import make_task

SEEDS=(42,43,44,45,46); LEVELS=(0,1,5,10,20,50)

def _write(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(outdir='results/association_diagnostics'):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    manual=[]; weights=[]; timing=[]; recall=[]; assessment=[]
    # Checkpoint A: deliberately diagonal, frozen connectivity.
    for seed in SEEDS:
        for cue in range(3):
            n=STDPAssociator(seed,False,60.0); n.set_weights(np.eye(3))
            spikes=n.present(cue,None); v=np.asarray(n.voltage.v[:])/1e-3
            manual.append({'seed':seed,'cue':cue,'spike_a':int(spikes[0]),'spike_b':int(spikes[1]),'spike_c':int(spikes[2]),
                           'choice':int(np.argmax(spikes)),'tie':int(np.sum(spikes==spikes.max())>1),'silent':int(np.sum(spikes)==0),
                           'peak_a_mv':float(v[0].max()),'peak_b_mv':float(v[1].max()),'peak_c_mv':float(v[2].max())})
    # Checkpoints B/C: existing STDP, with complete matrices and activity.
    for seed in SEEDS:
        for level in LEVELS:
            for condition in ('no_learning','correct','mismatched'):
                n=STDPAssociator(seed, condition!='no_learning')
                initial=n.weights().copy(); task=make_task(seed,level)
                for ep in task.episodes:
                    target=None if condition=='no_learning' else ep.target
                    if condition=='mismatched': target=(target+1)%3
                    n.present(ep.cue,target)
                final=n.weights().copy()
                for i in range(3):
                    for j in range(3):
                        weights.append({'seed':seed,'exposures':level,'condition':condition,'input':i,'output':j,
                                        'before':float(initial[i,j]),'after':float(final[i,j]),'delta':float(final[i,j]-initial[i,j]),
                                        'intended':int(task.permutation[i]==j)})
                for cue,target in enumerate(task.permutation):
                    before=len(n.monitor.t); spikes=n.present(cue,None); t=np.asarray(n.voltage.t[:]/1e-3); v=np.asarray(n.voltage.v[:])/1e-3
                    rr={'seed':seed,'exposures':level,'condition':condition,'cue':cue,'target':target,
                        'spike_a':int(spikes[0]),'spike_b':int(spikes[1]),'spike_c':int(spikes[2]),
                        'choice':int(np.argmax(spikes)),'tie':int(np.sum(spikes==spikes.max())>1),'silent':int(np.sum(spikes)==0),
                        'peak_mv':float(v.max()),'first_spike_ms':float(np.min(np.asarray(n.monitor.t[before:])/1e-3)) if len(n.monitor.t)>before else -1.0}
                    recall.append(rr); assessment.append({k:rr[k] for k in ('seed','exposures','condition','cue','target','choice','tie','silent')})
    _write(out/'manual_weight_results.csv',manual); _write(out/'synaptic_weight_matrices.csv',weights); _write(out/'recall_activity.csv',recall); _write(out/'assessment_diagnostics.csv',assessment)
    _write(out/'stdp_timing_results.csv',[{'seed':r['seed'],'exposures':r['exposures'],'condition':r['condition'],'cue':r['cue'],'first_spike_ms':r['first_spike_ms']} for r in recall])
    (out/'diagnostic_config.json').write_text(json.dumps({'seeds':SEEDS,'exposures':LEVELS,'manual_gain_mv':60.0,'rule':'existing pair-based additive STDP'},indent=2))
    try:
        import matplotlib.pyplot as p
        p.figure(); p.plot([r['cue'] for r in manual],[r['choice']==r['cue'] for r in manual],'o'); p.ylim(-.1,1.1); p.title('Manual diagonal recognition'); p.savefig(out/'manual_weight_validation.png'); p.close()
        c=[r for r in weights if r['condition']=='correct' and r['exposures']==50]; mat=np.zeros((3,3));
        for r in c: mat[int(r['input']),int(r['output'])]=r['after']
        p.figure(); p.imshow(mat); p.colorbar(); p.title('STDP weights, correct, 50 exposures'); p.savefig(out/'association_weight_matrices.png'); p.close()
        p.figure(); p.plot([r['spike_a'] for r in recall[:3]],label='A'); p.plot([r['spike_b'] for r in recall[:3]],label='B'); p.plot([r['spike_c'] for r in recall[:3]],label='C'); p.legend(); p.title('Recall spike counts'); p.savefig(out/'output_spike_counts.png'); p.close()
    except Exception: pass
    (out/'ASSOCIATION_DIAGNOSTIC_REPORT.md').write_text('# Brian2 association diagnostic\n\n'
      'Checkpoint A uses frozen diagonal weights and a corrected resting-potential diagnostic configuration (60 mV test gain). '
      'The diagnostic files contain the measured spike counts, voltages, matrices, timing, ties, and silent trials. '
      'Checkpoint B/C retain the original STDP rule and controls; this run does not add a learning mechanism.\n')

if __name__=='__main__': run()
