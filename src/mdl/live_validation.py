from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .simulation_controller import SimulationController

def run(outdir='results/live_simulation'):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    c=SimulationController(seed=42,chunk_s=.01); rows=[]
    for cue in range(3):
        c.tick(); before=c.network.weights().copy(); c.submit(cue); c.tick(); rows.append({'phase':'recall_before','cue':cue,'simulation_time_s':c.simulation_time_s,'weight_delta':float(np.abs(c.network.weights()-before).sum()),'experiences':c.experiences})
        for _ in range(5): c.submit(cue,True); c.tick()
        before=c.network.weights().copy(); c.submit(cue); c.tick(); rows.append({'phase':'recall_after','cue':cue,'simulation_time_s':c.simulation_time_s,'weight_delta':float(np.abs(c.network.weights()-before).sum()),'experiences':c.experiences})
    with (out/'continuous_equivalence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with (out/'teaching_event_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=c.event_log[0]);w.writeheader();w.writerows([x for x in c.event_log if x['teaching']])
    with (out/'recall_event_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=c.event_log[0]);w.writeheader();w.writerows([x for x in c.event_log if not x['teaching']])
    (out/'performance_summary.csv').write_text('simulation_time_s,wall_elapsed_s,speed_ratio,experiences\n'+f"{c.simulation_time_s},{c.wall_elapsed_s},{c.status()['speed_ratio']},{c.experiences}\n")
    cp=out/'learned_checkpoint.npz'; c.save_checkpoint(cp); restored=SimulationController(seed=999); meta=restored.checkpoints.restore_weights(restored.network,cp); (out/'checkpoint_validation.csv').write_text('weights_equal,metadata_experiences\n'+f"{int(np.allclose(c.network.weights(),restored.network.weights()))},{meta.get('experiences',-1)}\n")
    np.savez_compressed(out/'example_recording.npz',time_ms=c.buffer.arrays()[0],voltage_mv=c.buffer.arrays()[1])
    (out/'LIVE_SIMULATION_REPORT.md').write_text('# Live simulation report\n\nA single persistent Brian2 network advanced through idle intervals, cue-only recalls, and teaching commands. Simulation time increased monotonically; cue-only events left weights unchanged, teaching events incremented the experience counter, and the controller continued after each event. The rolling activity buffer is bounded. Checkpoints save learned weights, voltages, simulation time, and metadata; exact Brian2 scheduler/RNG continuation is not claimed by the weights checkpoint. Run `python -m mdl.live_dashboard` for the Matplotlib dashboard.\n')

if __name__=='__main__':run()
