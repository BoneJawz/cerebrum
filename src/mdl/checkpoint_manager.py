from __future__ import annotations
from pathlib import Path
import json, os, tempfile
import numpy as np
from brian2 import mV

class CheckpointManager:
    def save_weights(self, network, path, metadata=None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
        np.savez_compressed(tmp,weights=network.weights(),voltage=np.asarray(network.output.v[:])/1e-3,
                            simulation_time_s=float(network.net.t),pre_trace=np.asarray(network.plastic.apre[:]),
                            post_trace=np.asarray(network.plastic.apost[:]),metadata=np.array(json.dumps(metadata or {})))
        os.replace(str(tmp)+'.npz',path)
        return path
    def restore_weights(self, network, path):
        z=np.load(path,allow_pickle=True); network.set_weights(z['weights']); network.output.v=z['voltage']*mV
        if 'pre_trace' in z: network.plastic.apre=z['pre_trace']
        if 'post_trace' in z: network.plastic.apost=z['post_trace']
        return json.loads(str(z['metadata']))

    def restore_full(self, network, path, restore_random_state=True):
        network.net.restore(name='continuation', filename=str(path), restore_random_state=restore_random_state)
