import numpy as np
from mdl.network import STDPAssociator
from mdl.recorder import record_trial, save_recording

def test_recording_contains_aligned_voltage_and_spikes(tmp_path):
    n=STDPAssociator(seed=7, learning=False)
    trial=record_trial(n,0,None)
    assert trial['voltage_mv'].shape[0] == 3
    assert trial['time_ms'].shape[0] == trial['voltage_mv'].shape[1]
    path=tmp_path/'trial.npz'; save_recording(path,trial,{'phase':'recall'})
    loaded=np.load(path,allow_pickle=True)
    assert loaded['voltage_mv'].shape == trial['voltage_mv'].shape

def test_recording_does_not_change_frozen_weights():
    n=STDPAssociator(seed=8, learning=False); before=n.weights().copy(); record_trial(n,1,None)
    assert np.allclose(before,n.weights())
