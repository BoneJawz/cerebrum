import numpy as np
from mdl.simulation_controller import SimulationController

def _finish(c):
    for _ in range(20): c.tick()

def test_continuous_plasticity_gate_stays_open_during_idle():
    c=SimulationController(seed=31,chunk_s=.01); c.set_learning_mode('continuous'); c.tick()
    assert c.plasticity is True

def test_continuous_audio_only_can_change_weights_after_pair_event():
    c=SimulationController(seed=32,chunk_s=.01); c.set_learning_mode('continuous')
    c.submit_pair(0,0); _finish(c); before=c.network.weights().copy()
    c.submit_audio(0); _finish(c)
    assert np.abs(c.network.weights()-before).sum() > 0
    assert c.event_log[-1]['event_type']=='auditory_only'

def test_audio_mute_does_not_change_scheduled_neural_input():
    muted=SimulationController(seed=36,chunk_s=.01); audible=SimulationController(seed=36,chunk_s=.01)
    muted.audio_muted=True; audible.audio_muted=False
    muted.submit_audio(1); audible.submit_audio(1)
    _finish(muted); _finish(audible)
    assert np.array_equal(muted.network.input_monitor.count[:], audible.network.input_monitor.count[:])
    assert np.allclose(muted.network.weights(), audible.network.weights())

def test_full_checkpoint_restores_network_clock_and_weights(tmp_path):
    c=SimulationController(seed=33,chunk_s=.01); c.submit_pair(2,2); _finish(c)
    c.set_learning_mode('continuous'); c.set_environment_enabled(True); c.submit_audio(1)
    path=tmp_path/'continuation.b2'; c.save_full_checkpoint(path)
    restored=SimulationController(seed=33,chunk_s=.01); restored.restore_full_checkpoint(path)
    assert np.isclose(restored.simulation_time_s,c.simulation_time_s)
    assert np.allclose(restored.network.weights(),c.network.weights())
    assert restored.learning_mode=='continuous' and restored.environment_enabled
    assert restored.scheduler.snapshot()==c.scheduler.snapshot()
