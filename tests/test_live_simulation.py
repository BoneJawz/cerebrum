import numpy as np
from mdl.simulation_controller import SimulationController

def test_persistent_controller_advances_and_queues_without_rebuild():
    c=SimulationController(seed=11,chunk_s=.005); ident=id(c.network)
    c.tick(); t0=c.simulation_time_s; c.submit(0); c.submit(1,True)
    for _ in range(30): c.tick()
    assert id(c.network)==ident and c.simulation_time_s>t0 and c.paired_experiences==1

def test_pause_resume_and_cue_learning_gate():
    c=SimulationController(seed=12,chunk_s=.005); c.tick(); c.pause(); t=c.simulation_time_s; c.tick(); assert c.simulation_time_s==t
    c.resume(); before=c.network.weights().copy(); c.submit(0)
    for _ in range(15): c.tick()
    assert np.allclose(before,c.network.weights())
    c.submit(0,True)
    for _ in range(15): c.tick()
    assert not np.allclose(before,c.network.weights())

def test_rolling_buffer_is_bounded():
    c=SimulationController(seed=13,chunk_s=.001,buffer_samples=3)
    for _ in range(10): c.tick()
    assert len(c.buffer.times)<=3
