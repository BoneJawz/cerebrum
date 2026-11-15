import numpy as np
from mdl.network import STDPAssociator

def test_teacher_changes_weights_but_recall_has_no_teacher():
    net = STDPAssociator(seed=1, learning=True)
    before = net.weights().copy()
    net.present(0, 1)
    assert not np.allclose(before, net.weights())
    # The recall API supplies no target/teacher event.
    spikes = net.present(0, None)
    assert spikes.shape == (3,)

def test_learning_disabled_keeps_weights_fixed():
    net = STDPAssociator(seed=1, learning=False)
    before = net.weights().copy(); net.present(0, 1)
    assert np.allclose(before, net.weights())

def test_manual_diagonal_connectivity_discriminates():
    net = STDPAssociator(seed=2, learning=False, synaptic_gain=60.0)
    net.set_weights(np.eye(3))
    for cue in range(3):
        spikes = net.present(cue, None)
        assert int(np.argmax(spikes)) == cue
        assert spikes[cue] > 0 and np.sum(spikes) == spikes[cue]

def test_transient_reset_preserves_weights():
    net = STDPAssociator(seed=3, learning=True)
    before = net.weights().copy(); net.present(0, 1); learned = net.weights().copy()
    net.reset_transient()
    assert np.allclose(learned, net.weights())
    assert not np.allclose(before, learned)

def test_gain_changes_efficacy_not_relative_weights():
    net = STDPAssociator(seed=4, learning=False)
    net.set_weights(np.array([[.2,.1,.05],[.1,.3,.05],[.05,.1,.4]]))
    before = net.weights().copy(); net.set_gain(60.0); net.present(0, None)
    assert np.allclose(before, net.weights())
    assert float(net.plastic.gain / 1e-3) == 60.0
