import csv
from pathlib import Path
import numpy as np
from mdl.network import STDPAssociator

def test_weight_rows_follow_input_output_order():
    n=STDPAssociator(seed=5, learning=False)
    matrix=np.array([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]])
    n.set_weights(matrix)
    assert np.allclose(n.weights(), matrix)

def test_frozen_recall_keeps_weights_and_teacher_is_absent():
    n=STDPAssociator(seed=6, learning=False)
    n.set_weights(np.eye(3)); before=n.weights().copy(); n.set_gain(60); n.present(0, None)
    assert np.allclose(before, n.weights())
    assert n.present(1, None).shape == (3,)
