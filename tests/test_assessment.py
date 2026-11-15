import numpy as np
from mdl.assessment import choose

def test_candidate_choice_is_deterministic():
    assert choose(np.array([0., 2., 1.])) == 1
    assert choose(np.array([2., 2., 1.])) == 0
