import numpy as np
from mdl.graded import softmax

def test_softmax_preserves_ranking_and_normalizes():
    p=softmax([1.,2.,0.])
    assert np.isclose(p.sum(),1.)
    assert np.argmax(p)==1

def test_softmax_identical_responses_are_uniform():
    assert np.allclose(softmax([2.,2.,2.]), [1/3]*3)
