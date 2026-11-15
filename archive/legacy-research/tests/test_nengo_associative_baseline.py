import numpy as np

from sensory_lab.nengo_associative_baseline import (
    EXPOSURES, POSITIONS, _audio_features, _projection, _require_nengo,
)
from sensory_lab.spatial_environment import DigitSpatialVocabulary


def test_nengo_dependency_and_feature_projection_are_deterministic():
    nengo = _require_nengo()
    assert nengo.__version__ == "4.0.0"
    vocab = DigitSpatialVocabulary(42)
    rng = np.random.default_rng(7)
    first = _audio_features(vocab, -1, rng)
    second = _audio_features(vocab, -1, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (69,)
    matrix_a = _projection(42)
    matrix_b = _projection(42)
    np.testing.assert_array_equal(matrix_a, matrix_b)
    assert matrix_a.shape == (69, 16)


def test_exposure_schedule_is_experience_count_not_optimizer_epochs():
    assert EXPOSURES == (0, 1, 5, 10, 20, 50)
    assert POSITIONS == (-1, 0, 1)
    assert all(a < b for a, b in zip(EXPOSURES, EXPOSURES[1:]))
