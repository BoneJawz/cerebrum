import numpy as np

from sensory_lab.experiential_tokens import (
    TOKENS, _dual_ridge_predict, _observations, _variant,
)
from sensory_lab.liquid_readout import RidgeReadout
from sensory_lab.spatial_environment import DigitSpatialVocabulary


def test_token_probe_observations_are_raw_sensor_only():
    waveform = DigitSpatialVocabulary(seed=4).token("7")
    observations = _observations(waveform)
    assert len(observations) == 4
    assert set(observations[0].__dataclass_fields__) == {"t", "vision", "audio"}
    assert observations[0].vision.shape == (199,)
    assert all(not np.any(item.vision) for item in observations)
    assert all(item.audio.shape == (100,) for item in observations)
    assert all(np.isfinite(item.audio).all() for item in observations)


def test_token_probe_inventory_has_signs_and_all_decimal_digits():
    assert TOKENS == ("+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def test_waveform_nuisance_variants_are_seed_reproducible():
    waveform = DigitSpatialVocabulary(seed=5).token("3")
    a = _variant(waveform, np.random.default_rng(9))
    b = _variant(waveform, np.random.default_rng(9))
    c = _variant(waveform, np.random.default_rng(10))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == waveform.shape


def test_wide_sample_space_ridge_matches_existing_readout():
    rng = np.random.default_rng(16)
    train_x = rng.normal(size=(12, 40))
    train_y = rng.normal(size=12)
    test_x = rng.normal(size=(4, 40))
    expected = RidgeReadout(alpha=1.0).fit(train_x, train_y).predict(test_x).ravel()
    actual = _dual_ridge_predict(train_x, train_y, test_x)
    assert np.allclose(actual, expected, atol=1e-10)
