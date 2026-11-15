from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from brian2 import mV

from mdl import DistributedVisualAssociator, ModelConfig
from mdl.sensory import encode_visual_array
from mdl.state import restore_state, save_state


REFERENCE = Path(__file__).parent / "reference" / "distributed_seed123.npz"


def vertical_pattern() -> np.ndarray:
    pattern = np.zeros((5, 5), dtype=np.uint8)
    pattern[:, 2] = 1
    return pattern


def test_default_dimensions_fixed_topology_and_weight_bounds():
    model = DistributedVisualAssociator(seed=123)
    assert len(model.visual_input) == 25
    assert len(model.auditory_input) == 3
    assert len(model.associative) == 18
    assert model.weights().shape == (3, 18)
    assert model.visual_edges.shape[1] == 2
    assert len(model.visual_projection) == len(model.visual_edges)
    assert np.all(model.weights() >= 0.0) and np.all(model.weights() <= 1.0)
    assert model.config.rest_mv == -70.0
    assert model.config.threshold_mv == -50.0
    assert model.config.membrane_tau_ms == 10.0
    assert model.config.refractory_ms == 5.0
    assert model.config.timestep_ms == 0.1


def test_original_distributed_reference_run_is_preserved():
    reference = np.load(REFERENCE)
    model = DistributedVisualAssociator(seed=123)
    assert np.array_equal(model.visual_edges, reference["visual_edges"])
    assert np.allclose(model.weights(), reference["weights_initial"], rtol=0, atol=1e-12)

    first = model.run_event(cue=1, visual=vertical_pattern())
    assert np.allclose(first["voltage_mv"], reference["first_voltage_mv"], rtol=0, atol=1e-9)
    assert np.array_equal(first["spike_ids"], reference["first_spike_ids"])
    assert np.array_equal(first["spike_times_s"], reference["first_spike_times_s"])
    assert np.array_equal(first["auditory_ids"], reference["first_audio_ids"])
    assert np.array_equal(first["visual_ids"], reference["first_visual_ids"])

    model.rest(0.020)
    assert np.allclose(model.weights(), reference["weights_after_pair"], rtol=0, atol=1e-12)
    model.set_plasticity(False)
    recall = model.run_event(cue=1)
    assert np.allclose(recall["voltage_mv"], reference["recall_voltage_mv"], rtol=0, atol=1e-9)
    assert np.array_equal(recall["spike_ids"], reference["recall_spike_ids"])
    assert float(model.net.t) == pytest.approx(float(reference["final_time_s"]))


def test_no_learning_and_paired_learning_smoke_controls():
    pattern = vertical_pattern()
    frozen = DistributedVisualAssociator(seed=123, learning=False)
    frozen_before = frozen.weights()
    frozen.run_event(cue=1, visual=pattern)
    assert np.array_equal(frozen.weights(), frozen_before)

    paired = DistributedVisualAssociator(seed=123, learning=True)
    before = paired.weights()
    paired.run_event(cue=1, visual=pattern)
    after = paired.weights()
    assert np.abs(after - before).sum() > 0.0
    assert np.all((after >= 0.0) & (after <= 1.0))


def test_raw_sensory_events_reach_requested_channels():
    model = DistributedVisualAssociator(seed=7, learning=False)
    pattern = vertical_pattern()
    assert np.array_equal(encode_visual_array(pattern, (5, 5)), np.array([2, 7, 12, 17, 22]))
    result = model.run_event(cue=2, visual=pattern)
    assert np.array_equal(result["auditory_ids"], np.array([2]))
    assert np.array_equal(result["visual_ids"], np.array([2, 7, 12, 17, 22]))
    assert result["voltage_mv"].shape == (18, 600)


def test_continuous_run_keeps_network_state_and_time():
    model = DistributedVisualAssociator(seed=8)
    model.run_event(cue=1, visual=vertical_pattern())
    learned = model.weights()
    before_t = float(model.net.t)
    before_v = np.asarray(model.associative.v[:] / 1e-3, dtype=float)
    model.rest(0.010)
    after_v = np.asarray(model.associative.v[:] / 1e-3, dtype=float)
    assert float(model.net.t) > before_t
    assert np.array_equal(model.weights(), learned)
    assert not np.array_equal(after_v, np.full(18, -70.0))
    assert np.any(before_v != -70.0)


def test_seed_reproducibility_and_configurable_dimensions():
    a, b = DistributedVisualAssociator(seed=19), DistributedVisualAssociator(seed=19)
    assert np.array_equal(a.visual_edges, b.visual_edges)
    assert np.array_equal(a.weights(), b.weights())
    c = DistributedVisualAssociator(ModelConfig(seed=19, visual_shape=(4, 4),
                                                auditory_inputs=2, associative_neurons=12))
    assert len(c.visual_input) == 16
    assert len(c.auditory_input) == 2
    assert c.weights().shape == (2, 12)


def test_checkpoint_round_trip_continues_and_rejects_incompatible_config(tmp_path):
    original = DistributedVisualAssociator(seed=33)
    original.run_event(cue=1, visual=vertical_pattern())
    path = tmp_path / "state.b2"
    save_state(original, path)

    restored = DistributedVisualAssociator(seed=33)
    restore_state(restored, path)
    assert float(restored.net.t) == pytest.approx(float(original.net.t))
    assert np.array_equal(restored.weights(), original.weights())
    assert np.allclose(np.asarray(restored.associative.v[:] / mV),
                       np.asarray(original.associative.v[:] / mV))
    assert np.allclose(restored.auditory_stdp.apre[:], original.auditory_stdp.apre[:])
    assert np.allclose(restored.auditory_stdp.apost[:], original.auditory_stdp.apost[:])

    original.set_plasticity(False)
    restored.set_plasticity(False)
    left = original.run_event(cue=0)
    right = restored.run_event(cue=0)
    assert np.allclose(left["voltage_mv"], right["voltage_mv"])
    assert np.array_equal(left["spike_ids"], right["spike_ids"])

    incompatible = DistributedVisualAssociator(
        ModelConfig(seed=33, associative_neurons=17))
    with pytest.raises(ValueError, match="does not match"):
        restore_state(incompatible, path)
