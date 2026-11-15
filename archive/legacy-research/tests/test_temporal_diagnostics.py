import numpy as np

from sensory_lab.contracts import Observation
from sensory_lab.liquid import LiquidConfig, LiquidNetwork
from sensory_lab.temporal_diagnostics import (
    _dual_ridge, _make_trials, _new_network, _projection_matrix,
)


def test_dual_ridge_matches_primal_standardized_solution():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 5))
    y = rng.normal(size=(12, 3))
    xt = rng.normal(size=(4, 5))
    scores, _ = _dual_ridge(x, y, xt, 0.3)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-10] = 1.0
    z, zt = (x - mean) / scale, (xt - mean) / scale
    ymean = y.mean(0)
    coef = np.linalg.solve(z.T @ z + 0.3 * np.eye(z.shape[1]), z.T @ (y - ymean))
    np.testing.assert_allclose(scores, zt @ coef + ymean, atol=1e-10)


def test_three_position_partitions_are_reproducible_and_inputs_have_no_labels():
    counts = {"train": 2, "validation": 2, "test": 2}
    first, partitions, _ = _make_trials(22, counts)
    second, _, _ = _make_trials(22, counts)
    assert set(row["position"] for row in partitions) == {-1, 0, 1}
    ids = [{trial["trial_id"] for trial in first[split]} for split in counts]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for split in counts:
        for a, b in zip(first[split], second[split]):
            assert a["trial_id"] == b["trial_id"]
            np.testing.assert_array_equal(a["raw_audio"], b["raw_audio"])
            assert all(np.all(obs.vision == 0) for obs in a["observations"])
            assert all(not hasattr(obs, "position") for obs in a["observations"])


def test_fixed_projection_energy_is_matched_to_p0_audio_projection():
    seed = 35
    net = _new_network(seed, LiquidConfig())
    expected = np.sum(net._input_gains[199:] ** 2)
    for kind in ("P1", "P2"):
        matrix, _ = _projection_matrix(kind, net, seed)
        assert matrix.shape == (64, 69)
        np.testing.assert_allclose(np.sum(matrix ** 2), expected, rtol=1e-12)


def test_snapshot_reads_do_not_change_future_i0_trajectory():
    config = LiquidConfig()
    nets = [LiquidNetwork(268, seed=91, config=config, variant="I0", visual_dim=199)
            for _ in range(2)]
    obs0 = Observation(t=0.0, vision=np.zeros(199), audio=np.linspace(-1, 1, 100))
    obs1 = Observation(t=0.05, vision=np.zeros(199), audio=np.zeros(100))
    nets[0].ingest_observation(obs0)
    nets[1].ingest_observation(obs0)
    for time in (0.05, 0.10, 0.20):
        nets[0].snapshot(time)
    nets[0].ingest_observation(obs1)
    nets[1].ingest_observation(obs1)
    np.testing.assert_allclose(nets[0].snapshot(), nets[1].snapshot(), atol=0, rtol=0)
    assert nets[0].topology_summary() == nets[1].topology_summary()
