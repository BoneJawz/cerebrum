import dataclasses

import numpy as np

from sensory_lab import research_benchmark as rb
from sensory_lab.adaptive_dynamics import (
    DirectTrajectoryBaseline,
    EXTRAPOLATION_HORIZON,
    TrainableStateModel,
    latent_cluster_row,
    load_checkpoint,
    prepare_examples,
    save_checkpoint,
    train_model,
)


def tiny_split(seed=42):
    cfg = rb.ResearchConfig(seed=seed, seeds=2, train_sizes=(4,),
                            generalization_train_episodes=8,
                            validation_per_combo=2, test_per_combo=2)
    cfg = dataclasses.replace(cfg, horizons=tuple(range(1, 13)) + (EXTRAPOLATION_HORIZON,))
    return rb.make_split(cfg, seed)


def test_temporal_alignment_and_heldout_exclusion():
    split = tiny_split()
    assert {it["combination"] for it in split.train_pool}.isdisjoint({(-1, 2), (1, 3)})
    item = split.train_pool[0]
    direction, speed = item["combination"]
    initial = item["initial_position"]
    for h in range(1, 13):
        assert item["targets"][h] == initial + direction * speed * h


def test_model_uses_only_observed_prefix_and_resets_per_episode():
    split = tiny_split()
    x, _ = prepare_examples(split.train_pool[:2], "combined", split.world_config.pixels)
    model = TrainableStateModel(x.shape[-1], seed=11, kind="F")
    model.input_mean = x.mean((0, 1)); model.input_scale = x.std((0, 1))
    model.input_scale[model.input_scale < 1e-8] = 1
    pred = model.predict_rollout(x[:1], (3, 6, 9, 12))
    twice = model.predict_rollout(np.concatenate([x[:1], x[:1]]), (3, 6, 9, 12))
    for h in (3, 6, 9, 12):
        np.testing.assert_allclose(pred[h][0], twice[h][1])


def test_bptt_updates_trainable_transition_and_is_reproducible():
    split = tiny_split()
    x, y = prepare_examples(split.train_pool[:4], "combined", split.world_config.pixels)
    a = TrainableStateModel(x.shape[-1], seed=4, kind="F")
    b = TrainableStateModel(x.shape[-1], seed=4, kind="F")
    initial = a.params["a"].copy()
    sa = train_model(a, x, y, x, y, seed=8, epochs=20, batch_size=4)
    sb = train_model(b, x, y, x, y, seed=8, epochs=20, batch_size=4)
    assert sa.losses[-1] < sa.losses[0]
    assert not np.array_equal(initial, a.params["a"])
    np.testing.assert_array_equal(a.params["a"], b.params["a"])
    assert np.isfinite(sa.validation_mae)


def test_euler_rollout_is_finite_and_event_threshold_counts_skips():
    split = tiny_split()
    x, _ = prepare_examples(split.heldout_test, "combined", split.world_config.pixels)
    model = TrainableStateModel(x.shape[-1], seed=8, kind="G")
    model.input_mean = x.mean((0, 1)); model.input_scale = x.std((0, 1))
    model.input_scale[model.input_scale < 1e-8] = 1
    continuous = model.predict_rollout(x, (3, 6, 9, 12, 18))
    assert all(np.all(np.isfinite(v)) for v in continuous.values())
    fine = TrainableStateModel(x.shape[-1], seed=8, kind="G", dt=0.025, tau=0.1)
    fine.params = {k:v.copy() for k,v in model.params.items()}
    fine.input_mean, fine.input_scale = model.input_mean.copy(), model.input_scale.copy()
    fine.target_mean, fine.target_scale = model.target_mean, model.target_scale
    assert fine.substeps == 2
    refined = fine.predict_rollout(x, (3, 9, 18))
    assert all(np.all(np.isfinite(v)) for v in refined.values())
    event = TrainableStateModel(x.shape[-1], seed=8, kind="H")
    event.params = {k:v.copy() for k,v in model.params.items()}
    event.input_mean, event.input_scale = model.input_mean.copy(), model.input_scale.copy()
    event.target_mean, event.target_scale = model.target_mean, model.target_scale
    event.predict_rollout(x, (9,), event_threshold=1e6)
    assert event.opportunities == len(x) * 9
    assert event.skipped == event.opportunities


def test_direct_head_and_checkpoint_round_trip(tmp_path):
    split = tiny_split()
    x, y = prepare_examples(split.train_pool[:4], "combined", split.world_config.pixels)
    model = DirectTrajectoryBaseline(x.shape[-1], seed=2)
    model.train(x, y, seed=3, epochs=4, val_x=x, val_y=y)
    expected = model.predict_rollout(x, (3, 9, 12))
    path = tmp_path / "direct.npz"
    save_checkpoint(path, model, {"seed":3,"kind":"FF_direct"})
    restored = load_checkpoint(path, kind="FF_direct")
    actual = restored.predict_rollout(x, (3, 9, 12))
    for h in expected:
        np.testing.assert_allclose(expected[h], actual[h])


def test_latent_cluster_diagnostic_is_descriptive_and_finite():
    split=tiny_split()
    x,_=prepare_examples(split.observed_test,"combined",split.world_config.pixels)
    model=TrainableStateModel(x.shape[-1],seed=2,kind="F")
    model.input_mean=x.mean((0,1));model.input_scale=x.std((0,1));model.input_scale[model.input_scale<1e-8]=1
    row=latent_cluster_row(model,x,split.observed_test)
    assert -1 <= row["mean_silhouette"] <= 1
    assert row["mean_within_distance"] >= 0
