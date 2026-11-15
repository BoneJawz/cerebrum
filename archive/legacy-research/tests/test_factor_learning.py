import numpy as np

from sensory_lab.adaptive_dynamics import load_checkpoint, save_checkpoint
from sensory_lab.environment import WorldConfig
from sensory_lab.factor_environment import FactorWorld
from sensory_lab.factor_learning import (
    HELDOUT, _stage_commands, make_model, make_training_pools, prepare_factor_batch,
    split_start_positions,
    train_scheduled,
)


def test_strict_factor_stages_never_expose_heldout_motion_pairs():
    world = FactorWorld(WorldConfig(pixels=177, ticks=24))
    train_positions, _ = split_start_positions(world)
    pools = make_training_pools(world, 12, 500, "strict", train_positions, 1)
    seen = {(item["direction"], item["speed"])
            for stage in pools.values() for item in stage}
    assert not (seen & set(HELDOUT))


def test_transfer_speed_pretraining_explicitly_exposes_heldout_pairs():
    rng = np.random.default_rng(7)
    commands = _stage_commands("speed_focus", 600, "transfer", rng)
    assert set(HELDOUT) <= set(commands)
    assert all(sum(speed == level for _, speed in commands) == 200
               for level in (1, 2, 3))


def test_factor_batch_uses_only_requested_observation_prefix_and_future_targets():
    world = FactorWorld(WorldConfig(pixels=177, ticks=24))
    episode = world.episode(31, 1, 2, initial_position=80)
    item = {"episode": episode}
    x, y = prepare_factor_batch([item], world, 4, horizons=(1, 2, 3))
    assert x.shape[1] == 10
    assert np.all(x[:, 5:] == 0)
    assert np.all(np.isfinite(y))
    assert y[0, 0] == 82


def test_factor_data_generation_is_seed_reproducible_and_observations_hide_labels():
    world = FactorWorld(WorldConfig(pixels=177, ticks=24))
    left = world.episode(99, -1, 3, initial_position=80)
    again = world.episode(99, -1, 3, initial_position=80)
    assert left.cue_ids == again.cue_ids
    for a, b in zip(left.observations, again.observations):
        np.testing.assert_array_equal(a.vision, b.vision)
        np.testing.assert_array_equal(a.audio, b.audio)
        assert not hasattr(a, "direction")
        assert not hasattr(a, "speed")


def test_curriculum_and_shuffled_orders_keep_the_same_update_budget():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(6, 5, 3))
    y = rng.normal(size=(6, 12))
    labels = ["direction_focus", "direction_focus", "speed_focus",
              "speed_focus", "combined", "combined"]
    models = []
    for schedule in ("curriculum", "shuffled"):
        model = make_model("F", 3, 17)
        train_scheduled(model, x, y, labels, schedule, 22, epochs=2,
                        batch_size=2, val_x=x, val_y=y)
        models.append(model)
    assert models[0].opt_step == models[1].opt_step == 6


def test_ten_tick_direct_checkpoint_round_trip_keeps_sequence_length(tmp_path):
    model = make_model("FF_direct", 3, 23)
    x = np.random.default_rng(2).normal(size=(2, 10, 3))
    expected = model.predict_rollout(x, (1, 6, 12))
    path = tmp_path / "direct-10-tick.npz"
    save_checkpoint(path, model, {"sequence_length": 10})
    restored = load_checkpoint(path, kind="FF_direct")
    assert restored.sequence_length == 10
    actual = restored.predict_rollout(x, (1, 6, 12))
    for horizon in expected:
        np.testing.assert_allclose(actual[horizon], expected[horizon])


def test_recurrent_rollout_ignores_explicit_padding_without_reclassifying_zero_sensors():
    rng = np.random.default_rng(8)
    model = make_model("F", 3, 29)
    model.input_mean[:] = 0.25
    x = rng.normal(size=(2, 10, 3))
    x[:, 5:] = 0.0
    valid = np.zeros((2, 10), dtype=bool)
    valid[:, :5] = True
    padded = model.predict_rollout(x, (3, 9), observation_mask=valid)
    prefix = model.predict_rollout(x[:, :5], (3, 9))
    for horizon in padded:
        np.testing.assert_allclose(padded[horizon], prefix[horizon])
    valid_zero = x[:, :5].copy()
    valid_zero[:, 2] = 0.0
    assimilated = model.predict_rollout(valid_zero, (3,))
    skipped_mask = np.ones((2, 5), dtype=bool)
    skipped_mask[:, 2] = False
    skipped = model.predict_rollout(valid_zero, (3,), observation_mask=skipped_mask)
    assert not np.allclose(assimilated[3], skipped[3])
