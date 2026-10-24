import numpy as np

from sensory_lab.benchmark import _ablate
from sensory_lab.environment import LineWorld, MOTION_START_TICK, WorldConfig
from sensory_lab.research_benchmark import (
    ResearchConfig, extract_matrix, fit_validate_predict, make_split, run_all,
)


def small_config():
    return ResearchConfig(
        seed=3, seeds=2, train_sizes=(8, 12), generalization_train_episodes=12,
        validation_per_combo=2, test_per_combo=2,
        horizons=(3, 6, 9, 12), hidden=8, timed_repeats=2,
    )


def test_compositional_split_excludes_heldout_from_train_and_validation():
    config = small_config()
    split = make_split(config, 12)
    heldout = set(config.heldout_combinations)
    observed = {(d, v) for d in (-1, 1) for v in config.velocities} - heldout
    assert {x['combination'] for x in split.train_pool}.issubset(observed)
    assert {x['combination'] for x in split.validation}.issubset(observed)
    assert {x['combination'] for x in split.heldout_test} == heldout
    assert observed.issubset({x['combination'] for x in split.train_pool})
    assert split.train_pool[0]['trajectory_seed'] != split.validation[0]['trajectory_seed']
    assert split.train_pool[0]['trajectory_seed'] != split.heldout_test[0]['trajectory_seed']


def test_horizon_targets_and_sensor_inputs_are_temporally_separated():
    split = make_split(small_config(), 22)
    for item in split.heldout_test:
        d, v = item['direction'], item['velocity']
        x0 = item['initial_position']
        for horizon in (3, 6, 9, 12):
            frame = MOTION_START_TICK + horizon
            assert item['target_frames'][horizon] == frame
            assert item['targets'][horizon] == x0 + d * v * horizon
            assert frame > 4
            target = item['targets'][horizon]
            assert 2 <= target < len(item['observations'][0].vision) - 2
        assert all(set(obs.__dataclass_fields__) == {'t', 'vision', 'audio'}
                   for obs in item['observations'][:5])
        assert all(np.array_equal(obs.vision, item['observations'][0].vision)
                   for obs in item['observations'][:5])


def test_audio_encodes_speed_frequency_and_direction_phase():
    config = small_config()
    split = make_split(config, 24)
    phase = split.phase_by_direction
    assert phase[-1] == -phase[1]
    assert abs(abs(phase[1]) - np.pi/2) < 1e-12
    assert 120 + 180*3 > 120 + 180*2 > 120 + 180*1
    world = LineWorld(WorldConfig(pixels=129, ticks=18, noise_std=0.0))
    for direction in (-1, 1):
        for velocity in (1, 2, 3):
            observation = world.episode(7, direction, 120+180*velocity,
                                        initial_position=64, velocity=velocity,
                                        phase_offset=phase[direction])[0]
            bins = np.fft.rfftfreq(len(observation.audio), d=1/2000)
            peak_hz = bins[np.argmax(np.abs(np.fft.rfft(observation.audio)))]
            assert peak_hz == 120 + 180*velocity
            assert np.sign(observation.audio[0]) == np.sign(phase[direction])
    assert not hasattr(split.train_pool[0]['observations'][0], 'velocity')
    assert not hasattr(split.train_pool[0]['observations'][0], 'direction')


def test_supervised_readout_improves_seen_pair_prediction_over_constant():
    config = ResearchConfig(
        seed=42, seeds=2, train_sizes=(100,), generalization_train_episodes=100,
        validation_per_combo=10, test_per_combo=10, hidden=16, timed_repeats=2,
    )
    split = make_split(config, 42)
    train = split.train_pool[:100]
    evaluation = split.observed_test
    features, _ = extract_matrix("B", train, "combined", config, 42,
                                 split.world_config.pixels)
    validation, _ = extract_matrix("B", split.validation, "combined", config, 42,
                                   split.world_config.pixels)
    eval_features, _ = extract_matrix("B", evaluation, "combined", config, 42,
                                      split.world_config.pixels)
    horizon = 9
    y_train = np.asarray([[x['targets'][horizon]] for x in train])
    y_val = np.asarray([[x['targets'][horizon]] for x in split.validation])
    y_eval = np.asarray([x['targets'][horizon] for x in evaluation])
    prediction, _, _, _ = fit_validate_predict(
        features, y_train, validation, y_val, eval_features, config.alphas, "quadratic")
    constant = np.full_like(y_eval, np.mean(y_train))
    assert np.mean(np.abs(prediction[:, 0] - y_eval)) < np.mean(np.abs(constant - y_eval))


def test_ablation_zeros_only_the_requested_stream():
    observation = LineWorld(WorldConfig()).episode(4, 1, 300)[0]
    vision_only = _ablate(observation, 'vision_only')
    audio_only = _ablate(observation, 'audio_only')
    assert np.array_equal(vision_only.vision, observation.vision)
    assert np.count_nonzero(vision_only.audio) == 0
    assert np.array_equal(audio_only.audio, observation.audio)
    assert np.count_nonzero(audio_only.vision) == 0


def test_advanced_outputs_and_reproducibility(tmp_path):
    config = small_config()
    first = run_all(config, tmp_path / 'first')
    second = run_all(config, tmp_path / 'second')
    for name in ('generalization_results.csv', 'sample_efficiency_results.csv',
                 'compute_benchmark.csv', 'generalization_heatmap.png',
                 'learning_curves.png', 'latency_comparison.png', 'EXPERIMENT_REPORT.md'):
        assert (tmp_path / 'first' / name).exists()
        assert (tmp_path / 'second' / name).exists()
    g1 = {(r['model'],r['modality'],r['horizon_intervals'],r['evaluation_split']):r['mae_pixels_mean']
          for r in first['generalization']}
    g2 = {(r['model'],r['modality'],r['horizon_intervals'],r['evaluation_split']):r['mae_pixels_mean']
          for r in second['generalization']}
    assert g1 == g2
