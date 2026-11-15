import numpy as np

from sensory_lab.liquid import LiquidConfig, LiquidNetwork
from sensory_lab.spatial_grounding import all_combination_training_positions
from sensory_lab.spatial_environment import (
    CHUNK_SIZE, EPISODE_TICKS, FIELD_MIN, POSITIONS, SAMPLE_RATE,
    DigitSpatialVocabulary, SpatialVocabulary, make_digit_episode, make_episode,
    rotate_composition_split, visual_array,
)


def test_center_reference_and_single_occupied_retinal_cell():
    assert np.flatnonzero(visual_array(0)).tolist() == [10]
    for position in POSITIONS:
        retina = visual_array(position)
        assert retina.shape == (21,)
        assert retina.sum() == 1
        assert np.flatnonzero(retina)[0] == position - FIELD_MIN


def test_digit_vocabulary_uses_stable_sequential_sign_and_digit_tokens():
    vocabulary = DigitSpatialVocabulary(seed=5)
    assert vocabulary.token_sequence(87) == ["+", "8", "7"]
    assert vocabulary.token_sequence(-7) == ["-", "7"]
    assert vocabulary.token_sequence(0) == ["0"]
    assert len(vocabulary.instruction(87)) == 16 * CHUNK_SIZE
    assert len(vocabulary.instruction(-7)) == 16 * CHUNK_SIZE
    assert np.array_equal(vocabulary.token("8"), vocabulary.token("8"))
    assert not np.array_equal(vocabulary.token("8"), vocabulary.token("7"))


def test_repeated_digit_waveform_is_identical_across_sign_and_place_value():
    vocabulary = DigitSpatialVocabulary(seed=17)
    # Tick spans: sign 0:4, silence 4:6, then most-significant digit.
    token_three = vocabulary.token("3")
    assert np.array_equal(vocabulary.instruction(3)[6 * CHUNK_SIZE:10 * CHUNK_SIZE], token_three)
    assert np.array_equal(vocabulary.instruction(-3)[6 * CHUNK_SIZE:10 * CHUNK_SIZE], token_three)
    assert np.array_equal(vocabulary.instruction(30)[6 * CHUNK_SIZE:10 * CHUNK_SIZE], token_three)
    assert np.array_equal(vocabulary.instruction(13)[12 * CHUNK_SIZE:16 * CHUNK_SIZE], token_three)


def test_digit_episode_keeps_target_out_of_sensor_contract_and_maps_199_positions():
    vocabulary = DigitSpatialVocabulary(seed=5)
    assert np.flatnonzero(visual_array(-99, -99, 99)).tolist() == [0]
    assert np.flatnonzero(visual_array(0, -99, 99)).tolist() == [99]
    assert np.flatnonzero(visual_array(99, -99, 99)).tolist() == [198]
    both = make_digit_episode(-7, vocabulary)
    audio_only = make_digit_episode(-7, vocabulary, modality="audio")
    assert len(both) == 16
    assert set(both[0].__dataclass_fields__) == {"t", "vision", "audio"}
    assert all(o.vision.shape == (199,) and o.audio.shape == (CHUNK_SIZE,) for o in both)
    assert all(not np.any(o.vision) for o in audio_only)


def test_magnitude_waveform_is_identical_under_sign_intervention():
    vocab = SpatialVocabulary()
    positive = vocab.instruction(3)[6 * CHUNK_SIZE:10 * CHUNK_SIZE]
    negative = vocab.instruction(-3)[6 * CHUNK_SIZE:10 * CHUNK_SIZE]
    assert np.array_equal(positive, negative)
    assert np.array_equal(vocab.token("THREE"), vocab.token("THREE"))


def test_vocab_tokens_share_duration_and_rms_without_monotonic_centroid_code():
    rows = SpatialVocabulary().vocabulary_rows()
    assert {row["duration_seconds"] for row in rows} == {0.2}
    assert np.ptp([row["rms"] for row in rows]) < 1e-12
    magnitudes = [row for row in rows if row["token"] in {"ONE", "TWO", "THREE", "FOUR", "FIVE"}]
    levels = np.arange(1, 6)
    for property_name in ("peak", "mean_absolute_amplitude", "zero_crossing_rate", "spectral_centroid_hz"):
        values = np.asarray([row[property_name] for row in magnitudes])
        assert abs(np.corrcoef(levels, values)[0, 1]) < 0.8


def test_observation_contains_only_sensor_streams_and_audio_test_blanks_vision():
    episode = make_episode(3, SpatialVocabulary(), modality="audio")
    assert len(episode) == EPISODE_TICKS
    assert set(episode[0].__dataclass_fields__) == {"t", "vision", "audio"}
    assert all(not np.any(obs.vision) for obs in episode)
    assert all(obs.audio.shape == (CHUNK_SIZE,) for obs in episode)
    assert all(np.isfinite(obs.audio).all() for obs in episode)


def test_sequential_protocol_records_audio_before_visual_onset():
    episode = make_episode(2, mode="sequential", delay_seconds=0.2)
    assert episode[-1].t > episode[EPISODE_TICKS - 1].t
    assert all(not np.any(obs.vision) for obs in episode[:14])
    assert all(np.any(obs.vision) for obs in episode[14:])
    assert all(not np.any(obs.audio) for obs in episode[EPISODE_TICKS:])


def test_rotating_holdouts_keep_each_sign_and_magnitude_in_training():
    for split in range(10):
        heldout = set(rotate_composition_split(split))
        train = {(sign, magnitude) for magnitude in range(1, 6) for sign in (-1, 1)} - heldout
        assert len(heldout) == 2
        assert all((sign, magnitude) not in train for sign, magnitude in heldout)
        assert {-1, 1} <= {sign for sign, _ in train}
        assert set(range(1, 6)) <= {magnitude for _, magnitude in train}


def test_all_combinations_competence_training_includes_every_scored_position():
    assert all_combination_training_positions() == list(POSITIONS)


def test_reset_clears_transient_state_but_preserves_learned_parameters():
    config = LiquidConfig(neurons=8, initial_out_degree=2, max_out_degree=3,
        max_connections=24, max_sensory_targets=3)
    net = LiquidNetwork(90, seed=4, config=config, variant="I1", visual_dim=21)
    edge = next(edge for row in net.adjacency for edge in row.values())
    edge.weight = 0.271
    count = net.connection_count
    input_vector = np.zeros(90); input_vector[0] = 1
    net.inject(input_vector, 0.0)
    net.advance_to(0.1)
    net.reset_episode()
    assert net.connection_count == count
    assert net.connection(edge.source, edge.destination).weight == 0.271
    assert np.all(net.activation == 0)
    assert net.pending_events == 0

