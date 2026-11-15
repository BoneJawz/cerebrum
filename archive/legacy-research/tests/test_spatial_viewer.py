import numpy as np
import pytest

from sensory_lab.spatial_viewer import MODEL_CATALOG, REFERENCE_MODELS, SpatialViewerSession


def test_spatial_viewer_catalog_covers_task_compatible_a_to_i_models():
    ids = {item["id"] for item in MODEL_CATALOG}
    assert ids == {"A", "B", "C", "D", "E1", "FF_direct", "F", "G", "H",
                   "I0", "I1", "I2", "I2_rewire_fixed"}
    assert {item["id"] for item in REFERENCE_MODELS} == {"E2", "E3"}


def test_training_ui_exposes_truth_but_inputs_remain_sensor_only():
    session = SpatialViewerSession(seed=8, train_repetitions=1, test_per_position=1)
    example = session.training_example(0)
    assert "position" in example and "tokens" in example
    assert example["position"] == -99
    assert example["tokens"] == ["-", "9", "9"]
    assert all(set(obs) == {"t", "vision", "audio"} for obs in example["observations"])
    assert sum(example["observations"][0]["vision"]) == 1
    assert len(example["observations"][0]["vision"]) == 199
    assert len(example["observations"]) == 16


def test_viewer_spans_positions_in_sequential_order_and_fixed_duration():
    session = SpatialViewerSession(seed=8)
    assert [item["position"] for item in session.train] == list(range(-99, 100))
    assert session.training_example(186)["tokens"] == ["+", "8", "7"]
    assert session.metadata()["field_cells"] == 199
    assert session.metadata()["episode_ticks"] == 16
    assert np.array_equal(session.vocabulary.token("8"),
                          SpatialViewerSession(seed=8).vocabulary.token("8"))
    assert not np.array_equal(session.vocabulary.token("8"),
                              SpatialViewerSession(seed=9).vocabulary.token("8"))


def test_audio_only_test_payload_hides_target_and_reveal_requires_predictions():
    session = SpatialViewerSession(seed=9, train_repetitions=1, test_per_position=1)
    example = session.test_example(0)
    assert not ({"position", "target", "tokens", "direction"} & example.keys())
    assert example["vision_withheld"] is True
    assert all(not np.any(obs["vision"]) for obs in example["observations"])
    assert all(len(obs["vision"]) == 199 for obs in example["observations"])
    with pytest.raises(RuntimeError, match="predictions"):
        session.reveal(0)


def test_scorecard_does_not_score_unrevealed_predictions():
    session = SpatialViewerSession(seed=11)
    session.predictions[0] = [{"id": "A", "predicted_position": 0.0}]
    assert session.scorecard() == []
    session.reveal(0)
    assert session.scorecard() == [{"model": "A", "position_error": abs(session.test[0]["position"]),
                                    "within_one": abs(session.test[0]["position"]) <= 1.0}]


def test_viewer_requires_training_before_prediction():
    session = SpatialViewerSession(seed=10)
    with pytest.raises(RuntimeError, match="train the models"):
        session.predict(0)

