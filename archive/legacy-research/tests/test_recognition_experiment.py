import json

import numpy as np

from sensory_lab.liquid import LiquidNetwork
from sensory_lab.recognition_experiment import (
    CandidateMatcher, RecognitionConfig, _audio_assessment, _candidate_order,
    _frozen_view, _input_observations, _recognize, _teach, _visual_branch,
    _new_network, liquid_config,
)
from sensory_lab.spatial_environment import DigitSpatialVocabulary, visual_array


def _net(variant="I1"):
    return LiquidNetwork(268, seed=27, config=liquid_config(12),
                         variant=variant, visual_dim=199)


def test_candidate_retinas_are_valid_single_cell_arrays_and_token_is_reused():
    vocab = DigitSpatialVocabulary(42)
    np.testing.assert_array_equal(vocab.instruction(1)[600:1000],
                                  vocab.instruction(-1)[600:1000])
    for position in (-1, 0, 1):
        visual = visual_array(position, -99, 99)
        assert visual.shape == (199,)
        assert visual.sum() == 1
        observations = _input_observations(vocab, position, "both")
        assert len(observations) == 16
        assert all(obs.vision.shape == (199,) and obs.vision.sum() == 1 for obs in observations)
        assert all(not hasattr(obs, "position") and not hasattr(obs, "correct_index")
                   for obs in observations)


def test_candidate_order_is_seeded_and_balanced_across_slots():
    orders = [_candidate_order((-1, 0, 1), 12, i) for i in range(9)]
    assert orders == [_candidate_order((-1, 0, 1), 12, i) for i in range(9)]
    assert len(set(orders)) == 3
    for position in (-1, 0, 1):
        assert sorted(order.index(position) for order in orders) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_identical_candidate_branches_have_identical_state_and_time():
    vocab = DigitSpatialVocabulary(101)
    net = _net()
    before = json.dumps(net.state_dict(), sort_keys=True)
    snapshot, acoustic = _audio_assessment(net, vocab.instruction(-1))
    visual = visual_array(-1, -99, 99)
    a, branch_a = _visual_branch(snapshot, visual, 4)
    b, branch_b = _visual_branch(snapshot, visual, 4)
    np.testing.assert_array_equal(a, b)
    assert branch_a.time_seconds == branch_b.time_seconds
    assert branch_a.stats().processed_events == branch_b.stats().processed_events
    assert json.dumps(net.state_dict(), sort_keys=True) == before
    assert acoustic.shape == (12,)


def test_assessment_has_no_feedback_or_plasticity_and_score_is_order_invariant():
    vocab = DigitSpatialVocabulary(51)
    cfg = RecognitionConfig(seeds=(1,), exposures=(0, 1), variants=("I1",),
        pairings=("correct",), neurons=12, calibration_variants=2, test_variants=1)
    net = _net()
    before = json.dumps(net.state_dict(), sort_keys=True)
    waveform = vocab.instruction(1)
    result = _recognize(net, waveform, (-1, 0, 1), CandidateMatcher(), cfg,
        seed=5, trial_index=0)
    assert json.dumps(net.state_dict(), sort_keys=True) == before
    mapping = dict(zip(result["candidate_order"], result["scores"]))
    for index in (1, 2):
        repeated = _recognize(net, waveform, (-1, 0, 1), CandidateMatcher(), cfg,
            seed=5, trial_index=index)
        np.testing.assert_allclose([dict(zip(repeated["candidate_order"], repeated["scores"]))[p]
                                    for p in (-1, 0, 1)], [mapping[p] for p in (-1, 0, 1)])


def test_teaching_changes_only_permitted_adaptive_network_and_independent_reset_retains_weights():
    vocab = DigitSpatialVocabulary(51)
    rng = np.random.default_rng(8)
    adaptive = _net("I1")
    fixed = _net("I0")
    initial_fixed = [(e.source, e.destination, e.weight)
                     for row in fixed.adjacency for e in row.values()]
    initial_adaptive = [(e.source, e.destination, e.weight)
                        for row in adaptive.adjacency for e in row.values()]
    _teach(adaptive, vocab, 1, "correct", rng)
    _teach(fixed, vocab, 1, "correct", np.random.default_rng(8))
    after_fixed = [(e.source, e.destination, e.weight)
                   for row in fixed.adjacency for e in row.values()]
    after_adaptive = [(e.source, e.destination, e.weight)
                      for row in adaptive.adjacency for e in row.values()]
    assert after_fixed == initial_fixed
    assert after_adaptive != initial_adaptive
    probe = _frozen_view(adaptive, "independent", delay=0.5)
    assert probe.time_seconds == 0.5
    assert probe.pending_events == 0
    assert not probe.weight_plasticity and not probe.structural_plasticity
    assert [(e.source, e.destination, e.weight) for row in probe.adjacency
            for e in row.values()] == after_adaptive


def test_visual_projection_control_has_no_position_specific_exception():
    base = RecognitionConfig(seeds=(43,), exposures=(0, 1), neurons=24,
        calibration_variants=2, test_variants=1)
    corrected = RecognitionConfig(seeds=(43,), exposures=(0, 1), neurons=24,
        calibration_variants=2, test_variants=1, projection="P1")
    p0 = _new_network(43, "I0", base)
    p1 = _new_network(43, "I0", corrected)
    assert np.any(np.abs(p0._input_gains[:199]) <= p0.config.resting_threshold)
    assert np.all(np.abs(p1._input_gains[:199]) > p1.config.resting_threshold)
    np.testing.assert_allclose(np.sum(p0._input_gains[:199] ** 2),
                               np.sum(p1._input_gains[:199] ** 2), rtol=1e-12)
    np.testing.assert_array_equal(p0._input_gains[199:], p1._input_gains[199:])
    assert [(e.source, e.destination, e.weight) for row in p0.adjacency for e in row.values()] == [
        (e.source, e.destination, e.weight) for row in p1.adjacency for e in row.values()]
