import numpy as np

from sensory_lab.environment import MOTION_START_TICK
from sensory_lab.factor_diagnostics import validate_acoustic_vocabulary
from sensory_lab.factor_environment import FactorWorld, VOCABULARY


def test_five_waveform_ids_are_independent_and_speed_patterns_nonmonotonic():
    assert len({cue.concept_id for cue in VOCABULARY})==5
    assert len({cue.waveform_id for cue in VOCABULARY})==5
    speeds=sorted((int(c.concept_name[-1]),sum(c.frequencies_hz)/2) for c in VOCABULARY if c.category=="speed")
    means=[v for _,v in speeds]
    assert not (means[0]<means[1]<means[2])
    assert not (means[0]>means[1]>means[2])
    assert len({c.duration_ticks for c in VOCABULARY})==1
    assert len({c.nominal_amplitude for c in VOCABULARY})==1


def test_combined_cues_share_waveform_stream_and_are_temporally_aligned():
    world=FactorWorld()
    episode=world.episode(9,-1,2,task="combined")
    assert episode.cue_ids==("DIR_L","SPD_2")
    assert episode.cue_segments==(("DIR_L",0,2),("SILENCE",2,3),("SPD_2",3,5))
    assert len(episode.observations)==world.config.ticks
    assert all(o.audio.shape==(world.chunk,) for o in episode.observations)
    assert np.linalg.norm(episode.observations[0].audio)>0
    assert np.linalg.norm(episode.observations[1].audio)>0
    assert np.linalg.norm(episode.observations[3].audio)>0
    assert np.linalg.norm(episode.observations[4].audio)>0
    assert np.allclose(episode.observations[2].audio,0,atol=0.08)
    assert all(o.t==i/world.config.visual_hz for i,o in enumerate(episode.observations))
    assert not hasattr(episode.observations[0],"direction")


def test_existing_feature_transform_discriminates_all_five_ordered_cues():
    result=validate_acoustic_vocabulary(seed=7,repeats_train=12,repeats_test=8)
    assert result["accuracy"]>=0.95
    assert min(result["per_cue_accuracy"].values())>=0.9
    assert result["ordered_chunk_swap_feature_distance"]>0


def test_audio_counterfactual_changes_only_cue_waveform_not_scene_or_target():
    world=FactorWorld()
    left=world.episode(12,-1,2,task="combined")
    right=world.episode(12,1,2,task="combined")
    for a,b in zip(left.observations[:MOTION_START_TICK],right.observations[:MOTION_START_TICK]):
        np.testing.assert_array_equal(a.vision,b.vision)
    assert left.cue_ids[1]==right.cue_ids[1]=="SPD_2"
    assert not np.array_equal(left.observations[0].audio,right.observations[0].audio)
    assert all(np.array_equal(left.observations[k].audio,right.observations[k].audio) for k in (3,4))
