import numpy as np
from mdl.sensory_environment import ContinuousSensoryEnvironment, audible_cue, teacher_pattern
from mdl.stimulus_scheduler import StimulusScheduler

def test_environment_sequence_repeats_for_same_seed():
    a=ContinuousSensoryEnvironment(seed=17); b=ContinuousSensoryEnvironment(seed=17)
    assert [a.next_event() for _ in range(18)] == [b.next_event() for _ in range(18)]

def test_visual_teacher_code_is_one_hot_and_blank_when_absent():
    assert np.array_equal(teacher_pattern(1),[0,1,0])
    assert np.array_equal(teacher_pattern(None),[0,0,0])

def test_audible_representations_are_deterministic_and_distinct():
    a=audible_cue(0); same=audible_cue(0); b=audible_cue(1)
    assert np.array_equal(a.samples,same.samples)
    assert not np.array_equal(a.samples,b.samples)
    assert a.wav_bytes.startswith(b'RIFF')

def test_scheduler_accepts_audio_teacher_and_quiet_events_fifo():
    q=StimulusScheduler(); q.submit_event(0,None,'auditory_only'); q.submit_event(1,2,'paired_teaching'); q.submit_event(None,None,'quiet',.2)
    assert [q.pop().event_type for _ in range(3)] == ['auditory_only','paired_teaching','quiet']
