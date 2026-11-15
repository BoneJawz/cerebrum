from mdl.stimuli import make_task

def test_seeded_permutation_and_exposure_schedule():
    a = make_task(42, 5); b = make_task(42, 5)
    assert a == b
    assert len(a.episodes) == 15
    assert sorted(a.permutation) == [0, 1, 2]
