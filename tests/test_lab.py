import numpy as np
from sensory_lab.environment import LineWorld, WorldConfig
from sensory_lab.processors import PROCESSORS
from sensory_lab.benchmark import run, BenchmarkConfig, _ablate
from sensory_lab.viewer import ViewerSession

def test_reproducible_sensor_data():
    world = LineWorld(WorldConfig())
    a=world.episode(7,1,140)
    b=world.episode(7,1,140)
    assert all(np.array_equal(x.vision,y.vision) and np.array_equal(x.audio,y.audio) for x,y in zip(a,b))

def test_sensors_are_raw_and_separate():
    episode=LineWorld(WorldConfig()).episode(3,-1,340)
    assert len(episode[0].vision)==33
    assert set(np.unique(episode[0].vision)).issubset({0.,1.})
    assert len(episode[0].audio)==100
    assert 0 < np.std(episode[0].audio)
    assert episode[0].t < episode[1].t

def test_interchangeable_processors():
    ep=LineWorld(WorldConfig()).episode(1,1,140)
    for kind in PROCESSORS.values():
        model=kind(seed=5)
        xs=[model.step(o) for o in ep]
        assert len(xs)==len(ep)
        assert all(np.isfinite(x).all() for x in xs)
        assert model.updates>0

def test_benchmark_reproducible_predictions(tmp_path):
    cfg=BenchmarkConfig(24,12,8,16,1)
    a=run(cfg,output=tmp_path/'a.csv')
    b=run(cfg,output=tmp_path/'b.csv')
    assert [(v['seed'],v['case'],v['modality'],v['within_1_pixel_accuracy']) for v in a] == \
           [(v['seed'],v['case'],v['modality'],v['within_1_pixel_accuracy']) for v in b]
    assert (tmp_path/'a.csv').exists()

def test_each_single_modality_is_insufficient_by_construction():
    world=LineWorld(WorldConfig())
    # Same start and seed: the pre-movement visual input is identical while
    # changing the audio command changes the future position.
    left=world.episode(19,-1,340.,initial_position=16)
    right=world.episode(19,1,140.,initial_position=16)
    assert all(np.array_equal(_ablate(a,'audio_only').vision,
                              _ablate(b,'audio_only').vision)
               for a,b in zip(left[:5],right[:5]))
    slow=world.episode(19,1,140.,initial_position=16,velocity=1)
    fast=world.episode(19,1,340.,initial_position=16,velocity=3)
    assert all(np.array_equal(_ablate(a,'audio_only').vision,
                              _ablate(b,'audio_only').vision)
               for a,b in zip(slow[:5],fast[:5]))
    # Same direction, cue, seed, phase and noise but different initial scenes:
    # audio-only input is identical while the resulting positions differ.
    near=world.episode(19,1,140.,initial_position=12)
    far=world.episode(19,1,140.,initial_position=20)
    assert all(np.array_equal(_ablate(a,'vision_only').audio,
                              _ablate(b,'vision_only').audio)
               for a,b in zip(near[:5],far[:5]))
    center=lambda frame: np.sum(np.arange(len(frame.vision))*frame.vision)/np.sum(frame.vision)
    assert center(left[9]) != center(right[9])
    assert center(slow[9]) != center(fast[9])
    assert center(near[9]) != center(far[9])

def test_combined_task_metrics_and_outputs(tmp_path):
    rows=run(BenchmarkConfig(160,80,31,24,1),output=tmp_path/'multi.csv')
    assert len(rows)==36
    assert {row['modality'] for row in rows} == {'vision_only','audio_only','combined'}
    assert {row['task'] for row in rows} == {'direction_only','velocity_only','direction_and_velocity'}
    assert all(0 <= row['within_1_pixel_accuracy'] <= 1 for row in rows)
    assert all(row['active_updates'] > 0 and row['median_step_latency_us'] >= 0
               and row['peak_traced_bytes'] > 0 for row in rows)
    for task in ('direction_only','velocity_only','direction_and_velocity'):
        combined=[row['within_1_pixel_accuracy'] for row in rows
                  if row['task']==task and row['modality']=='combined']
        assert np.mean(combined) > 0.6
    combined_error=np.mean([row['mean_absolute_error_pixels'] for row in rows
                            if row['modality']=='combined'])
    ablation_error=np.mean([row['mean_absolute_error_pixels'] for row in rows
                            if row['modality']!='combined'])
    assert combined_error < ablation_error

def test_visual_movement_and_apparent_width():
    env=LineWorld(WorldConfig())
    a=env.episode(3,1,140)
    b=env.episode(3,-1,140)
    assert np.array_equal(a[0].vision,b[0].vision)
    assert not np.array_equal(a[-1].vision,b[-1].vision)

def test_discrete_velocity_scales_visual_displacement():
    world=LineWorld(WorldConfig())
    slow=world.episode(13,1,140.,initial_position=16,velocity=1)
    fast=world.episode(13,1,340.,initial_position=16,velocity=3)
    center=lambda frame: np.sum(np.arange(len(frame.vision))*frame.vision)/np.sum(frame.vision)
    assert center(slow[6])-center(slow[5]) == 1
    assert center(fast[6])-center(fast[5]) == 3

def test_viewer_keeps_heldout_truth_separate_until_reveal():
    session=ViewerSession(BenchmarkConfig(20,4,17,8,1))
    train=session.training_example(0)
    test=session.test_example(0)
    assert 'target_position' in train and 'direction' in train
    assert 'target' not in test and 'auditory_rule' not in test
    assert 'target' not in session.metadata()
    session.fit()
    predictions=session.predict(0)
    assert len(predictions)==12
    assert all('target' not in item and np.isfinite(item['predicted_position'])
               for item in predictions)
    truth=session.reveal(0)
    assert np.isfinite(truth['target_position'])
    assert truth['velocity'] in (1,2,3)
