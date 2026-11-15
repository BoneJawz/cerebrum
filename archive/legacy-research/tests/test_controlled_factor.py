import numpy as np

from sensory_lab import adaptive_dynamics as ad
from sensory_lab.controlled_factor import _batch, _episode_batch, _trace
from sensory_lab.environment import WorldConfig
from sensory_lab.factor_environment import FactorWorld, target_trajectory
from sensory_lab.factor_learning import split_start_positions


def test_instruction_window_contains_both_cues_and_no_motion_frame():
    world=FactorWorld(WorldConfig(pixels=177,ticks=32,visual_hz=20,audio_hz=2000,noise_std=0))
    ep=world.episode(1,-1,3,initial_position=88,task="combined",acoustic_config="B_independent")
    assert ep.cue_segments == (("DIR_L",0,2),("SILENCE",2,3),("SPD_3",3,5))
    assert np.any(ep.observations[0].audio) and np.any(ep.observations[1].audio)
    assert not np.any(ep.observations[2].audio)
    assert np.any(ep.observations[3].audio) and np.any(ep.observations[4].audio)
    assert all(np.array_equal(ep.observations[4].vision,ep.observations[5].vision) for _ in [0])


def test_windows_have_explicit_masks_and_correct_target_origins():
    world=FactorWorld(WorldConfig(pixels=177,ticks=32,visual_hz=20,audio_hz=2000,noise_std=0))
    tr,te=split_start_positions(world)
    item=_episode_batch(world,7,1,tr)[0]
    xa,ya,ma=_batch([item],4,world)
    xb,yb,mb=_batch([item],9,world)
    assert ma.sum()==5 and mb.sum()==10
    ep=item["episode"]
    assert ya[0,0] == target_trajectory(ep,4,(1,))[0]
    assert yb[0,0] == target_trajectory(ep,9,(1,))[0]
    assert xa[0,5:].sum()==0 and xb[0,10:].sum()==0


def test_masked_future_padding_does_not_change_assimilated_state():
    model=ad.TrainableStateModel(8,hidden=4,seed=2,kind="F")
    x=np.arange(2*5*8,dtype=float).reshape(2,5,8)/100
    mask=np.array([[1,1,0,0,0],[1,1,1,0,0]],dtype=bool)
    mean=x[mask].mean(0);scale=x[mask].std(0);scale[scale<1e-8]=1
    model.input_mean=mean;model.input_scale=scale
    assert _trace(model,x,mask,1)[0].shape==(2,)
    m1=model._encode(((x[0,:2]-mean)/scale)[:,None,:])[0]
    altered=x.copy();altered[0,2:]=1e8
    z=(altered[0]-mean)/scale;z[~mask[0]]=0
    m2=model._encode(z[:,None,:],mask[0,:,None])[0]
    assert np.allclose(m1,m2)


def test_masked_padding_has_no_effect_on_recurrent_training_gradient():
    model=ad.TrainableStateModel(8,hidden=4,seed=9,kind="F")
    x=np.arange(2*5*8,dtype=float).reshape(2,5,8)/100
    mask=np.array([[1,1,0,0,0],[1,1,1,0,0]],dtype=bool)
    y=np.ones((2,ad.TRAIN_HORIZON))*20
    loss1,g1,_=ad._batch_grad(model,x,y,mask)
    x[~mask]=1e9
    loss2,g2,_=ad._batch_grad(model,x,y,mask)
    assert loss1==loss2
    assert all(np.allclose(g1[k],g2[k]) for k in g1)


def test_trainable_model_resets_rollout_between_independent_calls():
    model=ad.TrainableStateModel(8,hidden=4,seed=3,kind="G")
    x=np.ones((1,3,8));mask=np.ones((1,3),bool)
    p1=model.predict_rollout(x,(1,3),observation_mask=mask)
    p2=model.predict_rollout(x,(1,3),observation_mask=mask)
    assert all(np.array_equal(p1[k],p2[k]) for k in p1)
