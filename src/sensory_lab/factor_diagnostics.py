"""Acoustic cue integrity checks and post-training factor diagnostics."""
from __future__ import annotations

import numpy as np

from .factor_environment import FactorWorld, VOCABULARY
from .processors import sensory_features


def acoustic_features(episode, pixels, start=0):
    """Preserve chunk order and discard visual coordinates for cue decoding."""
    return np.concatenate([sensory_features(o)[pixels:] for o in episode.observations[start:start+2]])


def validate_acoustic_vocabulary(seed=42, repeats_train=80, repeats_test=40):
    """Linear nearest-centroid cue probe using only ordered raw-audio features."""
    world=FactorWorld()
    vectors=[]; labels=[]
    for cue in VOCABULARY:
        for i in range(repeats_train+repeats_test):
            if cue.category=="direction":
                d=-1 if cue.concept_name=="LEFT" else 1
                ep=world.episode(seed+10000*len(labels)+i,d,1,task="direction_focus")
            else:
                speed=int(cue.concept_name[-1])
                ep=world.episode(seed+10000*len(labels)+i,(-1 if i%2==0 else 1),speed,task="speed_focus")
            vectors.append(acoustic_features(ep,world.config.pixels))
            labels.append(cue.concept_id)
    x=np.asarray(vectors); y=np.asarray(labels)
    per_class=repeats_train+repeats_test
    train_ids=np.concatenate([np.arange(k*per_class,k*per_class+repeats_train) for k in range(len(VOCABULARY))])
    test_ids=np.concatenate([np.arange(k*per_class+repeats_train,(k+1)*per_class) for k in range(len(VOCABULARY))])
    mean=x[train_ids].mean(0); scale=x[train_ids].std(0); scale[scale<1e-8]=1.
    z=(x-mean)/scale
    classes=[c.concept_id for c in VOCABULARY]
    centers=np.stack([z[train_ids][y[train_ids]==label].mean(0) for label in classes])
    pred=np.asarray(classes)[np.argmin(np.sum((z[test_ids,None,:]-centers[None,:,:])**2,axis=2),axis=1)]
    accuracy=float(np.mean(pred==y[test_ids]))
    confusion={}
    for label in classes:
        ids=y[test_ids]==label
        confusion[label]=float(np.mean(pred[ids]==label))
    # The actual sensory transform is chunkwise and keeps this order. This
    # verifies that swapping the two chunks changes the feature vector.
    ordered_change=0.0
    probe=world.episode(seed+909,direction=-1,speed=1,task="direction_focus")
    ordered=acoustic_features(probe,world.config.pixels)
    swapped=np.concatenate([sensory_features(probe.observations[k])[world.config.pixels:]
                            for k in (1,0)])
    ordered_change=float(np.linalg.norm(ordered-swapped))
    return {"accuracy":accuracy,"per_cue_accuracy":confusion,
            "n_train":len(train_ids),"n_test":len(test_ids),
            "ordered_chunk_swap_feature_distance":ordered_change}
