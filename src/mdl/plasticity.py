"""The established bounded pair-based STDP rule used by the model."""

AUDITORY_SYNAPSE_MODEL = """
w : 1
gain : volt (shared)
plasticity_enabled : 1 (shared)
dapre/dt = -apre/(tau*ms) : 1 (event-driven)
dapost/dt = -apost/(tau*ms) : 1 (event-driven)
tau : 1 (shared)
"""

AUDITORY_ON_PRE = """
v_post += w*gain
apre += plasticity_enabled*pre_increment
w = clip(w + plasticity_enabled*apost, weight_min, weight_max)
"""

AUDITORY_ON_POST = """
apost += plasticity_enabled*post_increment
w = clip(w + plasticity_enabled*apre, weight_min, weight_max)
"""
