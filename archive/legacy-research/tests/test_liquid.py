import json

import numpy as np
import pytest

from sensory_lab.contracts import Observation
from sensory_lab.liquid import LiquidConfig, LiquidNetwork
from sensory_lab.liquid_readout import RidgeReadout


def small_config(**changes):
    values=dict(neurons=8,initial_out_degree=2,max_out_degree=3,max_connections=24,
        max_sensory_targets=3,emit_threshold=.02,resting_threshold=.005,
        candidate_pairs_per_update=32,structural_interval_seconds=.05,
        minimum_connection_lifetime_seconds=.05,prune_after_inactive_seconds=.1,
        prune_weight_threshold=.9,prune_utility_threshold=1.0,
        transmission_delay_seconds=.01)
    values.update(changes)
    return LiquidConfig(**values)


def test_potential_connectivity_covers_every_directed_pair_without_self_edges():
    net=LiquidNetwork(12,seed=1,config=small_config())
    for source in range(8):
        for destination in range(8):
            assert net.potential_connection(source,destination)==(source!=destination)
    assert all(source not in row for source,row in enumerate(net.adjacency))
    assert all(len(row)<=net.config.max_out_degree for row in net.adjacency)


def test_self_connection_policy_is_configurable():
    cfg=small_config(allow_self_connections=True)
    net=LiquidNetwork(12,seed=2,config=cfg)
    assert net.potential_connection(3,3)
    if net.connection(3,3) is None:net.add_connection(3,3,.2)
    assert net.connection(3,3) is not None


def test_connection_creation_and_removal_enforce_budgets():
    net=LiquidNetwork(12,seed=3,config=small_config(max_out_degree=2,max_connections=16))
    source=0
    for d in range(8):
        if d!=source and net.connection(source,d):
            net.remove_connection(source,d)
    row=list(net.adjacency[source])
    if len(row):net.remove_connection(source,row[0])
    net.add_connection(source,7,.4)
    remaining=next(d for d in range(8) if d!=source and net.connection(source,d) is None)
    net.add_connection(source,remaining,.25)
    assert len(net.adjacency[source])<=2
    with pytest.raises(ValueError):net.add_connection(source,6,.3)
    assert net.connection_count<=16


def test_deleted_connection_invalidates_queued_message_safely():
    net=LiquidNetwork(12,seed=4,config=small_config())
    edge=next(edge for row in net.adjacency for edge in row.values())
    net._push(.01,edge.destination,.4,edge.source,edge.connection_id,.8)
    assert net.remove_connection(edge.source,edge.destination)
    before=net.activation.copy()
    net.advance_to(.02)
    assert np.array_equal(before,net.activation)
    assert net.stats().stale_events==1


def test_event_order_and_state_are_deterministic_for_fixed_seed():
    a=LiquidNetwork(12,seed=5,config=small_config(),variant="I1")
    b=LiquidNetwork(12,seed=5,config=small_config(),variant="I1")
    for tick in range(5):
        vector=np.zeros(12);vector[(tick*3)%12]=1.
        a.inject(vector,tick*.05);b.inject(vector,tick*.05)
    a.advance_to(.6);b.advance_to(.6)
    assert np.array_equal(a.snapshot(),b.snapshot())
    assert a.stats().processed_events==b.stats().processed_events
    assert a.stats().neuron_updates==b.stats().neuron_updates
    assert a.stats().propagated_events==b.stats().propagated_events
    assert [(e.source,e.destination,e.weight) for row in a.adjacency for e in row.values()] == [
        (e.source,e.destination,e.weight) for row in b.adjacency for e in row.values()]


def test_timestamped_updates_are_lazy_between_events_and_monotonic():
    net=LiquidNetwork(12,seed=6,config=small_config())
    x=np.zeros(12);x[0]=1
    net.inject(x,0.)
    updates=net.stats().neuron_updates
    snapshot0=net.snapshot(.1)
    assert net.stats().neuron_updates==updates
    net.advance_to(.1)
    assert np.all(np.isfinite(snapshot0))
    with pytest.raises(ValueError):net.advance_to(.05)


def test_activation_and_weights_stay_bounded_under_repeated_input():
    cfg=small_config(emit_threshold=.035,resting_threshold=.012,max_events_per_advance=100000)
    net=LiquidNetwork(12,seed=7,config=cfg,variant="I1")
    for tick in range(20):
        x=np.ones(12)*10
        net.inject(x,tick*.05)
    net.advance_to(5.)
    assert np.all(np.isfinite(net.activation))
    assert np.max(np.abs(net.activation))<=1.
    assert all(abs(e.weight)<=cfg.weight_limit for row in net.adjacency for e in row.values())


def test_variants_share_initial_graph_and_episode_reset_retains_learning():
    cfg=small_config()
    variants=[LiquidNetwork(12,seed=8,config=cfg,variant=v) for v in ("I0","I1","I2")]
    signatures=[[(e.source,e.destination,e.weight) for row in n.adjacency for e in row.values()] for n in variants]
    assert signatures[0]==signatures[1]==signatures[2]
    net=variants[1];x=np.zeros(12);x[1]=2
    net.inject(x,0.);net.advance_to(.2)
    weights=[e.weight for row in net.adjacency for e in row.values()]
    net.reset_episode()
    assert np.all(net.activation==0)
    assert weights==[e.weight for row in net.adjacency for e in row.values()]
    assert net.pending_events==0


def test_readout_snapshot_does_not_change_network_state():
    net=LiquidNetwork(12,seed=9,config=small_config())
    x=np.ones(12);net.inject(x,0.)
    before=net.state_dict()
    net.snapshot(.5)
    after=net.state_dict()
    assert before["activation"]==after["activation"]
    assert before["time_seconds"]==after["time_seconds"]


def test_state_snapshot_round_trips_queue_topology_and_rng():
    net=LiquidNetwork(12,seed=10,config=small_config(),variant="I2")
    x=np.ones(12);net.inject(x,0.)
    restored=LiquidNetwork.from_state_dict(json.loads(json.dumps(net.state_dict())))
    assert np.array_equal(net.snapshot(),restored.snapshot())
    assert net.pending_events==restored.pending_events
    assert net.topology_summary()==restored.topology_summary()
    assert net.sample_candidate_pairs(5)==restored.sample_candidate_pairs(5)


def test_observation_interface_only_accepts_raw_contract():
    net=LiquidNetwork(input_dim=70,visual_dim=2,seed=11,config=small_config())
    obs=Observation(t=0.,vision=np.array([1,0],dtype=np.int8),audio=np.array([.1,.2,.3]))
    with pytest.raises(ValueError):net.ingest_observation(obs)
    with pytest.raises(TypeError):net.ingest_observation({"direction":-1,"speed":2})
    valid=LiquidNetwork(input_dim=71,visual_dim=2,seed=11,config=small_config())
    assert valid.ingest_observation(obs)>0


def test_structural_updates_respect_degree_and_total_edge_limits():
    cfg=small_config(max_out_degree=3,max_connections=24,candidate_pairs_per_update=56,
        candidate_utility_threshold=0.,candidate_ema_rate=1.,structural_interval_seconds=.01,
        minimum_connection_lifetime_seconds=10.,prune_after_inactive_seconds=10.)
    net=LiquidNetwork(12,seed=12,config=cfg,variant="I2")
    for tick in range(50):
        x=np.ones(12)*((tick%2)*2-1)
        net.inject(x,tick*.05)
    net.advance_to(10.)
    assert net.connection_count<=cfg.max_connections
    assert max(map(len,net.adjacency))<=cfg.max_out_degree
    assert net.stats().structural_updates>0
    assert net.stats().formed_connections>0


def test_ridge_readout_is_independent_and_does_not_mutate_liquid():
    rng=np.random.default_rng(13);x=rng.normal(size=(30,8));y=np.eye(3)[np.arange(30)%3]
    readout=RidgeReadout(alpha=.1).fit(x,y)
    net=LiquidNetwork(12,seed=13,config=small_config());before=net.state_dict()
    prediction=readout.predict(x)
    after=net.state_dict()
    assert prediction.shape==(30,3)
    assert before["edges"]==after["edges"] and before["activation"]==after["activation"]

