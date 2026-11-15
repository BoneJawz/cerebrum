"""Sparse, event-scheduled liquid substrate (Model I research prototype).

This module contains dynamics and replaceable plasticity policies only. It
does not know task labels, cue meanings, target positions, or motion rules.
Simulation time is in seconds; wall-clock timing is measured by callers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import time
from typing import Any

import numpy as np

from .contracts import Observation
from .processors import sensory_features


@dataclass(frozen=True)
class LiquidConfig:
    neurons: int = 64
    initial_out_degree: int = 6
    max_out_degree: int = 8
    max_connections: int = 512
    allow_self_connections: bool = False
    tau_seconds: float = 0.20
    adaptation_tau_seconds: float = 0.60
    adaptation_gain: float = 0.04
    emit_threshold: float = 0.035
    resting_threshold: float = 0.012
    internal_interval_seconds: float = 0.05
    transmission_delay_seconds: float = 0.005
    max_queue_size: int = 100_000
    max_events_per_advance: int = 250_000
    max_sensory_targets: int = 8
    weight_limit: float = 1.0
    weight_learning_rate: float = 0.003
    weight_decay: float = 0.001
    structural_interval_seconds: float = 0.25
    candidate_pairs_per_update: int = 96
    candidate_utility_threshold: float = 0.0005
    candidate_ema_rate: float = 0.25
    minimum_connection_lifetime_seconds: float = 1.0
    prune_after_inactive_seconds: float = 3.0
    prune_weight_threshold: float = 0.015
    prune_utility_threshold: float = 0.008

    def __post_init__(self):
        if self.neurons < 2:
            raise ValueError("neurons must be at least two")
        if self.initial_out_degree < 0 or self.max_out_degree < self.initial_out_degree:
            raise ValueError("max_out_degree must be >= initial_out_degree >= 0")
        if self.max_connections < self.neurons * self.initial_out_degree:
            raise ValueError("max_connections must fit the initial graph")
        if self.tau_seconds <= 0 or self.adaptation_tau_seconds <= 0:
            raise ValueError("time constants must be positive")
        if self.internal_interval_seconds <= 0 or self.structural_interval_seconds <= 0:
            raise ValueError("scheduler intervals must be positive")
        if not 0 < self.weight_limit <= 1:
            raise ValueError("weight_limit must be in (0, 1]")
        if not 1 <= self.max_sensory_targets <= self.neurons:
            raise ValueError("max_sensory_targets must be between one and neuron count")
        if self.max_queue_size < 1 or self.max_events_per_advance < 1:
            raise ValueError("event limits must be positive")


@dataclass
class Connection:
    source: int
    destination: int
    weight: float
    created_at: float
    last_active_at: float
    utility: float
    connection_id: int
    delay_seconds: float


@dataclass(frozen=True)
class LiquidStats:
    processed_events: int
    sensory_events: int
    propagated_events: int
    internal_events: int
    stale_events: int
    neuron_updates: int
    structural_updates: int
    formed_connections: int
    pruned_connections: int
    queue_drops: int
    readout_scans: int
    weight_updates: int
    weight_update_seconds: float
    structural_seconds: float


class LiquidNetwork:
    """Directed sparse graph with lazy leaky dynamics and a stable event queue.

    Between scheduled events, an activation is mathematically defined to decay
    exponentially. That decay is evaluated only when the cell receives an
    event, has a scheduled internal event, or is read for a state snapshot.
    Snapshot reads inspect all cells but do not mutate the liquid or propagate
    messages; they are accounted separately from neuron updates.
    """

    VARIANTS = {"I0", "I1", "I2", "I2_rewire_fixed"}

    def __init__(self, input_dim: int, seed: int = 0, config: LiquidConfig | None = None,
                 variant: str = "I0", visual_dim: int | None = None):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.config = config or LiquidConfig()
        self.input_dim = int(input_dim)
        self.visual_dim = int(visual_dim if visual_dim is not None else input_dim)
        if not 0 <= self.visual_dim <= input_dim:
            raise ValueError("visual_dim must be within input dimension")
        self.variant = variant
        self.weight_plasticity = variant in {"I1", "I2"}
        self.structural_plasticity = variant in {"I2", "I2_rewire_fixed"}
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        n = self.config.neurons

        # Per-cell persistent dynamic variables. Activity is bounded in [-1, 1].
        self.activation = np.zeros(n, dtype=np.float64)
        self.recent_incoming = np.zeros(n, dtype=np.float64)
        self.adaptation = np.zeros(n, dtype=np.float64)
        self.last_update_time = np.zeros(n, dtype=np.float64)
        self.activity_trace = np.zeros(n, dtype=np.float64)
        self.activity_trace_time = np.zeros(n, dtype=np.float64)
        self.last_activity_time = np.full(n, -np.inf, dtype=np.float64)
        self.neuron_update_counts = np.zeros(n, dtype=np.int64)
        self.next_internal_time = np.full(n, np.inf, dtype=np.float64)
        self._internal_token = np.zeros(n, dtype=np.int64)

        # Sparse adjacency is the only persistent weight representation.
        self.adjacency: list[dict[int, Connection]] = [dict() for _ in range(n)]
        self._next_connection_id = 0
        self._queue: list[tuple[float, int, int, float, int, int, float, int, int]] = []
        self._sequence = 0
        self.time_seconds = 0.0
        self._last_structural_time = 0.0
        self._candidate_utility: dict[tuple[int, int], tuple[float, int]] = {}
        self._stats = {k: 0 for k in LiquidStats.__dataclass_fields__}

        # Every sensory coordinate has one fixed, seed-controlled projection
        # target and signed gain. Audio and vision are kept in separate maps.
        self._input_targets = self.rng.integers(0, n, size=self.input_dim)
        gain = 0.35 / math.sqrt(max(1, self.input_dim / n))
        self._input_gains = self.rng.uniform(-gain, gain, size=self.input_dim)
        self._initialise_graph()

    def _initialise_graph(self):
        n = self.config.neurons
        for source in range(n):
            choices = np.arange(n)
            if not self.config.allow_self_connections:
                choices = choices[choices != source]
            destinations = self.rng.choice(choices, size=self.config.initial_out_degree, replace=False)
            for destination in sorted(int(x) for x in destinations):
                weight = float(self.rng.uniform(-0.22, 0.22))
                self._add_connection(source, destination, weight, 0.0)

    @property
    def connection_count(self) -> int:
        return sum(len(row) for row in self.adjacency)

    @property
    def pending_events(self) -> int:
        return len(self._queue)

    def connection(self, source: int, destination: int) -> Connection | None:
        self._validate_neuron(source); self._validate_neuron(destination)
        return self.adjacency[source].get(destination)

    def potential_connection(self, source: int, destination: int) -> bool:
        """Whether the pair is eligible, independent of current topology."""
        self._validate_neuron(source); self._validate_neuron(destination)
        return self.config.allow_self_connections or source != destination

    def _validate_neuron(self, neuron: int):
        if not isinstance(neuron, (int, np.integer)) or not 0 <= int(neuron) < self.config.neurons:
            raise IndexError("neuron id is outside this liquid")

    def _add_connection(self, source: int, destination: int, weight: float, now: float,
                        delay_seconds: float | None = None) -> Connection:
        self._validate_neuron(source); self._validate_neuron(destination)
        if not self.potential_connection(source, destination):
            raise ValueError("self-connections are disabled")
        if destination in self.adjacency[source]:
            raise ValueError("connection already exists")
        if len(self.adjacency[source]) >= self.config.max_out_degree:
            raise ValueError("source reached max_out_degree")
        if self.connection_count >= self.config.max_connections:
            raise ValueError("liquid reached max_connections")
        delay = self.config.transmission_delay_seconds if delay_seconds is None else float(delay_seconds)
        if delay < 0:
            raise ValueError("transmission delay cannot be negative")
        weight = float(np.clip(weight, -self.config.weight_limit, self.config.weight_limit))
        edge = Connection(source, destination, weight, float(now), float(now), 0.0,
                          self._next_connection_id, delay)
        self._next_connection_id += 1
        self.adjacency[source][destination] = edge
        return edge

    def add_connection(self, source: int, destination: int, weight: float = 0.1,
                       now: float | None = None, delay_seconds: float | None = None) -> Connection:
        """Create an edge subject to self, degree, and total-budget limits."""
        return self._add_connection(source, destination, weight,
            self.time_seconds if now is None else now, delay_seconds)

    def remove_connection(self, source: int, destination: int) -> bool:
        """Remove an edge; queued transmissions become harmless stale events."""
        self._validate_neuron(source); self._validate_neuron(destination)
        return self.adjacency[source].pop(destination, None) is not None

    def sample_candidate_pairs(self, count: int | None = None) -> list[tuple[int, int]]:
        """Uniformly sample from all ordered potential pairs without dense weights."""
        n = self.config.neurons
        per_source = n if self.config.allow_self_connections else n - 1
        total = n * per_source
        count = self.config.candidate_pairs_per_update if count is None else int(count)
        if count < 0:
            raise ValueError("candidate count must be nonnegative")
        raw = self.rng.choice(total, size=min(total, count), replace=False)
        out = []
        for value in raw:
            source, offset = divmod(int(value), per_source)
            destination = offset if self.config.allow_self_connections or offset < source else offset + 1
            out.append((source, destination))
        return out

    def _push(self, when: float, destination: int, amplitude: float, source: int,
              connection_id: int, presynaptic_activation: float, kind: int = 0, token: int = 0):
        if len(self._queue) >= self.config.max_queue_size:
            self._stats["queue_drops"] += 1
            return False
        self._sequence += 1
        heapq.heappush(self._queue, (float(when), self._sequence, int(destination),
            float(amplitude), int(source), int(connection_id), float(presynaptic_activation),
            int(kind), int(token)))
        return True

    def _decayed_activity(self, neuron: int, now: float) -> float:
        dt = max(0.0, now - self.activity_trace_time[neuron])
        return float(self.activity_trace[neuron] * math.exp(-dt / self.config.tau_seconds))

    def _schedule_internal(self, neuron: int, when: float):
        if abs(self.activation[neuron]) <= self.config.resting_threshold:
            return
        if when >= self.next_internal_time[neuron] - 1e-12:
            return
        self._internal_token[neuron] += 1
        token = int(self._internal_token[neuron])
        self.next_internal_time[neuron] = when
        self._push(when, neuron, 0.0, -2, -1, 0.0, kind=1, token=token)

    def _update_neuron(self, neuron: int, now: float, incoming: float) -> float:
        cfg = self.config
        elapsed = max(0.0, float(now) - self.last_update_time[neuron])
        prior = self.activation[neuron] * math.exp(-elapsed / cfg.tau_seconds)
        adapted = self.adaptation[neuron] * math.exp(-elapsed / cfg.adaptation_tau_seconds)
        new = float(np.tanh(prior + incoming - adapted))
        self.recent_incoming[neuron] = (self.recent_incoming[neuron] * math.exp(-elapsed / cfg.tau_seconds)
                                        + abs(float(incoming)))
        self.adaptation[neuron] = min(1.0, adapted + cfg.adaptation_gain * abs(new))
        self.activation[neuron] = float(np.clip(new, -1.0, 1.0))
        self.last_update_time[neuron] = float(now)
        self.last_activity_time[neuron] = float(now)
        trace = self._decayed_activity(neuron, now)
        self.activity_trace[neuron] = trace + abs(self.activation[neuron])
        self.activity_trace_time[neuron] = float(now)
        self._stats["neuron_updates"] += 1
        self.neuron_update_counts[neuron] += 1
        return self.activation[neuron], abs(self.activation[neuron] - prior)

    def _adapt_weight(self, edge: Connection, pre: float, post: float, now: float):
        started=time.perf_counter()
        cfg = self.config
        correlation = float(np.clip(pre * post, -1.0, 1.0))
        edge.weight = float(np.clip(edge.weight + cfg.weight_learning_rate *
            (correlation - cfg.weight_decay * edge.weight), -cfg.weight_limit, cfg.weight_limit))
        edge.utility = (1.0 - cfg.candidate_ema_rate) * edge.utility + cfg.candidate_ema_rate * abs(correlation)
        edge.last_active_at = float(now)
        self._stats["weight_updates"] += 1
        self._stats["weight_update_seconds"] += time.perf_counter()-started

    def _emit(self, source: int, activation: float, now: float):
        for destination in sorted(self.adjacency[source]):
            edge = self.adjacency[source][destination]
            amplitude = activation * edge.weight
            if abs(amplitude) < self.config.resting_threshold:
                continue
            if self._push(now + edge.delay_seconds, destination, amplitude, source,
                          edge.connection_id, activation):
                self._stats["propagated_events"] += 1

    def _candidate_update(self, now: float):
        started=time.perf_counter()
        cfg = self.config
        # Prune conservatively: both long inactivity and low strength/utility
        # must hold after the minimum edge lifetime.
        for source, row in enumerate(self.adjacency):
            for destination, edge in list(row.items()):
                old_enough = now - edge.created_at >= cfg.minimum_connection_lifetime_seconds
                inactive = now - edge.last_active_at >= cfg.prune_after_inactive_seconds
                weak = abs(edge.weight) <= cfg.prune_weight_threshold
                useless = edge.utility <= cfg.prune_utility_threshold
                if old_enough and inactive and weak and useless:
                    self.remove_connection(source, destination)
                    self._stats["pruned_connections"] += 1

        n = cfg.neurons
        for source, destination in self.sample_candidate_pairs():
            if destination in self.adjacency[source]:
                edge = self.adjacency[source][destination]
                score = abs(self._decayed_activity(source, now) * self._decayed_activity(destination, now))
                edge.utility = (1.0 - cfg.candidate_ema_rate) * edge.utility + cfg.candidate_ema_rate * score
                continue
            key = (source, destination)
            score = abs(self._decayed_activity(source, now) * self._decayed_activity(destination, now))
            prior, count = self._candidate_utility.get(key, (0.0, 0))
            utility = (1.0 - cfg.candidate_ema_rate) * prior + cfg.candidate_ema_rate * score
            self._candidate_utility[key] = (utility, count + 1)
            if (utility >= cfg.candidate_utility_threshold and count + 1 >= 2
                    and len(self.adjacency[source]) < cfg.max_out_degree
                    and self.connection_count < cfg.max_connections):
                sign = 1.0 if self._decayed_activity(source, now) * self._decayed_activity(destination, now) >= 0 else -1.0
                magnitude = (0.1 if self.variant == "I2_rewire_fixed" else
                    max(cfg.prune_weight_threshold * 2,
                        min(0.2, utility / max(cfg.candidate_utility_threshold, 1e-9) * 0.08)))
                self._add_connection(source, destination, sign * magnitude, now)
                self._stats["formed_connections"] += 1
                del self._candidate_utility[key]
        # Bound exploratory statistics independent of possible-pair count.
        if len(self._candidate_utility) > 4096:
            keep = sorted(self._candidate_utility.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)[:2048]
            self._candidate_utility = dict(keep)
        self._stats["structural_updates"] += 1
        self._stats["structural_seconds"] += time.perf_counter()-started
        self._last_structural_time = float(now)

    def _maybe_structural_update(self, now: float):
        if (self.structural_plasticity and
                now - self._last_structural_time >= self.config.structural_interval_seconds):
            self._candidate_update(now)

    def advance_to(self, time_seconds: float) -> int:
        """Process queued events in timestamp/sequence order through ``time``."""
        now = float(time_seconds)
        if not math.isfinite(now) or now < self.time_seconds - 1e-12:
            raise ValueError("simulation time must be finite and monotonic")
        processed = 0
        while self._queue and self._queue[0][0] <= now and processed < self.config.max_events_per_advance:
            (event_time, _, destination, amplitude, source, edge_id,
             pre_activation, kind, token) = heapq.heappop(self._queue)
            processed += 1
            if kind == 1:
                if token != self._internal_token[destination] or event_time != self.next_internal_time[destination]:
                    self._stats["stale_events"] += 1
                    continue
                self.next_internal_time[destination] = np.inf
                post, change = self._update_neuron(destination, event_time, 0.0)
                self._stats["internal_events"] += 1
                if change >= self.config.emit_threshold:
                    self._emit(destination, post, event_time)
                self._schedule_internal(destination, event_time + self.config.internal_interval_seconds)
            else:
                edge = None
                if source >= 0:
                    edge = self.adjacency[source].get(destination)
                    if edge is None or edge.connection_id != edge_id:
                        self._stats["stale_events"] += 1
                        continue
                post, change = self._update_neuron(destination, event_time, amplitude)
                if edge is not None:
                    edge.last_active_at = event_time
                    if self.weight_plasticity:
                        self._adapt_weight(edge, pre_activation, post, event_time)
                if change >= self.config.emit_threshold:
                    self._emit(destination, post, event_time)
                self._schedule_internal(destination, event_time + self.config.internal_interval_seconds)
            self._maybe_structural_update(event_time)
        if self._queue and processed >= self.config.max_events_per_advance and self._queue[0][0] <= now:
            raise RuntimeError("max_events_per_advance exceeded; event cascade is not bounded")
        self.time_seconds = now
        self._stats["processed_events"] += processed
        return processed

    def inject(self, features: np.ndarray, time_seconds: float, modality: str = "all") -> int:
        """Project generic sensory features into a bounded number of input events."""
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (self.input_dim,) or not np.all(np.isfinite(x)):
            raise ValueError(f"features must be finite with shape ({self.input_dim},)")
        if modality not in {"all", "vision", "audio"}:
            raise ValueError("modality must be all, vision, or audio")
        now = float(time_seconds)
        if now < self.time_seconds - 1e-12:
            raise ValueError("sensory event time must be monotonic")
        self.advance_to(now)
        drives = np.zeros(self.config.neurons, dtype=np.float64)
        if modality == "all":
            indices = np.arange(self.input_dim)
        elif modality == "vision":
            indices = np.arange(self.visual_dim)
        else:
            indices = np.arange(self.visual_dim, self.input_dim)
        np.add.at(drives, self._input_targets[indices], x[indices] * self._input_gains[indices])
        candidates = np.flatnonzero(np.abs(drives) > self.config.resting_threshold)
        if len(candidates) > self.config.max_sensory_targets:
            order = np.argsort(np.abs(drives[candidates]))[-self.config.max_sensory_targets:]
            candidates = candidates[order]
        accepted = 0
        for neuron in sorted(int(i) for i in candidates):
            if self._push(now, neuron, float(drives[neuron]), -1, -1, 0.0):
                accepted += 1
        self._stats["sensory_events"] += accepted
        self.advance_to(now)
        return accepted

    def ingest_observation(self, observation: Observation):
        """Incrementally encode one raw observation; no labels are accepted."""
        if not isinstance(observation, Observation):
            raise TypeError("expected Observation(t, vision, audio)")
        vector = sensory_features(observation)
        if vector.shape != (self.input_dim,):
            raise ValueError(f"configured input_dim {self.input_dim} != sensory feature size {len(vector)}")
        return self.inject(vector, observation.t)

    def snapshot(self, time_seconds: float | None = None) -> np.ndarray:
        """Read analytically decayed activations without mutating state."""
        now = self.time_seconds if time_seconds is None else float(time_seconds)
        if now < self.time_seconds - 1e-12:
            raise ValueError("snapshot time cannot precede current simulation time")
        elapsed = np.maximum(0.0, now - self.last_update_time)
        state = self.activation * np.exp(-elapsed / self.config.tau_seconds)
        self._stats["readout_scans"] += 1
        return state.copy()

    def reset_episode(self):
        """Reset fast state/queue while retaining learned weights and topology."""
        # Convert edge times to the new episode's zero-based simulation clock,
        # preserving accumulated age/inactivity across episodes.
        offset=self.time_seconds
        if offset:
            for row in self.adjacency:
                for edge in row.values():
                    edge.created_at-=offset;edge.last_active_at-=offset
            self._last_structural_time-=offset
        self.activation.fill(0.0); self.recent_incoming.fill(0.0); self.adaptation.fill(0.0)
        self.last_update_time.fill(0.0); self.activity_trace.fill(0.0); self.activity_trace_time.fill(0.0)
        self.last_activity_time.fill(-np.inf); self.next_internal_time.fill(np.inf); self._internal_token.fill(0)
        self._queue.clear(); self.time_seconds=0.0

    def topology_summary(self) -> dict[str, Any]:
        edges=[edge for row in self.adjacency for edge in row.values()]
        indegree=np.zeros(self.config.neurons,dtype=int)
        for edge in edges:indegree[edge.destination]+=1
        outdegree=np.array([len(row) for row in self.adjacency],dtype=int)
        weights=np.array([e.weight for e in edges],dtype=float)
        lifetimes=np.array([max(0.0,self.time_seconds-e.created_at) for e in edges],dtype=float)
        current=self.activation*np.exp(-np.maximum(0.0,self.time_seconds-self.last_update_time)/self.config.tau_seconds)
        return {"active_connections":len(edges),"mean_in_degree":float(indegree.mean()),
            "mean_out_degree":float(outdegree.mean()),"active_neurons":int(np.count_nonzero(np.abs(current)>self.config.resting_threshold)),
            "inactive_neuron_fraction":float(np.mean(np.abs(current)<=self.config.resting_threshold)),
            "update_count_min":int(self.neuron_update_counts.min()),
            "update_count_mean":float(self.neuron_update_counts.mean()),
            "update_count_max":int(self.neuron_update_counts.max()),
            "update_counts_by_neuron":self.neuron_update_counts.tolist(),
            "mean_abs_weight":float(np.mean(np.abs(weights))) if len(weights) else 0.0,
            "weight_std":float(np.std(weights)) if len(weights) else 0.0,
            "mean_lifetime_seconds":float(lifetimes.mean()) if len(lifetimes) else 0.0,
            "degree_out":outdegree.tolist(),"degree_in":indegree.tolist()}

    def stats(self) -> LiquidStats:
        return LiquidStats(**self._stats)

    def state_dict(self) -> dict[str, Any]:
        """Full JSON-compatible simulator snapshot, including RNG and queued events."""
        edges=[asdict(edge) for row in self.adjacency for edge in row.values()]
        return {"variant":self.variant,"seed":self.seed,"input_dim":self.input_dim,
            "visual_dim":self.visual_dim,"weight_plasticity":self.weight_plasticity,
            "structural_plasticity":self.structural_plasticity,
            "config":asdict(self.config),"edges":edges,
            "activation":self.activation.tolist(),"recent_incoming":self.recent_incoming.tolist(),
            "adaptation":self.adaptation.tolist(),"last_update_time":self.last_update_time.tolist(),
            "activity_trace":self.activity_trace.tolist(),"activity_trace_time":self.activity_trace_time.tolist(),
            "last_activity_time":self.last_activity_time.tolist(),
            "neuron_update_counts":self.neuron_update_counts.tolist(),"next_internal_time":self.next_internal_time.tolist(),
            "internal_token":self._internal_token.tolist(),"input_targets":self._input_targets.tolist(),
            "input_gains":self._input_gains.tolist(),"time_seconds":self.time_seconds,
            "last_structural_time":self._last_structural_time,"next_connection_id":self._next_connection_id,
            "sequence":self._sequence,"queue":[list(e) for e in self._queue],
            "candidate_utility":[[a,b,v,c] for (a,b),(v,c) in self._candidate_utility.items()],
            "rng_state":self.rng.bit_generator.state,"stats":asdict(self.stats())}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "LiquidNetwork":
        """Restore a snapshot without changing its topology or random stream."""
        net=cls(state["input_dim"],state["seed"],LiquidConfig(**state["config"]),
                state["variant"],state["visual_dim"])
        net.weight_plasticity=bool(state["weight_plasticity"])
        net.structural_plasticity=bool(state["structural_plasticity"])
        net.adjacency=[{} for _ in range(net.config.neurons)]
        for raw in state["edges"]:
            edge=Connection(**raw);net.adjacency[edge.source][edge.destination]=edge
        for name in ("activation","recent_incoming","adaptation","last_update_time",
                     "activity_trace","activity_trace_time","last_activity_time",
                     "neuron_update_counts","next_internal_time","_internal_token","_input_targets","_input_gains"):
            key=name[1:] if name.startswith("_") else name
            getattr(net,name)[:] = np.asarray(state[key],dtype=getattr(net,name).dtype)
        net.time_seconds=float(state["time_seconds"])
        net._last_structural_time=float(state["last_structural_time"])
        net._next_connection_id=int(state["next_connection_id"])
        net._sequence=int(state["sequence"])
        net._queue=[tuple(e) for e in state["queue"]];heapq.heapify(net._queue)
        net._candidate_utility={(int(a),int(b)):(float(v),int(c)) for a,b,v,c in state["candidate_utility"]}
        net.rng.bit_generator.state=state["rng_state"]
        timer_fields={"weight_update_seconds","structural_seconds"}
        net._stats.update({k:(float(v) if k in timer_fields else int(v)) for k,v in state["stats"].items()})
        return net
