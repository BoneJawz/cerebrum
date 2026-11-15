"""Always-on dashboard for the distributed 5x5 sensory configuration."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from collections import deque
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from .activity_buffer import ActivityBuffer
from .audio_player import AudibleCuePlayer
from .distributed_network import DistributedVisualAssociator
from brian2 import ms, second, mV
from .distributed_visual_experiment import patterns
from .sensory_environment import audible_cue


class DistributedLiveDashboard:
    """Keyboard-driven persistent simulation; plot refresh never sets Brian dt."""

    def __init__(self, seed=42, window_s=2.0, chunk_s=0.010):
        self.seed, self.chunk_s = int(seed), float(chunk_s)
        self.window_ms = float(window_s) * 1000.0
        self.network = DistributedVisualAssociator(seed)
        self.buffer = ActivityBuffer(20000)
        self.queue = deque()
        self.running = True
        self.mode = "paired_gated"
        self.phase = "teach"
        self.audio_muted = False
        self.audio_amplitude = .22
        self.player = AudibleCuePlayer()
        self.pattern_set = patterns()
        self.pattern_names = list(self.pattern_set)
        self.selected_pattern = 0
        self.current = None
        self.current_visual = np.zeros((5, 5), dtype=np.uint8)
        self.last_visual_name = None
        self.current_cue = None
        self.current_until_s = 0.0
        self.experiences = 0
        self.paired_experiences = 0
        self.event_log = []
        self.auto_enabled = False
        self.auto_rng = np.random.default_rng(np.random.SeedSequence([seed, 640]))
        self.next_auto_s = 0.0
        self.monitor_cursors = {"audio": 0, "visual": 0, "assoc": 0}
        self.wall_start = time.perf_counter()
        self.network.set_plasticity(False)

        self.fig = plt.figure(figsize=(15, 14), constrained_layout=True)
        grid = self.fig.add_gridspec(4, 2, height_ratios=(1.0, 1.6, 1.35, 2.2))
        self.audio_ax = self.fig.add_subplot(grid[0, 0])
        self.visual_ax = self.fig.add_subplot(grid[0, 1])
        self.voltage_ax = self.fig.add_subplot(grid[1, :])
        self.raster_ax = self.fig.add_subplot(grid[2, 0])
        self.weight_ax = self.fig.add_subplot(grid[2, 1])
        self.graph_ax = self.fig.add_subplot(grid[3, :])
        self.audio_ax.set_title("Human-audible representation of cue")
        self.audio_line, = self.audio_ax.plot([], [], color="tab:purple")
        self.audio_ax.set_xlabel("Audio time (s)"); self.audio_ax.set_ylabel("Amplitude")
        self.audio_label = self.audio_ax.text(.02, .9, "Quiet", transform=self.audio_ax.transAxes, va="top")
        self.audio_ax.set_ylim(-.27, .27)
        self.visual_ax.set_title("Actual 5 × 5 binary visual array")
        self.visual_image = self.visual_ax.imshow(self.current_visual, vmin=0, vmax=1, cmap="Greens")
        self.visual_ax.set_xticks(range(5)); self.visual_ax.set_yticks(range(5))
        self.visual_ax.set_xticks(np.arange(-.5, 5, 1), minor=True); self.visual_ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        self.visual_ax.grid(which="minor", color="#777", linewidth=.7)
        self.visual_label = self.visual_ax.text(.02, .98, "No visual event", transform=self.visual_ax.transAxes, va="top")
        self.voltage_lines = [self.voltage_ax.plot([], [], lw=1, label=f"Assoc {i}")[0]
                              for i in range(self.network.n_associative)]
        self.voltage_ax.axhline(-70, color="black", linestyle=":", label="rest")
        self.voltage_ax.axhline(-50, color="red", linestyle="--", label="threshold")
        self.voltage_ax.set_title("Shared associative population membrane voltage")
        self.voltage_ax.set_xlabel("Simulation time (ms)"); self.voltage_ax.set_ylabel("mV")
        self.voltage_ax.set_ylim(-75, 5); self.voltage_ax.legend(ncol=7, fontsize=6, loc="upper right")
        self.raster_ax.set_title("Sensory and associative spike events")
        self.raster_ax.set_xlabel("Simulation time (ms)")
        ylabels = ([f"Audio {c}" for c in "ABC"]
                   + [f"Visual {i}" if i % 10 == 0 or i == 24 else "" for i in range(25)]
                   + [f"Assoc {i}" if i % 6 == 0 or i == 17 else "" for i in range(18)])
        self.raster_labels = ylabels
        self.raster_ax.set_yticks(range(46), ylabels, fontsize=6)
        self.weight_image = self.weight_ax.imshow(self.network.weights(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        self.weight_ax.set_title("Auditory → associative STDP weights")
        self.weight_ax.set_xlabel("Associative neuron"); self.weight_ax.set_ylabel("Auditory cue channel")
        self.weight_ax.set_yticks(range(3), list("ABC"))
        self.fig.colorbar(self.weight_image, ax=self.weight_ax, fraction=.046, pad=.04, label="Weight")
        self._build_network_graph()
        self.status = self.fig.suptitle("Teach: T • Test: R • A/B/C audio • Alt+A/B/C paired • 1/2/3 choose array • J visual • N auto • 0/G/P plasticity • Space pause • S save • F checkpoint")
        self.fig.canvas.mpl_connect("key_press_event", self.key)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.player.stop())
        self.anim = FuncAnimation(self.fig, self.update, interval=25, cache_frame_data=False)

    @property
    def simulation_time_s(self):
        return float(self.network.net.t / second)

    def enqueue(self, cue=None, visual_idx=None, kind="manual"):
        pattern = None if visual_idx is None else self.pattern_set[self.pattern_names[int(visual_idx)]].copy()
        self.queue.append({"cue": cue, "visual_idx": visual_idx, "pattern": pattern, "kind": kind})

    def _start(self, event):
        start = self.simulation_time_s
        cue, pattern = event["cue"], event["pattern"]
        audio_ids = np.array([], dtype=int) if cue is None else np.array([cue], dtype=int)
        audio_times = np.array([]) * second if cue is None else np.array([start + .010]) * second
        visual_ids = np.array([], dtype=int) if pattern is None else np.flatnonzero(pattern.reshape(-1))
        visual_times = np.array([]) * second if not len(visual_ids) else np.full(len(visual_ids), start + .025) * second
        self.network.auditory_input.set_spikes(audio_ids, audio_times, period=0 * second)
        self.network.visual_input.set_spikes(visual_ids, visual_times, period=0 * second)
        paired = cue is not None and pattern is not None
        plasticity_enabled = self.phase == "teach" and (
            self.mode == "continuous" or (self.mode == "paired_gated" and paired))
        self.network.set_plasticity(plasticity_enabled)
        self.current = {**event, "start_s": start, "end_s": start + .060, "paired": paired}
        self.current_cue = cue
        self.current_visual = np.zeros((5, 5), dtype=np.uint8) if pattern is None else pattern
        if pattern is not None:
            self.last_visual_name = self.pattern_names[int(event["visual_idx"])]
        self.current_until_s = start + .060
        if cue is not None and not self.audio_muted:
            self.player.play(cue, self.audio_amplitude)
        self.event_log.append({"sequence": len(self.event_log), "simulation_start_s": start,
                               "event_type": event["kind"], "auditory_channel": cue,
                               "visual_array": "" if pattern is None else "".join(map(str, pattern.reshape(-1).tolist())),
                               "visual_pattern_name": "" if event["visual_idx"] is None else self.pattern_names[event["visual_idx"]],
                               "phase": self.phase, "plasticity_mode": self.mode,
                               "plasticity_enabled": int(plasticity_enabled),
                               "auditory_onset_s": "" if cue is None else start + .010,
                               "visual_onset_s": "" if pattern is None else start + .025,
                               "simulation_end_s": ""})

    def _capture(self):
        vm = self.network.voltage
        times = np.asarray(vm.t[:] / ms, dtype=float)
        volts = np.asarray(vm.v[:] / mV, dtype=float)
        for k, ts in enumerate(times):
            self.buffer.append(float(ts), volts[:, k], ())
        vm.resize(0)
        events = []
        for group, offset, cursor_key in ((self.network.auditory_spikes, 0, "audio"),
                                           (self.network.visual_spikes, 3, "visual"),
                                           (self.network.associative_spikes, 28, "assoc")):
            old = self.monitor_cursors[cursor_key]
            new = len(group.t)
            ids = np.asarray(group.i[old:new], dtype=int)
            tms = np.asarray(group.t[old:new] / ms, dtype=float)
            events.extend((float(t), int(i + offset)) for t, i in zip(tms, ids))
            self.monitor_cursors[cursor_key] = new
        self.buffer.spikes.extend(events)

    def _build_network_graph(self):
        """Build static topology once; animation updates only colors and widths."""
        self.audio_xy = np.column_stack((np.full(3, .10), np.linspace(.65, -.65, 3)))
        self.visual_xy = np.asarray([[1.00 + .16 * (cell % 5), .80 - .40 * (cell // 5)]
                                     for cell in range(25)], dtype=float)
        self.assoc_xy = np.column_stack((np.full(self.network.n_associative, 2.75),
                                         np.linspace(1.0, -1.0, self.network.n_associative)))
        self.audio_edges = np.asarray([(src, dst) for src in range(3)
                                       for dst in range(self.network.n_associative)], dtype=int)
        self.visual_edges = self.network.visual_edges.copy()
        self.audio_segments = np.asarray([[self.audio_xy[src], self.assoc_xy[dst]]
                                          for src, dst in self.audio_edges])
        self.visual_segments = np.asarray([[self.visual_xy[src], self.assoc_xy[dst]]
                                           for src, dst in self.visual_edges])
        self.visual_edge_artist = LineCollection(self.visual_segments, colors="#7b9b91",
                                                 linewidths=.55, alpha=.28, zorder=1)
        self.audio_edge_artist = LineCollection(self.audio_segments, colors="#536b9a",
                                                linewidths=.8, alpha=.8, zorder=2)
        self.graph_ax.add_collection(self.visual_edge_artist)
        self.graph_ax.add_collection(self.audio_edge_artist)
        self.audio_nodes = self.graph_ax.scatter(self.audio_xy[:, 0], self.audio_xy[:, 1],
                                                 s=95, marker="o", c="#c6ddf2",
                                                 edgecolors="#263548", linewidths=1.0, zorder=5)
        self.visual_nodes = self.graph_ax.scatter(self.visual_xy[:, 0], self.visual_xy[:, 1],
                                                  s=54, marker="s", c="#e5eee9",
                                                  edgecolors="#263548", linewidths=.65, zorder=5)
        self.assoc_nodes = self.graph_ax.scatter(self.assoc_xy[:, 0], self.assoc_xy[:, 1],
                                                 s=66, marker="o", c="#d7e5f4",
                                                 edgecolors="#263548", linewidths=.8, zorder=5)
        for i, (x, y) in enumerate(self.audio_xy):
            self.graph_ax.text(x - .16, y, "ABC"[i], ha="right", va="center", fontsize=9, weight="bold")
        for i, (x, y) in enumerate(self.visual_xy):
            self.graph_ax.text(x, y, str(i), ha="center", va="center", fontsize=5.5,
                               color="#263548", zorder=6)
        for label, x in (("Auditory input", .10), ("5 × 5 visual sensor", 1.32),
                         ("Shared associative population", 2.75)):
            self.graph_ax.text(x, 1.18, label, ha="center", va="bottom", fontsize=9, weight="bold")
        self.graph_ax.text(.52, -.91, "fixed sparse visual projection", color="#526c63", fontsize=7)
        self.graph_ax.text(.84, -1.14, "auditory STDP • color/width = weight • gold = recent pre→post spikes",
                           color="#42577b", fontsize=8)
        self.graph_ax.set_xlim(-.35, 3.08); self.graph_ax.set_ylim(-1.32, 1.38)
        self.graph_ax.set_axis_off()
        self.graph_ax.set_title("Live neural network and directed connections", pad=6)
        self.audio_weight_cmap = plt.get_cmap("viridis")
        self.audio_weight_norm = Normalize(vmin=0.0, vmax=1.0)
        scalar = ScalarMappable(norm=self.audio_weight_norm, cmap=self.audio_weight_cmap)
        scalar.set_array([])
        self.fig.colorbar(scalar, ax=self.graph_ax, fraction=.018, pad=.01,
                          location="right", label="Auditory synaptic weight")

    def _update_network_graph(self):
        now_ms = self.simulation_time_s * 1000.0
        recent = [(t, channel) for t, channel in self.buffer.spikes if now_ms - t <= 35.0]
        by_channel = {}
        for t, channel in recent:
            by_channel.setdefault(channel, []).append(t)
        active_ids = {channel for _, channel in recent}
        audio_active = np.array([i in active_ids for i in range(3)])
        visual_spike_active = np.array([i + 3 in active_ids for i in range(25)])
        assoc_spike_active = np.array([i + 28 in active_ids for i in range(self.network.n_associative)])

        self.audio_nodes.set_facecolors(["#ffbf47" if active else "#c6ddf2" for active in audio_active])
        self.audio_nodes.set_edgecolors(["#d97706" if active else "#263548" for active in audio_active])
        pattern_cells = self.current_visual.reshape(-1).astype(bool)
        self.visual_nodes.set_facecolors(["#65b985" if active else "#edf2ef" for active in pattern_cells])
        self.visual_nodes.set_edgecolors(["#f59e0b" if active else "#263548" for active in visual_spike_active])
        self.visual_nodes.set_linewidths([1.8 if active else .65 for active in visual_spike_active])
        voltage = np.asarray(self.network.associative.v[:] / mV, dtype=float)
        voltage_fraction = np.clip((voltage + 70.0) / 25.0, 0.0, 1.0)
        self.assoc_nodes.set_facecolors(plt.get_cmap("coolwarm")(voltage_fraction))
        self.assoc_nodes.set_edgecolors(["#f59e0b" if active else "#263548" for active in assoc_spike_active])
        self.assoc_nodes.set_linewidths([2.2 if active else .8 for active in assoc_spike_active])
        self.audio_nodes.set_sizes([125 if active else 95 for active in audio_active])
        self.visual_nodes.set_sizes([78 if spiked else (62 if pattern_cells[i] else 48)
                                     for i, spiked in enumerate(visual_spike_active)])
        self.assoc_nodes.set_sizes([92 if active else 66 for active in assoc_spike_active])

        weights = self.network.weights().reshape(-1)
        audio_colors = self.audio_weight_cmap(self.audio_weight_norm(weights))
        audio_widths = .55 + 2.2 * weights
        for edge_idx, (src, dst) in enumerate(self.audio_edges):
            source_times = by_channel.get(int(src), ())
            target_times = by_channel.get(int(dst + 28), ())
            paired_spike = any(pre <= post <= pre + 30.0
                               for pre in source_times for post in target_times)
            if paired_spike:
                audio_colors[edge_idx] = (1.0, .58, .05, 1.0)
                audio_widths[edge_idx] = 2.8
        self.audio_edge_artist.set_color(audio_colors)
        self.audio_edge_artist.set_linewidth(audio_widths)

        visual_colors = np.tile(np.array([.48, .61, .56, .24]), (len(self.visual_edges), 1))
        visual_widths = np.full(len(self.visual_edges), .55)
        for edge_idx, (src, dst) in enumerate(self.visual_edges):
            source_times = by_channel.get(int(src + 3), ())
            target_times = by_channel.get(int(dst + 28), ())
            paired_spike = any(pre <= post <= pre + 30.0
                               for pre in source_times for post in target_times)
            if paired_spike:
                visual_colors[edge_idx] = (1.0, .58, .05, 1.0)
                visual_widths[edge_idx] = 2.0
        self.visual_edge_artist.set_color(visual_colors)
        self.visual_edge_artist.set_linewidth(visual_widths)

    def tick(self):
        if not self.running:
            return
        now = self.simulation_time_s
        if self.current is None:
            if self.auto_enabled and now >= self.next_auto_s and not self.queue:
                cue = int(self.auto_rng.integers(0, 3)); visual = int(self.auto_rng.integers(0, 3))
                self.enqueue(cue, visual, "seeded_auto_pair")
                self.next_auto_s = now + .25
            if self.queue:
                self._start(self.queue.popleft())
        if self.current is None:
            self.network.set_plasticity(self.phase == "teach" and self.mode == "continuous")
            step = self.chunk_s
        else:
            step = min(self.chunk_s, max(self.current["end_s"] - now, 0.0))
        if step > 0:
            self.network.net.run(step * second)
            self._capture()
        if self.current is not None and self.simulation_time_s >= self.current["end_s"] - 1e-12:
            if self.current["cue"] is not None: self.experiences += 1
            if self.current["paired"]:
                self.paired_experiences += 1
            self.event_log[-1]["simulation_end_s"] = self.simulation_time_s
            self.current = None
            self.network.set_plasticity(self.phase == "teach" and self.mode == "continuous")

    def key(self, event):
        key = event.key
        if not key: return
        low = key.lower()
        if low == "t":
            self.phase = "teach"
            self.status.set_text("TEACH phase • plasticity follows 0/G/P mode")
        elif low == "r":
            self.phase = "test"
            self.network.set_plasticity(False)
            self.status.set_text("TEST phase • plasticity forced off; A/B/C recall is auditory-only")
        elif low in ("alt+a", "alt+b", "alt+c"):
            cue = ord(low[-1]) - 97; self.enqueue(cue, cue, "paired_sensory")
        elif low in ("shift+a", "shift+b", "shift+c"):
            cue = ord(low[-1]) - 97; self.enqueue(cue, self.selected_pattern, "paired_sensory")
        elif low in ("a", "b", "c"):
            self.enqueue(ord(low) - 97, None, "auditory_only")
        elif key in ("1", "2", "3"):
            self.selected_pattern = int(key) - 1
            self.status.set_text(f"Selected {self.pattern_names[self.selected_pattern]} • 0/G/P plasticity • Space pause")
        elif low == "j": self.enqueue(None, self.selected_pattern, "visual_only")
        elif low == "n":
            self.auto_enabled = not self.auto_enabled
            self.status.set_text(f"Seeded automatic paired stream {'ON' if self.auto_enabled else 'OFF'}")
        elif low in ("0", "g", "p"):
            self.mode = {"0": "off", "g": "paired_gated", "p": "continuous"}[low]
            self.network.set_plasticity(self.phase == "teach" and self.mode == "continuous")
            self.status.set_text(f"Plasticity mode: {self.mode}; phase={self.phase}")
        elif key == " ": self.running = not self.running
        elif low == "m":
            self.audio_muted = not self.audio_muted; self.player.muted = self.audio_muted
            if self.audio_muted: self.player.stop()
        elif low == "s": self.save()
        elif low == "f": self.checkpoint()
        elif key in ("+", "="): self.audio_amplitude = min(.8, self.audio_amplitude + .05)
        elif key == "-": self.audio_amplitude = max(.02, self.audio_amplitude - .05)

    def save(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = Path("results/distributed_live") / stamp
        path.mkdir(parents=True, exist_ok=True)
        times_ms, voltage_mv, spikes = self.buffer.arrays()
        np.savez_compressed(path / "learned_weights.npz", weights=self.network.weights(),
                            visual_edges=self.network.visual_edges, seed=self.seed,
                            simulation_time_s=self.simulation_time_s)
        spike_array = np.asarray(spikes, dtype=float).reshape(-1, 2) if spikes else np.empty((0, 2))
        np.savez_compressed(path / "activity_recording.npz", times_ms=times_ms,
                            voltage_mv=voltage_mv, spike_time_ms_and_channel=spike_array)
        with (path / "activity.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.event_log[0]) if self.event_log else ["sequence"])
            w.writeheader(); w.writerows(self.event_log)
        self.status.set_text(f"Saved session under {path}")

    def checkpoint(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = Path("results/distributed_live") / f"checkpoint-{stamp}.b2"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.network.net.store("distributed_live", filename=str(path))
        self.status.set_text(f"Saved Brian2 checkpoint {path}")

    def update(self, _frame):
        self.tick()
        times, voltages, spikes = self.buffer.arrays()
        if len(times):
            keep = times >= times[-1] - self.window_ms
            times, voltages = times[keep], voltages[keep]
            for i, line in enumerate(self.voltage_lines):
                line.set_data(times, voltages[:, i])
            self.voltage_ax.set_xlim(max(0, times[-1] - self.window_ms), max(10, times[-1]))
            recent = [(t, n) for t, n in spikes if t >= times[0]]
            self.raster_ax.clear()
            self.raster_ax.set_title("Actual auditory, visual-sensor, and associative spikes")
            self.raster_ax.set_xlabel("Simulation time (ms)")
            self.raster_ax.set_yticks(range(46), self.raster_labels, fontsize=6)
            if recent:
                self.raster_ax.eventplot([[t for t, n in recent if n == i] for i in range(46)], lineoffsets=range(46), linewidths=.6)
        self.weight_image.set_data(self.network.weights())
        self._update_network_graph()
        self.visual_image.set_data(self.current_visual)
        if self.current is not None and self.current["pattern"] is not None:
            self.visual_label.set_text(f"Input received: {self.last_visual_name} (raw binary array)")
        elif self.last_visual_name is not None:
            self.visual_label.set_text(f"Last input: {self.last_visual_name}")
        else:
            self.visual_label.set_text("No visual event")
        if self.current_cue is not None and self.simulation_time_s <= self.current_until_s + .18:
            tone = audible_cue(self.current_cue, amplitude=self.audio_amplitude)
            sample_times = np.arange(len(tone.samples)) / tone.sample_rate_hz
            self.audio_line.set_data(sample_times, tone.samples)
            self.audio_ax.set_xlim(0, max(.01, sample_times[-1]))
            self.audio_label.set_text(f"Cue {'ABC'[self.current_cue]} • {'muted' if self.audio_muted else 'play requested'}")
        else:
            self.audio_line.set_data([], []); self.audio_label.set_text("Quiet")
        status = "running" if self.running else "paused"
        event_label = "idle" if self.current is None else self.current["kind"]
        self.status.set_text(f"{status} • phase={self.phase} • plasticity={self.mode} • event={event_label} • sim={self.simulation_time_s:.2f}s • wall={time.perf_counter()-self.wall_start:.1f}s • experiences={self.experiences} paired={self.paired_experiences} queue={len(self.queue)} auto={'on' if self.auto_enabled else 'off'}")
        return self.voltage_lines

    def show(self):
        plt.show()

