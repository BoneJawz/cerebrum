from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .audio_player import AudibleCuePlayer
from .sensory_environment import audible_cue, teacher_pattern
from .simulation_controller import SimulationController


class LiveDashboard:
    def __init__(self, controller=None, window_s=2.0):
        self.controller = controller or SimulationController()
        self.window_ms = float(window_s) * 1000
        self.player = AudibleCuePlayer()
        self.controller.audio_play_callback = self._play_audio
        self.selected_visual = None

        self.fig = plt.figure(figsize=(13, 11), constrained_layout=True)
        grid = self.fig.add_gridspec(4, 2, height_ratios=(1.0, 2.5, 1.15, 1.4))
        self.audio_ax = self.fig.add_subplot(grid[0, 0])
        self.visual_ax = self.fig.add_subplot(grid[0, 1])
        self.voltage_ax = self.fig.add_subplot(grid[1, :])
        self.raster_ax = self.fig.add_subplot(grid[2, :])
        self.weight_ax = self.fig.add_subplot(grid[3, 0])
        self.response_ax = self.fig.add_subplot(grid[3, 1])

        self.lines = [self.voltage_ax.plot([], [], label=f"Output {'ABC'[i]}", color=f"C{i}", lw=1.5)[0] for i in range(3)]
        self.voltage_ax.axhline(-70, color="black", linestyle=":", label="rest −70 mV")
        self.voltage_ax.axhline(-50, color="red", linestyle="--", label="threshold −50 mV")
        self.voltage_ax.set_ylabel("Membrane voltage (mV)")
        self.voltage_ax.set_xlabel("Simulation time (ms)")
        self.voltage_ax.set_ylim(-75, 5)
        self.voltage_ax.legend(loc="upper right", ncol=2, fontsize=8)
        self.voltage_ax.set_title("Neural activity")

        self.audio_ax.set_title("Audible representation of synthetic cue")
        self.audio_ax.set_xlabel("Audio time (s)")
        self.audio_ax.set_ylabel("Amplitude")
        self.audio_line, = self.audio_ax.plot([], [], color="tab:purple")
        self.audio_label = self.audio_ax.text(.02, .90, "Quiet", transform=self.audio_ax.transAxes, va="top")
        self.audio_ax.set_ylim(-0.27, 0.27)

        self.visual_ax.set_title("Binary visual input (3-cell field)\n(fixed sensory relay; separate teacher control)")
        self.visual_image = self.visual_ax.imshow(np.zeros((1, 3)), aspect="auto", vmin=0, vmax=1, cmap="Greens")
        self.visual_ax.set_xticks([0, 1, 2], ["A", "B", "C"])
        self.visual_ax.set_yticks([])
        self.visual_ax.set_xlabel("Visual sensor cell")
        self.visual_label = self.visual_ax.text(.02, .90, "No visual input", transform=self.visual_ax.transAxes, va="top")

        self.raster_ax.set_title("Actual spike events: audio input, visual sensor, teacher, output")
        self.raster_ax.set_xlabel("Simulation time (ms)")
        self.raster_ax.set_yticks(range(12), ["Audio A", "Audio B", "Audio C", "Visual A", "Visual B", "Visual C", "Teacher A", "Teacher B", "Teacher C", "Output A", "Output B", "Output C"], fontsize=8)

        self.weight_image = self.weight_ax.imshow(self.controller.network.weights(), vmin=0, vmax=1, cmap="viridis")
        self.weight_ax.set_title("Auditory → output synaptic weights")
        self.weight_ax.set_xlabel("Output neuron")
        self.weight_ax.set_ylabel("Auditory input neuron")
        self.weight_ax.set_xticks([0, 1, 2], list("ABC"))
        self.weight_ax.set_yticks([0, 1, 2], list("ABC"))
        self.fig.colorbar(self.weight_image, ax=self.weight_ax, fraction=.046, pad=.04, label="Weight (0–1)")

        self.response_ax.axis("off")
        self.response_text = self.response_ax.text(0, 1, "Waiting for a sensory event", va="top", family="monospace", fontsize=8)
        self.status_text = self.fig.suptitle("Keyboard: A/B/C audio • Alt+A/B/C pair visual array • Shift+A/B/C legacy teacher • 1/2/3 choose visual cell • J visual only • N sensory stream • 0/G/P plasticity • Space pause • M mute")
        self.fig.canvas.mpl_connect("key_press_event", self.key)
        self.anim = FuncAnimation(self.fig, self.update, interval=25, cache_frame_data=False)
        self.fig.canvas.mpl_connect("close_event", lambda _event: self.player.stop())

    def _play_audio(self, cue, amplitude):
        self.player.amplitude = amplitude
        self.player.muted = self.controller.audio_muted
        self.player.play(cue, amplitude)

    def _queue_audio(self, cue):
        self.controller.submit_audio(cue)
        self._set_status(f"Queued Sound {'ABC'[cue]} — auditory neuron receives a spike pattern")

    def _queue_pair(self, cue):
        target = cue if self.selected_visual is None else self.selected_visual
        self.controller.submit_pair(cue, target)
        self._set_status(f"Queued Sound {'ABC'[cue]} + teacher target {'ABC'[target]}")

    def _queue_sensory_pair(self, cue):
        position = cue if self.selected_visual is None else self.selected_visual
        self.controller.submit_sensory_pair(cue, teacher_pattern(position))
        self._set_status(f"Queued Sound {'ABC'[cue]} + binary visual array at cell {'ABC'[position]}")

    def _set_status(self, message):
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def key(self, event):
        key = event.key
        if not key:
            return
        lowered = key.lower()
        if lowered in ("alt+a", "alt+b", "alt+c"):
            self._queue_sensory_pair(ord(lowered[-1]) - ord("a"))
            return
        if lowered in ("a", "b", "c"):
            cue = ord(lowered) - ord("a")
            if key.isupper() or lowered.startswith("shift+"):
                self._queue_pair(cue)
            else:
                self._queue_audio(cue)
        elif lowered in ("shift+a", "shift+b", "shift+c"):
            self._queue_pair(ord(lowered[-1]) - ord("a"))
        elif key in ("1", "2", "3"):
            self.selected_visual = int(key) - 1
            self._set_status(f"Selected visual cell {'ABC'[self.selected_visual]}; Alt+A/B/C uses its sensory array, Shift+A/B/C its legacy teacher")
        elif lowered == "v":
            target = 0 if self.selected_visual is None else self.selected_visual
            self.controller.submit_visual(target)
            self._set_status(f"Queued visual teacher-only event for {'ABC'[target]}")
        elif lowered == "j":
            position = 0 if self.selected_visual is None else self.selected_visual
            self.controller.submit_visual_array(teacher_pattern(position))
            self._set_status(f"Queued visual sensory array only at cell {'ABC'[position]}")
        elif lowered == "e":
            self.controller.set_environment_enabled(not self.controller.environment_enabled)
            self._set_status(f"Seeded sensory stream {'on' if self.controller.environment_enabled else 'off'}")
        elif lowered == "n":
            self.controller.set_sensory_environment_enabled(not self.controller.sensory_environment_enabled)
            self._set_status(f"Visual sensory experience stream {'on' if self.controller.sensory_environment_enabled else 'off'}")
        elif lowered == "0":
            self.controller.set_learning_mode("off")
        elif lowered == "g":
            self.controller.set_learning_mode("teaching_gated")
        elif lowered == "p":
            self.controller.set_learning_mode("continuous")
        elif lowered == "m":
            self.controller.audio_muted = not self.controller.audio_muted
            self.player.muted = self.controller.audio_muted
            if self.controller.audio_muted:
                self.player.stop()
            self._set_status(f"Audio {'muted' if self.controller.audio_muted else 'unmuted'}; neural events are unchanged")
        elif key in ("+", "="):
            self.controller.audio_amplitude = min(.8, self.controller.audio_amplitude + .05)
        elif key == "-":
            self.controller.audio_amplitude = max(.02, self.controller.audio_amplitude - .05)
        elif key == " ":
            self.controller.pause() if self.controller.running else self.controller.resume()
        elif lowered == "s":
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self.controller.save_checkpoint(f"results/live_simulation/learned-{stamp}.npz")
            self.controller.save_event_log(f"results/live_simulation/events-{stamp}.csv")
            self._set_status("Saved learned parameters and event log")
        elif lowered == "f":
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self.controller.save_full_checkpoint(f"results/live_simulation/continuation-{stamp}.b2")
            self._set_status("Saved Brian2 full-state continuation checkpoint")
        elif lowered == "r":
            self.controller.reset_transient()
            self._set_status("Manually reset transient membrane state; weights and simulation time retained")

    def _update_sensory_panels(self):
        now = self.controller.simulation_time_s
        cue = self.controller.current_auditory
        if cue is not None and now <= self.controller.current_audio_until_s:
            audio = audible_cue(cue, amplitude=self.controller.audio_amplitude)
            t = np.arange(len(audio.samples)) / audio.sample_rate_hz
            self.audio_line.set_data(t, audio.samples)
            self.audio_ax.set_xlim(0, max(t[-1], .01))
            amp = max(float(np.max(np.abs(audio.samples))), .01) * 1.1
            self.audio_ax.set_ylim(-amp, amp)
            self.audio_label.set_text(f"Sound {'ABC'[cue]} · {audio.frequency_hz:.1f} Hz · {'muted' if self.controller.audio_muted else 'play requested'}")
        else:
            self.audio_line.set_data([], [])
            self.audio_label.set_text("Quiet")
        target = self.controller.current_visual
        visual_active = self.controller.current_visual_onset_s <= now <= self.controller.current_visual_until_s
        if visual_active and self.controller.current_visual_kind == "sensory_array":
            pattern = self.controller.current_visual_pattern
            self.visual_image.set_data(pattern[None, :])
            self.visual_label.set_text(f"Sensory array received: {''.join(map(str, pattern.tolist()))}")
        elif visual_active and target is not None:
            self.visual_image.set_data(teacher_pattern(target)[None, :])
            self.visual_label.set_text(f"Separate teacher signal → output {'ABC'[target]}")
        else:
            self.visual_image.set_data(np.zeros((1, 3)))
            self.visual_label.set_text("No visual input")

    def update(self, _):
        self.controller.tick()
        times, voltages, spikes = self.controller.buffer.arrays()
        if len(times):
            keep = times >= times[-1] - self.window_ms
            times, voltages = times[keep], voltages[keep]
            for idx, line in enumerate(self.lines):
                line.set_data(times, voltages[:, idx])
            xmin = max(0.0, times[-1] - self.window_ms)
            self.voltage_ax.set_xlim(xmin, max(xmin + 10.0, times[-1]))
            recent = [(ts, neuron) for ts, neuron in spikes if ts >= times[0]]
            self.raster_ax.clear()
            self.raster_ax.set_title("Actual spike events: audio input, visual sensor, teacher, output")
            self.raster_ax.set_xlabel("Simulation time (ms)")
            self.raster_ax.set_yticks(range(12), ["Audio A", "Audio B", "Audio C", "Visual A", "Visual B", "Visual C", "Teacher A", "Teacher B", "Teacher C", "Output A", "Output B", "Output C"], fontsize=8)
            if recent:
                self.raster_ax.eventplot([[ts for ts, neuron in recent if neuron == idx] for idx in range(12)], lineoffsets=range(12), colors=[f"C{i%3}" for i in range(12)])
        self.weight_image.set_data(self.controller.network.weights())
        self._update_sensory_panels()
        response = self.controller.last_response
        if response is not None:
            p = response["peak_mv"]
            s = response["spike_counts"]
            self.response_text.set_text(
                f"Last audio cue: {self.controller.current_auditory}\n"
                f"Peak depolarization (mV): A {p[0]:.2f} | B {p[1]:.2f} | C {p[2]:.2f}\n"
                f"Preferred output: {'ABC'[response['prediction']]}   margin: {response['margin_mv']:.2f} mV\n"
                f"Spikes: A {s[0]} | B {s[1]} | C {s[2]} | silent={response['silent']} tie={response['spike_tie']}\n"
                f"Sensory visual path is fixed; teacher pathway is separate"
            )
        status = self.controller.status()
        self.status_text.set_text(
            f"{status['learning_mode']} plasticity={status['plasticity']} | sim={status['simulation_time_s']:.2f}s wall={status['wall_elapsed_s']:.1f}s "
            f"rate={status['speed_ratio']:.2f}x | events={status['experiences']} paired={status['paired_experiences']} "
            f"queue={status['pending']} legacy-stream={'on' if status['environment_enabled'] else 'off'} "
            f"visual-stream={'on' if status['sensory_environment_enabled'] else 'off'}"
        )
        return self.lines

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--distributed", action="store_true",
                        help="use the separate 5x5 distributed visual-association network")
    parser.add_argument("--preview", type=Path, help="save a static initial dashboard screenshot and exit")
    args = parser.parse_args()
    if args.distributed:
        from .distributed_live_dashboard import DistributedLiveDashboard
        dashboard = DistributedLiveDashboard(args.seed, args.window)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            dashboard.enqueue(0, 0, "preview_pair")
            # Capture the immediate sensory and postsynaptic activity while it
            # is still visible in the live network graph.
            for _ in range(4): dashboard.update(0)
            dashboard.anim.event_source.stop()
            dashboard.anim._draw_was_started = True
            dashboard.fig.savefig(args.preview, dpi=130)
        else:
            dashboard.show()
        return
    dashboard = LiveDashboard(SimulationController(args.seed), args.window)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        dashboard.controller.set_learning_mode("continuous")
        dashboard.controller.audio_muted = True
        dashboard.controller.submit_sensory_pair(0, teacher_pattern(0))
        dashboard.controller.tick()
        dashboard.update(0)
        dashboard.anim.event_source.stop()
        dashboard.anim._draw_was_started = True
        dashboard.fig.savefig(args.preview, dpi=130)
    else:
        dashboard.show()


if __name__ == "__main__":
    main()
