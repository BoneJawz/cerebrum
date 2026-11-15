from __future__ import annotations

import sys
import os
import tempfile

from .sensory_environment import audible_cue


class AudibleCuePlayer:
    """Best-effort asynchronous Windows playback for cue representations."""

    def __init__(self):
        self.available = sys.platform == "win32"
        self.muted = False
        self.amplitude = 0.22
        self.last_request_wall_time = None
        self.last_cue = None
        self._wave_bytes = None
        self._wave_path = None
        if self.available:
            import winsound
            self._winsound = winsound
        else:
            self._winsound = None

    def play(self, cue: int, amplitude: float | None = None):
        self.last_cue = cue
        self.last_request_wall_time = __import__("time").perf_counter()
        if self.muted or not self.available:
            return False
        amp = self.amplitude if amplitude is None else float(amplitude)
        self._wave_bytes = audible_cue(cue, amplitude=amp).wav_bytes
        self.stop()
        fd, self._wave_path = tempfile.mkstemp(prefix="mdl-cue-", suffix=".wav")
        with os.fdopen(fd, "wb") as stream:
            stream.write(self._wave_bytes)
        self._winsound.PlaySound(self._wave_path, self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
        return True

    def stop(self):
        if self.available:
            self._winsound.PlaySound(None, self._winsound.SND_PURGE)
        if self._wave_path and os.path.exists(self._wave_path):
            os.unlink(self._wave_path)
        self._wave_path = None
