from __future__ import annotations
from collections import deque
import numpy as np

class ActivityBuffer:
    """Bounded rolling snapshots for live displays; older data is discarded."""
    def __init__(self,max_samples=20000): self.max_samples=max_samples; self.times=deque(maxlen=max_samples); self.voltages=deque(maxlen=max_samples); self.spikes=deque(maxlen=max_samples)
    def append(self,time_ms,voltage,spikes=()): self.times.append(float(time_ms)); self.voltages.append(np.asarray(voltage,dtype=float).copy()); self.spikes.extend((float(t),int(i)) for t,i in spikes)
    def arrays(self): return np.asarray(self.times),np.asarray(self.voltages),tuple(self.spikes)
    def clear(self): self.times.clear();self.voltages.clear();self.spikes.clear()
