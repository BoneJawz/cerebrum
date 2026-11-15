"""Independent regularized linear readout for liquid-state diagnostics."""
from __future__ import annotations

import numpy as np


class RidgeReadout:
    """Multi-output ridge regression fitted only on supplied state snapshots."""
    def __init__(self, alpha: float = 1.0):
        if alpha < 0:raise ValueError("alpha must be nonnegative")
        self.alpha=float(alpha);self.mean=None;self.scale=None;self.coef=None;self.intercept=None

    def fit(self, states: np.ndarray, targets: np.ndarray):
        x=np.asarray(states,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
        if x.ndim!=2 or y.ndim not in (1,2) or len(x)!=len(y):
            raise ValueError("states must be (examples,features) and targets must align")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("readout data must be finite")
        if y.ndim==1:y=y[:,None]
        self.mean=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-10]=1.
        z=(x-self.mean)/self.scale;self.intercept=y.mean(0);yc=y-self.intercept
        gram=z.T@z+self.alpha*np.eye(z.shape[1])
        self.coef=np.linalg.solve(gram,z.T@yc)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        if self.coef is None:raise RuntimeError("readout is not fitted")
        x=np.asarray(states,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.mean):raise ValueError("state dimension does not match fitted readout")
        return ((x-self.mean)/self.scale)@self.coef+self.intercept
