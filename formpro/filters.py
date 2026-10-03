"""One-Euro filter for landmark smoothing (Casiez et al., CHI 2012)."""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter over a fixed-shape numpy array."""

    def __init__(
        self,
        shape: tuple[int, ...],
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.shape = shape
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: np.ndarray | None = None
        self._dx_prev = np.zeros(shape, dtype=np.float32)
        self._t_prev_s: float | None = None

    def reset(self) -> None:
        """Forget history. Called when tracking is lost, so the filter does not drag the
        skeleton from the old pose toward the new one across the gap.
        """
        self._x_prev = None
        self._dx_prev = np.zeros(self.shape, dtype=np.float32)
        self._t_prev_s = None

    def __call__(self, x: np.ndarray, timestamp_s: float) -> np.ndarray:
        if x.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {x.shape}")
        x = x.astype(np.float32, copy=False)

        if self._x_prev is None or self._t_prev_s is None:
            self._x_prev = x.copy()
            self._t_prev_s = timestamp_s
            return x

        dt = timestamp_s - self._t_prev_s
        if dt <= 0:
            # Duplicate or out-of-order timestamp: hold the previous estimate.
            return self._x_prev.copy()
        # Clamp dt so stalls and near-duplicate stamps don't break the filter.
        dt = min(max(dt, 1e-3), 0.5)

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha_array(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat.astype(np.float32, copy=False)
        self._dx_prev = dx_hat.astype(np.float32, copy=False)
        self._t_prev_s = timestamp_s
        return self._x_prev.copy()


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _alpha_array(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return (1.0 / (1.0 + tau / dt)).astype(np.float32)
