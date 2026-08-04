"""One-Euro filter for landmark smoothing.

BlazePose output jitters by a few millimetres frame to frame even when the lifter is
motionless. That noise is harmless for a static pose but not for us: Phase 3 differentiates
hip Y to segment the eccentric and concentric phases, and differentiation amplifies
high-frequency noise. Smoothing therefore belongs in ingestion, upstream of any analysis.

A One-Euro filter is used rather than a fixed low-pass because its cutoff adapts to speed:
heavy smoothing while the lifter is holding a position, light smoothing during a fast
ascent. A fixed filter would have to choose between visible jitter at the bottom of the
squat and lag at the top.

Reference: Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems", CHI 2012.
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter over a fixed-shape numpy array.

    Parameters
    ----------
    min_cutoff:
        Cutoff frequency (Hz) at zero speed. Lower = smoother but laggier when still.
    beta:
        Speed coefficient. Higher = less lag during fast motion, more jitter.
    d_cutoff:
        Cutoff for the derivative estimate itself.
    """

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
        """Forget history — call when tracking is lost so the filter does not drag the
        skeleton from the old pose toward the new one across the gap."""
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
            # Duplicate or out-of-order timestamp: hold the previous estimate rather
            # than dividing by zero when computing the derivative.
            return self._x_prev.copy()
        # Clamp so a long stall (window drag, USB hiccup) doesn't produce a huge dt that
        # makes alpha ~1 and defeats the filter, nor a tiny dt that spikes the derivative.
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
