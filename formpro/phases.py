"""Rep cycle segmentation from hip trajectory.

Splits the lift into ``setup / eccentric / bottom / concentric / recovery`` by tracking
the vertical position of the hips over time. The live segmenter must emit exactly the
vocabulary the reference dataset uses, or Phase 5 would align a live ``bottom`` against
a reference ``eccentric`` and report the mismatch as a form error.

Velocity, not frame deltas
--------------------------
Hip velocity is computed as a least-squares slope over a fixed **time** window using
capture timestamps, never as a difference over frame counts. Two reasons, and the first
is structural: Phase 2 deliberately drops frames when inference falls behind, so
consecutive frames are not evenly spaced and a per-frame delta would read a dropped
frame as a sudden acceleration. The second is that ``error_good_morning`` is defined by
the hips rising faster than the shoulders, which is a physical rate; it is only
meaningful in units of per-second.

Both the position and the velocity are body-size normalized. Hip height arrives as a
fraction of the lifter's own leg length, so velocity is in leg-lengths per second and a
single threshold set covers every lifter.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import PhaseConfig
from .kinematics import KinematicFrame
from .schema import Phase

log = logging.getLogger(__name__)

_EPS = 1e-9


class VelocityTracker:
    """Least-squares slope of a signal over a trailing time window."""

    def __init__(self, window_ms: int, max_gap_ms: int) -> None:
        self._window_ms = window_ms
        self._max_gap_ms = max_gap_ms
        self._samples: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, timestamp_ms: int, value: float) -> float:
        """Add a sample and return the current slope per second, or ``nan``."""
        if math.isnan(value):
            return math.nan

        if self._samples:
            gap = timestamp_ms - self._samples[-1][0]
            # A long gap means tracking was lost or the pipeline stalled. Fitting across
            # it would invent a velocity from two unrelated positions.
            if gap > self._max_gap_ms or gap < 0:
                self._samples.clear()

        self._samples.append((float(timestamp_ms), float(value)))
        cutoff = timestamp_ms - self._window_ms
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if len(self._samples) < 3:
            return math.nan

        t = np.array([s[0] for s in self._samples]) / 1000.0
        y = np.array([s[1] for s in self._samples])
        t_var = float(((t - t.mean()) ** 2).sum())
        if t_var < _EPS:
            return math.nan
        return float(((t - t.mean()) * (y - y.mean())).sum() / t_var)


@dataclass(frozen=True)
class RepSegment:
    """One completed rep, closed out when the lifter returns to standing."""

    index: int
    start_frame_id: int
    end_frame_id: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    min_hip_height_norm: float

    @property
    def duration_ms(self) -> int:
        return self.end_timestamp_ms - self.start_timestamp_ms


@dataclass
class SegmentationResult:
    phase: Phase
    hip_velocity: float
    rep_index: int
    completed_rep: RepSegment | None = None


@dataclass
class _RepState:
    start_frame_id: int
    start_timestamp_ms: int
    min_hip_height_norm: float = field(default=math.inf)


class PhaseSegmenter:
    """Hysteretic state machine over hip height and hip velocity.

    Every transition must hold for ``min_dwell_frames`` consecutive frames before it
    commits. Without that, noise around a threshold produces a burst of phase flips, and
    a phase that flickers is worse than one that lags: Phase 5 keys its comparison
    window off the phase, so a single spurious ``concentric`` frame mid-descent pulls in
    the wrong reference segment entirely.
    """

    def __init__(self, config: PhaseConfig) -> None:
        self.config = config
        self._velocity = VelocityTracker(config.velocity_window_ms, config.max_gap_ms)
        self._phase = Phase.SETUP
        self._candidate = Phase.SETUP
        self._dwell = 0
        self._rep_index = 0
        self._rep: _RepState | None = None
        self._armed = False

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Full reset, e.g. a new set."""
        self._velocity.reset()
        self._phase = Phase.SETUP
        self._candidate = Phase.SETUP
        self._dwell = 0
        self._rep = None
        self._armed = False

    def on_tracking_lost(self) -> None:
        """The subject left the frame or the pose dropped out.

        Any rep in progress is abandoned rather than closed. Stitching across a gap
        would fabricate a rep whose middle was never observed, and a fabricated rep
        scored against the reference set produces a confident, wrong verdict.

        Disarming matters as much as discarding: after a dropout the lifter is usually
        still mid-rep, and without this the segmenter would immediately open a *new* rep
        from wherever it regained tracking and later close it as though complete. That
        partial rep would be scored on the fraction that happened to be visible, so a
        descent that was never seen could not be faulted. A new rep may only begin from
        a standing position this segmenter actually observed.
        """
        if self._rep is not None:
            log.debug("rep %d abandoned: tracking lost", self._rep_index + 1)
        self._velocity.reset()
        self._rep = None
        self._phase = Phase.SETUP
        self._candidate = Phase.SETUP
        self._dwell = 0
        self._armed = False

    # -- main entry point ------------------------------------------------------

    def update(self, frame: KinematicFrame) -> SegmentationResult:
        velocity = self._velocity.update(frame.timestamp_ms, frame.hip_height_norm)
        height = frame.hip_height_norm

        if self._rep is not None and not math.isnan(height):
            self._rep.min_hip_height_norm = min(self._rep.min_hip_height_norm, height)

        # Before calibration completes there is no normalized height, so there is no
        # trustworthy velocity either. Hold in setup rather than guess.
        if math.isnan(velocity) or math.isnan(height):
            return SegmentationResult(self._phase, velocity, self._rep_index)

        # Seeing the lifter stood up is what licenses the next rep. This also covers
        # startup: if the app opens with someone already in the hole, that rep is not
        # scored, because its descent was never observed.
        if height >= self.config.standing_height and abs(velocity) < self.config.still_velocity:
            self._armed = True

        desired = self._desired(height, velocity)
        completed = self._commit(desired, frame)
        return SegmentationResult(self._phase, velocity, self._rep_index, completed)

    # -- internals -------------------------------------------------------------

    def _desired(self, height: float, velocity: float) -> Phase:
        cfg = self.config
        moving_down = velocity <= -cfg.move_velocity
        moving_up = velocity >= cfg.move_velocity
        still = abs(velocity) < cfg.still_velocity

        if self._phase is Phase.SETUP:
            return Phase.ECCENTRIC if (moving_down and self._armed) else Phase.SETUP

        if self._phase is Phase.ECCENTRIC:
            if moving_up:
                # Bounced straight out of the hole with no pause, which is the norm.
                return Phase.CONCENTRIC
            if still and height < cfg.descended_height:
                return Phase.BOTTOM
            return Phase.ECCENTRIC

        if self._phase is Phase.BOTTOM:
            return Phase.CONCENTRIC if moving_up else Phase.BOTTOM

        if self._phase is Phase.CONCENTRIC:
            if height >= cfg.standing_height and not moving_up:
                return Phase.RECOVERY
            if moving_down:
                # Lost the rep and sank back down.
                return Phase.ECCENTRIC
            return Phase.CONCENTRIC

        if self._phase is Phase.RECOVERY:
            if moving_down:
                return Phase.ECCENTRIC
            return Phase.SETUP if still else Phase.RECOVERY

        return self._phase

    def _commit(self, desired: Phase, frame: KinematicFrame) -> RepSegment | None:
        if desired is self._phase:
            self._candidate = desired
            self._dwell = 0
            return None

        if desired is not self._candidate:
            self._candidate = desired
            self._dwell = 1
            return None

        self._dwell += 1
        if self._dwell < self.config.min_dwell_frames:
            return None

        previous = self._phase
        self._phase = desired
        self._dwell = 0
        return self._on_enter(previous, desired, frame)

    def _on_enter(
        self, previous: Phase, current: Phase, frame: KinematicFrame
    ) -> RepSegment | None:
        if current is Phase.ECCENTRIC and previous in (Phase.SETUP, Phase.RECOVERY):
            self._rep = _RepState(
                start_frame_id=frame.frame_id,
                start_timestamp_ms=frame.timestamp_ms,
                min_hip_height_norm=frame.hip_height_norm,
            )
            return None

        if current is Phase.RECOVERY and self._rep is not None:
            rep = self._rep
            self._rep = None
            self._rep_index += 1
            segment = RepSegment(
                index=self._rep_index,
                start_frame_id=rep.start_frame_id,
                end_frame_id=frame.frame_id,
                start_timestamp_ms=rep.start_timestamp_ms,
                end_timestamp_ms=frame.timestamp_ms,
                min_hip_height_norm=rep.min_hip_height_norm,
            )
            log.info(
                "rep %d complete: %d ms, depth %.2f leg-lengths",
                segment.index, segment.duration_ms, segment.min_hip_height_norm,
            )
            return segment
        return None

    # -- introspection ---------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def rep_count(self) -> int:
        return self._rep_index
