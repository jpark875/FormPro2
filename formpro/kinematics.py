"""Converts 3D landmarks into the reference angle set.

Angles attenuate Z by z_weight; segment lengths use full 3D. The reference corpus must
come from this engine at the same camera angle so the attenuation bias cancels.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import KinematicsConfig
from .schema import SIDE_LANDMARKS, FormLabel, Phase, PoseFrame, Side

log = logging.getLogger(__name__)

_EPS = 1e-6
_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)

#: Feature layout shared by live and reference data. Order is part of the contract.
FEATURE_ORDER: tuple[str, ...] = (
    "camera_near.hip_flexion",
    "camera_near.knee_flexion",
    "camera_near.ankle_dorsiflexion",
    "camera_near.back_to_vertical",
    "camera_far.hip_flexion",
    "camera_far.knee_flexion",
    "camera_far.ankle_dorsiflexion",
    "camera_far.back_to_vertical",
    "global.knee_to_hip_width_ratio",
)

SIDE_ANGLE_FIELDS: tuple[str, ...] = (
    "hip_flexion", "knee_flexion", "ankle_dorsiflexion", "back_to_vertical",
)


@dataclass(frozen=True)
class SideAngles:
    """The four sagittal measures for one side, in degrees."""

    hip_flexion: float
    knee_flexion: float
    ankle_dorsiflexion: float
    back_to_vertical: float

    def to_dict(self) -> dict[str, float]:
        return {f: round(float(getattr(self, f)), 4) for f in SIDE_ANGLE_FIELDS}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SideAngles:
        return cls(**{f: float(raw[f]) for f in SIDE_ANGLE_FIELDS})

    @property
    def complete(self) -> bool:
        return not any(math.isnan(getattr(self, f)) for f in SIDE_ANGLE_FIELDS)

    @classmethod
    def unmeasured(cls) -> SideAngles:
        return cls(math.nan, math.nan, math.nan, math.nan)


@dataclass(frozen=True)
class GlobalMetrics:
    """Measures that are inherently bilateral and have no per-side form."""

    knee_to_hip_width_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {"knee_to_hip_width_ratio": round(float(self.knee_to_hip_width_ratio), 4)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GlobalMetrics:
        return cls(knee_to_hip_width_ratio=float(raw["knee_to_hip_width_ratio"]))


@dataclass(frozen=True)
class BodyProportions:
    """Calibrated segment geometry for one lifter, in metres and dimensionless ratios."""

    femur_m: float
    tibia_m: float
    torso_m: float
    femur_to_torso_ratio: float
    tibia_to_femur_ratio: float
    samples: int

    @property
    def leg_length_m(self) -> float:
        return self.femur_m + self.tibia_m

    def to_dict(self) -> dict[str, float]:
        return {
            "femur_to_torso_ratio": round(self.femur_to_torso_ratio, 4),
            "tibia_to_femur_ratio": round(self.tibia_to_femur_ratio, 4),
        }


@dataclass(frozen=True)
class KinematicFrame:
    """One normalized frame, produced by this engine for live data and by
    ``dataset_loader`` for reference data. Identical type on both paths.
    """

    frame_id: int
    timestamp_ms: int
    camera_near: SideAngles
    camera_far: SideAngles
    global_metrics: GlobalMetrics
    phase: Phase | None = None
    form_label: FormLabel | None = None
    near_side: Side | None = None
    hip_height_norm: float = math.nan

    def to_json_frame(self) -> dict[str, Any]:
        """Serialize into the reference dataset's frame shape."""
        frame: dict[str, Any] = {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "phase": self.phase.value if self.phase else None,
            "angles": {
                "camera_near": self.camera_near.to_dict(),
                "camera_far": self.camera_far.to_dict(),
                "global": self.global_metrics.to_dict(),
            },
        }
        if self.form_label is not None:
            frame["form_label"] = self.form_label.value
        return frame

    @property
    def near_complete(self) -> bool:
        """Whether the well-observed side is fully measured."""
        return self.camera_near.complete

    def with_phase(self, phase: Phase) -> KinematicFrame:
        return replace(self, phase=phase)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Included angle between two vectors, in degrees on [0, 180]."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return math.nan
    cosine = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _segment(pose: PoseFrame, a: int, b: int, z_weight: float) -> np.ndarray:
    """Vector from landmark ``a`` to ``b`` with the depth axis attenuated."""
    v = pose.world_xyz[b].astype(np.float64) - pose.world_xyz[a].astype(np.float64)
    return np.array([v[0], v[1], v[2] * z_weight])


def _length_m(pose: PoseFrame, a: int, b: int) -> float:
    """True 3D segment length, never attenuated. See the module docstring."""
    return float(np.linalg.norm(pose.world_xyz[b] - pose.world_xyz[a]))


class SideResolver:
    """Decides which anatomical side faces the camera, with hysteresis."""

    def __init__(self, config: KinematicsConfig) -> None:
        self._alpha = config.side_ema_alpha
        self._hysteresis = config.side_hysteresis_m
        self._score: float | None = None
        self._side: Side | None = None

    def reset(self) -> None:
        self._score = None
        self._side = None

    @property
    def side(self) -> Side | None:
        return self._side

    def update(self, pose: PoseFrame) -> Side | None:
        depths = {}
        for side, joints in SIDE_LANDMARKS.items():
            zs = [float(pose.world(joints[j])[2]) for j in ("hip", "knee", "ankle")]
            depths[side] = sum(zs) / len(zs)

        # Positive means the left side is nearer the camera.
        raw = depths[Side.LEFT] - depths[Side.RIGHT]
        self._score = raw if self._score is None else (
            self._alpha * raw + (1.0 - self._alpha) * self._score
        )

        if self._side is None:
            self._side = Side.LEFT if self._score >= 0 else Side.RIGHT
        elif self._side is Side.LEFT and self._score < -self._hysteresis:
            log.debug("camera-near side switched to RIGHT (score %.3f)", self._score)
            self._side = Side.RIGHT
        elif self._side is Side.RIGHT and self._score > self._hysteresis:
            log.debug("camera-near side switched to LEFT (score %.3f)", self._score)
            self._side = Side.LEFT
        return self._side


class ProportionCalibrator:
    """Estimates the lifter's segment geometry over a rolling window."""

    def __init__(self, config: KinematicsConfig) -> None:
        self._window = config.calibration_window_frames
        self._min_frames = config.calibration_min_frames
        self._femur: deque[float] = deque(maxlen=self._window)
        self._tibia: deque[float] = deque(maxlen=self._window)
        self._torso: deque[float] = deque(maxlen=self._window)

    def reset(self) -> None:
        self._femur.clear()
        self._tibia.clear()
        self._torso.clear()

    @property
    def calibrated(self) -> bool:
        return len(self._femur) >= self._min_frames and len(self._torso) >= self._min_frames

    @property
    def progress(self) -> float:
        """0..1, for a calibration prompt in the UI."""
        if self._min_frames <= 0:
            return 1.0
        return min(1.0, min(len(self._femur), len(self._torso)) / self._min_frames)

    def update(self, pose: PoseFrame, min_visibility: float) -> None:
        for joints in SIDE_LANDMARKS.values():
            def visible(*names: str, joints: dict = joints) -> bool:
                return all(pose.is_visible(joints[n], min_visibility) for n in names)

            if visible("hip", "knee"):
                self._femur.append(_length_m(pose, joints["hip"], joints["knee"]))
            if visible("knee", "ankle"):
                self._tibia.append(_length_m(pose, joints["knee"], joints["ankle"]))
            if visible("shoulder", "hip"):
                self._torso.append(_length_m(pose, joints["shoulder"], joints["hip"]))

    def proportions(self) -> BodyProportions | None:
        if not self.calibrated:
            return None
        femur = float(np.median(self._femur))
        tibia = float(np.median(self._tibia)) if self._tibia else math.nan
        torso = float(np.median(self._torso))
        if femur < _EPS or torso < _EPS:
            return None
        return BodyProportions(
            femur_m=femur,
            tibia_m=tibia,
            torso_m=torso,
            femur_to_torso_ratio=femur / torso,
            tibia_to_femur_ratio=math.nan if math.isnan(tibia) else tibia / femur,
            samples=len(self._femur),
        )


class KinematicsEngine:
    """Converts ``PoseFrame`` into ``KinematicFrame``."""

    def __init__(self, config: KinematicsConfig, min_visibility: float) -> None:
        self.config = config
        self.min_visibility = min_visibility
        self._sides = SideResolver(config)
        self._calibrator = ProportionCalibrator(config)
        self._proportions: BodyProportions | None = None

    def reset(self) -> None:
        self._sides.reset()
        self._calibrator.reset()
        self._proportions = None

    @property
    def proportions(self) -> BodyProportions | None:
        """Frozen once calibration completes; see ProportionCalibrator."""
        return self._proportions

    @property
    def calibration_progress(self) -> float:
        return 1.0 if self._proportions else self._calibrator.progress

    def update(self, pose: PoseFrame) -> KinematicFrame | None:
        """Normalize one pose frame, or return ``None`` if the core joints are missing."""
        core = (
            SIDE_LANDMARKS[Side.LEFT]["hip"], SIDE_LANDMARKS[Side.RIGHT]["hip"],
            SIDE_LANDMARKS[Side.LEFT]["knee"], SIDE_LANDMARKS[Side.RIGHT]["knee"],
        )
        if pose.missing(core, self.min_visibility):
            return None

        near = self._sides.update(pose)
        if near is None:
            return None

        self._calibrator.update(pose, self.min_visibility)
        if self._proportions is None:
            self._proportions = self._calibrator.proportions()

        return KinematicFrame(
            frame_id=pose.index,
            timestamp_ms=pose.timestamp_ms,
            camera_near=self._side_angles(pose, near),
            camera_far=self._side_angles(pose, near.other),
            global_metrics=GlobalMetrics(self._width_ratio(pose)),
            near_side=near,
            hip_height_norm=self._hip_height_norm(pose),
        )

    def _side_angles(self, pose: PoseFrame, side: Side) -> SideAngles:
        joints = SIDE_LANDMARKS[side]
        zw = self.config.z_weight

        def ok(*names: str) -> bool:
            return all(pose.is_visible(joints[n], self.min_visibility) for n in names)

        torso = _segment(pose, joints["hip"], joints["shoulder"], zw)

        hip = knee = ankle = back = math.nan
        if ok("hip", "shoulder", "knee"):
            hip = angle_between(torso, _segment(pose, joints["hip"], joints["knee"], zw))
        if ok("knee", "hip", "ankle"):
            knee = angle_between(
                _segment(pose, joints["knee"], joints["hip"], zw),
                _segment(pose, joints["knee"], joints["ankle"], zw),
            )
        if ok("ankle", "knee", "heel", "toe"):
            ankle = angle_between(
                _segment(pose, joints["ankle"], joints["knee"], zw),
                _segment(pose, joints["heel"], joints["toe"], zw),
            )
        if ok("hip", "shoulder"):
            back = angle_between(torso, _UP)

        return SideAngles(hip, knee, ankle, back)

    def _width_ratio(self, pose: PoseFrame) -> float:
        """Inter-knee X separation over inter-hip X separation."""
        left, right = SIDE_LANDMARKS[Side.LEFT], SIDE_LANDMARKS[Side.RIGHT]
        knee_sep = abs(float(pose.world(left["knee"])[0] - pose.world(right["knee"])[0]))
        hip_sep = abs(float(pose.world(left["hip"])[0] - pose.world(right["hip"])[0]))
        if hip_sep < _EPS:
            return math.nan
        return knee_sep / hip_sep

    def _hip_height_norm(self, pose: PoseFrame) -> float:
        """Hip height above the ankles as a fraction of the lifter's own leg length."""
        proportions = self._proportions
        if proportions is None or proportions.leg_length_m < _EPS:
            return math.nan
        left, right = SIDE_LANDMARKS[Side.LEFT], SIDE_LANDMARKS[Side.RIGHT]
        hip_y = 0.5 * (pose.world(left["hip"])[1] + pose.world(right["hip"])[1])
        ankle_y = 0.5 * (pose.world(left["ankle"])[1] + pose.world(right["ankle"])[1])
        return float(hip_y - ankle_y) / proportions.leg_length_m


def to_feature_vector(frame: KinematicFrame) -> np.ndarray:
    """Flatten a frame into ``FEATURE_ORDER``, in native units. May contain ``nan``."""
    near, far = frame.camera_near, frame.camera_far
    return np.array(
        [
            near.hip_flexion, near.knee_flexion, near.ankle_dorsiflexion, near.back_to_vertical,
            far.hip_flexion, far.knee_flexion, far.ankle_dorsiflexion, far.back_to_vertical,
            frame.global_metrics.knee_to_hip_width_ratio,
        ],
        dtype=np.float64,
    )


def feature_weights(config: KinematicsConfig) -> dict[str, float]:
    """Per-feature multipliers for the distance metric, keyed by feature name."""
    far = config.camera_far_weight
    scale = config.width_ratio_scale
    return {
        "camera_near.hip_flexion": 1.0,
        "camera_near.knee_flexion": 1.0,
        "camera_near.ankle_dorsiflexion": 1.0,
        "camera_near.back_to_vertical": 1.0,
        "camera_far.hip_flexion": far,
        "camera_far.knee_flexion": far,
        "camera_far.ankle_dorsiflexion": far,
        "camera_far.back_to_vertical": far,
        "global.knee_to_hip_width_ratio": scale,
    }


def feature_weight_vector(config: KinematicsConfig) -> np.ndarray:
    """``feature_weights`` flattened into ``FEATURE_ORDER`` for vectorized maths."""
    weights = feature_weights(config)
    return np.array([weights[name] for name in FEATURE_ORDER], dtype=np.float64)
