"""Typed configuration objects loaded from ``configs/*.yaml``.

Config is parsed once at startup into frozen dataclasses. Modules receive the section
they need rather than a dict, so a typo in the YAML fails loudly at load time instead of
surfacing as a ``None`` threshold in the middle of a set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "squat.yaml"

T = TypeVar("T")


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"
    warmup_frames: int = 5
    read_timeout_s: float = 2.0


@dataclass(frozen=True)
class SmoothingConfig:
    enabled: bool = True
    min_cutoff: float = 1.2
    beta: float = 0.35
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class PoseConfig:
    variant: str = "heavy"
    model_dir: str = "models"
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    min_visibility: float = 0.5
    smoothing: SmoothingConfig = SmoothingConfig()

    def model_path(self, root: Path = PROJECT_ROOT) -> Path:
        """Absolute path to the ``.task`` binary for the configured variant."""
        directory = Path(self.model_dir)
        if not directory.is_absolute():
            directory = root / directory
        return directory / f"pose_landmarker_{self.variant}.task"


@dataclass(frozen=True)
class KinematicsConfig:
    #: Depth-axis attenuation applied when measuring angles. 1.0 is a true 3D angle,
    #: 0.0 projects onto the image plane. See the kinematics module docstring for why
    #: neither extreme suits a 45-degree view. Segment *lengths* ignore this.
    z_weight: float = 0.6

    side_ema_alpha: float = 0.15
    side_hysteresis_m: float = 0.02

    calibration_window_frames: int = 150
    calibration_min_frames: int = 45

    #: (femur_to_torso_ratio, max back_to_vertical degrees) anchors, interpolated
    #: linearly. Fallback only, until Phase 5 derives the band from the corpus.
    back_angle_anchors: tuple[tuple[float, float], ...] = ((0.85, 38.0), (1.30, 52.0))

    #: Distance-metric weight for the partially occluded camera-far side.
    camera_far_weight: float = 0.35
    #: Rescales the dimensionless width ratio into the angles' degree range.
    width_ratio_scale_deg: float = 60.0


@dataclass(frozen=True)
class PhaseConfig:
    velocity_window_ms: int = 150
    #: Longer than this between frames and the velocity fit is discarded rather than
    #: interpolated across the gap.
    max_gap_ms: int = 250

    #: Leg-lengths per second. Body-size normalized, so one set of thresholds fits all.
    move_velocity: float = 0.15
    still_velocity: float = 0.06

    #: Hip height as a fraction of leg length: ~1.0 standing, ~0.5 at depth.
    standing_height: float = 0.95
    descended_height: float = 0.90

    min_dwell_frames: int = 3


@dataclass(frozen=True)
class DatasetConfig:
    root: str = "data/reference"
    expected_exercise: str = "barbell_back_squat"
    accepted_camera_angles: tuple[str, ...] = ("45_oblique_anterior",)
    legacy_camera_angles: tuple[str, ...] = ("45_oblique",)
    max_timestamp_gap_ms: int = 250
    #: Warn below this spread in femur_to_torso_ratio across the corpus; a narrow
    #: corpus makes the build-adjusted band nominally dynamic but effectively fixed.
    min_ratio_span: float = 0.15

    def resolved_root(self, project_root: Path = PROJECT_ROOT) -> Path:
        path = Path(self.root)
        return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class AppConfig:
    exercise: str = "barbell_back_squat"
    camera: CameraConfig = CameraConfig()
    pose: PoseConfig = PoseConfig()
    kinematics: KinematicsConfig = KinematicsConfig()
    phases: PhaseConfig = PhaseConfig()
    dataset: DatasetConfig = DatasetConfig()

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: expected a top-level mapping")
        return _from_mapping(cls, raw, context=path.name)


def _from_mapping(cls: type[T], raw: Mapping[str, Any], context: str) -> T:
    """Recursively build a dataclass from a mapping, rejecting unknown keys.

    Silently ignoring an unrecognised key is the failure mode that lets a renamed
    threshold sit dead in the YAML for weeks, so unknown keys are an error.
    """
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {sorted(unknown)} in section '{cls.__name__}'; "
            f"expected any of {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        # `field.type` is a string here (PEP 563), so resolve nested sections via the
        # runtime type of the field's default instance instead of the annotation.
        default = getattr(cls, name, None)
        if isinstance(value, Mapping) and is_dataclass(default):
            kwargs[name] = _from_mapping(type(default), value, context)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]
