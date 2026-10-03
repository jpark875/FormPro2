"""Typed configuration loaded from configs/*.yaml. Unknown keys are rejected."""

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
    #: Pace file sources at their recorded frame rate (cameras are already rate-limited).
    pace_file_playback: bool = True


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
    #: Z attenuation for angles: 1.0 is true 3D, 0.0 is image-plane. Lengths ignore this.
    z_weight: float = 0.6

    side_ema_alpha: float = 0.15
    side_hysteresis_m: float = 0.02

    calibration_window_frames: int = 150
    calibration_min_frames: int = 45

    #: Distance-metric weight for the partially occluded camera-far side.
    camera_far_weight: float = 0.35

    #: A width-ratio deviation of `delta` counts as `degrees` of joint-angle deviation.
    width_ratio_equivalent_degrees: float = 15.0
    width_ratio_equivalent_delta: float = 0.1

    @property
    def width_ratio_scale(self) -> float:
        """Degrees of angle deviation per unit of width-ratio deviation."""
        if self.width_ratio_equivalent_delta <= 0:
            raise ValueError("width_ratio_equivalent_delta must be positive")
        return self.width_ratio_equivalent_degrees / self.width_ratio_equivalent_delta


@dataclass(frozen=True)
class PhaseConfig:
    velocity_window_ms: int = 150
    #: Gaps longer than this reset the velocity fit.
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
    #: Warn when the corpus spans less than this in femur_to_torso_ratio.
    min_ratio_span: float = 0.15

    def resolved_root(self, project_root: Path = PROJECT_ROOT) -> Path:
        path = Path(self.root)
        return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class AnalyzerConfig:
    #: Percentiles of optimal-form reference frames that define the acceptable band.
    band_low_percentile: float = 5.0
    band_high_percentile: float = 95.0
    #: Fewer reference frames than this yields no band, and no finding.
    min_band_samples: int = 8

    #: Degrees added to each side of a band to absorb pose-estimator jitter.
    noise_allowance_deg: float = 2.0

    #: Frames a deviation must persist to surface, and to linger once cleared.
    finding_hold_frames: int = 4
    finding_decay_frames: int = 10

    #: Sakoe-Chiba band as a fraction of sequence length.
    dtw_band_ratio: float = 0.2
    #: Minimum margin between best and runner-up DTW label.
    min_confidence_margin: float = 0.12


@dataclass(frozen=True)
class AppConfig:
    exercise: str = "barbell_back_squat"
    camera: CameraConfig = CameraConfig()
    pose: PoseConfig = PoseConfig()
    kinematics: KinematicsConfig = KinematicsConfig()
    phases: PhaseConfig = PhaseConfig()
    dataset: DatasetConfig = DatasetConfig()
    analyzer: AnalyzerConfig = AnalyzerConfig()

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
    """Recursively build a dataclass from a mapping, rejecting unknown keys."""
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
        # field.type is a string under PEP 563; use the default instance's type instead.
        default = getattr(cls, name, None)
        if isinstance(value, Mapping) and is_dataclass(default):
            kwargs[name] = _from_mapping(type(default), value, context)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]
