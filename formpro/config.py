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
class AppConfig:
    exercise: str = "barbell_back_squat"
    camera: CameraConfig = CameraConfig()
    pose: PoseConfig = PoseConfig()

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
