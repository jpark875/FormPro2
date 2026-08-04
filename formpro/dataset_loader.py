"""Phase 4: reference dataset ingestion.

Loads a directory of labelled reference sequences into ``KinematicFrame`` objects — the
same type ``kinematics.py`` produces for live data. That identity is the point: Phase 5
compares live against reference without a translation layer sitting between them, where
a units mismatch or a field rename could hide indefinitely.

Validation is strict and loud
-----------------------------
Every field is checked on load and a bad file raises rather than being skipped with a
warning. A silently dropped reference file does not break anything visibly; it just
quietly removes an anchor point from the corpus, and the tolerance band Phase 5
interpolates then narrows around whichever builds happened to survive. Failing at load
is the only way that surfaces.

Corpus, not file
----------------
Phase 5 needs the acceptable band for a lifter of a particular build, which requires
reference subjects at several ``femur_to_torso_ratio`` values to interpolate between.
``ReferenceCorpus`` therefore indexes sequences by ratio and exposes bracketing lookup,
so a live lifter at 1.05 is placed between the 0.95 and 1.12 references rather than
snapped to whichever is closest.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DatasetConfig
from .kinematics import (
    FEATURE_ORDER,
    GlobalMetrics,
    KinematicFrame,
    SideAngles,
    to_feature_vector,
)
from .schema import FormLabel, Phase

log = logging.getLogger(__name__)

_ANGLE_GROUPS = ("camera_near", "camera_far", "global")
_REQUIRED_METADATA = ("exercise", "camera_angle", "dataset_type")
_REQUIRED_PROPORTIONS = ("femur_to_torso_ratio", "tibia_to_femur_ratio")


class DatasetError(ValueError):
    """A reference file is malformed or inconsistent with the configured contract."""

    def __init__(self, path: Path | str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = str(path)


@dataclass(frozen=True)
class ReferenceMetadata:
    exercise: str
    camera_angle: str
    dataset_type: str
    fps_target: float | None = None


@dataclass(frozen=True)
class SubjectProportions:
    """The reference subject's build. Not the live lifter's."""

    femur_to_torso_ratio: float
    tibia_to_femur_ratio: float


@dataclass(frozen=True)
class ReferenceSequence:
    """One labelled reference file, parsed and vectorized."""

    path: Path
    metadata: ReferenceMetadata
    proportions: SubjectProportions
    frames: tuple[KinematicFrame, ...]
    features: np.ndarray      # (N, len(FEATURE_ORDER)) float64, native units
    timestamps_ms: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def femur_to_torso_ratio(self) -> float:
        return self.proportions.femur_to_torso_ratio

    @property
    def phases(self) -> tuple[Phase, ...]:
        return tuple(f.phase for f in self.frames)  # type: ignore[misc]

    @property
    def labels(self) -> tuple[FormLabel, ...]:
        return tuple(f.form_label for f in self.frames)  # type: ignore[misc]

    @property
    def duration_ms(self) -> int:
        if not len(self.timestamps_ms):
            return 0
        return int(self.timestamps_ms[-1] - self.timestamps_ms[0])

    def slice_phase(self, phase: Phase) -> np.ndarray:
        """Indices of frames in the given phase.

        Phase 5 aligns like against like — a live concentric against reference
        concentric frames — rather than warping a whole rep against a whole rep, which
        would let a slow descent absorb a fast, broken ascent.
        """
        return np.array([i for i, f in enumerate(self.frames) if f.phase is phase], dtype=int)

    def slice_label(self, label: FormLabel) -> np.ndarray:
        return np.array(
            [i for i, f in enumerate(self.frames) if f.form_label is label], dtype=int
        )

    def label_transitions(self) -> tuple[tuple[int, FormLabel], ...]:
        """Frame indices where the label changes, with the label it changes to.

        A good-morning file is expected to read ``optimal_form`` through the eccentric
        and switch at the concentric; this exposes that boundary directly.
        """
        out: list[tuple[int, FormLabel]] = []
        previous: FormLabel | None = None
        for i, frame in enumerate(self.frames):
            if frame.form_label is not previous:
                out.append((i, frame.form_label))  # type: ignore[arg-type]
                previous = frame.form_label
        return tuple(out)


@dataclass(frozen=True)
class ReferenceCorpus:
    """Every reference sequence available, indexed for interpolation by build."""

    sequences: tuple[ReferenceSequence, ...]

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator[ReferenceSequence]:
        return iter(self.sequences)

    @property
    def ratios(self) -> tuple[float, ...]:
        return tuple(s.femur_to_torso_ratio for s in self.sequences)

    @property
    def ratio_span(self) -> tuple[float, float]:
        ratios = self.ratios
        return (min(ratios), max(ratios)) if ratios else (math.nan, math.nan)

    def with_label(self, label: FormLabel) -> tuple[ReferenceSequence, ...]:
        """Sequences containing at least one frame of this label."""
        return tuple(s for s in self.sequences if label in s.labels)

    def with_dataset_type(self, dataset_type: str) -> tuple[ReferenceSequence, ...]:
        return tuple(s for s in self.sequences if s.metadata.dataset_type == dataset_type)

    def bracketing(
        self, femur_to_torso_ratio: float
    ) -> tuple[ReferenceSequence | None, ReferenceSequence | None]:
        """The nearest reference below and above a live lifter's ratio.

        Either side may be ``None`` when the lifter falls outside the corpus. Phase 5
        must treat a one-sided bracket as extrapolation and widen its tolerance
        accordingly, rather than trusting a band anchored on one side only.
        """
        if not self.sequences or math.isnan(femur_to_torso_ratio):
            return None, None
        ordered = sorted(self.sequences, key=lambda s: s.femur_to_torso_ratio)
        below = [s for s in ordered if s.femur_to_torso_ratio <= femur_to_torso_ratio]
        above = [s for s in ordered if s.femur_to_torso_ratio >= femur_to_torso_ratio]
        return (below[-1] if below else None, above[0] if above else None)

    def coverage_report(self) -> str:
        """Human-readable summary for the loader CLI."""
        if not self.sequences:
            return "corpus is empty"
        low, high = self.ratio_span
        labels = sorted({label.value for s in self.sequences for label in set(s.labels)})
        return (
            f"{len(self.sequences)} sequences, "
            f"{sum(len(s) for s in self.sequences)} frames, "
            f"femur_to_torso_ratio {low:.2f}-{high:.2f}, "
            f"labels: {', '.join(labels)}"
        )


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def load_sequence(path: Path | str, config: DatasetConfig) -> ReferenceSequence:
    """Parse and validate one reference JSON file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(path, f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(path, "expected a JSON object at the top level")

    metadata = _parse_metadata(path, raw.get("metadata"), config)
    proportions = _parse_proportions(path, raw.get("subject_proportions"))
    frames = _parse_frames(path, raw.get("frames"), config)

    features = np.vstack([to_feature_vector(f) for f in frames])
    timestamps = np.array([f.timestamp_ms for f in frames], dtype=np.int64)

    if np.isnan(features).any():
        bad = sorted({FEATURE_ORDER[c] for c in np.unique(np.argwhere(np.isnan(features))[:, 1])})
        raise DatasetError(path, f"reference data may not contain nulls; affected: {bad}")

    return ReferenceSequence(
        path=path,
        metadata=metadata,
        proportions=proportions,
        frames=frames,
        features=features,
        timestamps_ms=timestamps,
    )


def load_corpus(
    root: Path | str | None = None,
    config: DatasetConfig | None = None,
    pattern: str = "*.json",
) -> ReferenceCorpus:
    """Load every reference file under ``root``.

    An empty directory raises. A reference-driven analyzer with no references would
    otherwise start up looking healthy and pass every rep.
    """
    config = config or DatasetConfig()
    root = Path(root) if root is not None else config.resolved_root()
    if not root.is_dir():
        raise DatasetError(root, "reference directory does not exist")

    paths = sorted(p for p in root.glob(pattern) if p.is_file())
    if not paths:
        raise DatasetError(root, f"no files matching {pattern!r}")

    sequences = tuple(load_sequence(p, config) for p in paths)
    corpus = ReferenceCorpus(sequences)
    log.info("loaded corpus from %s: %s", root, corpus.coverage_report())

    low, high = corpus.ratio_span
    if len(sequences) > 1 and (high - low) < config.min_ratio_span:
        log.warning(
            "corpus spans only %.3f in femur_to_torso_ratio (%.2f-%.2f); the "
            "build-adjusted tolerance band has too few anchors to interpolate between "
            "and will behave as a fixed threshold",
            high - low, low, high,
        )
    return corpus


def _parse_metadata(path: Path, raw: Any, config: DatasetConfig) -> ReferenceMetadata:
    if not isinstance(raw, dict):
        raise DatasetError(path, "missing or malformed 'metadata'")
    _require_keys(path, raw, _REQUIRED_METADATA, "metadata")

    exercise = str(raw["exercise"])
    if exercise != config.expected_exercise:
        raise DatasetError(
            path, f"exercise is {exercise!r}, expected {config.expected_exercise!r}"
        )

    angle = str(raw["camera_angle"])
    if angle in config.legacy_camera_angles:
        log.warning(
            "%s: camera_angle %r is a legacy alias; rewrite as %r",
            path.name, angle, config.accepted_camera_angles[0],
        )
    elif angle not in config.accepted_camera_angles:
        # Comparing against a differently-framed recording is worse than having no
        # reference at all, because the mismatch presents as consistent form error.
        raise DatasetError(
            path,
            f"camera_angle is {angle!r}, expected one of "
            f"{list(config.accepted_camera_angles)}",
        )

    fps = raw.get("fps_target")
    return ReferenceMetadata(
        exercise=exercise,
        camera_angle=angle,
        dataset_type=str(raw["dataset_type"]),
        fps_target=None if fps is None else float(fps),
    )


def _parse_proportions(path: Path, raw: Any) -> SubjectProportions:
    if not isinstance(raw, dict):
        raise DatasetError(path, "missing or malformed 'subject_proportions'")
    _require_keys(path, raw, _REQUIRED_PROPORTIONS, "subject_proportions")
    values = {}
    for key in _REQUIRED_PROPORTIONS:
        value = _as_float(path, raw[key], f"subject_proportions.{key}")
        if not (0.0 < value < 10.0):
            raise DatasetError(path, f"subject_proportions.{key} is implausible: {value}")
        values[key] = value
    return SubjectProportions(**values)


def _parse_frames(path: Path, raw: Any, config: DatasetConfig) -> tuple[KinematicFrame, ...]:
    if not isinstance(raw, list) or not raw:
        raise DatasetError(path, "'frames' must be a non-empty list")

    frames: list[KinematicFrame] = []
    previous_ts: int | None = None

    for position, item in enumerate(raw):
        where = f"frames[{position}]"
        if not isinstance(item, dict):
            raise DatasetError(path, f"{where} is not an object")
        _require_keys(path, item, ("frame_id", "timestamp_ms", "phase", "angles"), where)

        timestamp = int(_as_float(path, item["timestamp_ms"], f"{where}.timestamp_ms"))
        if previous_ts is not None:
            if timestamp <= previous_ts:
                raise DatasetError(
                    path,
                    f"{where}.timestamp_ms is {timestamp}, not greater than the previous "
                    f"{previous_ts}; velocity is undefined on non-monotonic time",
                )
            gap = timestamp - previous_ts
            if gap > config.max_timestamp_gap_ms:
                raise DatasetError(
                    path,
                    f"{where}: {gap} ms gap exceeds max_timestamp_gap_ms "
                    f"({config.max_timestamp_gap_ms}); a rep cannot be reconstructed "
                    f"across a dropout this long",
                )
        previous_ts = timestamp

        angles = item["angles"]
        if not isinstance(angles, dict):
            raise DatasetError(path, f"{where}.angles is not an object")
        _require_keys(path, angles, _ANGLE_GROUPS, f"{where}.angles")

        frames.append(
            KinematicFrame(
                frame_id=int(_as_float(path, item["frame_id"], f"{where}.frame_id")),
                timestamp_ms=timestamp,
                camera_near=_parse_side(path, angles["camera_near"], f"{where}.camera_near"),
                camera_far=_parse_side(path, angles["camera_far"], f"{where}.camera_far"),
                global_metrics=_parse_global(path, angles["global"], f"{where}.global"),
                phase=_parse_enum(path, Phase, item["phase"], f"{where}.phase"),
                form_label=_parse_enum(
                    path, FormLabel, item.get("form_label"), f"{where}.form_label"
                ),
            )
        )
    return tuple(frames)


def _parse_side(path: Path, raw: Any, where: str) -> SideAngles:
    if not isinstance(raw, dict):
        raise DatasetError(path, f"{where} is not an object")
    from .kinematics import SIDE_ANGLE_FIELDS

    _require_keys(path, raw, SIDE_ANGLE_FIELDS, where)
    values = {}
    for key in SIDE_ANGLE_FIELDS:
        value = _as_float(path, raw[key], f"{where}.{key}")
        if not (0.0 <= value <= 180.0):
            raise DatasetError(
                path, f"{where}.{key} is {value}, outside the valid 0-180 degree range"
            )
        values[key] = value
    return SideAngles(**values)


def _parse_global(path: Path, raw: Any, where: str) -> GlobalMetrics:
    if not isinstance(raw, dict):
        raise DatasetError(path, f"{where} is not an object")
    _require_keys(path, raw, ("knee_to_hip_width_ratio",), where)
    ratio = _as_float(path, raw["knee_to_hip_width_ratio"], f"{where}.knee_to_hip_width_ratio")
    if not (0.0 < ratio < 5.0):
        raise DatasetError(path, f"{where}.knee_to_hip_width_ratio is implausible: {ratio}")
    return GlobalMetrics(knee_to_hip_width_ratio=ratio)


def _parse_enum(path: Path, enum_cls: type, value: Any, where: str) -> Any:
    if value is None:
        if enum_cls is FormLabel:
            raise DatasetError(path, f"{where} is required on reference data")
        raise DatasetError(path, f"{where} is required")
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = [member.value for member in enum_cls]
        raise DatasetError(path, f"{where} is {value!r}; expected one of {allowed}") from exc


def _require_keys(path: Path, raw: dict, keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise DatasetError(path, f"{where} is missing required key(s): {missing}")


def _as_float(path: Path, value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(path, f"{where} must be a number, got {type(value).__name__}")
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        raise DatasetError(path, f"{where} is not finite")
    return result


def summarize(sequences: Sequence[ReferenceSequence]) -> str:
    lines = []
    for seq in sequences:
        phases = {p.value for p in seq.phases}
        lines.append(
            f"  {seq.path.name}: {len(seq)} frames, {seq.duration_ms} ms, "
            f"ratio {seq.femur_to_torso_ratio:.2f}, type {seq.metadata.dataset_type}, "
            f"phases {sorted(phases)}"
        )
    return "\n".join(lines)
