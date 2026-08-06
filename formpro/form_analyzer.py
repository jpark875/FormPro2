"""Phase 5: real-time comparison against the reference corpus.

This module is an inference engine over Phase 4's evidence. It contains no baseline
thresholds and no fallback constants: every bound it applies is computed at runtime from
reference frames labelled ``optimal_form``, then adjusted to the live lifter's build.
If the corpus is empty the analyzer refuses to start rather than degrading to guesses,
because an analyzer that silently falls back to hardcoded numbers looks identical to one
that is working and passes every rep.

How a bound is produced
-----------------------
1. Every reference frame labelled ``optimal_form`` is pooled by subject build, giving one
   ``ThresholdProfile`` per distinct ``femur_to_torso_ratio`` in the corpus. Error files
   contribute too: the clean eccentric of a good-morning rep is still optimal evidence.
2. Within a profile, frames are grouped by phase, and each feature gets a percentile band.
   Bands are per phase because the acceptable knee angle at the bottom has nothing to do
   with the acceptable knee angle during setup.
3. At evaluation time the live lifter's ratio is passed to ``ReferenceCorpus.bracketing``.
   Between two profiles the bands are blended by distance. Outside the corpus they are
   projected along the trend of the two nearest profiles, with a warning, because a lifter
   with an unusual build is exactly the case where a corpus-average bound would be wrong.

Thresholds are evidence. Interpretation is domain knowledge
-----------------------------------------------------------
``ERROR_SIGNATURES`` maps a feature deviating in a particular direction during a
particular phase onto a named error. That table carries no numbers — it says *which*
error a deviation means, never *how far* is too far. The magnitudes come entirely from
the corpus. Keeping the two separate is what allows the bounds to stay fully dynamic
while the messages stay specific enough to act on.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import AnalyzerConfig, KinematicsConfig
from .dataset_loader import ReferenceCorpus, ReferenceSequence
from .kinematics import (
    FEATURE_ORDER,
    KinematicFrame,
    feature_weights,
    to_feature_vector,
)
from .phases import VelocityTracker
from .schema import FormLabel, Phase

log = logging.getLogger(__name__)

_EPS = 1e-9


class AnalyzerError(RuntimeError):
    """The analyzer cannot operate on the corpus it was given."""


# ---------------------------------------------------------------------------
# interpretation table (no magnitudes here, only meanings)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorSignature:
    feature: str
    #: -1 when falling below the corpus band is the fault, +1 when exceeding it is.
    direction: int
    phases: tuple[Phase, ...]
    label: FormLabel
    message: str
    #: Additionally require the feature to be increasing, for errors defined by a rate.
    requires_rising: bool = False


ERROR_SIGNATURES: tuple[ErrorSignature, ...] = (
    # Knees tracking inward relative to the hips. The ratio is the only frontal-plane
    # feature, and it is meaningful throughout the loaded portion of the lift.
    ErrorSignature(
        "global.knee_to_hip_width_ratio", -1,
        (Phase.ECCENTRIC, Phase.BOTTOM, Phase.CONCENTRIC),
        FormLabel.KNEE_VALGUS, "KNEE VALGUS DETECTED",
    ),
    # Hips outrunning the shoulders. The schema carries no shoulder height, but it does
    # not need to: if the hips rise faster than the shoulders the torso necessarily
    # becomes more horizontal, so a back angle that is both above the corpus band and
    # still opening during the ascent is exactly that failure.
    ErrorSignature(
        "camera_near.back_to_vertical", +1, (Phase.CONCENTRIC,),
        FormLabel.GOOD_MORNING, "HIPS RISING TOO FAST", requires_rising=True,
    ),
    ErrorSignature(
        "camera_near.back_to_vertical", +1, (Phase.ECCENTRIC, Phase.BOTTOM),
        FormLabel.GOOD_MORNING, "EXCESSIVE FORWARD LEAN",
    ),
    # Not enough flexion at the bottom: the hip crease never got below the knee.
    ErrorSignature(
        "camera_near.knee_flexion", +1, (Phase.BOTTOM,),
        FormLabel.HIGH_SQUAT, "SQUAT DEEPER",
    ),
    ErrorSignature(
        "camera_near.hip_flexion", +1, (Phase.BOTTOM,),
        FormLabel.HIGH_SQUAT, "HIP CREASE ABOVE KNEE",
    ),
    # Heel lift reads as an unexpectedly large ankle angle: as the heel rises, the
    # heel-to-toe axis tilts away from the shin, opening the measured angle, while normal
    # dorsiflexion during a descent closes it.
    ErrorSignature(
        "camera_near.ankle_dorsiflexion", +1, (Phase.BOTTOM, Phase.CONCENTRIC),
        FormLabel.HEEL_LIFT, "HEELS LIFTING - WEIGHT FORWARD",
    ),
)


# ---------------------------------------------------------------------------
# corpus-derived bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBand:
    lower: float
    upper: float
    mean: float
    samples: int

    def deviation(self, value: float) -> float:
        """Signed distance outside the band; 0 inside it."""
        if math.isnan(value):
            return math.nan
        if value < self.lower:
            return value - self.lower
        if value > self.upper:
            return value - self.upper
        return 0.0

    def widened(self, allowance: float) -> FeatureBand:
        return FeatureBand(self.lower - allowance, self.upper + allowance,
                           self.mean, self.samples)


@dataclass(frozen=True)
class ThresholdProfile:
    """Acceptable bands for one subject build, keyed by (phase, feature)."""

    femur_to_torso_ratio: float
    bands: dict[tuple[Phase, str], FeatureBand]

    def band(self, phase: Phase, feature: str) -> FeatureBand | None:
        return self.bands.get((phase, feature))


@dataclass(frozen=True)
class BandSource:
    """Where the bounds in play came from, so the UI can be honest about confidence."""

    mode: str  # "interpolated" | "extrapolated" | "single_profile"
    lower_ratio: float
    upper_ratio: float
    blend: float

    @property
    def extrapolated(self) -> bool:
        return self.mode != "interpolated"

    def describe(self) -> str:
        if self.mode == "interpolated":
            return f"interpolated {self.lower_ratio:.2f}-{self.upper_ratio:.2f}"
        if self.mode == "single_profile":
            return f"single profile {self.lower_ratio:.2f}"
        return f"EXTRAPOLATED from {self.lower_ratio:.2f}-{self.upper_ratio:.2f}"


def build_profiles(corpus: ReferenceCorpus, config: AnalyzerConfig) -> list[ThresholdProfile]:
    """One profile per distinct subject build in the corpus.

    Only ``optimal_form`` frames contribute. Frames labelled with an error describe what
    the lifter must not do, so folding them into the acceptable band would widen it to
    admit the very thing being detected.
    """
    pooled: dict[float, list[KinematicFrame]] = {}
    for sequence in corpus:
        ratio = round(sequence.femur_to_torso_ratio, 4)
        for frame in sequence.frames:
            if frame.form_label is FormLabel.OPTIMAL:
                pooled.setdefault(ratio, []).append(frame)

    profiles: list[ThresholdProfile] = []
    for ratio, frames in sorted(pooled.items()):
        bands: dict[tuple[Phase, str], FeatureBand] = {}
        by_phase: dict[Phase, list[KinematicFrame]] = {}
        for frame in frames:
            if frame.phase is not None:
                by_phase.setdefault(frame.phase, []).append(frame)

        for phase, phase_frames in by_phase.items():
            if len(phase_frames) < config.min_band_samples:
                continue
            matrix = np.vstack([to_feature_vector(f) for f in phase_frames])
            for index, feature in enumerate(FEATURE_ORDER):
                column = matrix[:, index]
                column = column[~np.isnan(column)]
                if len(column) < config.min_band_samples:
                    continue
                bands[(phase, feature)] = FeatureBand(
                    lower=float(np.percentile(column, config.band_low_percentile)),
                    upper=float(np.percentile(column, config.band_high_percentile)),
                    mean=float(column.mean()),
                    samples=len(column),
                )
        if bands:
            profiles.append(ThresholdProfile(ratio, bands))

    if not profiles:
        raise AnalyzerError(
            "no usable threshold profiles: the corpus has no optimal_form frames with at "
            f"least {config.min_band_samples} samples per phase. The analyzer derives "
            "every bound from this evidence and has no baseline to fall back on."
        )
    return profiles


class ThresholdModel:
    """Resolves corpus profiles into bounds for one lifter's build."""

    def __init__(self, profiles: Sequence[ThresholdProfile]) -> None:
        if not profiles:
            raise AnalyzerError("at least one threshold profile is required")
        self.profiles = sorted(profiles, key=lambda p: p.femur_to_torso_ratio)
        self._warned: set[float] = set()

    @property
    def ratios(self) -> tuple[float, ...]:
        return tuple(p.femur_to_torso_ratio for p in self.profiles)

    def resolve(self, femur_to_torso_ratio: float) -> tuple[ThresholdProfile, BandSource]:
        """Bands for this build, by interpolation inside the corpus or projection outside."""
        if math.isnan(femur_to_torso_ratio):
            raise AnalyzerError("cannot resolve thresholds before calibration completes")

        if len(self.profiles) == 1:
            only = self.profiles[0]
            self._warn_once(
                femur_to_torso_ratio,
                "[Extrapolation Warning] corpus holds a single build "
                f"({only.femur_to_torso_ratio:.2f}); lifter is {femur_to_torso_ratio:.2f}. "
                "Bounds cannot be adjusted to build until the corpus spans a range.",
            )
            return only, BandSource("single_profile", only.femur_to_torso_ratio,
                                    only.femur_to_torso_ratio, 0.0)

        lower, upper, blend, mode = self._bracket(femur_to_torso_ratio)
        if mode != "interpolated":
            self._warn_once(
                femur_to_torso_ratio,
                f"[Extrapolation Warning] lifter ratio {femur_to_torso_ratio:.2f} lies "
                f"outside the corpus span {self.ratios[0]:.2f}-{self.ratios[-1]:.2f}. "
                f"Projecting the trend from {lower.femur_to_torso_ratio:.2f} and "
                f"{upper.femur_to_torso_ratio:.2f}; treat findings as lower confidence.",
            )

        return (
            ThresholdProfile(femur_to_torso_ratio, _blend_bands(lower, upper, blend)),
            BandSource(mode, lower.femur_to_torso_ratio, upper.femur_to_torso_ratio, blend),
        )

    def _bracket(
        self, ratio: float
    ) -> tuple[ThresholdProfile, ThresholdProfile, float, str]:
        below = [p for p in self.profiles if p.femur_to_torso_ratio <= ratio]
        above = [p for p in self.profiles if p.femur_to_torso_ratio >= ratio]

        if below and above:
            lower, upper = below[-1], above[0]
            mode = "interpolated"
        elif above:
            # Below the corpus: project down the trend of the two lowest profiles.
            lower, upper, mode = self.profiles[0], self.profiles[1], "extrapolated"
        else:
            lower, upper, mode = self.profiles[-2], self.profiles[-1], "extrapolated"

        span = upper.femur_to_torso_ratio - lower.femur_to_torso_ratio
        # blend > 1 or < 0 is what turns the same lerp into a linear projection.
        blend = 0.0 if abs(span) < _EPS else (ratio - lower.femur_to_torso_ratio) / span
        return lower, upper, blend, mode

    def _warn_once(self, ratio: float, message: str) -> None:
        key = round(ratio, 2)
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message)


def _blend_bands(
    lower: ThresholdProfile, upper: ThresholdProfile, t: float
) -> dict[tuple[Phase, str], FeatureBand]:
    """Blend two profiles at position ``t``.

    ``t`` inside [0, 1] interpolates; outside it the same expression extrapolates along
    the line through the two profiles, which is what keeps an unusual build on real
    corpus evidence rather than a fallback constant.
    """
    out: dict[tuple[Phase, str], FeatureBand] = {}
    for key, low_band in lower.bands.items():
        high_band = upper.bands.get(key)
        if high_band is None:
            out[key] = low_band
            continue
        blended_lower = low_band.lower + t * (high_band.lower - low_band.lower)
        blended_upper = low_band.upper + t * (high_band.upper - low_band.upper)
        if blended_upper < blended_lower:
            # A steep projection can invert the band; keep it degenerate but ordered.
            blended_lower, blended_upper = blended_upper, blended_lower
        out[key] = FeatureBand(
            lower=blended_lower,
            upper=blended_upper,
            mean=low_band.mean + t * (high_band.mean - low_band.mean),
            samples=min(low_band.samples, high_band.samples),
        )
    for key, high_band in upper.bands.items():
        out.setdefault(key, high_band)
    return out


# ---------------------------------------------------------------------------
# dynamic time warping
# ---------------------------------------------------------------------------


def dtw_distance(
    a: np.ndarray, b: np.ndarray, weights: np.ndarray, band_ratio: float = 0.2
) -> float:
    """Weighted DTW between two feature sequences, normalized by path length.

    A Sakoe-Chiba band bounds how far the alignment may warp. Without it a slow live
    descent could align against a fast reference ascent and report a small distance
    between two quite different movements.

    ``nan`` features are skipped per-pair rather than poisoning the whole cell, since the
    camera-far side is legitimately unmeasured for stretches of a rep.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return math.inf

    band = max(abs(n - m) + 1, int(round(band_ratio * max(n, m))))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            delta = a[i - 1] - b[j - 1]
            usable = ~np.isnan(delta)
            if not usable.any():
                local = 0.0
            else:
                scaled = delta[usable] * weights[usable]
                local = float(np.sqrt((scaled ** 2).sum() / usable.sum()))
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    total = cost[n, m]
    return math.inf if not np.isfinite(total) else float(total) / (n + m)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    label: FormLabel
    message: str
    feature: str
    phase: Phase
    observed: float
    band: FeatureBand
    #: Deviation past the band, expressed in degree-equivalent units so a valgus finding
    #: and a back-angle finding can be ranked against each other.
    severity: float

    def detail(self) -> str:
        return (
            f"{self.feature.split('.')[-1]} {self.observed:.1f} "
            f"vs {self.band.lower:.1f}-{self.band.upper:.1f}"
        )


@dataclass
class AnalysisResult:
    phase: Phase
    findings: tuple[Finding, ...] = ()
    band_source: BandSource | None = None
    #: DTW verdict for the most recently completed phase segment, when confident.
    segment_label: FormLabel | None = None
    segment_confidence: float = 0.0

    @property
    def label(self) -> FormLabel:
        """Worst active finding, or optimal when nothing is firing."""
        if not self.findings:
            return FormLabel.OPTIMAL
        return max(self.findings, key=lambda f: f.severity).label

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass
class _Pending:
    """Debounce state for one error label."""

    hits: int = 0
    misses: int = 0
    active: bool = False
    finding: Finding | None = None


# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------


class FormAnalyzer:
    """Compares live normalized frames against corpus-derived bounds."""

    def __init__(
        self,
        corpus: ReferenceCorpus,
        analyzer_config: AnalyzerConfig,
        kinematics_config: KinematicsConfig,
    ) -> None:
        if len(corpus) == 0:
            raise AnalyzerError(
                "the reference corpus is empty. Every bound this analyzer applies is "
                "derived from reference data; it has no baseline to fall back on."
            )
        self.config = analyzer_config
        self.corpus = corpus
        self.weights = feature_weights(kinematics_config)
        self.weight_vector = np.array(
            [self.weights[name] for name in FEATURE_ORDER], dtype=np.float64
        )
        self.model = ThresholdModel(build_profiles(corpus, analyzer_config))

        self._profile: ThresholdProfile | None = None
        self._source: BandSource | None = None
        self._resolved_for: float = math.nan

        self._pending: dict[FormLabel, _Pending] = {}
        self._lean = VelocityTracker(window_ms=200, max_gap_ms=400)
        self._segment: list[KinematicFrame] = []
        self._segment_phase: Phase | None = None
        self._segment_label: FormLabel | None = None
        self._segment_confidence = 0.0

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        self._pending.clear()
        self._lean.reset()
        self._segment.clear()
        self._segment_phase = None
        self._segment_label = None
        self._segment_confidence = 0.0

    @property
    def band_source(self) -> BandSource | None:
        return self._source

    # -- main entry point ------------------------------------------------------

    def update(self, frame: KinematicFrame, femur_to_torso_ratio: float) -> AnalysisResult:
        """Evaluate one live frame. ``frame.phase`` must already be set."""
        phase = frame.phase or Phase.SETUP
        self._ensure_profile(femur_to_torso_ratio)
        self._accumulate_segment(frame, phase)

        rising = self._lean.update(frame.timestamp_ms, frame.camera_near.back_to_vertical)
        observed = self._evaluate(frame, phase, rising)
        findings = self._debounce(observed)

        return AnalysisResult(
            phase=phase,
            findings=findings,
            band_source=self._source,
            segment_label=self._segment_label,
            segment_confidence=self._segment_confidence,
        )

    # -- bounds ----------------------------------------------------------------

    def _ensure_profile(self, ratio: float) -> None:
        if math.isnan(ratio):
            self._profile, self._source = None, None
            return
        # Proportions are frozen once calibrated, so this resolves once per session.
        if self._profile is None or abs(ratio - self._resolved_for) > 1e-6:
            self._profile, self._source = self.model.resolve(ratio)
            self._resolved_for = ratio
            log.info(
                "thresholds resolved for femur_to_torso_ratio %.3f (%s)",
                ratio, self._source.describe(),
            )

    def _evaluate(
        self, frame: KinematicFrame, phase: Phase, rising: float
    ) -> dict[FormLabel, Finding]:
        if self._profile is None:
            return {}

        values = dict(zip(FEATURE_ORDER, to_feature_vector(frame)))
        allowance = self.config.noise_allowance_deg
        worst: dict[FormLabel, Finding] = {}

        for signature in ERROR_SIGNATURES:
            if phase not in signature.phases:
                continue
            if signature.requires_rising and not (rising > 0):
                continue

            value = values[signature.feature]
            if math.isnan(value):
                # Occluded joint: withhold the finding rather than infer one from a
                # coordinate the model guessed.
                continue

            band = self._profile.band(phase, signature.feature)
            if band is None:
                continue

            weight = self.weights[signature.feature]
            widened = band.widened(allowance / weight if weight > _EPS else allowance)
            deviation = widened.deviation(value)
            if math.isnan(deviation) or deviation == 0.0:
                continue
            if (deviation > 0) != (signature.direction > 0):
                continue

            severity = abs(deviation) * weight
            existing = worst.get(signature.label)
            if existing is None or severity > existing.severity:
                worst[signature.label] = Finding(
                    label=signature.label,
                    message=signature.message,
                    feature=signature.feature,
                    phase=phase,
                    observed=value,
                    band=widened,
                    severity=severity,
                )
        return worst

    def _debounce(self, observed: dict[FormLabel, Finding]) -> tuple[Finding, ...]:
        for label in set(self._pending) | set(observed):
            state = self._pending.setdefault(label, _Pending())
            if label in observed:
                state.hits += 1
                state.misses = 0
                state.finding = observed[label]
                if state.hits >= self.config.finding_hold_frames:
                    state.active = True
            else:
                state.misses += 1
                state.hits = 0
                if state.misses >= self.config.finding_decay_frames:
                    state.active = False
                    state.finding = None

        active = [s.finding for s in self._pending.values() if s.active and s.finding]
        return tuple(sorted(active, key=lambda f: f.severity, reverse=True))

    # -- sequence classification -----------------------------------------------

    def _accumulate_segment(self, frame: KinematicFrame, phase: Phase) -> None:
        if phase is not self._segment_phase:
            if self._segment_phase in (Phase.ECCENTRIC, Phase.CONCENTRIC) and self._segment:
                self._classify(self._segment_phase, tuple(self._segment))
            self._segment_phase = phase
            self._segment = []
        if phase in (Phase.ECCENTRIC, Phase.CONCENTRIC):
            self._segment.append(frame)

    def _classify(self, phase: Phase, segment: Sequence[KinematicFrame]) -> None:
        """Nearest-label DTW over reference segments of the same phase.

        A second opinion alongside the band checks, and the one that can catch a
        whole-shape problem that no single frame violates: a rep where every angle stays
        inside its band but the coordination between them is wrong.
        """
        if len(segment) < 4:
            return
        live = np.vstack([to_feature_vector(f) for f in segment])

        best: dict[FormLabel, float] = {}
        for sequence in self.corpus:
            for label, reference in _label_segments(sequence, phase):
                if len(reference) < 4:
                    continue
                distance = dtw_distance(
                    live, reference, self.weight_vector, self.config.dtw_band_ratio
                )
                if distance < best.get(label, math.inf):
                    best[label] = distance

        if len(best) < 2:
            self._segment_label, self._segment_confidence = None, 0.0
            return

        ranked = sorted(best.items(), key=lambda kv: kv[1])
        (top_label, top), (_, runner_up) = ranked[0], ranked[1]
        if not math.isfinite(top) or not math.isfinite(runner_up) or runner_up < _EPS:
            self._segment_label, self._segment_confidence = None, 0.0
            return

        margin = (runner_up - top) / runner_up
        if margin < self.config.min_confidence_margin:
            # Two labels fit about equally well; reporting either would be a coin toss.
            self._segment_label, self._segment_confidence = None, 0.0
        else:
            self._segment_label, self._segment_confidence = top_label, margin


def _label_segments(
    sequence: ReferenceSequence, phase: Phase
) -> list[tuple[FormLabel, np.ndarray]]:
    """Contiguous runs of one label within one phase, as feature matrices."""
    runs: list[tuple[FormLabel, np.ndarray]] = []
    current: list[int] = []
    current_label: FormLabel | None = None

    for index, frame in enumerate(sequence.frames):
        matches = frame.phase is phase
        if matches and frame.form_label is current_label and current:
            current.append(index)
            continue
        if current and current_label is not None:
            runs.append((current_label, sequence.features[current]))
        current = [index] if matches else []
        current_label = frame.form_label if matches else None

    if current and current_label is not None:
        runs.append((current_label, sequence.features[current]))
    return runs
