"""Phase 5 tests: corpus-derived bounds, build adjustment, and error detection.

The recurring assertion across this file is that nothing is hardcoded. Change the corpus
and the bounds must move with it; empty the corpus and the analyzer must refuse to run
rather than fall back to a constant.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from formpro.config import AnalyzerConfig, DatasetConfig, KinematicsConfig
from formpro.dataset_loader import ReferenceCorpus, load_corpus, load_sequence
from formpro.form_analyzer import (
    ERROR_SIGNATURES,
    AnalyzerError,
    FormAnalyzer,
    ThresholdModel,
    build_profiles,
    dtw_distance,
)
from formpro.kinematics import (
    FEATURE_ORDER,
    GlobalMetrics,
    KinematicFrame,
    SideAngles,
    feature_weight_vector,
)
from formpro.schema import FormLabel, Phase

KIN = KinematicsConfig()
ANALYZER = AnalyzerConfig(finding_hold_frames=2, finding_decay_frames=3)

# A nominal optimal rep, per phase: (hip, knee, ankle, back, width ratio).
OPTIMAL = {
    Phase.SETUP: (178.0, 178.0, 89.0, 3.0, 1.05),
    Phase.ECCENTRIC: (130.0, 130.0, 80.0, 25.0, 1.04),
    Phase.BOTTOM: (75.0, 78.0, 66.0, 34.0, 1.02),
    Phase.CONCENTRIC: (128.0, 128.0, 79.0, 27.0, 1.03),
    Phase.RECOVERY: (176.0, 176.0, 88.0, 5.0, 1.05),
}


def frame_dict(frame_id, timestamp_ms, phase, values, label, jitter=0.0):
    hip, knee, ankle, back, ratio = values
    near = {
        "hip_flexion": hip + jitter,
        "knee_flexion": knee + jitter,
        "ankle_dorsiflexion": ankle + jitter,
        "back_to_vertical": max(0.0, back + jitter),
    }
    far = {k: v + 0.4 for k, v in near.items()}
    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "phase": phase.value,
        "angles": {"camera_near": near, "camera_far": far,
                   "global": {"knee_to_hip_width_ratio": ratio}},
        "form_label": label.value,
    }


def reference_doc(ratio: float, *, back_offset: float = 0.0, repeats: int = 12) -> dict:
    """A clean rep for a subject of the given build.

    ``back_offset`` shifts every back_to_vertical value, which is how a longer-femur
    subject is represented: their optimal lean genuinely is further forward.
    """
    frames, frame_id, timestamp = [], 0, 0
    for phase in (Phase.SETUP, Phase.ECCENTRIC, Phase.BOTTOM,
                  Phase.CONCENTRIC, Phase.RECOVERY):
        hip, knee, ankle, back, width = OPTIMAL[phase]
        for i in range(repeats):
            jitter = (i % 4 - 1.5) * 0.8
            frames.append(
                frame_dict(frame_id, timestamp, phase,
                           (hip, knee, ankle, back + back_offset, width),
                           FormLabel.OPTIMAL, jitter)
            )
            frame_id += 1
            timestamp += 33
    return {
        "metadata": {
            "exercise": "barbell_back_squat",
            "camera_angle": "45_oblique_anterior",
            "dataset_type": "reference_optimal",
            "fps_target": 30,
        },
        "subject_proportions": {
            "femur_to_torso_ratio": ratio,
            "tibia_to_femur_ratio": 0.85,
        },
        "frames": frames,
    }


def write_corpus(tmp_path, builds) -> ReferenceCorpus:
    for index, (ratio, back_offset) in enumerate(builds):
        doc = reference_doc(ratio, back_offset=back_offset)
        (tmp_path / f"subject_{index}.json").write_text(json.dumps(doc), encoding="utf-8")
    return load_corpus(tmp_path, DatasetConfig())


def live_frame(phase, values, frame_id=0, timestamp_ms=0):
    hip, knee, ankle, back, ratio = values
    return KinematicFrame(
        frame_id=frame_id, timestamp_ms=timestamp_ms,
        camera_near=SideAngles(hip, knee, ankle, back),
        camera_far=SideAngles(hip + 0.4, knee + 0.4, ankle + 0.4, back + 0.4),
        global_metrics=GlobalMetrics(ratio),
        phase=phase,
    )


def feed(analyzer, phase, values, ratio, count=6, start_ms=0):
    """Push several frames so the debounce has a chance to latch."""
    result = None
    for i in range(count):
        result = analyzer.update(
            live_frame(phase, values, frame_id=i, timestamp_ms=start_ms + i * 33), ratio
        )
    return result


# -- profile construction ------------------------------------------------------


def test_bands_come_from_the_corpus(tmp_path):
    corpus = write_corpus(tmp_path, [(0.90, 0.0)])
    profiles = build_profiles(corpus, ANALYZER)

    assert len(profiles) == 1
    band = profiles[0].band(Phase.BOTTOM, "camera_near.knee_flexion")
    assert band is not None
    # The fixture's bottom knee angle is 78 with +-1.2 of jitter.
    assert band.lower == pytest.approx(76.8, abs=0.5)
    assert band.upper == pytest.approx(79.2, abs=0.5)
    assert band.samples == 12


def test_bands_track_a_changed_corpus(tmp_path):
    """The load-bearing property: bounds are evidence, not constants."""
    shallow = build_profiles(write_corpus(tmp_path, [(0.90, 0.0)]), ANALYZER)
    for path in tmp_path.glob("*.json"):
        path.unlink()
    leaned = build_profiles(write_corpus(tmp_path, [(0.90, 18.0)]), ANALYZER)

    a = shallow[0].band(Phase.BOTTOM, "camera_near.back_to_vertical")
    b = leaned[0].band(Phase.BOTTOM, "camera_near.back_to_vertical")
    assert b.upper - a.upper == pytest.approx(18.0, abs=0.5)


def test_error_frames_do_not_widen_the_acceptable_band(tmp_path):
    doc = reference_doc(0.90)
    for frame in doc["frames"]:
        if frame["phase"] == Phase.CONCENTRIC.value:
            frame["form_label"] = FormLabel.GOOD_MORNING.value
            frame["angles"]["camera_near"]["back_to_vertical"] = 70.0
    (tmp_path / "mixed.json").write_text(json.dumps(doc), encoding="utf-8")
    corpus = load_corpus(tmp_path, DatasetConfig())

    profiles = build_profiles(corpus, ANALYZER)
    # The error frames were the only concentric evidence, so no concentric band exists
    # rather than one stretched to admit a 70-degree lean.
    assert profiles[0].band(Phase.CONCENTRIC, "camera_near.back_to_vertical") is None
    assert profiles[0].band(Phase.BOTTOM, "camera_near.back_to_vertical") is not None


def test_optimal_frames_inside_an_error_file_still_count(tmp_path):
    """A good-morning rep has a clean eccentric, and that is real optimal evidence."""
    doc = reference_doc(0.90)
    doc["metadata"]["dataset_type"] = "reference_good_morning_error"
    for frame in doc["frames"]:
        if frame["phase"] == Phase.CONCENTRIC.value:
            frame["form_label"] = FormLabel.GOOD_MORNING.value
    (tmp_path / "gm.json").write_text(json.dumps(doc), encoding="utf-8")

    profiles = build_profiles(load_corpus(tmp_path, DatasetConfig()), ANALYZER)
    assert profiles[0].band(Phase.ECCENTRIC, "camera_near.hip_flexion") is not None


def test_corpus_without_optimal_frames_refuses_to_build(tmp_path):
    doc = reference_doc(0.90)
    for frame in doc["frames"]:
        frame["form_label"] = FormLabel.KNEE_VALGUS.value
    (tmp_path / "bad.json").write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(AnalyzerError, match="no baseline to fall back on"):
        build_profiles(load_corpus(tmp_path, DatasetConfig()), ANALYZER)


def test_empty_corpus_refuses_to_analyze():
    with pytest.raises(AnalyzerError, match="empty"):
        FormAnalyzer(ReferenceCorpus(()), ANALYZER, KIN)


# -- build adjustment ----------------------------------------------------------


def test_thresholds_interpolate_between_bracketing_builds(tmp_path):
    corpus = write_corpus(tmp_path, [(0.85, 0.0), (1.25, 20.0)])
    model = ThresholdModel(build_profiles(corpus, ANALYZER))

    profile, source = model.resolve(1.05)
    assert source.mode == "interpolated"
    assert source.blend == pytest.approx(0.5)

    low, _ = model.resolve(0.85)
    high, _ = model.resolve(1.25)
    key = (Phase.BOTTOM, "camera_near.back_to_vertical")
    midpoint = 0.5 * (low.bands[key].upper + high.bands[key].upper)
    assert profile.bands[key].upper == pytest.approx(midpoint, abs=0.01)


def test_longer_femur_is_allowed_more_lean_when_the_corpus_says_so(tmp_path):
    corpus = write_corpus(tmp_path, [(0.85, 0.0), (1.25, 20.0)])
    model = ThresholdModel(build_profiles(corpus, ANALYZER))
    key = (Phase.BOTTOM, "camera_near.back_to_vertical")

    short_femur, _ = model.resolve(0.90)
    long_femur, _ = model.resolve(1.20)
    assert long_femur.bands[key].upper > short_femur.bands[key].upper + 10


def test_out_of_corpus_build_extrapolates_along_the_trend(tmp_path, caplog):
    corpus = write_corpus(tmp_path, [(0.90, 0.0), (1.10, 10.0)])
    model = ThresholdModel(build_profiles(corpus, ANALYZER))
    key = (Phase.BOTTOM, "camera_near.back_to_vertical")

    inside_low, _ = model.resolve(0.90)
    inside_high, _ = model.resolve(1.10)
    beyond, source = model.resolve(1.30)

    assert source.mode == "extrapolated"
    assert source.extrapolated
    assert "[Extrapolation Warning]" in caplog.text

    # Projected, not clamped: 1.30 is one further step of the same slope.
    step = inside_high.bands[key].upper - inside_low.bands[key].upper
    assert beyond.bands[key].upper == pytest.approx(
        inside_high.bands[key].upper + step, abs=0.5
    )


def test_extrapolation_below_the_corpus_projects_downward(tmp_path):
    corpus = write_corpus(tmp_path, [(0.90, 0.0), (1.10, 10.0)])
    model = ThresholdModel(build_profiles(corpus, ANALYZER))
    key = (Phase.BOTTOM, "camera_near.back_to_vertical")

    inside, _ = model.resolve(0.90)
    below, source = model.resolve(0.70)
    assert source.mode == "extrapolated"
    assert below.bands[key].upper < inside.bands[key].upper


def test_never_falls_back_to_a_baseline_for_an_extreme_build(tmp_path, caplog):
    """An extreme ratio must still be answered from corpus evidence."""
    corpus = write_corpus(tmp_path, [(0.90, 0.0), (1.10, 10.0)])
    analyzer = FormAnalyzer(corpus, ANALYZER, KIN)
    result = feed(analyzer, Phase.BOTTOM, OPTIMAL[Phase.BOTTOM], 1.80)

    assert result.band_source.extrapolated
    assert "[Extrapolation Warning]" in caplog.text
    # A projected band, not the corpus's own values.
    projected = analyzer.model.resolve(1.80)[0]
    key = (Phase.BOTTOM, "camera_near.back_to_vertical")
    assert projected.bands[key].upper > 60


def test_single_profile_corpus_warns_rather_than_inventing_a_trend(tmp_path, caplog):
    corpus = write_corpus(tmp_path, [(0.90, 0.0)])
    model = ThresholdModel(build_profiles(corpus, ANALYZER))
    _, source = model.resolve(1.40)
    assert source.mode == "single_profile"
    assert "[Extrapolation Warning]" in caplog.text


def test_resolve_before_calibration_raises(tmp_path):
    corpus = write_corpus(tmp_path, [(0.90, 0.0), (1.10, 10.0)])
    model = ThresholdModel(build_profiles(corpus, ANALYZER))
    with pytest.raises(AnalyzerError, match="calibration"):
        model.resolve(math.nan)


# -- detection -----------------------------------------------------------------


def analyzer_for(tmp_path):
    corpus = write_corpus(tmp_path, [(0.85, 0.0), (1.25, 20.0)])
    return FormAnalyzer(corpus, ANALYZER, KIN)


def test_clean_rep_reports_optimal(tmp_path):
    analyzer = analyzer_for(tmp_path)
    for phase in (Phase.ECCENTRIC, Phase.BOTTOM, Phase.CONCENTRIC):
        values = list(OPTIMAL[phase])
        values[3] += 10.0  # midway between the two corpus builds
        result = feed(analyzer, phase, tuple(values), 1.05)
        assert result.ok, f"{phase} flagged: {[f.message for f in result.findings]}"
        assert result.label is FormLabel.OPTIMAL


def test_knee_valgus_is_detected(tmp_path):
    analyzer = analyzer_for(tmp_path)
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0
    values[4] = 0.72  # knees well inside the hips
    result = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)

    assert result.label is FormLabel.KNEE_VALGUS
    assert "KNEE VALGUS" in result.findings[0].message


def test_high_squat_is_detected(tmp_path):
    analyzer = analyzer_for(tmp_path)
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0
    values[0] += 45.0  # never reached depth
    values[1] += 45.0
    result = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)
    assert result.label is FormLabel.HIGH_SQUAT


def test_good_morning_requires_the_back_angle_to_be_opening(tmp_path):
    """The rate is what distinguishes it; a static lean is a different fault."""
    analyzer = analyzer_for(tmp_path)
    base = list(OPTIMAL[Phase.CONCENTRIC])
    base[3] += 10.0

    # Back angle high but steady: the rising-rate signature must not fire.
    steady = list(base)
    steady[3] += 30.0
    result = feed(analyzer, Phase.CONCENTRIC, tuple(steady), 1.05, count=10)
    assert "HIPS RISING TOO FAST" not in [f.message for f in result.findings]

    # Back angle high and still opening through the ascent.
    analyzer.reset()
    result = None
    for i in range(12):
        rising = list(base)
        rising[3] += 20.0 + i * 2.5
        result = analyzer.update(
            live_frame(Phase.CONCENTRIC, tuple(rising), frame_id=i, timestamp_ms=i * 33),
            1.05,
        )
    assert result.label is FormLabel.GOOD_MORNING
    assert "HIPS RISING TOO FAST" in [f.message for f in result.findings]


def test_heel_lift_is_detected_as_an_opening_ankle(tmp_path):
    analyzer = analyzer_for(tmp_path)
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0
    values[2] += 25.0  # heel rises, tilting the foot axis away from the shin
    result = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)
    assert result.label is FormLabel.HEEL_LIFT


def test_occluded_feature_withholds_its_finding(tmp_path):
    analyzer = analyzer_for(tmp_path)
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0
    values[4] = math.nan  # width ratio unmeasured
    result = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)
    assert FormLabel.KNEE_VALGUS not in [f.label for f in result.findings]


def test_findings_are_debounced(tmp_path):
    """A single bad frame must not flash a cue at the lifter."""
    analyzer = FormAnalyzer(
        write_corpus(tmp_path, [(0.85, 0.0), (1.25, 20.0)]),
        AnalyzerConfig(finding_hold_frames=5, finding_decay_frames=3), KIN,
    )
    bad = list(OPTIMAL[Phase.BOTTOM])
    bad[3] += 10.0
    bad[4] = 0.70

    first = analyzer.update(live_frame(Phase.BOTTOM, tuple(bad)), 1.05)
    assert first.ok, "surfaced a finding from a single frame"

    result = feed(analyzer, Phase.BOTTOM, tuple(bad), 1.05, count=6, start_ms=33)
    assert not result.ok


def test_findings_are_ranked_by_severity(tmp_path):
    analyzer = analyzer_for(tmp_path)
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0
    values[4] = 0.60          # severe valgus
    values[2] += 6.0          # mild ankle deviation
    result = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)

    severities = [f.severity for f in result.findings]
    assert severities == sorted(severities, reverse=True)
    assert result.findings[0].label is FormLabel.KNEE_VALGUS


def test_valgus_and_angle_deviations_are_ranked_on_one_scale(tmp_path):
    """Without variance-equivalent weighting the ratio would always rank last."""
    analyzer = analyzer_for(tmp_path)
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0
    values[4] -= 0.10         # 0.1 of ratio ...
    result_ratio = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)

    analyzer.reset()
    values = list(OPTIMAL[Phase.BOTTOM])
    values[3] += 10.0 + 15.0  # ... against 15 degrees of angle
    result_angle = feed(analyzer, Phase.BOTTOM, tuple(values), 1.05)

    assert result_ratio.findings and result_angle.findings
    assert result_ratio.findings[0].severity == pytest.approx(
        result_angle.findings[0].severity, rel=0.25
    )


# -- signature table -----------------------------------------------------------


def test_signature_table_carries_no_magnitudes():
    """Interpretation is domain knowledge; magnitudes must come from the corpus."""
    for signature in ERROR_SIGNATURES:
        assert signature.feature in FEATURE_ORDER
        assert signature.direction in (-1, 1)
        assert signature.phases
        for value in vars(signature).values():
            assert not isinstance(value, float), "a float in the table is a threshold"


def test_every_error_label_has_a_signature():
    covered = {s.label for s in ERROR_SIGNATURES}
    expected = set(FormLabel) - {FormLabel.OPTIMAL}
    assert covered == expected


# -- dtw -----------------------------------------------------------------------


def test_dtw_is_zero_for_identical_sequences():
    weights = feature_weight_vector(KIN)
    seq = np.tile(np.arange(9, dtype=float), (10, 1))
    assert dtw_distance(seq, seq, weights) == pytest.approx(0.0)


def test_dtw_tolerates_a_time_stretch():
    """Same movement performed slower should stay close."""
    weights = feature_weight_vector(KIN)
    base = np.linspace(180, 80, 20)[:, None] * np.ones((1, 9))
    slow = np.linspace(180, 80, 30)[:, None] * np.ones((1, 9))
    different = np.linspace(180, 150, 20)[:, None] * np.ones((1, 9))

    assert dtw_distance(base, slow, weights) < dtw_distance(base, different, weights)


def test_dtw_skips_nan_features_without_poisoning_the_path():
    weights = feature_weight_vector(KIN)
    a = np.tile(np.arange(9, dtype=float), (8, 1))
    b = a.copy()
    b[:, 4:8] = np.nan  # camera-far side unmeasured
    assert math.isfinite(dtw_distance(a, b, weights))


def test_dtw_handles_empty_input():
    weights = feature_weight_vector(KIN)
    assert dtw_distance(np.zeros((0, 9)), np.ones((5, 9)), weights) == math.inf


def test_sequence_classification_prefers_the_matching_label(tmp_path):
    """Whole-shape check: catches coordination faults no single frame violates."""
    good = reference_doc(0.90)
    bad = reference_doc(0.90)
    for frame in bad["frames"]:
        if frame["phase"] == Phase.CONCENTRIC.value:
            frame["form_label"] = FormLabel.GOOD_MORNING.value
            frame["angles"]["camera_near"]["back_to_vertical"] += 30.0
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    corpus = load_corpus(tmp_path, DatasetConfig())

    analyzer = FormAnalyzer(corpus, ANALYZER, KIN)
    values = list(OPTIMAL[Phase.CONCENTRIC])
    values[3] += 30.0
    for i in range(12):
        analyzer.update(live_frame(Phase.CONCENTRIC, tuple(values), i, i * 33), 0.90)
    # Leaving the phase closes the segment and triggers classification.
    result = analyzer.update(live_frame(Phase.RECOVERY, OPTIMAL[Phase.RECOVERY], 12, 400), 0.90)

    assert result.segment_label is FormLabel.GOOD_MORNING
    assert result.segment_confidence > 0


def test_sequence_loaded_from_disk_matches_itself(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(reference_doc(0.90)), encoding="utf-8")
    sequence = load_sequence(path, DatasetConfig())
    weights = feature_weight_vector(KIN)
    assert dtw_distance(sequence.features, sequence.features, weights) == pytest.approx(0.0)
