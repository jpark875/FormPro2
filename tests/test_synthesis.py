"""Phase 7 tests: the build-warping model.

The claim under test is that lean scales with femur-to-torso ratio, and that warping
preserves everything the model does not speak to.
"""

from __future__ import annotations

import math

import pytest

from formpro.kinematics import GlobalMetrics, KinematicFrame, SideAngles
from formpro.schema import FormLabel, Phase
from formpro.synthesis import (
    DEFAULT_TARGET_RATIOS,
    build_document,
    required_back_angle,
    shin_angle_from_vertical,
    thigh_angle_from_vertical,
    warp_frames,
)

K = 0.85  # tibia/femur


def frame(hip=75.0, knee=78.0, ankle=66.0, back=34.0, ratio=1.02, i=0):
    return KinematicFrame(
        frame_id=i, timestamp_ms=i * 33,
        camera_near=SideAngles(hip, knee, ankle, back),
        camera_far=SideAngles(hip + 0.5, knee + 0.5, ankle + 0.5, back + 0.5),
        global_metrics=GlobalMetrics(ratio),
        phase=Phase.BOTTOM, form_label=FormLabel.OPTIMAL,
    )


# -- angle conversions ---------------------------------------------------------


def test_standing_maps_to_vertical_segments():
    shin = shin_angle_from_vertical(90.0)
    assert shin == pytest.approx(0.0)
    assert thigh_angle_from_vertical(180.0, shin) == pytest.approx(0.0)


def test_deep_squat_maps_to_plausible_segment_angles():
    shin = shin_angle_from_vertical(66.0)
    thigh = thigh_angle_from_vertical(78.0, shin)
    assert shin == pytest.approx(24.0)
    # Thigh near horizontal at the bottom of a deep squat.
    assert thigh == pytest.approx(78.0)


def test_standing_needs_no_lean_for_any_build():
    for ratio in (0.70, 1.00, 1.35):
        angle, infeasible = required_back_angle(ratio, 0.0, 0.0, K)
        assert angle == pytest.approx(0.0)
        assert not infeasible


def test_longer_femur_requires_more_lean():
    """The single biomechanical claim the whole module rests on."""
    shin = shin_angle_from_vertical(66.0)
    thigh = thigh_angle_from_vertical(78.0, shin)
    angles = [required_back_angle(r, thigh, shin, K)[0] for r in (0.70, 1.00, 1.35)]
    assert angles == sorted(angles)
    assert angles[2] > angles[0] + 10


def test_infeasible_position_is_reported_not_silently_clamped():
    # A very long femur with a near-horizontal thigh cannot be balanced by any lean.
    angle, infeasible = required_back_angle(3.0, 89.0, 0.0, K)
    assert infeasible
    assert angle == pytest.approx(90.0)


# -- warping -------------------------------------------------------------------


def test_warping_to_the_same_ratio_is_a_no_op():
    frames = (frame(),)
    warped, report = warp_frames(frames, 1.00, 1.00, K)
    assert warped[0].camera_near == frames[0].camera_near
    assert report.max_back_shift_deg == pytest.approx(0.0)


def test_longer_femur_leans_further_forward():
    original = frame()
    warped, report = warp_frames((original,), 1.00, 1.30, K)
    assert warped[0].camera_near.back_to_vertical > original.camera_near.back_to_vertical
    assert report.mean_back_shift_deg > 0


def test_shorter_femur_stands_more_upright():
    original = frame()
    warped, report = warp_frames((original,), 1.00, 0.75, K)
    assert warped[0].camera_near.back_to_vertical < original.camera_near.back_to_vertical
    # The reported mean must carry its sign, or a reduction in lean reads as an increase.
    assert report.mean_back_shift_deg < 0
    assert report.max_back_shift_deg > 0


def test_hip_flexion_moves_opposite_to_the_back_angle():
    """They are two views of one linkage: 180 = back + thigh + hip_flexion."""
    original = frame()
    warped, _ = warp_frames((original,), 1.00, 1.25, K)

    back_delta = warped[0].camera_near.back_to_vertical - original.camera_near.back_to_vertical
    hip_delta = warped[0].camera_near.hip_flexion - original.camera_near.hip_flexion
    assert hip_delta == pytest.approx(-back_delta, abs=1e-6)


def test_leg_and_frontal_measures_are_untouched():
    """The model rotates the torso; it makes no claim about the shin or the knees."""
    original = frame()
    warped, _ = warp_frames((original,), 1.00, 1.30, K)

    assert warped[0].camera_near.knee_flexion == original.camera_near.knee_flexion
    assert warped[0].camera_near.ankle_dorsiflexion == original.camera_near.ankle_dorsiflexion
    assert warped[0].global_metrics == original.global_metrics
    assert warped[0].phase is original.phase
    assert warped[0].timestamp_ms == original.timestamp_ms
    assert warped[0].form_label is original.form_label


def test_both_sides_receive_the_same_shift():
    original = frame()
    warped, _ = warp_frames((original,), 1.00, 1.30, K)
    near_delta = warped[0].camera_near.back_to_vertical - original.camera_near.back_to_vertical
    far_delta = warped[0].camera_far.back_to_vertical - original.camera_far.back_to_vertical
    assert near_delta == pytest.approx(far_delta)


def test_standing_frames_barely_move():
    """Build affects the loaded positions, not standing at the top of the rep."""
    standing = frame(hip=178.0, knee=178.0, ankle=89.0, back=3.0)
    warped, report = warp_frames((standing,), 1.00, 1.35, K)
    assert report.max_back_shift_deg < 3.0
    assert warped[0].camera_near.back_to_vertical == pytest.approx(3.0, abs=3.0)


def test_shift_is_monotonic_across_the_default_ratios():
    original = frame()
    leans = [
        warp_frames((original,), 1.00, r, K)[0][0].camera_near.back_to_vertical
        for r in DEFAULT_TARGET_RATIOS
    ]
    assert leans == sorted(leans)


def test_angles_stay_inside_the_legal_range():
    """The loader rejects anything outside 0-180, so the warp must never emit it."""
    original = frame(hip=40.0, knee=60.0, ankle=55.0, back=70.0)
    for ratio in (0.3, 0.7, 1.35, 3.0):
        warped, _ = warp_frames((original,), 1.00, ratio, K)
        near = warped[0].camera_near
        assert 0.0 <= near.back_to_vertical <= 180.0
        assert 0.0 <= near.hip_flexion <= 180.0


def test_nan_angles_survive_without_becoming_numbers():
    unmeasured = KinematicFrame(
        frame_id=0, timestamp_ms=0,
        camera_near=SideAngles(math.nan, math.nan, math.nan, math.nan),
        camera_far=SideAngles(math.nan, math.nan, math.nan, math.nan),
        global_metrics=GlobalMetrics(1.0), phase=Phase.SETUP,
    )
    warped, report = warp_frames((unmeasured,), 1.00, 1.30, K)
    assert math.isnan(warped[0].camera_near.back_to_vertical)
    assert report.max_back_shift_deg == pytest.approx(0.0)


def test_rejects_nonsense_ratios():
    with pytest.raises(ValueError):
        warp_frames((frame(),), 0.0, 1.0, K)


# -- document assembly ---------------------------------------------------------


def test_document_records_its_synthetic_provenance():
    frames, _ = warp_frames((frame(),), 1.00, 1.30, K)
    document = build_document(
        frames, 1.30, K, "source.json", 1.00, "45_oblique_anterior",
        "barbell_back_squat", 30.0,
    )
    metadata = document["metadata"]
    assert metadata["dataset_type"] == "reference_optimal_synthetic"
    assert metadata["synthesized_from"] == "source.json"
    assert metadata["synthesized_from_ratio"] == 1.00
    assert document["subject_proportions"]["femur_to_torso_ratio"] == 1.30
