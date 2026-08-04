"""Phase 3 tests: geometry, side resolution, calibration and rep segmentation.

Poses are synthesised directly in world space (Y-up, metres, hip-origin) so the maths is
checked against known geometry rather than against whatever the model happens to emit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from formpro.config import KinematicsConfig, PhaseConfig
from formpro.kinematics import (
    FEATURE_ORDER,
    GlobalMetrics,
    KinematicFrame,
    KinematicsEngine,
    SideAngles,
    angle_between,
    back_angle_allowance,
    feature_weights,
    to_feature_vector,
)
from formpro.phases import PhaseSegmenter, VelocityTracker
from formpro.schema import LM, Phase, PoseFrame, Side

from .support import STANDING, make_pose


def engine(**overrides) -> KinematicsEngine:
    return KinematicsEngine(KinematicsConfig(**overrides), min_visibility=0.5)


# -- geometry ------------------------------------------------------------------


def test_angle_between_basics():
    assert angle_between(np.array([0, 1, 0]), np.array([0, 1, 0])) == pytest.approx(0)
    assert angle_between(np.array([0, 1, 0]), np.array([1, 0, 0])) == pytest.approx(90)
    assert angle_between(np.array([0, 1, 0]), np.array([0, -1, 0])) == pytest.approx(180)
    assert math.isnan(angle_between(np.array([0, 0, 0]), np.array([0, 1, 0])))


def test_standing_pose_produces_extended_angles():
    """180/180/90/0 is the convention the reference dataset encodes."""
    eng = engine(calibration_min_frames=1)
    frame = eng.update(make_pose())
    assert frame is not None

    near = frame.camera_near
    assert near.hip_flexion == pytest.approx(180.0, abs=0.01)
    assert near.knee_flexion == pytest.approx(180.0, abs=0.01)
    assert near.ankle_dorsiflexion == pytest.approx(90.0, abs=0.01)
    assert near.back_to_vertical == pytest.approx(0.0, abs=0.01)
    assert near.complete


def test_descending_decreases_flexion_and_increases_back_angle():
    eng = engine(calibration_min_frames=1)
    # Bottom position: knees travel forward over the toes (+z is the facing
    # direction, per the heel/toe geometry), hips drop and shift back, torso leans
    # forward to keep the bar over midfoot.
    bottom = eng.update(
        make_pose(
            joints={
                "shoulder": (0.09, -0.05, 0.30),
                "hip": (0.09, -0.45, -0.15),
                "knee": (0.09, -0.55, 0.30),
            }
        )
    )
    assert bottom is not None
    near = bottom.camera_near
    assert near.hip_flexion < 120
    assert near.knee_flexion < 120
    assert near.ankle_dorsiflexion < 90
    assert near.back_to_vertical > 25


def test_width_ratio_is_one_when_knees_track_over_hips():
    frame = engine(calibration_min_frames=1).update(make_pose())
    assert frame.global_metrics.knee_to_hip_width_ratio == pytest.approx(1.0, abs=1e-4)


def test_valgus_drives_width_ratio_below_one():
    frame = engine(calibration_min_frames=1).update(make_pose(knee_x=0.045))
    assert frame.global_metrics.knee_to_hip_width_ratio == pytest.approx(0.5, abs=1e-4)


def test_width_ratio_is_invariant_to_camera_yaw():
    """The ratio's whole purpose: both axes foreshorten together, so it cancels.

    Rotating the subject about the vertical axis simulates a different camera yaw. The
    absolute knee separation in X changes; the ratio must not.
    """
    eng = engine(calibration_min_frames=1)
    straight = eng.update(make_pose(knee_x=0.07)).global_metrics.knee_to_hip_width_ratio

    for yaw_deg in (20.0, 45.0, 60.0):
        pose = make_pose(knee_x=0.07)
        theta = math.radians(yaw_deg)
        cos, sin = math.cos(theta), math.sin(theta)
        rot = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        rotated = PoseFrame(
            index=0, timestamp_ms=0,
            world_xyz=(pose.world_xyz @ rot.T).astype(np.float32),
            image_xyz=pose.image_xyz, visibility=pose.visibility, presence=pose.presence,
        )
        got = engine(calibration_min_frames=1).update(rotated).global_metrics
        assert got.knee_to_hip_width_ratio == pytest.approx(straight, abs=1e-4)


def test_occluded_joint_yields_nan_not_zero():
    frame = engine(calibration_min_frames=1).update(
        make_pose(hidden=(LM.LEFT_ANKLE, LM.RIGHT_ANKLE))
    )
    assert math.isnan(frame.camera_near.knee_flexion)
    assert math.isnan(frame.camera_near.ankle_dorsiflexion)
    # The hip is still measurable and must survive.
    assert not math.isnan(frame.camera_near.hip_flexion)
    assert not frame.camera_near.complete


def test_missing_core_joints_drops_the_frame():
    assert engine().update(make_pose(hidden=(LM.LEFT_KNEE, LM.RIGHT_KNEE))) is None


# -- side resolution -----------------------------------------------------------


def test_near_side_is_the_one_closer_to_camera():
    """World Z is positive toward the camera."""
    eng = engine(calibration_min_frames=1)
    frame = eng.update(make_pose(left_dz=0.30))
    assert frame.near_side is Side.LEFT

    eng = engine(calibration_min_frames=1)
    frame = eng.update(make_pose(right_dz=0.30))
    assert frame.near_side is Side.RIGHT


def test_near_side_does_not_flip_on_noise():
    """A side that flips mid-rep splices two limbs into one time series."""
    eng = engine(calibration_min_frames=1, side_hysteresis_m=0.05, side_ema_alpha=0.2)
    rng = np.random.default_rng(0)
    sides = []
    for i in range(60):
        jitter = float(rng.normal(0, 0.01))
        sides.append(eng.update(make_pose(index=i, left_dz=0.02 + jitter)).near_side)
    assert len(set(sides)) == 1, "near side flickered under noise"


def test_near_side_switches_on_sustained_change():
    eng = engine(calibration_min_frames=1, side_hysteresis_m=0.02, side_ema_alpha=0.3)
    for i in range(30):
        eng.update(make_pose(index=i, left_dz=0.20))
    assert eng.update(make_pose(index=30, left_dz=0.20)).near_side is Side.LEFT
    for i in range(31, 80):
        frame = eng.update(make_pose(index=i, right_dz=0.20))
    assert frame.near_side is Side.RIGHT


# -- proportions ---------------------------------------------------------------


def test_proportions_match_the_synthesised_geometry():
    eng = engine(calibration_min_frames=10)
    for i in range(20):
        eng.update(make_pose(index=i))
    proportions = eng.proportions
    assert proportions is not None
    assert proportions.femur_m == pytest.approx(0.45, abs=1e-3)
    assert proportions.tibia_m == pytest.approx(0.40, abs=1e-3)
    assert proportions.torso_m == pytest.approx(0.50, abs=1e-3)
    assert proportions.femur_to_torso_ratio == pytest.approx(0.90, abs=1e-3)
    assert proportions.tibia_to_femur_ratio == pytest.approx(0.8889, abs=1e-3)


def test_proportion_median_rejects_outlier_frames():
    """Depth spikes must not move the estimate; that is why it is a median."""
    eng = engine(calibration_min_frames=10)
    for i in range(30):
        if i % 7 == 0:
            eng.update(make_pose(index=i, joints={"knee": (0.09, -1.60, 0.0)}))
        else:
            eng.update(make_pose(index=i))
    assert eng.proportions.femur_m == pytest.approx(0.45, abs=1e-3)


def test_proportions_are_frozen_once_calibrated():
    eng = engine(calibration_min_frames=5)
    for i in range(10):
        eng.update(make_pose(index=i))
    first = eng.proportions
    for i in range(10, 200):
        eng.update(make_pose(index=i, joints={"knee": (0.09, -0.60, 0.0)}))
    assert eng.proportions is first


def test_hip_height_norm_is_one_when_standing():
    eng = engine(calibration_min_frames=1)
    frame = eng.update(make_pose())
    # leg length 0.85, hip sits 0.85 above the ankles.
    assert frame.hip_height_norm == pytest.approx(1.0, abs=1e-3)


def test_hip_height_norm_is_body_size_invariant():
    """A tall and a short lifter at the same relative depth read the same."""
    def depth_for(scale: float) -> float:
        eng = engine(calibration_min_frames=1)
        joints = {k: (x * scale, y * scale, z * scale) for k, (x, y, z) in STANDING.items()}
        eng.update(make_pose(joints=joints))
        squat = dict(joints)
        squat["hip"] = (joints["hip"][0], joints["ankle"][1] + 0.5 * 0.85 * scale, 0.0)
        return eng.update(make_pose(index=1, joints=squat)).hip_height_norm

    assert depth_for(0.8) == pytest.approx(depth_for(1.3), abs=1e-3)


# -- tolerance band and features ----------------------------------------------


def test_back_angle_allowance_interpolates_and_clamps():
    anchors = ((0.85, 38.0), (1.30, 52.0))
    assert back_angle_allowance(0.85, anchors) == pytest.approx(38.0)
    assert back_angle_allowance(1.30, anchors) == pytest.approx(52.0)
    assert back_angle_allowance(1.075, anchors) == pytest.approx(45.0)
    # Outside the anchored range, clamp rather than extrapolate.
    assert back_angle_allowance(0.50, anchors) == pytest.approx(38.0)
    assert back_angle_allowance(2.00, anchors) == pytest.approx(52.0)
    assert math.isnan(back_angle_allowance(math.nan, anchors))


def test_long_femur_is_allowed_more_forward_lean():
    anchors = ((0.85, 38.0), (1.30, 52.0))
    assert back_angle_allowance(1.25, anchors) > back_angle_allowance(0.90, anchors)


def test_feature_vector_matches_declared_order():
    frame = KinematicFrame(
        frame_id=1, timestamp_ms=0,
        camera_near=SideAngles(110.5, 125.0, 85.2, 45.1),
        camera_far=SideAngles(111.0, 124.5, 86.0, 45.3),
        global_metrics=GlobalMetrics(0.95),
    )
    vector = to_feature_vector(frame)
    assert len(vector) == len(FEATURE_ORDER) == 9
    assert vector[0] == pytest.approx(110.5)
    assert vector[4] == pytest.approx(111.0)
    assert vector[8] == pytest.approx(0.95)


def test_feature_weights_downrank_far_side_and_rescale_ratio():
    weights = feature_weights(KinematicsConfig(camera_far_weight=0.35, width_ratio_scale_deg=60.0))
    assert list(weights[:4]) == [1.0] * 4
    assert list(weights[4:8]) == [0.35] * 4
    assert weights[8] == 60.0


# -- velocity ------------------------------------------------------------------


def test_velocity_tracker_measures_per_second_not_per_frame():
    tracker = VelocityTracker(window_ms=200, max_gap_ms=250)
    slope = math.nan
    for i in range(10):
        slope = tracker.update(i * 33, 1.0 - 0.30 * (i * 0.033))
    assert slope == pytest.approx(-0.30, abs=0.02)


def test_velocity_is_unaffected_by_dropped_frames():
    """Phase 2 drops frames under load; a per-frame delta would read that as speed."""
    even = VelocityTracker(window_ms=300, max_gap_ms=400)
    uneven = VelocityTracker(window_ms=300, max_gap_ms=400)

    even_slope = uneven_slope = math.nan
    for i in range(20):
        t = i * 33
        even_slope = even.update(t, 1.0 - 0.4 * t / 1000.0)
    for t in (0, 33, 99, 132, 231, 264, 297, 396, 429, 462, 528, 561):
        uneven_slope = uneven.update(t, 1.0 - 0.4 * t / 1000.0)

    assert even_slope == pytest.approx(-0.4, abs=0.02)
    assert uneven_slope == pytest.approx(-0.4, abs=0.02)


def test_velocity_resets_across_a_long_gap():
    tracker = VelocityTracker(window_ms=200, max_gap_ms=250)
    for i in range(6):
        tracker.update(i * 33, 1.0)
    assert math.isnan(tracker.update(5_000, 0.5)), "fitted a velocity across a dropout"


# -- phase segmentation --------------------------------------------------------


def squat_trajectory(fps: int = 30) -> list[tuple[int, float]]:
    """Standing, descend, brief pause, ascend, stand."""
    dt = 1000 // fps
    samples: list[tuple[int, float]] = []
    t = 0

    def hold(seconds: float, height: float) -> None:
        nonlocal t
        for _ in range(int(seconds * fps)):
            samples.append((t, height))
            t += dt

    def ramp(seconds: float, start: float, end: float) -> None:
        nonlocal t
        steps = int(seconds * fps)
        for i in range(steps):
            samples.append((t, start + (end - start) * i / steps))
            t += dt

    hold(1.0, 1.0)
    ramp(1.2, 1.0, 0.55)
    hold(0.4, 0.55)
    ramp(1.2, 0.55, 1.0)
    hold(1.0, 1.0)
    return samples


def run_segmenter(samples, config: PhaseConfig | None = None):
    segmenter = PhaseSegmenter(config or PhaseConfig())
    observed, reps = [], []
    for i, (timestamp, height) in enumerate(samples):
        frame = KinematicFrame(
            frame_id=i, timestamp_ms=timestamp,
            camera_near=SideAngles.unmeasured(), camera_far=SideAngles.unmeasured(),
            global_metrics=GlobalMetrics(1.0), hip_height_norm=height,
        )
        result = segmenter.update(frame)
        if not observed or observed[-1] is not result.phase:
            observed.append(result.phase)
        if result.completed_rep:
            reps.append(result.completed_rep)
    return segmenter, observed, reps


def test_segmenter_walks_the_full_rep_cycle():
    segmenter, observed, reps = run_segmenter(squat_trajectory())
    assert observed == [
        Phase.SETUP, Phase.ECCENTRIC, Phase.BOTTOM, Phase.CONCENTRIC,
        Phase.RECOVERY, Phase.SETUP,
    ]
    assert segmenter.rep_count == 1
    assert len(reps) == 1
    assert reps[0].min_hip_height_norm == pytest.approx(0.55, abs=0.02)
    assert reps[0].duration_ms > 2000


def test_touch_and_go_rep_skips_the_bottom_hold():
    samples = [s for s in squat_trajectory()]
    # Remove the pause so the lifter bounces straight out of the hole.
    trimmed = [s for i, s in enumerate(samples) if not (36 <= i < 48)]
    _, observed, reps = run_segmenter(trimmed)
    assert Phase.ECCENTRIC in observed and Phase.CONCENTRIC in observed
    assert len(reps) == 1


def test_multiple_reps_are_counted():
    trajectory = squat_trajectory()
    samples, offset = [], 0
    for _ in range(3):
        samples.extend((t + offset, h) for t, h in trajectory)
        offset = samples[-1][0] + 33
    segmenter, _, reps = run_segmenter(samples)
    assert segmenter.rep_count == 3
    assert len(reps) == 3


def test_tracking_loss_abandons_the_rep_instead_of_closing_it():
    """A rep whose middle was never seen must not be scored."""
    samples = squat_trajectory()
    segmenter = PhaseSegmenter(PhaseConfig())
    reps = []
    for i, (timestamp, height) in enumerate(samples):
        if 45 <= i < 60:
            segmenter.on_tracking_lost()
            continue
        result = segmenter.update(
            KinematicFrame(
                frame_id=i, timestamp_ms=timestamp,
                camera_near=SideAngles.unmeasured(), camera_far=SideAngles.unmeasured(),
                global_metrics=GlobalMetrics(1.0), hip_height_norm=height,
            )
        )
        if result.completed_rep:
            reps.append(result.completed_rep)
    assert reps == [], "closed a rep across a tracking dropout"


def test_rep_starting_mid_descent_is_not_scored():
    """App opened with the lifter already descending: that rep was never fully seen."""
    trajectory = squat_trajectory()
    # Drop the opening standing hold, so the very first frame is already moving down.
    segmenter, _, reps = run_segmenter(trajectory[30:])
    assert reps == []
    assert segmenter.rep_count == 0

    # The following rep, entered from an observed standing position, does count.
    resumed = trajectory[30:] + [(t + trajectory[-1][0] + 33, h) for t, h in trajectory]
    segmenter, _, reps = run_segmenter(resumed)
    assert segmenter.rep_count == 1


def test_segmenter_holds_setup_before_calibration_supplies_a_height():
    segmenter = PhaseSegmenter(PhaseConfig())
    for i in range(30):
        result = segmenter.update(
            KinematicFrame(
                frame_id=i, timestamp_ms=i * 33,
                camera_near=SideAngles.unmeasured(), camera_far=SideAngles.unmeasured(),
                global_metrics=GlobalMetrics(1.0), hip_height_norm=math.nan,
            )
        )
    assert result.phase is Phase.SETUP
    assert segmenter.rep_count == 0
