"""Synthetic pose fixtures shared by the Phase 3 and integration tests.

Poses are built directly in world space (Y-up, metres, hip-origin) so the geometry under
test is known exactly, rather than depending on what the pose model happens to emit for
a given video.

The standing geometry is a vertical stack per side, which makes the extended angles come
out at exactly 180 / 180 / 90 / 0 and gives femur 0.45 m, tibia 0.40 m, torso 0.50 m.
"""

from __future__ import annotations

import math

import numpy as np

from formpro.schema import LM, NUM_LANDMARKS, SIDE_LANDMARKS, PoseFrame, Side

TIBIA_M = 0.40
FEMUR_M = 0.45
TORSO_M = 0.50

_ANKLE = (0.09, -0.85, 0.0)
_HEEL = (0.09, -0.90, -0.03)
_TOE = (0.09, -0.90, 0.15)

#: Segment angles from vertical at the bottom of the squat, in degrees:
#: shin forward over the toes, thigh back, torso forward over the knees.
BOTTOM_ANGLES = (40.0, 75.0, 50.0)


def joints_at(
    shin_deg: float, thigh_deg: float, torso_deg: float
) -> dict[str, tuple[float, float, float]]:
    """Build a skeleton by forward kinematics from three segment angles.

    Interpolating joint *positions* between two poses does not preserve segment
    lengths — a linear midpoint between a straight and a bent leg shortens the femur by
    over a centimetre. Driving the fixture from angles instead keeps every segment
    rigid at every point in the trajectory, which is what makes the calibration
    assertions meaningful.

    ``+z`` is the direction the lifter faces, per the heel-to-toe geometry.
    """
    shin, thigh, torso = map(math.radians, (shin_deg, thigh_deg, torso_deg))
    ax, ay, az = _ANKLE
    knee = (ax, ay + TIBIA_M * math.cos(shin), az + TIBIA_M * math.sin(shin))
    hip = (
        knee[0],
        knee[1] + FEMUR_M * math.cos(thigh),
        knee[2] - FEMUR_M * math.sin(thigh),
    )
    shoulder = (
        hip[0],
        hip[1] + TORSO_M * math.cos(torso),
        hip[2] + TORSO_M * math.sin(torso),
    )
    return {
        "shoulder": shoulder, "hip": hip, "knee": knee,
        "ankle": _ANKLE, "heel": _HEEL, "toe": _TOE,
    }


STANDING: dict[str, tuple[float, float, float]] = joints_at(0.0, 0.0, 0.0)
BOTTOM: dict[str, tuple[float, float, float]] = joints_at(*BOTTOM_ANGLES)


def make_pose(
    index: int = 0,
    timestamp_ms: int = 0,
    joints: dict | None = None,
    left_dz: float = 0.0,
    right_dz: float = 0.0,
    knee_x: float | None = None,
    visibility: float = 1.0,
    hidden: tuple[LM, ...] = (),
) -> PoseFrame:
    """Build a ``PoseFrame``. ``joints`` overrides individual landmarks by name."""
    joints = {**STANDING, **(joints or {})}
    world = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    for side, lookup in SIDE_LANDMARKS.items():
        sign = -1.0 if side is Side.LEFT else 1.0
        dz = left_dz if side is Side.LEFT else right_dz
        for name, landmark in lookup.items():
            x, y, z = joints[name]
            if name == "knee" and knee_x is not None:
                x = knee_x
            world[int(landmark)] = (sign * x, y, z + dz)

    vis = np.full(NUM_LANDMARKS, visibility, dtype=np.float32)
    for landmark in hidden:
        vis[int(landmark)] = 0.0

    return PoseFrame(
        index=index,
        timestamp_ms=timestamp_ms,
        world_xyz=world,
        image_xyz=np.full((NUM_LANDMARKS, 3), 0.5, dtype=np.float32),
        visibility=vis,
        presence=np.full(NUM_LANDMARKS, visibility, dtype=np.float32),
    )


def lerp_joints(t: float) -> dict[str, tuple[float, float, float]]:
    """Pose at descent fraction ``t``, interpolated in angle space so limbs stay rigid."""
    return joints_at(*(t * angle for angle in BOTTOM_ANGLES))
