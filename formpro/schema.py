"""Data contracts shared by every stage of the pipeline.

This module is deliberately dependency-light (numpy only). Phases 3–5 import the
``PoseFrame`` contract; nothing here may import MediaPipe, so the pose backend stays
swappable and the kinematics/analysis code stays unit-testable without a camera.

Coordinate conventions
----------------------
``image_xyz``  normalized [0,1], origin top-left, **Y down**. Rendering only.
``world_xyz``  metres, origin at the hip midpoint, X right, **Y up**, Z toward camera.
               All biomechanics use this space.

MediaPipe emits world landmarks Y-down / Z-away-from-camera. ``PoseFrame.from_mediapipe``
negates both axes so world space is a conventional right-handed Y-up system. Getting this
wrong silently inverts every depth and back-angle calculation downstream, so the flip
happens exactly once, here, at the boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

NUM_LANDMARKS = 33


class Side(IntEnum):
    """Anatomical side. Distinct from camera-near/camera-far, which is a viewing
    relationship resolved per frame by the kinematics engine."""

    LEFT = 0
    RIGHT = 1

    @property
    def other(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


class Phase(str, Enum):
    """Rep cycle vocabulary.

    Shared verbatim between the live segmenter and the reference dataset. Both sides
    of the comparison must speak the same words or Phase 5 would be aligning a live
    ``bottom`` against a reference ``eccentric`` and calling the difference form.
    """

    SETUP = "setup"           # standing, un-racking, bracing
    ECCENTRIC = "eccentric"   # descent
    BOTTOM = "bottom"         # the hole; velocity approaching zero
    CONCENTRIC = "concentric"  # ascent
    RECOVERY = "recovery"     # return to standing


class FormLabel(str, Enum):
    """Per-frame form classification.

    Evaluated per frame, not per file: a good-morning squat typically has a clean
    eccentric and breaks only once the concentric begins, so one sequence legitimately
    transitions between labels partway through.
    """

    OPTIMAL = "optimal_form"
    HIGH_SQUAT = "error_high_squat"
    KNEE_VALGUS = "error_knee_valgus"
    GOOD_MORNING = "error_good_morning"
    HEEL_LIFT = "error_heel_lift"


class LM(IntEnum):
    """BlazePose 33-landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31  # toe
    RIGHT_FOOT_INDEX = 32


#: The only joints the squat analysis depends on. Visibility gating and the reference
#: dataset schema are both defined over this subset, not all 33 landmarks — a lifter's
#: wrists and face may be occluded or out of frame without invalidating a rep.
SQUAT_JOINTS: tuple[LM, ...] = (
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_KNEE, LM.RIGHT_KNEE,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
    LM.LEFT_HEEL, LM.RIGHT_HEEL,
    LM.LEFT_FOOT_INDEX, LM.RIGHT_FOOT_INDEX,
)

#: Per-side landmark lookup, so kinematics can be written once and applied to whichever
#: side turns out to be nearest the camera.
SIDE_LANDMARKS: dict[Side, dict[str, LM]] = {
    Side.LEFT: {
        "shoulder": LM.LEFT_SHOULDER, "hip": LM.LEFT_HIP, "knee": LM.LEFT_KNEE,
        "ankle": LM.LEFT_ANKLE, "heel": LM.LEFT_HEEL, "toe": LM.LEFT_FOOT_INDEX,
    },
    Side.RIGHT: {
        "shoulder": LM.RIGHT_SHOULDER, "hip": LM.RIGHT_HIP, "knee": LM.RIGHT_KNEE,
        "ankle": LM.RIGHT_ANKLE, "heel": LM.RIGHT_HEEL, "toe": LM.RIGHT_FOOT_INDEX,
    },
}

#: Bone connectivity for the tracked subset, used by the overlay renderer.
SQUAT_SKELETON: tuple[tuple[LM, LM], ...] = (
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_ANKLE, LM.LEFT_HEEL),
    (LM.RIGHT_ANKLE, LM.RIGHT_HEEL),
    (LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX),
    (LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX),
    (LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
    (LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
)


@dataclass(frozen=True)
class Frame:
    """A raw camera frame with a capture timestamp.

    ``timestamp_ms`` is monotonic and measured from stream start — never wall-clock,
    which can jump backwards and would corrupt both MediaPipe's tracker state and the
    velocity terms used for phase segmentation in Phase 4.
    """

    index: int
    timestamp_ms: int
    image: np.ndarray  # HxWx3, BGR, uint8

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        h, w = self.image.shape[:2]
        return w, h


@dataclass(frozen=True)
class PoseFrame:
    """One frame's worth of pose, in both coordinate spaces."""

    index: int
    timestamp_ms: int
    world_xyz: np.ndarray     # (33, 3) float32 — metres, hip-origin, Y up
    image_xyz: np.ndarray     # (33, 3) float32 — normalized, Y down
    visibility: np.ndarray    # (33,)   float32 — 0..1, landmark not occluded
    presence: np.ndarray      # (33,)   float32 — 0..1, landmark inside the frame

    def __post_init__(self) -> None:
        for name, arr, shape in (
            ("world_xyz", self.world_xyz, (NUM_LANDMARKS, 3)),
            ("image_xyz", self.image_xyz, (NUM_LANDMARKS, 3)),
            ("visibility", self.visibility, (NUM_LANDMARKS,)),
            ("presence", self.presence, (NUM_LANDMARKS,)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")

    # -- accessors -------------------------------------------------------------

    def world(self, joint: LM) -> np.ndarray:
        """3D metric position of one joint, shape (3,)."""
        return self.world_xyz[int(joint)]

    def image(self, joint: LM) -> np.ndarray:
        """Normalized image position of one joint, shape (3,)."""
        return self.image_xyz[int(joint)]

    def pixel(self, joint: LM, size: tuple[int, int]) -> tuple[int, int]:
        """Joint position in pixels for a frame of ``size`` = (width, height)."""
        width, height = size
        x, y = self.image_xyz[int(joint), :2]
        return int(round(x * width)), int(round(y * height))

    def world_midpoint(self, a: LM, b: LM) -> np.ndarray:
        return 0.5 * (self.world(a) + self.world(b))

    # -- quality gating --------------------------------------------------------

    def is_visible(self, joint: LM, threshold: float) -> bool:
        i = int(joint)
        return bool(self.visibility[i] >= threshold and self.presence[i] >= threshold)

    def missing(self, joints: Iterable[LM], threshold: float) -> tuple[LM, ...]:
        """Which of ``joints`` fall below the confidence threshold.

        Callers should suppress any finding that depends on a missing joint rather than
        reporting a form error derived from a guessed coordinate — a false "knees caving
        in" cue costs more trust than saying nothing for a few frames.
        """
        return tuple(j for j in joints if not self.is_visible(j, threshold))

    def is_analysable(self, threshold: float, joints: Sequence[LM] = SQUAT_JOINTS) -> bool:
        return not self.missing(joints, threshold)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_mediapipe(
        cls,
        index: int,
        timestamp_ms: int,
        world_landmarks: Sequence,
        image_landmarks: Sequence,
    ) -> PoseFrame:
        """Convert MediaPipe landmark lists into a ``PoseFrame``.

        This is the one place the MediaPipe axis convention is translated; see the
        module docstring.
        """
        world = _landmarks_to_array(world_landmarks)
        image = _landmarks_to_array(image_landmarks)

        # Y-down -> Y-up, Z-away -> Z-toward-camera. Image space is left as-is.
        world[:, 1] *= -1.0
        world[:, 2] *= -1.0

        visibility = _attr_to_array(image_landmarks, "visibility")
        presence = _attr_to_array(image_landmarks, "presence")

        return cls(
            index=index,
            timestamp_ms=timestamp_ms,
            world_xyz=world,
            image_xyz=image,
            visibility=visibility,
            presence=presence,
        )

    def with_world(self, world_xyz: np.ndarray) -> PoseFrame:
        """Copy carrying replacement world coordinates (used by the smoothing filter)."""
        return PoseFrame(
            index=self.index,
            timestamp_ms=self.timestamp_ms,
            world_xyz=world_xyz,
            image_xyz=self.image_xyz,
            visibility=self.visibility,
            presence=self.presence,
        )


def _landmarks_to_array(landmarks: Sequence) -> np.ndarray:
    out = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
    if len(landmarks) != NUM_LANDMARKS:
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
    for i, lm in enumerate(landmarks):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def _attr_to_array(landmarks: Sequence, attr: str) -> np.ndarray:
    out = np.empty(NUM_LANDMARKS, dtype=np.float32)
    for i, lm in enumerate(landmarks):
        value = getattr(lm, attr, None)
        out[i] = 0.0 if value is None else float(value)
    return out
