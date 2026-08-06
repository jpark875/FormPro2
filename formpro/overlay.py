"""Phase 6: OpenCV rendering of the skeleton and the telemetry HUD.

Kept out of ``app.py`` so the main loop stays readable and so drawing can be exercised
without a camera. Everything here is pure: it takes state and a canvas and returns a
canvas, holding no pipeline references of its own.

Display mirroring happens at the very end, after the skeleton is drawn. Mirroring the
frame before inference would swap the lifter's anatomical left and right and invert every
per-side finding, so the flip is applied to the finished composite instead. The HUD text
is drawn after the flip so it stays readable.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .form_analyzer import AnalysisResult
from .kinematics import BodyProportions, KinematicFrame
from .schema import LM, SIDE_LANDMARKS, Phase, PoseFrame, Side

# BGR. The near/far split is the one thing a user must be able to confirm at a glance,
# so it gets the strongest contrast in the palette.
NEAR = (120, 255, 120)
FAR = (150, 120, 60)
OCCLUDED = (70, 70, 70)
GOOD = (110, 230, 130)
WARN = (0, 190, 255)
BAD = (70, 70, 255)
TEXT = (245, 245, 245)
MUTED = (170, 170, 170)
PANEL = (28, 28, 28)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Bones drawn per side, so each can take the near or far colour.
_SIDE_BONES = (
    ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
    ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe"),
)


def draw_skeleton(
    image: np.ndarray, pose: PoseFrame, near_side: Side | None, min_visibility: float
) -> np.ndarray:
    """Draw the tracked joints, colouring the camera-near side distinctly."""
    size = (image.shape[1], image.shape[0])

    for side, joints in SIDE_LANDMARKS.items():
        colour = NEAR if side is near_side else FAR
        for a_name, b_name in _SIDE_BONES:
            a, b = joints[a_name], joints[b_name]
            visible = (
                pose.is_visible(a, min_visibility) and pose.is_visible(b, min_visibility)
            )
            cv2.line(image, pose.pixel(a, size), pose.pixel(b, size),
                     colour if visible else OCCLUDED, 3 if visible else 1, cv2.LINE_AA)
        for name, landmark in joints.items():
            if name in ("shoulder", "hip", "knee", "ankle", "heel", "toe"):
                visible = pose.is_visible(landmark, min_visibility)
                cv2.circle(image, pose.pixel(landmark, size), 5 if visible else 3,
                           colour if visible else OCCLUDED, -1, cv2.LINE_AA)

    # The pelvis and shoulder girdle span both sides, so they get a neutral colour.
    for a, b in ((LM.LEFT_HIP, LM.RIGHT_HIP), (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)):
        cv2.line(image, pose.pixel(a, size), pose.pixel(b, size), TEXT, 2, cv2.LINE_AA)
    return image


def _text(image, string, origin, colour, scale=0.6, thickness=1):
    cv2.putText(image, string, origin, _FONT, scale, colour, thickness, cv2.LINE_AA)


def _panel(image, top_left, bottom_right, alpha=0.65):
    """Translucent backing so text stays legible over a bright gym floor."""
    x0, y0 = top_left
    x1, y1 = bottom_right
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = image[y0:y1, x0:x1]
    backing = np.full(region.shape, PANEL, dtype=np.uint8)
    image[y0:y1, x0:x1] = cv2.addWeighted(backing, alpha, region, 1 - alpha, 0)


def draw_hud(
    image: np.ndarray,
    *,
    proportions: BodyProportions | None,
    calibration_progress: float,
    kinematic: KinematicFrame | None,
    analysis: AnalysisResult | None,
    phase: Phase,
    rep_count: int,
    fps: float,
    inference_ms: float,
    dropped: int,
    tracking: bool,
) -> np.ndarray:
    """Telemetry panel plus the coaching cue."""
    width = image.shape[1]
    _panel(image, (0, 0), (width, 108))

    if proportions is not None:
        build = (
            f"femur/torso {proportions.femur_to_torso_ratio:.2f}   "
            f"tibia/femur {proportions.tibia_to_femur_ratio:.2f}"
        )
        build_colour = TEXT
    else:
        build = f"calibrating build  {calibration_progress * 100:3.0f}%"
        build_colour = WARN
    _text(image, build, (14, 30), build_colour, 0.62)

    _text(image, f"phase  {phase.value.upper()}", (14, 58), TEXT, 0.62)
    _text(image, f"reps  {rep_count}", (260, 58), TEXT, 0.62)

    if kinematic is not None and not math.isnan(kinematic.hip_height_norm):
        _text(image, f"depth  {kinematic.hip_height_norm:.2f}", (380, 58), MUTED, 0.55)

    _text(
        image,
        f"{fps:4.1f} fps   {inference_ms:4.1f} ms   dropped {dropped}",
        (14, 86), MUTED, 0.5,
    )

    if analysis is not None and analysis.band_source is not None:
        source = analysis.band_source
        _text(
            image, source.describe(), (width - 320, 30),
            WARN if source.extrapolated else MUTED, 0.5,
        )

    _draw_cue(image, analysis, tracking)
    return image


def _draw_cue(image: np.ndarray, analysis: AnalysisResult | None, tracking: bool) -> None:
    """The one thing a lifter mid-set can actually read."""
    height, width = image.shape[:2]

    if not tracking:
        _panel(image, (0, height - 76), (width, height))
        _text(image, "NO LIFTER DETECTED", (14, height - 32), WARN, 0.9, 2)
        return
    if analysis is None:
        _panel(image, (0, height - 76), (width, height))
        _text(image, "CALIBRATING", (14, height - 32), WARN, 0.9, 2)
        return

    if analysis.ok:
        _panel(image, (0, height - 76), (width, height))
        _text(image, "OPTIMAL FORM", (14, height - 32), GOOD, 0.9, 2)
        return

    findings = analysis.findings[:3]
    block = 44 * len(findings) + 26
    _panel(image, (0, height - block), (width, height))
    for index, finding in enumerate(findings):
        baseline = height - block + 36 + index * 44
        _text(image, finding.message, (14, baseline), BAD, 0.85, 2)
        _text(image, finding.detail(), (14, baseline + 18), MUTED, 0.45)


def compose(
    image: np.ndarray,
    *,
    pose: PoseFrame | None,
    near_side: Side | None,
    min_visibility: float,
    mirror: bool,
    **hud,
) -> np.ndarray:
    """Draw skeleton, mirror if requested, then draw the HUD on top."""
    if pose is not None:
        draw_skeleton(image, pose, near_side, min_visibility)
    if mirror:
        image = cv2.flip(image, 1)
    return draw_hud(image, **hud)
