"""Phase 7: synthesise reference profiles for builds the corpus does not cover.

The analyzer projects a linear trend when a lifter falls outside the corpus. Projection
is better than clamping, but it is still the weakest evidence in the system, and a lifter
at 0.70 against anchors at 0.85 and 1.25 is being judged by a line drawn well past its
last data point. Widening the corpus narrows how often that happens.

The model
---------
Work in the sagittal plane with the ankle as origin, and hold the requirement that the
bar stays over the midfoot. Let ``F`` be femur, ``T`` torso, ``S`` tibia, and let each
segment's angle be measured from vertical. Reading horizontal offsets along the chain:

    knee is forward of the ankle by      S*sin(shin)
    hip is behind the knee by            F*sin(thigh)
    shoulder is forward of the hip by    T*sin(back)

Setting the shoulder back over the ankle and dividing through by ``T`` so only the ratio
matters, with ``r = F/T`` and ``k = S/F``:

    sin(back) = r * (sin(thigh) - k*sin(shin))

That single line is the whole biomechanical claim: **the lean a lifter needs scales with
femur-to-torso ratio.** A longer femur pushes the hips further back for the same knee
bend, so the torso must incline further to bring the bar back over the midfoot.

The remaining angles follow from the existing convention. ``ankle_dorsiflexion`` is the
shin against the foot's long axis, so ``shin = 90 - ankle_dorsiflexion``. ``knee_flexion``
is the included angle at the knee, which gives ``thigh = 180 - knee_flexion - shin``. And
``hip_flexion``, the included angle between torso and thigh, is ``180 - back - thigh``,
so any change in back angle moves hip flexion by the same amount in the opposite
direction.

What this deliberately does not model
-------------------------------------
The shin and knee are held fixed and only the torso is rotated. A real lifter with a
longer femur would also change knee travel, stance and bar path. Modelling that needs
assumptions about individual mobility that this prototype has no basis for.

Rather than replace the recorded angles with model output, the model is evaluated at the
source ratio and at the target ratio and only the **difference** is applied. The recording
keeps its own character - its jitter, its timing, its idiosyncrasies - and only the
build-driven component is shifted. It also means a bad absolute model still produces a
usable relative warp.

Synthetic profiles are not measurements
---------------------------------------
A generated profile encodes this model's assumption, not an observed lifter. If the model
is wrong, the analyzer will apply a wrong band with exactly the same confidence it applies
a measured one. Generated files are marked ``reference_optimal_synthetic`` with provenance
in their metadata so they can always be told apart, and they should be replaced by real
recordings as those become available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .kinematics import KinematicFrame, SideAngles

_EPS = 1e-9

#: Builds worth covering. Spans roughly the range seen in adult lifters, from a
#: short-femur/long-torso build to the long-femur build that struggles to stay upright.
DEFAULT_TARGET_RATIOS: tuple[float, ...] = (
    0.70, 0.75, 0.80, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20, 1.30, 1.35,
)


@dataclass(frozen=True)
class WarpReport:
    """What the warp actually did, so the CLI can report rather than assert."""

    target_ratio: float
    frames: int
    #: Signed, so a shorter-femur warp is visibly a reduction in lean rather than
    #: looking identical to a longer-femur one.
    mean_back_shift_deg: float
    #: Largest absolute shift, for spotting a warp that has gone somewhere extreme.
    max_back_shift_deg: float
    clamped_frames: int
    infeasible_frames: int


def shin_angle_from_vertical(ankle_dorsiflexion: float) -> float:
    """Shin inclination in degrees. 0 is vertical, positive is knee forward of ankle."""
    return 90.0 - ankle_dorsiflexion


def thigh_angle_from_vertical(knee_flexion: float, shin_angle: float) -> float:
    """Thigh inclination in degrees. 0 is vertical, positive is hip behind the knee."""
    return 180.0 - knee_flexion - shin_angle


def required_back_angle(
    femur_to_torso_ratio: float, thigh_angle: float, shin_angle: float,
    tibia_to_femur_ratio: float,
) -> tuple[float, bool]:
    """Back angle from vertical that puts the bar over the midfoot.

    Returns the angle in degrees and whether the position was infeasible - that is,
    whether the required sine exceeded 1, meaning no torso inclination can bring the bar
    back over the midfoot for that build and that leg configuration.
    """
    horizontal = (
        math.sin(math.radians(thigh_angle))
        - tibia_to_femur_ratio * math.sin(math.radians(shin_angle))
    )
    sine = femur_to_torso_ratio * horizontal
    infeasible = abs(sine) > 1.0
    return math.degrees(math.asin(max(-1.0, min(1.0, sine)))), infeasible


def back_angle_shift(
    frame: KinematicFrame,
    source_ratio: float,
    target_ratio: float,
    tibia_to_femur_ratio: float,
) -> tuple[float, bool]:
    """Degrees of extra forward lean this build needs, relative to the source lifter."""
    near = frame.camera_near
    if math.isnan(near.ankle_dorsiflexion) or math.isnan(near.knee_flexion):
        return 0.0, False

    shin = shin_angle_from_vertical(near.ankle_dorsiflexion)
    thigh = thigh_angle_from_vertical(near.knee_flexion, shin)

    at_source, bad_source = required_back_angle(
        source_ratio, thigh, shin, tibia_to_femur_ratio
    )
    at_target, bad_target = required_back_angle(
        target_ratio, thigh, shin, tibia_to_femur_ratio
    )
    return at_target - at_source, (bad_source or bad_target)


def _shift_side(side: SideAngles, delta: float) -> tuple[SideAngles, bool]:
    """Apply the lean shift to one side, keeping angles inside the legal 0-180 range."""
    if math.isnan(side.back_to_vertical) or math.isnan(side.hip_flexion):
        return side, False

    back = side.back_to_vertical + delta
    # hip_flexion = 180 - back - thigh, and the thigh is unchanged, so it moves by -delta.
    hip = side.hip_flexion - delta

    clamped = not (0.0 <= back <= 180.0 and 0.0 <= hip <= 180.0)
    return (
        replace(
            side,
            back_to_vertical=min(180.0, max(0.0, back)),
            hip_flexion=min(180.0, max(0.0, hip)),
        ),
        clamped,
    )


def warp_frames(
    frames: tuple[KinematicFrame, ...],
    source_ratio: float,
    target_ratio: float,
    tibia_to_femur_ratio: float,
) -> tuple[tuple[KinematicFrame, ...], WarpReport]:
    """Re-express a recorded rep as the same rep performed by a different build."""
    if source_ratio <= 0 or target_ratio <= 0:
        raise ValueError("femur_to_torso ratios must be positive")

    warped: list[KinematicFrame] = []
    shifts: list[float] = []
    clamped = infeasible = 0

    for frame in frames:
        delta, bad = back_angle_shift(
            frame, source_ratio, target_ratio, tibia_to_femur_ratio
        )
        infeasible += int(bad)
        shifts.append(delta)

        near, near_clamped = _shift_side(frame.camera_near, delta)
        # The far side sees the same torso, so it takes the same shift.
        far, far_clamped = _shift_side(frame.camera_far, delta)
        clamped += int(near_clamped or far_clamped)

        # knee_flexion, ankle_dorsiflexion and the width ratio are untouched: the model
        # rotates the torso only, and the width ratio is a frontal-plane measure that
        # sagittal proportions do not affect.
        warped.append(replace(frame, camera_near=near, camera_far=far))

    signed = shifts or [0.0]
    return tuple(warped), WarpReport(
        target_ratio=target_ratio,
        frames=len(warped),
        mean_back_shift_deg=sum(signed) / len(signed),
        max_back_shift_deg=max(abs(s) for s in signed),
        clamped_frames=clamped,
        infeasible_frames=infeasible,
    )


def build_document(
    frames: tuple[KinematicFrame, ...],
    target_ratio: float,
    tibia_to_femur_ratio: float,
    source_name: str,
    source_ratio: float,
    camera_angle: str,
    exercise: str,
    fps_target: float | None,
) -> dict:
    """Assemble a reference file in the Phase 4 schema.

    Frames are serialized through ``KinematicFrame.to_json_frame()``, the same call the
    live path uses, so a generated file cannot drift from what the loader expects.
    """
    metadata = {
        "exercise": exercise,
        "camera_angle": camera_angle,
        "dataset_type": "reference_optimal_synthetic",
        # Provenance, so synthetic anchors are never mistaken for recorded ones.
        "synthesized_from": source_name,
        "synthesized_from_ratio": round(source_ratio, 4),
        "synthesis_model": "midfoot_balance_v1",
    }
    if fps_target is not None:
        metadata["fps_target"] = fps_target

    return {
        "metadata": metadata,
        "subject_proportions": {
            "femur_to_torso_ratio": round(target_ratio, 4),
            "tibia_to_femur_ratio": round(tibia_to_femur_ratio, 4),
        },
        "frames": [frame.to_json_frame() for frame in frames],
    }
