"""Write a synthetic starter corpus so the app can run before any real recording exists.

    python scripts/seed_reference.py [--out data/reference]

Simulates one clean squat on a rigid skeleton, runs it through the same kinematics and
phase pipeline as live input, then widens it across builds with dataset_generator's
warp. Replace these files with real recordings as soon as you have them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formpro.config import AppConfig  # noqa: E402
from formpro.dataset_loader import DatasetError, load_sequence  # noqa: E402
from formpro.kinematics import KinematicsEngine  # noqa: E402
from formpro.phases import PhaseSegmenter  # noqa: E402
from formpro.schema import FormLabel  # noqa: E402
from formpro.simulate import squat_poses  # noqa: E402
from formpro.synthesis import DEFAULT_TARGET_RATIOS, build_document, warp_frames  # noqa: E402

FPS = 30
CAMERA_ANGLE = "45_oblique_anterior"
EXERCISE = "barbell_back_squat"


def record_base(config: AppConfig):
    """Run the simulated rep through the live pipeline; return usable frames and the build."""
    engine = KinematicsEngine(config.kinematics, min_visibility=config.pose.min_visibility)
    segmenter = PhaseSegmenter(config.phases)
    frames = []
    for pose in squat_poses(fps=FPS):
        kinematic = engine.update(pose)
        if kinematic is None:
            continue
        frames.append(kinematic.with_phase(segmenter.update(kinematic).phase))

    usable = [f for f in frames if f.near_complete and not math.isnan(f.hip_height_norm)]
    if engine.proportions is None or not usable:
        raise RuntimeError("the simulated rep did not calibrate; check the kinematics config")
    return tuple(usable), engine.proportions


def slug(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="default: the configured corpus")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing seed files")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    out_dir = args.out or config.dataset.resolved_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, proportions = record_base(config)
    base_ratio = proportions.femur_to_torso_ratio
    labelled = tuple(replace(f, form_label=FormLabel.OPTIMAL) for f in frames)

    written = 0
    for target in sorted({*DEFAULT_TARGET_RATIOS, round(base_ratio, 2)}):
        destination = out_dir / f"seed_{slug(target)}.json"
        if destination.exists() and not args.force:
            print(f"{destination.name}: exists, skipping (use --force)")
            continue
        warped, _ = warp_frames(labelled, base_ratio, target, proportions.tibia_to_femur_ratio)
        document = build_document(
            frames=warped,
            target_ratio=target,
            tibia_to_femur_ratio=proportions.tibia_to_femur_ratio,
            source_name="simulated_squat",
            source_ratio=base_ratio,
            camera_angle=CAMERA_ANGLE,
            exercise=EXERCISE,
            fps_target=FPS,
        )
        destination.write_text(json.dumps(document, indent=1), encoding="utf-8")
        try:
            load_sequence(destination, config.dataset)
        except DatasetError as exc:
            destination.unlink(missing_ok=True)
            print(f"{destination.name}: failed validation, removed: {exc}", file=sys.stderr)
            return 1
        written += 1

    print(f"wrote {written} file(s) to {out_dir}")
    print("These are simulated, not measured lifters. Verify with scripts/validate_dataset.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
