"""Phase 2 diagnostic viewer — verifies camera + pose ingestion end to end.

This is NOT the application UI. It draws the tracked-joint wireframe and a telemetry HUD
so you can confirm the ingestion stage works and check that your camera placement keeps
all twelve squat joints visible before any analysis exists to consume them. The real
overlay lands in Phase 6 (``app.py``).

    python scripts/smoke_test_pose.py [--config configs/squat.yaml] [--source 0]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formpro.config import AppConfig  # noqa: E402
from formpro.schema import LM, SQUAT_JOINTS, SQUAT_SKELETON  # noqa: E402
from formpro.video_processor import AnalysisFrame, VideoProcessor  # noqa: E402

GREEN = (0, 220, 80)
AMBER = (0, 190, 255)
RED = (60, 60, 235)
WHITE = (245, 245, 245)


def draw(image, af: AnalysisFrame, min_visibility: float, mirror: bool):
    size = af.frame.size
    pose = af.pose

    if pose is not None:
        for a, b in SQUAT_SKELETON:
            va = pose.is_visible(a, min_visibility)
            vb = pose.is_visible(b, min_visibility)
            colour = GREEN if (va and vb) else AMBER
            cv2.line(image, pose.pixel(a, size), pose.pixel(b, size), colour, 2, cv2.LINE_AA)
        for joint in SQUAT_JOINTS:
            visible = pose.is_visible(joint, min_visibility)
            cv2.circle(image, pose.pixel(joint, size), 5,
                       GREEN if visible else RED, -1, cv2.LINE_AA)

    # Mirror only now that inference is done — see capture.py on why never before.
    if mirror:
        image = cv2.flip(image, 1)

    lines = [
        f"{af.fps:5.1f} fps   inference {af.inference_ms:5.1f} ms   dropped {af.dropped}",
    ]
    if pose is None:
        lines.append("no subject detected")
        status = RED
    else:
        missing = pose.missing(SQUAT_JOINTS, min_visibility)
        if missing:
            lines.append("occluded: " + ", ".join(LM(j).name.lower() for j in missing))
            status = AMBER
        else:
            hip = pose.world_midpoint(LM.LEFT_HIP, LM.RIGHT_HIP)
            ankle = pose.world_midpoint(LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
            lines.append(f"all 12 squat joints tracked   hip height {hip[1] - ankle[1]:+.3f} m")
            status = GREEN

    cv2.rectangle(image, (0, 0), (image.shape[1], 20 + 24 * len(lines)), (25, 25, 25), -1)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (12, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    WHITE if i == 0 else status, 1, cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source", default=None,
                        help="camera index or video file path (overrides config)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="disable display mirroring")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = AppConfig.load(args.config)
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        config = replace(config, camera=replace(config.camera, source=source))

    min_visibility = config.pose.min_visibility
    mirror = not args.no_mirror

    with VideoProcessor(config) as processor:
        for af in processor.stream():
            canvas = draw(af.frame.image.copy(), af, min_visibility, mirror)
            cv2.imshow("FormPro2 - Phase 2 ingestion check", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                processor.reset()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
