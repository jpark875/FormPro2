"""FormPro2 main loop: real-time barbell back squat form analysis.

    python app.py [--config configs/squat.yaml] [--reference data/reference]

Keys: q or Esc to quit, r to reset the session (new lifter or new set).

The loop is deliberately thin. Every stage is constructed once, then each frame walks the
pipeline in order: capture and pose (Phase 2), normalization (Phase 3), rep segmentation,
comparison against the corpus (Phase 5), render (Phase 6). Anything more interesting than
plumbing belongs in the module that owns it.

The corpus is loaded before the camera opens. Phase 5 derives every bound it applies from
reference data and has no baseline to fall back on, so a missing or malformed corpus is a
startup failure, not a degraded mode. Starting anyway would produce an application that
looks like it is working and silently passes every rep.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from formpro import overlay  # noqa: E402
from formpro.config import AppConfig  # noqa: E402
from formpro.dataset_loader import DatasetError, load_corpus  # noqa: E402
from formpro.form_analyzer import AnalyzerError, FormAnalyzer  # noqa: E402
from formpro.kinematics import KinematicsEngine  # noqa: E402
from formpro.phases import PhaseSegmenter  # noqa: E402
from formpro.schema import Phase  # noqa: E402
from formpro.video_processor import VideoProcessor  # noqa: E402

log = logging.getLogger("formpro.app")

WINDOW = "FormPro2 - barbell back squat"


class Session:
    """Per-lifter state: everything reset by pressing 'r'."""

    def __init__(self, config: AppConfig, analyzer: FormAnalyzer) -> None:
        self.config = config
        self.analyzer = analyzer
        self.kinematics = KinematicsEngine(config.kinematics, config.pose.min_visibility)
        self.segmenter = PhaseSegmenter(config.phases)
        self.phase = Phase.SETUP
        self.rep_count = 0

    def reset(self) -> None:
        self.kinematics.reset()
        self.segmenter.reset()
        self.analyzer.reset()
        self.phase = Phase.SETUP
        self.rep_count = 0
        log.info("session reset")


def run(config: AppConfig, reference_root: Path | None, mirror: bool) -> int:
    try:
        corpus = load_corpus(reference_root, config.dataset)
    except DatasetError as exc:
        log.error("cannot load the reference corpus: %s", exc)
        log.error(
            "Phase 5 derives every threshold from this corpus and has no fallback. "
            "Check the directory, or run: python scripts/validate_dataset.py"
        )
        return 2

    try:
        analyzer = FormAnalyzer(corpus, config.analyzer, config.kinematics)
    except AnalyzerError as exc:
        log.error("cannot build the analyzer: %s", exc)
        return 2

    log.info("corpus: %s", corpus.coverage_report())
    log.info("threshold profiles at ratios: %s",
             ", ".join(f"{r:.2f}" for r in analyzer.model.ratios))

    session = Session(config, analyzer)
    min_visibility = config.pose.min_visibility

    with VideoProcessor(config) as processor:
        for analysis_frame in processor.stream():
            pose = analysis_frame.pose
            kinematic = None
            result = None

            if pose is None:
                # A dropout must break the rep rather than be stitched over; the
                # segmenter also disarms so the next rep starts from a seen standing
                # position rather than mid-descent.
                session.segmenter.on_tracking_lost()
                session.phase = Phase.SETUP
            else:
                kinematic = session.kinematics.update(pose)
                if kinematic is not None:
                    segmentation = session.segmenter.update(kinematic)
                    session.phase = segmentation.phase
                    session.rep_count = session.segmenter.rep_count
                    kinematic = kinematic.with_phase(segmentation.phase)

                    proportions = session.kinematics.proportions
                    if proportions is not None:
                        result = analyzer.update(
                            kinematic, proportions.femur_to_torso_ratio
                        )

                    if segmentation.completed_rep is not None:
                        rep = segmentation.completed_rep
                        log.info(
                            "rep %d: %d ms, depth %.2f, verdict %s",
                            rep.index, rep.duration_ms, rep.min_hip_height_norm,
                            result.label.value if result else "unknown",
                        )

            canvas = overlay.compose(
                analysis_frame.frame.image.copy(),
                pose=pose,
                near_side=kinematic.near_side if kinematic else None,
                min_visibility=min_visibility,
                mirror=mirror,
                proportions=session.kinematics.proportions,
                calibration_progress=session.kinematics.calibration_progress,
                kinematic=kinematic,
                analysis=result,
                phase=session.phase,
                rep_count=session.rep_count,
                fps=analysis_frame.fps,
                inference_ms=analysis_frame.inference_ms,
                dropped=analysis_frame.dropped,
                tracking=pose is not None,
            )

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                session.reset()
                processor.reset()

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time barbell back squat form analysis. "
                    "Keys: q or Esc to quit, r to reset the session.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None,
                        help="reference corpus directory (default: from config)")
    parser.add_argument("--source", default=None,
                        help="camera index or video file path (overrides config)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="disable display mirroring")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig.load(args.config)
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        config = replace(config, camera=replace(config.camera, source=source))

    return run(config, args.reference, mirror=not args.no_mirror)


if __name__ == "__main__":
    raise SystemExit(main())
