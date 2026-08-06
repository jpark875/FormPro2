"""FormPro2 web prototype: MJPEG video plus a JSON telemetry feed.

    python server.py [--host 0.0.0.0] [--port 8000] [--reference data/reference]

Then open http://127.0.0.1:8000.

Why a worker thread rather than analysing inside the request
------------------------------------------------------------
There is one camera and one pipeline, but potentially several viewers. The pipeline
therefore runs in its own thread and publishes the latest annotated frame; HTTP handlers
only read that. Running inference per request would mean two browser tabs competing for
one camera, and a page refresh restarting a lifter's calibration mid-set.

The publish slot holds exactly one frame, the same drop-old policy the capture stage
uses. A viewer on a slow connection falls behind by dropping frames rather than by
accumulating a backlog of stale ones, so what it shows stays current.

Split of responsibilities with the desktop UI
---------------------------------------------
The MJPEG stream carries the skeleton only. Telemetry lives in the HTML panel, where it
is selectable, readable at any size, and does not consume video bandwidth to redraw text
that has not changed. ``overlay.draw_skeleton`` is shared with the desktop path; the HUD
drawing is not used here.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from formpro import overlay
from formpro.config import AppConfig
from formpro.dataset_loader import DatasetError, ReferenceCorpus, load_corpus
from formpro.form_analyzer import AnalyzerError, FormAnalyzer
from formpro.kinematics import SIDE_ANGLE_FIELDS, KinematicsEngine
from formpro.phases import PhaseSegmenter
from formpro.schema import Phase
from formpro.video_processor import VideoProcessor

log = logging.getLogger("formpro.server")

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "index.html"
BOUNDARY = "formprofr"
LOG_CAPACITY = 200


def _placeholder(text: str, width: int = 960, height: int = 540) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class PipelineWorker(threading.Thread):
    """Runs the full pipeline and publishes the latest frame and telemetry."""

    def __init__(
        self,
        config: AppConfig,
        corpus: ReferenceCorpus,
        analyzer: FormAnalyzer,
        mirror: bool = True,
        jpeg_quality: int = 80,
    ) -> None:
        super().__init__(name="formpro-pipeline", daemon=True)
        self.config = config
        self.corpus = corpus
        self.analyzer = analyzer
        self.mirror = mirror
        self.jpeg_quality = int(jpeg_quality)

        self.kinematics = KinematicsEngine(config.kinematics, config.pose.min_visibility)
        self.segmenter = PhaseSegmenter(config.phases)

        self._condition = threading.Condition()
        self._frame_seq = 0
        self._jpeg: bytes | None = None
        self._state: dict[str, Any] = {
            "connected": False, "tracking": False, "error": None,
        }
        self._log: deque[dict[str, Any]] = deque(maxlen=LOG_CAPACITY)
        self._log_seq = 0
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        #: Seconds a viewer's stream waits on a stalled pipeline before closing.
        self.stream_idle_timeout_s = 10.0

        # Latched so an event is logged on transition, not once per frame.
        self._active_labels: set[str] = set()
        self._was_tracking = True
        self._was_calibrated = False
        self._warned_extrapolation = False

        self._publish(_placeholder("starting camera"))
        self._emit("info", "server started")

    # -- publishing ------------------------------------------------------------

    def _publish(self, canvas: np.ndarray) -> None:
        ok, buffer = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        with self._condition:
            self._frame_seq += 1
            self._jpeg = buffer.tobytes()
            self._condition.notify_all()

    def wait_for_frame(self, since: int, timeout: float = 2.0) -> tuple[int, bytes] | None:
        """Block until a frame newer than ``since`` exists."""
        with self._condition:
            if self._frame_seq <= since:
                self._condition.wait(timeout)
            if self._jpeg is None or self._frame_seq <= since:
                return None
            return self._frame_seq, self._jpeg

    def _emit(self, level: str, text: str) -> None:
        self._log_seq += 1
        self._log.appendleft({
            "seq": self._log_seq,
            "t": round(time.monotonic() - self._started_at, 1),
            "level": level,
            "text": text,
        })

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._condition:
            state = dict(self._state)
            entries = [e for e in self._log if e["seq"] > since]
        low, high = self.corpus.ratio_span
        state["log"] = entries
        state["log_seq"] = self._log_seq
        state["corpus"] = {
            "sequences": len(self.corpus),
            "ratio_low": round(low, 2),
            "ratio_high": round(high, 2),
            "profiles": len(self.analyzer.model.ratios),
        }
        return state

    def reset(self) -> None:
        self.kinematics.reset()
        self.segmenter.reset()
        self.analyzer.reset()
        self._active_labels.clear()
        self._was_calibrated = False
        self._warned_extrapolation = False
        self._emit("info", "session reset")

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            # Wake any viewer blocked on a frame so streams end promptly on shutdown.
            self._condition.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # -- main loop -------------------------------------------------------------

    def run(self) -> None:
        try:
            with VideoProcessor(self.config) as processor:
                with self._condition:
                    self._state["connected"] = True
                self._emit("info", "camera connected")
                for analysis_frame in processor.stream():
                    if self._stop.is_set():
                        break
                    self._process(analysis_frame)
        except Exception as exc:  # surfaced in the UI rather than dying silently
            log.exception("pipeline stopped")
            with self._condition:
                self._state["connected"] = False
                self._state["error"] = str(exc)
            self._emit("error", f"pipeline stopped: {exc}")
            self._publish(_placeholder(f"pipeline error: {exc}"))

    def _process(self, analysis_frame) -> None:
        pose = analysis_frame.pose
        kinematic = None
        result = None

        if pose is None:
            self.segmenter.on_tracking_lost()
            if self._was_tracking:
                self._emit("warn", "tracking lost")
                self._active_labels.clear()
            self._was_tracking = False
        else:
            if not self._was_tracking:
                self._emit("info", "tracking regained")
            self._was_tracking = True

            kinematic = self.kinematics.update(pose)
            if kinematic is not None:
                segmentation = self.segmenter.update(kinematic)
                kinematic = kinematic.with_phase(segmentation.phase)

                proportions = self.kinematics.proportions
                if proportions is not None:
                    if not self._was_calibrated:
                        self._was_calibrated = True
                        self._emit(
                            "info",
                            f"calibrated: femur/torso {proportions.femur_to_torso_ratio:.2f}",
                        )
                    result = self.analyzer.update(
                        kinematic, proportions.femur_to_torso_ratio
                    )
                    self._log_findings(result)

                if segmentation.completed_rep is not None:
                    rep = segmentation.completed_rep
                    verdict = result.label.value if result else "unknown"
                    self._emit(
                        "ok" if verdict == "optimal_form" else "error",
                        f"rep {rep.index} complete: {rep.duration_ms} ms, "
                        f"depth {rep.min_hip_height_norm:.2f}, {verdict}",
                    )

        canvas = analysis_frame.frame.image.copy()
        if pose is not None:
            overlay.draw_skeleton(
                canvas, pose,
                kinematic.near_side if kinematic else None,
                self.config.pose.min_visibility,
            )
        if self.mirror:
            canvas = cv2.flip(canvas, 1)
        self._publish(canvas)

        with self._condition:
            self._state.update(self._telemetry(analysis_frame, kinematic, result))

    def _log_findings(self, result) -> None:
        current = {f.label.value: f for f in result.findings}
        for label, finding in current.items():
            if label not in self._active_labels:
                self._emit("error", f"{finding.message} ({finding.detail()})")
        for label in self._active_labels - set(current):
            self._emit("ok", f"cleared: {label}")
        self._active_labels = set(current)

        source = result.band_source
        if source is not None and source.extrapolated and not self._warned_extrapolation:
            self._warned_extrapolation = True
            self._emit("warn", f"thresholds {source.describe().lower()}")

    def _telemetry(self, analysis_frame, kinematic, result) -> dict[str, Any]:
        proportions = self.kinematics.proportions
        phase = kinematic.phase if kinematic else self.segmenter.phase
        angles = {}
        if kinematic is not None:
            for field in SIDE_ANGLE_FIELDS:
                value = getattr(kinematic.camera_near, field)
                angles[field] = None if value != value else round(float(value), 1)
            ratio = kinematic.global_metrics.knee_to_hip_width_ratio
            angles["knee_to_hip_width_ratio"] = (
                None if ratio != ratio else round(float(ratio), 3)
            )

        findings = [
            {
                "label": f.label.value,
                "message": f.message,
                "detail": f.detail(),
                "severity": round(f.severity, 1),
            }
            for f in (result.findings if result else ())
        ]

        depth = kinematic.hip_height_norm if kinematic else float("nan")
        return {
            "connected": True,
            "error": None,
            "tracking": analysis_frame.pose is not None,
            "calibrated": proportions is not None,
            "calibration_progress": round(self.kinematics.calibration_progress, 3),
            "femur_to_torso_ratio": (
                round(proportions.femur_to_torso_ratio, 3) if proportions else None
            ),
            "tibia_to_femur_ratio": (
                round(proportions.tibia_to_femur_ratio, 3) if proportions else None
            ),
            "phase": (phase or Phase.SETUP).value,
            "rep_count": self.segmenter.rep_count,
            "depth": None if depth != depth else round(float(depth), 3),
            "near_side": (
                kinematic.near_side.name.lower()
                if kinematic and kinematic.near_side else None
            ),
            "angles": angles,
            "findings": findings,
            "status": "optimal" if (result and result.ok) else (
                "error" if findings else "idle"
            ),
            "band_source": (
                {
                    "text": result.band_source.describe(),
                    "extrapolated": result.band_source.extrapolated,
                }
                if result and result.band_source else None
            ),
            "fps": round(analysis_frame.fps, 1),
            "inference_ms": round(analysis_frame.inference_ms, 1),
            "dropped": analysis_frame.dropped,
            "segment_label": (
                result.segment_label.value if result and result.segment_label else None
            ),
        }


def mjpeg_stream(worker: PipelineWorker) -> Iterator[bytes]:
    """Yield multipart JPEG parts until the pipeline stalls or the client leaves.

    The idle bound is load-bearing rather than defensive. A generator that loops without
    ever yielding cannot receive ``GeneratorExit``, so if the pipeline stops producing
    frames, every disconnected viewer would leave a thread spinning forever and the
    server would slowly consume its threadpool. Ending the response instead lets the
    page's ``onerror`` handler reconnect, which is also the behaviour a viewer wants
    when the camera comes back.
    """
    last = 0
    idle = 0.0
    while not worker.stopped:
        got = worker.wait_for_frame(last, timeout=1.0)
        if got is None:
            idle += 1.0
            if idle >= worker.stream_idle_timeout_s:
                log.info("closing MJPEG stream: no frames for %.0fs", idle)
                return
            continue
        idle = 0.0
        last, jpeg = got
        yield (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode() + jpeg + b"\r\n"


def create_app(worker: PipelineWorker) -> FastAPI:
    app = FastAPI(title="FormPro2", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(TEMPLATE, media_type="text/html")

    @app.get("/video")
    def video() -> StreamingResponse:
        return StreamingResponse(
            mjpeg_stream(worker),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/telemetry")
    def telemetry(since: int = 0) -> JSONResponse:
        return JSONResponse(worker.snapshot(since))

    @app.post("/api/reset")
    def reset() -> JSONResponse:
        worker.reset()
        return JSONResponse({"ok": True})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FormPro2 web prototype: MJPEG video and a telemetry panel.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--source", default=None,
                        help="camera index or video file path (overrides config)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality, 1-100")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.load(args.config)
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        config = replace(config, camera=replace(config.camera, source=source))

    try:
        corpus = load_corpus(args.reference, config.dataset)
        analyzer = FormAnalyzer(corpus, config.analyzer, config.kinematics)
    except (DatasetError, AnalyzerError) as exc:
        log.error("%s", exc)
        log.error(
            "Every threshold is derived from the corpus and there is no fallback. "
            "Generate one with: python scripts/dataset_generator.py <source.json>"
        )
        return 2

    log.info("corpus: %s", corpus.coverage_report())

    import uvicorn

    worker = PipelineWorker(
        config, corpus, analyzer, mirror=not args.no_mirror, jpeg_quality=args.quality
    )
    worker.start()
    try:
        uvicorn.run(create_app(worker), host=args.host, port=args.port, log_level="warning")
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
