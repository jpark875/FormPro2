"""Camera stream to pose stream. pose is None when nobody is in shot."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from .capture import CameraStream
from .config import AppConfig
from .pose_estimator import BlazePoseEstimator, PoseBackend
from .schema import Frame, PoseFrame

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisFrame:
    """One camera frame plus its pose (if a person was found) and timing telemetry."""

    frame: Frame
    pose: PoseFrame | None
    inference_ms: float
    fps: float
    dropped: int

    @property
    def has_pose(self) -> bool:
        return self.pose is not None


class VideoProcessor:
    """Ties capture and pose estimation into a single iterable stream."""

    def __init__(self, config: AppConfig, backend: PoseBackend | None = None) -> None:
        self.config = config
        self._camera = CameraStream(config.camera)
        self._backend = backend
        self._owns_backend = backend is None
        self._fps = 0.0
        self._last_emit_s: float | None = None
        self._started = False

    def start(self) -> VideoProcessor:
        if self._started:
            raise RuntimeError("processor already started")
        # Load the model before opening the camera.
        if self._backend is None:
            self._backend = BlazePoseEstimator(self.config.pose)
        self._camera.start()
        self._started = True
        return self

    def stop(self) -> None:
        self._camera.stop()
        if self._owns_backend and self._backend is not None:
            self._backend.close()
            self._backend = None
        self._started = False

    def __enter__(self) -> VideoProcessor:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def stream(self) -> Iterator[AnalysisFrame]:
        if not self._started:
            raise RuntimeError("call start() (or use as a context manager) first")
        assert self._backend is not None

        for frame in self._camera.frames():
            t0 = time.perf_counter()
            pose = self._backend.estimate(frame)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            yield AnalysisFrame(
                frame=frame,
                pose=pose,
                inference_ms=inference_ms,
                fps=self._tick(),
                dropped=self._camera.dropped,
            )

    def reset(self) -> None:
        """Clear tracker state, e.g. at the start of a new set."""
        if self._backend is not None:
            self._backend.reset()

    def _tick(self) -> float:
        """End-to-end throughput, EMA-smoothed so the HUD readout is stable."""
        now = time.perf_counter()
        if self._last_emit_s is not None:
            dt = now - self._last_emit_s
            if dt > 0:
                instant = 1.0 / dt
                self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant
        self._last_emit_s = now
        return self._fps
