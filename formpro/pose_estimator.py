"""MediaPipe BlazePose backend.

3D world landmarks are required rather than merely convenient: a lateral 2D view cannot
see knee valgus, and metric hip-centred coordinates are what let segment lengths survive
the lifter moving toward or away from the camera.

Sits behind the PoseBackend Protocol, so another model can replace it provided it emits
the same PoseFrame contract.

Runs in VIDEO mode rather than LIVE_STREAM. The async callback would decouple result from
frame and force timestamp re-association, and capture already bounds latency, so every
PoseFrame stays unambiguously paired with the image it came from.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .config import PoseConfig
from .filters import OneEuroFilter
from .schema import NUM_LANDMARKS, Frame, PoseFrame

log = logging.getLogger(__name__)

MODEL_VARIANTS = ("lite", "full", "heavy")


@runtime_checkable
class PoseBackend(Protocol):
    """Anything that turns a camera frame into a ``PoseFrame``."""

    def estimate(self, frame: Frame) -> PoseFrame | None:
        """Return the pose for this frame, or ``None`` if no person was detected."""
        ...

    def reset(self) -> None:
        """Drop tracker/filter state (new session, or subject left the frame)."""
        ...

    def close(self) -> None:
        ...


class BlazePoseEstimator:
    """MediaPipe ``PoseLandmarker`` wrapped to emit ``PoseFrame`` objects."""

    def __init__(self, config: PoseConfig, model_path: Path | None = None) -> None:
        # Imported lazily so that schema/kinematics tests do not pay MediaPipe's
        # multi-second import cost or require the native library at all.
        from mediapipe import Image, ImageFormat
        from mediapipe.tasks.python import BaseOptions, vision

        self._Image = Image
        self._ImageFormat = ImageFormat

        if config.variant not in MODEL_VARIANTS:
            raise ValueError(
                f"unknown pose variant {config.variant!r}; expected one of {MODEL_VARIANTS}"
            )

        path = Path(model_path) if model_path is not None else config.model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"pose model not found at {path}. Download it with:\n"
                f"    python scripts/fetch_model.py --variant {config.variant}"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,  # single lifter; multi-person adds cost and ambiguity here
            min_pose_detection_confidence=config.min_detection_confidence,
            min_pose_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.config = config
        log.info("loaded pose model: %s", path.name)

        self._filter: OneEuroFilter | None = None
        if config.smoothing.enabled:
            self._filter = OneEuroFilter(
                shape=(NUM_LANDMARKS, 3),
                min_cutoff=config.smoothing.min_cutoff,
                beta=config.smoothing.beta,
                d_cutoff=config.smoothing.d_cutoff,
            )

        self._last_timestamp_ms = -1
        self._closed = False

    # -- PoseBackend -----------------------------------------------------------

    def estimate(self, frame: Frame) -> PoseFrame | None:
        if self._closed:
            raise RuntimeError("estimator is closed")

        # detect_for_video requires strictly increasing timestamps; two frames can share
        # a millisecond at high capture rates, so nudge rather than let MediaPipe raise.
        timestamp_ms = max(frame.timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or not result.pose_world_landmarks:
            # Subject left the frame: clear filter history so the skeleton does not
            # interpolate across the gap when they return.
            if self._filter is not None:
                self._filter.reset()
            return None

        pose = PoseFrame.from_mediapipe(
            index=frame.index,
            timestamp_ms=frame.timestamp_ms,
            world_landmarks=result.pose_world_landmarks[0],
            image_landmarks=result.pose_landmarks[0],
        )

        if self._filter is not None:
            # Smooth world space only. Image space is for drawing, where the raw
            # landmarks track the video more faithfully.
            smoothed = self._filter(pose.world_xyz, frame.timestamp_ms / 1000.0)
            pose = pose.with_world(smoothed.astype(np.float32, copy=False))

        return pose

    def reset(self) -> None:
        if self._filter is not None:
            self._filter.reset()

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> BlazePoseEstimator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
