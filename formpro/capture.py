"""Threaded camera capture with drop-old buffering.

Why a thread and not a plain ``cap.read()`` loop: OpenCV's ``VideoCapture`` holds an
internal FIFO. If inference takes longer than the frame interval — which it will on the
``heavy`` pose model — a synchronous read loop drains that FIFO in order and the analysed
frame falls further and further behind the lifter. Corrective feedback delivered 800 ms
late is worse than no feedback, because the user has already left the position being
critiqued.

So the grab loop runs in its own thread and keeps a **one-slot** buffer: a new frame
overwrites an unconsumed one and increments a drop counter. Latency stays bounded at one
frame interval regardless of inference speed, and the drop count is surfaced so a machine
that is too slow to run this model reports that fact instead of quietly lagging.

This stage does **not** mirror the image. Flipping before inference would swap the lifter's
anatomical left and right, inverting per-side findings such as knee valgus. Display
mirroring is a render-time concern (Phase 6).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator

import cv2

from .config import CameraConfig
from .schema import Frame

log = logging.getLogger(__name__)

_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "ffmpeg": cv2.CAP_FFMPEG,
}


class CameraError(RuntimeError):
    """The camera could not be opened or died mid-stream."""


class CameraStream:
    """Background reader over a webcam (or a video file, for replay).

    Usage::

        with CameraStream(cfg) as cam:
            for frame in cam.frames():
                ...
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

        self._start_ns = 0
        self._grabbed = 0
        self._dropped = 0

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> CameraStream:
        if self._thread is not None:
            raise RuntimeError("stream already started")

        backend = _BACKENDS.get(self.config.backend)
        if backend is None:
            raise ValueError(
                f"unknown camera backend {self.config.backend!r}; "
                f"expected one of {sorted(_BACKENDS)}"
            )

        cap = cv2.VideoCapture(self.config.source, backend)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"could not open camera source {self.config.source!r} "
                f"(backend={self.config.backend}). Check that no other application holds "
                f"the device and that camera access is enabled in system privacy settings."
            )

        # Requests, not guarantees — the driver picks the nearest supported mode.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # honoured by some backends, harmless elsewhere

        self._cap = cap
        self._drain_warmup(cap)

        self._start_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._run, name="camera-grab", daemon=True)
        self._thread.start()

        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        log.info("camera open: %s @ %s fps (requested %sx%s)", actual, cap.get(cv2.CAP_PROP_FPS),
                 self.config.width, self.config.height)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> CameraStream:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- reading ---------------------------------------------------------------

    def read(self, timeout: float | None = None) -> Frame | None:
        """Return the most recent frame, or ``None`` if none arrived within the timeout."""
        if self._error is not None:
            raise CameraError("camera thread failed") from self._error
        timeout = self.config.read_timeout_s if timeout is None else timeout
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise CameraError("camera thread failed") from self._error
            return None

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source ends or ``stop()`` is called."""
        while not self._stop.is_set():
            frame = self.read()
            if frame is None:
                if self._stop.is_set():
                    break
                log.warning("no frame within %.1fs", self.config.read_timeout_s)
                continue
            yield frame

    # -- stats -----------------------------------------------------------------

    @property
    def grabbed(self) -> int:
        """Frames pulled off the sensor."""
        return self._grabbed

    @property
    def dropped(self) -> int:
        """Frames discarded because the consumer was still busy.

        A steadily rising count means inference is slower than the capture rate — expected
        and healthy on the heavy model; the pipeline stays live rather than falling behind.
        """
        return self._dropped

    # -- internals -------------------------------------------------------------

    def _playback_interval(self, cap: cv2.VideoCapture) -> float:
        """Seconds to wait between frames, or 0 to read as fast as the source allows.

        Only file sources are paced. A camera already delivers at its own rate, and
        adding a sleep there would fight the driver.
        """
        if isinstance(self.config.source, int) or not self.config.pace_file_playback:
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 1000:
            log.warning("file source reports no usable frame rate; replaying at full speed")
            return 0.0
        log.info("pacing file playback at %.1f fps", fps)
        return 1.0 / float(fps)

    def _drain_warmup(self, cap: cv2.VideoCapture) -> None:
        for _ in range(max(0, self.config.warmup_frames)):
            if not cap.read()[0]:
                break

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None
        index = 0
        interval = self._playback_interval(cap)
        due = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    log.info("capture source ended")
                    break
                timestamp_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
                frame = Frame(index=index, timestamp_ms=int(timestamp_ms), image=image)
                index += 1
                self._grabbed += 1
                self._publish(frame)

                if interval:
                    due += interval
                    delay = due - time.monotonic()
                    if delay > 0:
                        # Event.wait, not sleep, so stop() is still responsive.
                        self._stop.wait(delay)
                    else:
                        # Decoding fell behind real time; resync rather than
                        # accumulating an ever-growing debt.
                        due = time.monotonic()
        except BaseException as exc:  # surfaced to the consumer via read()
            self._error = exc
            log.exception("camera thread crashed")
        finally:
            self._stop.set()

    def _publish(self, frame: Frame) -> None:
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        # Consumer is still busy: evict the stale frame and publish the fresh one.
        try:
            self._queue.get_nowait()
            self._dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1
