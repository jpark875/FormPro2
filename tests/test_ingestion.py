"""Phase 2 tests. No camera and no MediaPipe required — the pose backend is faked."""

from __future__ import annotations

import numpy as np
import pytest

from formpro.config import AppConfig, CameraConfig, PoseConfig, SmoothingConfig
from formpro.filters import OneEuroFilter
from formpro.schema import LM, NUM_LANDMARKS, Frame, PoseFrame


class _FakeLandmark:
    def __init__(self, x, y, z, visibility=1.0, presence=1.0):
        self.x, self.y, self.z = x, y, z
        self.visibility, self.presence = visibility, presence


def _landmarks(fn):
    return [_FakeLandmark(*fn(i)) for i in range(NUM_LANDMARKS)]


# -- schema --------------------------------------------------------------------


def test_world_axes_are_flipped_to_y_up_z_toward_camera():
    """The single most load-bearing conversion in the pipeline."""
    world = _landmarks(lambda i: (1.0, 2.0, 3.0))
    image = _landmarks(lambda i: (0.5, 0.5, 0.0))
    pose = PoseFrame.from_mediapipe(0, 0, world, image)

    np.testing.assert_allclose(pose.world(LM.NOSE), [1.0, -2.0, -3.0])
    # Image space keeps MediaPipe's Y-down orientation for rendering.
    np.testing.assert_allclose(pose.image(LM.NOSE), [0.5, 0.5, 0.0])


def test_pixel_projection_uses_frame_size():
    image = _landmarks(lambda i: (0.25, 0.75, 0.0))
    pose = PoseFrame.from_mediapipe(0, 0, _landmarks(lambda i: (0, 0, 0)), image)
    assert pose.pixel(LM.LEFT_HIP, (1280, 720)) == (320, 540)


def test_missing_reports_low_confidence_joints():
    image = _landmarks(
        lambda i: (0.5, 0.5, 0.0, 0.2 if i == LM.LEFT_KNEE else 0.9, 0.9)
    )
    pose = PoseFrame.from_mediapipe(0, 0, _landmarks(lambda i: (0, 0, 0)), image)

    assert pose.missing([LM.LEFT_KNEE, LM.RIGHT_KNEE], 0.5) == (LM.LEFT_KNEE,)
    assert not pose.is_analysable(0.5)
    assert pose.is_analysable(0.1)


def test_presence_gates_independently_of_visibility():
    """A landmark outside the frame is unusable even if the model calls it visible."""
    image = _landmarks(lambda i: (0.5, 0.5, 0.0, 0.99, 0.1))
    pose = PoseFrame.from_mediapipe(0, 0, _landmarks(lambda i: (0, 0, 0)), image)
    assert not pose.is_visible(LM.LEFT_ANKLE, 0.5)


def test_rejects_wrong_landmark_count():
    with pytest.raises(ValueError):
        PoseFrame.from_mediapipe(0, 0, _landmarks(lambda i: (0, 0, 0))[:10],
                                 _landmarks(lambda i: (0, 0, 0)))


# -- filters -------------------------------------------------------------------


def test_one_euro_passes_first_sample_through():
    f = OneEuroFilter(shape=(2, 3), min_cutoff=1.0, beta=0.0)
    x = np.ones((2, 3), dtype=np.float32)
    np.testing.assert_allclose(f(x, 0.0), x)


def test_one_euro_attenuates_jitter_around_a_static_pose():
    rng = np.random.default_rng(0)
    truth = np.zeros((33, 3), dtype=np.float32)
    f = OneEuroFilter(shape=(33, 3), min_cutoff=1.0, beta=0.0)

    raw_err, filt_err = [], []
    for i in range(120):
        noisy = truth + rng.normal(0, 0.01, (33, 3)).astype(np.float32)
        out = f(noisy, i / 30.0)
        if i > 30:  # let it converge
            raw_err.append(np.abs(noisy).mean())
            filt_err.append(np.abs(out).mean())

    assert np.mean(filt_err) < 0.5 * np.mean(raw_err)


def test_one_euro_tracks_a_ramp_without_unbounded_lag():
    f = OneEuroFilter(shape=(1,), min_cutoff=1.0, beta=0.5)
    for i in range(90):
        out = f(np.array([i * 0.01], dtype=np.float32), i / 30.0)
    assert abs(out[0] - 0.89) < 0.05


def test_one_euro_holds_on_non_increasing_timestamps():
    f = OneEuroFilter(shape=(1,))
    f(np.array([0.0], dtype=np.float32), 1.0)
    first = f(np.array([5.0], dtype=np.float32), 1.5)
    repeat = f(np.array([99.0], dtype=np.float32), 1.5)
    np.testing.assert_allclose(first, repeat)


def test_reset_clears_history():
    f = OneEuroFilter(shape=(1,), min_cutoff=1.0)
    f(np.array([0.0], dtype=np.float32), 0.0)
    f(np.array([0.0], dtype=np.float32), 1.0)
    f.reset()
    np.testing.assert_allclose(f(np.array([7.0], dtype=np.float32), 2.0), [7.0])


# -- config --------------------------------------------------------------------


def test_config_loads_shipped_yaml():
    cfg = AppConfig.load()
    assert cfg.exercise == "barbell_back_squat"
    assert cfg.camera.width == 1280
    assert cfg.pose.smoothing.enabled is True
    assert cfg.pose.model_path().name == "pose_landmarker_heavy.task"


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("camera:\n  widht: 640\n", encoding="utf-8")
    with pytest.raises(ValueError, match="widht"):
        AppConfig.load(path)


def test_config_nested_override(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text("pose:\n  smoothing:\n    beta: 0.9\n", encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.pose.smoothing.beta == 0.9
    assert cfg.pose.smoothing.min_cutoff == SmoothingConfig().min_cutoff


# -- video_processor wiring ----------------------------------------------------


class _FakeBackend:
    """Stands in for BlazePose: returns a pose for every third frame."""

    def __init__(self):
        self.seen: list[int] = []
        self.resets = 0

    def estimate(self, frame: Frame):
        self.seen.append(frame.index)
        if frame.index % 3 == 2:
            return None
        return PoseFrame.from_mediapipe(
            frame.index, frame.timestamp_ms,
            _landmarks(lambda i: (0.0, float(frame.index), 0.0)),
            _landmarks(lambda i: (0.5, 0.5, 0.0)),
        )

    def reset(self):
        self.resets += 1

    def close(self):
        pass


def test_processor_streams_video_file(tmp_path):
    import cv2

    from formpro.video_processor import VideoProcessor

    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    writer.release()

    cfg = AppConfig(
        camera=CameraConfig(source=str(clip), width=160, height=120,
                            warmup_frames=0, read_timeout_s=1.0),
        pose=PoseConfig(smoothing=SmoothingConfig(enabled=False)),
    )
    backend = _FakeBackend()
    with VideoProcessor(cfg, backend=backend) as processor:
        frames = list(processor.stream())

    assert len(frames) >= 10
    assert [f.has_pose for f in frames[:3]] == [True, True, False]
    # Timestamps are monotonic — required by MediaPipe and by phase segmentation.
    stamps = [f.frame.timestamp_ms for f in frames]
    assert stamps == sorted(stamps)
    assert all(f.inference_ms >= 0 for f in frames)


def test_processor_reset_forwards_to_backend():
    from formpro.video_processor import VideoProcessor

    backend = _FakeBackend()
    processor = VideoProcessor(AppConfig(), backend=backend)
    processor.reset()
    assert backend.resets == 1
