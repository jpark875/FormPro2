"""Phase 6 tests: web transport and telemetry.

The pipeline itself is faked. What matters here is that one camera feeds many viewers,
that telemetry serialises cleanly, and that the server refuses to run without a corpus.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server as server_module
from formpro.config import AnalyzerConfig, AppConfig, DatasetConfig, KinematicsConfig
from formpro.dataset_loader import load_corpus
from formpro.form_analyzer import FormAnalyzer
from formpro.schema import FormLabel, Phase
from server import BOUNDARY, PipelineWorker, create_app

from .test_form_analyzer import reference_doc


@pytest.fixture
def worker(tmp_path):
    for index, (ratio, offset) in enumerate(((0.85, 0.0), (1.25, 20.0))):
        (tmp_path / f"s{index}.json").write_text(
            json.dumps(reference_doc(ratio, back_offset=offset)), encoding="utf-8"
        )
    corpus = load_corpus(tmp_path, DatasetConfig())
    analyzer = FormAnalyzer(corpus, AnalyzerConfig(), KinematicsConfig())
    made = PipelineWorker(AppConfig(), corpus, analyzer)
    # The pipeline thread is never started here, so no new frames arrive; keep the
    # idle bound short so the stream test closes quickly instead of waiting 10s.
    made.stream_idle_timeout_s = 1.0
    yield made
    made.stop()


@pytest.fixture
def client(worker):
    with TestClient(create_app(worker)) as made:
        yield made


# -- publishing ----------------------------------------------------------------


def test_placeholder_frame_exists_before_the_camera_opens(worker):
    """A viewer connecting during startup must get pixels, not a hang."""
    got = worker.wait_for_frame(0, timeout=1.0)
    assert got is not None
    seq, jpeg = got
    assert seq >= 1
    assert jpeg.startswith(b"\xff\xd8")  # JPEG SOI


def test_publish_keeps_only_the_newest_frame(worker):
    """Drop-old, matching the capture stage: a slow viewer skips, never backlogs."""
    for value in (10, 20, 30):
        worker._publish(np.full((32, 32, 3), value, dtype=np.uint8))
    first = worker.wait_for_frame(0, timeout=1.0)
    second = worker.wait_for_frame(0, timeout=1.0)
    assert first[0] == second[0], "served two different frames from one slot"


def test_wait_for_frame_times_out_without_new_data(worker):
    seq, _ = worker.wait_for_frame(0, timeout=1.0)
    assert worker.wait_for_frame(seq, timeout=0.2) is None


def test_many_viewers_share_one_frame(worker):
    """Two tabs must not mean two cameras."""
    worker._publish(np.zeros((16, 16, 3), dtype=np.uint8))
    results = []

    def viewer():
        results.append(worker.wait_for_frame(0, timeout=1.0))

    threads = [threading.Thread(target=viewer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(r is not None for r in results)
    assert len({r[0] for r in results}) == 1


# -- endpoints -----------------------------------------------------------------


def test_index_serves_the_template(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FormPro2" in response.text
    assert 'id="video"' in response.text


def test_telemetry_is_json_serialisable(client):
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    payload = response.json()
    assert "corpus" in payload
    assert payload["corpus"]["profiles"] >= 1
    assert payload["corpus"]["ratio_low"] == pytest.approx(0.85)
    json.dumps(payload)  # nothing exotic leaked into the response


def test_telemetry_log_is_incremental(client):
    first = client.get("/api/telemetry?since=0").json()
    assert first["log"], "startup produced no log entries"
    highest = first["log_seq"]

    second = client.get(f"/api/telemetry?since={highest}").json()
    assert second["log"] == [], "resent log entries the client already had"


def test_reset_emits_an_event(client):
    before = client.get("/api/telemetry").json()["log_seq"]
    assert client.post("/api/reset").json() == {"ok": True}
    after = client.get(f"/api/telemetry?since={before}").json()
    assert any("reset" in entry["text"] for entry in after["log"])


def test_video_stream_yields_multipart_jpeg(client):
    with client.stream("GET", "/video") as response:
        assert response.status_code == 200
        assert BOUNDARY in response.headers["content-type"]
        chunks = b""
        for chunk in response.iter_bytes():
            chunks += chunk
            if b"\xff\xd8" in chunks and len(chunks) > 512:
                break
    assert f"--{BOUNDARY}".encode() in chunks
    assert b"Content-Type: image/jpeg" in chunks


def test_stalled_pipeline_closes_the_stream_instead_of_spinning(worker):
    """A generator that never yields can never be told the client left."""
    worker.stream_idle_timeout_s = 1.0
    parts = list(server_module.mjpeg_stream(worker))
    # One placeholder frame, then the idle bound ends the response.
    assert len(parts) == 1
    assert b"Content-Type: image/jpeg" in parts[0]


def test_stopping_the_worker_ends_open_streams(worker):
    worker.stop()
    assert list(server_module.mjpeg_stream(worker)) == []


# -- event latching ------------------------------------------------------------


class _Result:
    def __init__(self, findings, band_source=None):
        self.findings = tuple(findings)
        self.band_source = band_source
        self.segment_label = None

    @property
    def ok(self):
        return not self.findings

    @property
    def label(self):
        return FormLabel.KNEE_VALGUS if self.findings else FormLabel.OPTIMAL


class _Finding:
    def __init__(self, label, message):
        self.label = label
        self.message = message
        self.severity = 10.0

    def detail(self):
        return "detail"


def test_a_finding_logs_once_not_once_per_frame(worker):
    """A cue held for a hundred frames must not write a hundred log lines."""
    finding = _Finding(FormLabel.KNEE_VALGUS, "KNEE VALGUS DETECTED")
    for _ in range(50):
        worker._log_findings(_Result([finding]))

    entries = [e for e in worker.snapshot()["log"] if "VALGUS" in e["text"]]
    assert len(entries) == 1


def test_clearing_a_finding_is_logged(worker):
    finding = _Finding(FormLabel.KNEE_VALGUS, "KNEE VALGUS DETECTED")
    worker._log_findings(_Result([finding]))
    worker._log_findings(_Result([]))
    texts = [e["text"] for e in worker.snapshot()["log"]]
    assert any(t.startswith("cleared:") for t in texts)


def test_extrapolation_is_reported_once(worker):
    class _Source:
        extrapolated = True

        def describe(self):
            return "EXTRAPOLATED from 0.85-1.25"

    for _ in range(10):
        worker._log_findings(_Result([], _Source()))
    entries = [e for e in worker.snapshot()["log"] if "extrapolated" in e["text"].lower()]
    assert len(entries) == 1


def test_reset_clears_latched_state(worker):
    finding = _Finding(FormLabel.KNEE_VALGUS, "KNEE VALGUS DETECTED")
    worker._log_findings(_Result([finding]))
    worker.reset()
    assert worker._active_labels == set()

    worker._log_findings(_Result([finding]))
    entries = [e for e in worker.snapshot()["log"] if "VALGUS" in e["text"]]
    assert len(entries) == 2, "post-reset recurrence was swallowed"


# -- startup contract ----------------------------------------------------------


def test_server_refuses_to_start_without_a_corpus(tmp_path, monkeypatch, capsys):
    """Same contract as the desktop app: no corpus means no analysis, not silence."""
    monkeypatch.setattr(
        "sys.argv", ["server.py", "--reference", str(tmp_path / "missing")]
    )
    assert server_module.main() == 2


def test_telemetry_survives_a_frame_with_no_pose(worker):
    """Serialisation must not assume a lifter is present."""
    class _Frame:
        pose = None
        fps = 30.0
        inference_ms = 12.0
        dropped = 3

    payload = worker._telemetry(_Frame(), None, None)
    json.dumps(payload)
    assert payload["tracking"] is False
    assert payload["femur_to_torso_ratio"] is None
    assert payload["phase"] in {p.value for p in Phase}
