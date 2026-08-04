"""Phase 4 tests: reference ingestion, validation strictness and corpus indexing."""

from __future__ import annotations

import copy
import json

import pytest

from formpro.config import DatasetConfig
from formpro.dataset_loader import (
    DatasetError,
    ReferenceCorpus,
    load_corpus,
    load_sequence,
)
from formpro.kinematics import GlobalMetrics, KinematicFrame, SideAngles
from formpro.schema import FormLabel, Phase


def reference_doc(
    *,
    ratio: float = 1.12,
    dataset_type: str = "reference_optimal",
    camera_angle: str = "45_oblique_anterior",
    frames: int = 12,
) -> dict:
    return {
        "metadata": {
            "exercise": "barbell_back_squat",
            "camera_angle": camera_angle,
            "dataset_type": dataset_type,
            "fps_target": 30,
        },
        "subject_proportions": {
            "femur_to_torso_ratio": ratio,
            "tibia_to_femur_ratio": 0.85,
        },
        "frames": [
            {
                "frame_id": 100 + i,
                "timestamp_ms": 3000 + i * 33,
                "phase": "eccentric" if i < frames // 2 else "concentric",
                "angles": {
                    "camera_near": {
                        "hip_flexion": 170.0 - i * 4,
                        "knee_flexion": 168.0 - i * 4,
                        "ankle_dorsiflexion": 90.0 - i,
                        "back_to_vertical": 10.0 + i * 2,
                    },
                    "camera_far": {
                        "hip_flexion": 171.0 - i * 4,
                        "knee_flexion": 167.0 - i * 4,
                        "ankle_dorsiflexion": 90.5 - i,
                        "back_to_vertical": 10.5 + i * 2,
                    },
                    "global": {"knee_to_hip_width_ratio": 1.05 - i * 0.01},
                },
                "form_label": "optimal_form" if i < frames // 2 else "error_good_morning",
            }
            for i in range(frames)
        ],
    }


def write(tmp_path, doc, name="ref.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


CONFIG = DatasetConfig()


# -- happy path ----------------------------------------------------------------


def test_loads_a_valid_sequence(tmp_path):
    sequence = load_sequence(write(tmp_path, reference_doc()), CONFIG)

    assert len(sequence) == 12
    assert sequence.metadata.exercise == "barbell_back_squat"
    assert sequence.metadata.fps_target == 30
    assert sequence.femur_to_torso_ratio == pytest.approx(1.12)
    assert sequence.features.shape == (12, 9)
    assert sequence.duration_ms == 11 * 33


def test_parsed_frames_are_the_same_type_the_live_engine_emits(tmp_path):
    """The invariant that removes any translation layer between live and reference."""
    sequence = load_sequence(write(tmp_path, reference_doc()), CONFIG)
    frame = sequence.frames[0]
    assert isinstance(frame, KinematicFrame)
    assert isinstance(frame.camera_near, SideAngles)
    assert isinstance(frame.global_metrics, GlobalMetrics)
    assert frame.phase is Phase.ECCENTRIC
    assert frame.form_label is FormLabel.OPTIMAL


def test_live_frame_round_trips_through_the_reference_schema(tmp_path):
    """A frame the engine produces must serialize into a file the loader accepts."""
    live = KinematicFrame(
        frame_id=142, timestamp_ms=4733,
        camera_near=SideAngles(110.5, 125.0, 85.2, 45.1),
        camera_far=SideAngles(111.0, 124.5, 86.0, 45.3),
        global_metrics=GlobalMetrics(0.95),
        phase=Phase.CONCENTRIC, form_label=FormLabel.GOOD_MORNING,
    )
    doc = reference_doc(frames=1)
    doc["frames"] = [live.to_json_frame()]

    parsed = load_sequence(write(tmp_path, doc), CONFIG).frames[0]
    assert parsed.frame_id == live.frame_id
    assert parsed.timestamp_ms == live.timestamp_ms
    assert parsed.camera_near == live.camera_near
    assert parsed.phase is Phase.CONCENTRIC
    assert parsed.form_label is FormLabel.GOOD_MORNING


def test_mixed_labels_within_one_file_are_supported(tmp_path):
    """A good-morning rep is clean on the way down and breaks on the way up."""
    sequence = load_sequence(write(tmp_path, reference_doc()), CONFIG)
    transitions = sequence.label_transitions()
    assert [label for _, label in transitions] == [
        FormLabel.OPTIMAL, FormLabel.GOOD_MORNING
    ]
    assert transitions[1][0] == 6


def test_phase_and_label_slicing(tmp_path):
    sequence = load_sequence(write(tmp_path, reference_doc()), CONFIG)
    assert len(sequence.slice_phase(Phase.ECCENTRIC)) == 6
    assert len(sequence.slice_phase(Phase.CONCENTRIC)) == 6
    assert len(sequence.slice_phase(Phase.BOTTOM)) == 0
    assert len(sequence.slice_label(FormLabel.GOOD_MORNING)) == 6


def test_legacy_camera_angle_still_loads(tmp_path, caplog):
    sequence = load_sequence(
        write(tmp_path, reference_doc(camera_angle="45_oblique")), CONFIG
    )
    assert sequence.metadata.camera_angle == "45_oblique"
    assert "legacy alias" in caplog.text


# -- validation strictness -----------------------------------------------------


def _near(doc: dict) -> dict:
    return doc["frames"][3]["angles"]["camera_near"]


def _global(doc: dict) -> dict:
    return doc["frames"][3]["angles"]["global"]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d["metadata"].__setitem__("camera_angle", "lateral"), "camera_angle"),
        (lambda d: d["metadata"].__setitem__("exercise", "front_squat"), "exercise"),
        (lambda d: d["metadata"].pop("dataset_type"), "dataset_type"),
        (lambda d: d.pop("subject_proportions"), "subject_proportions"),
        (lambda d: d["subject_proportions"].pop("tibia_to_femur_ratio"), "tibia_to_femur"),
        (lambda d: d["frames"][3].__setitem__("phase", "descending"), "phase"),
        (lambda d: d["frames"][3].__setitem__("form_label", "sloppy"), "form_label"),
        (lambda d: d["frames"][3].pop("form_label"), "form_label"),
        (lambda d: d["frames"][3].pop("timestamp_ms"), "timestamp_ms"),
        (lambda d: d["frames"][3]["angles"].pop("camera_far"), "camera_far"),
        (lambda d: _near(d).pop("hip_flexion"), "hip_flexion"),
        (lambda d: _near(d).update(knee_flexion=200.0), "0-180"),
        (lambda d: _near(d).update(knee_flexion=-5.0), "0-180"),
        (lambda d: _global(d).update(knee_to_hip_width_ratio=0.0), "implausible"),
        (lambda d: d["frames"].clear(), "non-empty"),
    ],
)
def test_malformed_files_are_rejected(tmp_path, mutate, expected):
    doc = reference_doc()
    mutate(doc)
    with pytest.raises(DatasetError, match=expected):
        load_sequence(write(tmp_path, doc), CONFIG)


def test_non_monotonic_timestamps_are_rejected(tmp_path):
    doc = reference_doc()
    doc["frames"][5]["timestamp_ms"] = doc["frames"][4]["timestamp_ms"]
    with pytest.raises(DatasetError, match="not greater than"):
        load_sequence(write(tmp_path, doc), CONFIG)


def test_oversized_timestamp_gap_is_rejected(tmp_path):
    doc = reference_doc()
    for frame in doc["frames"][6:]:
        frame["timestamp_ms"] += 5000
    with pytest.raises(DatasetError, match="gap exceeds"):
        load_sequence(write(tmp_path, doc), CONFIG)


def test_invalid_json_is_reported_with_the_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_sequence(path, CONFIG)


def test_string_where_a_number_belongs_is_rejected(tmp_path):
    doc = reference_doc()
    doc["frames"][2]["angles"]["camera_near"]["hip_flexion"] = "170"
    with pytest.raises(DatasetError, match="must be a number"):
        load_sequence(write(tmp_path, doc), CONFIG)


# -- corpus --------------------------------------------------------------------


def test_corpus_loads_a_directory(tmp_path):
    for i, ratio in enumerate((0.88, 1.05, 1.24)):
        write(tmp_path, reference_doc(ratio=ratio), f"subject_{i}.json")
    corpus = load_corpus(tmp_path, CONFIG)

    assert len(corpus) == 3
    assert corpus.ratio_span == (0.88, 1.24)
    assert "3 sequences" in corpus.coverage_report()


def test_corpus_brackets_a_live_ratio_for_interpolation(tmp_path):
    for i, ratio in enumerate((0.88, 1.05, 1.24)):
        write(tmp_path, reference_doc(ratio=ratio), f"subject_{i}.json")
    corpus = load_corpus(tmp_path, CONFIG)

    below, above = corpus.bracketing(1.12)
    assert below.femur_to_torso_ratio == pytest.approx(1.05)
    assert above.femur_to_torso_ratio == pytest.approx(1.24)


def test_corpus_bracketing_reports_one_sided_for_out_of_range_lifters(tmp_path):
    """Phase 5 must know it is extrapolating rather than interpolating."""
    write(tmp_path, reference_doc(ratio=1.05), "a.json")
    corpus = load_corpus(tmp_path, CONFIG)

    below, above = corpus.bracketing(0.70)
    assert below is None and above is not None

    below, above = corpus.bracketing(1.60)
    assert below is not None and above is None


def test_empty_reference_directory_raises(tmp_path):
    with pytest.raises(DatasetError, match="no files matching"):
        load_corpus(tmp_path, CONFIG)


def test_missing_reference_directory_raises(tmp_path):
    with pytest.raises(DatasetError, match="does not exist"):
        load_corpus(tmp_path / "nope", CONFIG)


def test_narrow_ratio_span_warns(tmp_path, caplog):
    """One body type in the corpus means the band cannot really be interpolated."""
    for i, ratio in enumerate((1.10, 1.12, 1.11)):
        write(tmp_path, reference_doc(ratio=ratio), f"s{i}.json")
    load_corpus(tmp_path, CONFIG)
    assert "too few anchors" in caplog.text


def test_selecting_sequences_by_label_and_type(tmp_path):
    write(tmp_path, reference_doc(dataset_type="reference_optimal"), "a.json")
    doc = reference_doc(dataset_type="reference_good_morning_error")
    write(tmp_path, doc, "b.json")
    corpus = load_corpus(tmp_path, CONFIG)

    assert len(corpus.with_dataset_type("reference_optimal")) == 1
    assert len(corpus.with_label(FormLabel.GOOD_MORNING)) == 2
    assert len(corpus.with_label(FormLabel.HEEL_LIFT)) == 0


def test_empty_corpus_reports_rather_than_crashing():
    corpus = ReferenceCorpus(())
    assert corpus.coverage_report() == "corpus is empty"
    assert corpus.bracketing(1.0) == (None, None)


def test_deep_copy_of_doc_is_independent(tmp_path):
    """Guards the test helper itself, since every case mutates a shared shape."""
    original = reference_doc()
    clone = copy.deepcopy(original)
    clone["frames"][0]["frame_id"] = 999
    assert original["frames"][0]["frame_id"] == 100
