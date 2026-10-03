"""The seeding script must produce a corpus the loader and analyzer accept."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from formpro.config import AnalyzerConfig, DatasetConfig, KinematicsConfig
from formpro.dataset_loader import load_corpus
from formpro.form_analyzer import FormAnalyzer

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "seed_reference.py"


def load_script():
    spec = importlib.util.spec_from_file_location("seed_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeded_corpus_loads_and_spans_builds(tmp_path, monkeypatch):
    seed = load_script()
    monkeypatch.setattr("sys.argv", ["seed_reference.py", "--out", str(tmp_path)])

    assert seed.main() == 0

    corpus = load_corpus(tmp_path, DatasetConfig())
    low, high = corpus.ratio_span
    assert len(corpus) >= 10
    assert low <= 0.75 and high >= 1.3


def test_rerun_skips_existing_files_unless_forced(tmp_path, monkeypatch, capsys):
    seed = load_script()
    monkeypatch.setattr("sys.argv", ["seed_reference.py", "--out", str(tmp_path)])
    seed.main()
    first = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.json")}

    seed.main()

    assert "skipping" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.json")} == first


def test_the_simulated_rep_scores_clean_against_its_own_corpus(tmp_path, monkeypatch):
    seed = load_script()
    monkeypatch.setattr("sys.argv", ["seed_reference.py", "--out", str(tmp_path)])
    seed.main()
    from formpro.config import AppConfig

    frames, proportions = seed.record_base(AppConfig.load(None))
    analyzer = FormAnalyzer(
        load_corpus(tmp_path, DatasetConfig()),
        AnalyzerConfig(finding_hold_frames=2),
        KinematicsConfig(),
    )

    verdicts = [analyzer.update(f, proportions.femur_to_torso_ratio) for f in frames]

    scored = [v for v in verdicts if v.band_source is not None]
    assert scored
    assert all(v.ok for v in scored)
    assert not any(math.isnan(f.hip_height_norm) for f in frames)
