"""Calibration stats + harness (design spec §17)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from voice_evals import calibrate as cal
from voice_evals.calibrate import (
    pearson,
    quadratic_weighted_kappa,
    run_calibration,
    spearman,
    _load_golden,
    _resolve_clip,
)
from voice_evals.config import Config
from voice_evals.models import AudioClip, ClipMetadata
from voice_evals.scorers.judge.base_judge import DIMENSIONS, BaseJudgeScorer

pytestmark = pytest.mark.local


def test_kappa_perfect_and_chance():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # constant judge vs varied human => no better than chance
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [3, 3, 3, 3, 3]) == pytest.approx(0.0)


def test_pearson_spearman():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [10, 20, 25, 40]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_load_golden_filters_comments_and_bad_rows(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text(
        "clip_id,dimension,human_score\n"
        "# a comment,naturalness,5\n"
        "callA,naturalness,4\n"
        "callA,bogusdim,3\n"        # unknown dimension -> dropped
        "callA,prosody,9\n"          # out of range -> dropped
        "callB,fluency,2\n"
    )
    g = _load_golden(p)
    assert g["callA"] == {"naturalness": 4}
    assert g["callB"] == {"fluency": 2}


def test_resolve_clip_by_name_and_prefix():
    clips = [
        AudioClip("abcdef123456", __import__("pathlib").Path("call_x.wav"),
                  __import__("pathlib").Path("call_x.wav"), None, None, None, 16000, 1.0, 1, ClipMetadata()),
    ]
    assert _resolve_clip("call_x.wav", clips) is clips[0]
    assert _resolve_clip("call_x", clips) is clips[0]
    assert _resolve_clip("abcdef", clips) is clips[0]  # prefix
    assert _resolve_clip("nope", clips) is None


class _MockJudge(BaseJudgeScorer):
    def _call_model(self, system, block, wav_bytes, mime, nudge):
        d = {dim: {"reasoning": "r", "score": 4} for dim in DIMENSIONS}
        d.update({"naturalness": {"reasoning": "r", "score": 5}, "overall": 4,
                  "notable_timestamps": [], "summary": "s", "unscoreable": False,
                  "unscoreable_reason": None, "rubric_version": "v1"})
        return json.dumps(d)


def test_run_calibration_end_to_end(config, tmp_path, monkeypatch):
    corpus = config.corpus_dir
    corpus.mkdir(parents=True, exist_ok=True)
    for i, nm in enumerate("abcd"):  # distinct content per clip
        f0 = 150 + 30 * i
        sf.write(corpus / f"{nm}.wav",
                 (0.1 * np.sin(2 * np.pi * f0 * np.linspace(0, 1, 16000))).astype("float32"), 16000)
    gp = tmp_path / "golden.csv"
    with gp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "dimension", "human_score"])
        for nm, nat in zip("abcd", [5, 4, 5, 4]):
            w.writerow([f"{nm}.wav", "naturalness", nat])

    monkeypatch.setattr(cal, "build_scorer", lambda name, cfg: _MockJudge(cfg))
    report = run_calibration(config, gp, out_dir=tmp_path / "out")
    assert report.n_clips == 4
    nat = [d for d in report.per_dimension if d.dimension == "naturalness"][0]
    assert nat.n == 4
    assert (tmp_path / "out" / "calibration_report.json").exists()
    assert "naturalness" in report.render_text()
