"""Reporting: HTML/Parquet/JSON outputs (design spec §14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_evals.aggregate import Aggregator
from voice_evals.config import Config, GateSpec
from voice_evals.models import AudioClip, ClipMetadata, EvalRun, ScoreResult
from voice_evals.report import render_html, write_outputs

pytestmark = pytest.mark.local


def _run():
    clip = AudioClip("c1", Path("a.wav"), Path("a.wav"), None, None, None, 16000, 2.0, 1,
                     ClipMetadata(agent_version="v1"))
    results = [
        ScoreResult("c1", "dnsmos", "acoustic", {"dnsmos_ovrl": 2.0, "dnsmos_sig": 3.4}),
        ScoreResult("c1", "audio_judge", "judge", {"judge_overall": 4.0},
                    raw={"naturalness": {"score": 4, "reasoning": "warm"},
                         "summary": "good", "notable_timestamps": ["0:14 — pause"]}),
        ScoreResult("c1", "nisqa", "acoustic", {}, error="weights missing"),
    ]
    return EvalRun("run1", Path("corpus"), {}, results, "t0", "t1", clips=[clip])


def _cfg():
    cfg = Config()
    cfg.gates.absolute = {"dnsmos_ovrl": GateSpec(min=2.8)}
    return cfg


def test_render_html_contains_sections():
    run = _run()
    agg = Aggregator(_cfg()).evaluate(run)
    html = render_html(run, agg, _cfg())
    for needle in ["voice-evals report", "Per-metric summary", "Gate violations",
                   "Regression vs baseline", "Worst offenders", "Per-clip detail",
                   "<svg", "<audio", "judge_overall"]:
        assert needle in html, needle


def test_write_outputs_creates_three_files(tmp_path):
    run = _run()
    cfg = _cfg()
    agg = Aggregator(cfg).evaluate(run)
    out = write_outputs(run, agg, cfg, out_dir=tmp_path / "run")
    assert (out / "report.html").exists()
    assert (out / "run.json").exists()
    assert (out / "results.parquet").exists() or (out / "results.csv").exists()


def test_parquet_readable(tmp_path):
    pd = pytest.importorskip("pandas")
    run = _run()
    cfg = _cfg()
    agg = Aggregator(cfg).evaluate(run)
    out = write_outputs(run, agg, cfg, out_dir=tmp_path / "run")
    pq = out / "results.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
        assert "metric" in df.columns and "value" in df.columns and "agent_version" in df.columns
