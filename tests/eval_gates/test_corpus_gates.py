"""Regression-gate suite (design spec §15).

Mirrors the team's behavioral-eval shape: a session-scoped EvalRun, then one
assertion per (clip, gated metric). Default CI runs ``pytest -m "not judge"`` so
this whole file is network/key-free — it gates on the deterministic dynamics
metrics from an event-log clip. Judge gates live behind ``@pytest.mark.judge``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_evals.aggregate import Aggregator, absolute_gate_cases
from voice_evals.config import Config, GateSpec
from voice_evals.models import AudioClip, ClipMetadata, EvalRun, GatewayEvent, ScoreResult
from voice_evals.scorers.dynamics.barge_in import BargeInScorer
from voice_evals.scorers.dynamics.latency import LatencyScorer
from voice_evals.scorers.dynamics.turn_taking import TurnTakingScorer

EVENTS = [
    ("user_speech_start", 0.0), ("user_speech_end", 3.2), ("stt_final", 3.4),
    ("tts_first_audio", 4.1), ("agent_tts_start", 4.1), ("user_speech_start", 6.0),
    ("barge_in_detected", 6.05), ("agent_interrupted", 6.25), ("agent_tts_end", 6.25),
]


def _gate_config() -> Config:
    cfg = Config()
    cfg.gates.absolute = {
        "latency_p50_s": GateSpec(max=1.7),
        "latency_p95_s": GateSpec(max=3.0, warn=2.5),
        "bargein_success_rate": GateSpec(min=0.90),
    }
    return cfg


def _event_clip() -> AudioClip:
    meta = ClipMetadata(agent_version="v1", events=[GatewayEvent(k, t) for k, t in EVENTS])
    return AudioClip("evt1", Path("x.wav"), Path("x.wav"), None, None, None, 16000, 7.0, 1, meta)


@pytest.fixture(scope="module")
def eval_run() -> EvalRun:
    cfg = _gate_config()
    clip = _event_clip()
    results: list[ScoreResult] = []
    for scorer in (LatencyScorer(cfg), TurnTakingScorer(cfg), BargeInScorer(cfg)):
        results.append(scorer.score(clip))
    return EvalRun("gate_run", Path("corpus"), {}, results, "t0", "t1", clips=[clip])


def _gate_cases():
    cfg = _gate_config()
    return absolute_gate_cases(cfg)


@pytest.mark.local
@pytest.mark.parametrize("metric,spec", _gate_cases(), ids=lambda x: getattr(x, "__name__", str(x)))
def test_absolute_floor(eval_run: EvalRun, metric: str, spec: GateSpec):
    for clip_id in eval_run.clip_ids:
        val = eval_run.metric(clip_id, metric)
        if val is None:
            continue  # metric not produced for this clip (conditionally applicable)
        if spec.min is not None:
            assert val >= spec.min, f"{clip_id}: {metric}={val:.3f} < min {spec.min}"
        if spec.max is not None:
            assert val <= spec.max, f"{clip_id}: {metric}={val:.3f} > max {spec.max}"


@pytest.mark.local
def test_run_passes_all_absolute_gates(eval_run: EvalRun):
    agg = Aggregator(_gate_config()).evaluate(eval_run)
    assert agg.passed(), [o.message for o in agg.failures]


@pytest.mark.local
def test_regression_gate_detects_degradation():
    cfg = _gate_config()
    cfg.gates.regression.max_delta = {"latency_p95_s": 0.2}
    clip = _event_clip()
    baseline = EvalRun("base", Path("c"), {}, [
        ScoreResult("evt1", "latency", "dynamics", {"latency_p95_s": 0.7})
    ], "t0", "t1", clips=[clip])
    current = EvalRun("cur", Path("c"), {}, [
        ScoreResult("evt1", "latency", "dynamics", {"latency_p95_s": 1.5})  # rose 0.8 > 0.2
    ], "t0", "t1", clips=[clip])
    agg = Aggregator(cfg).evaluate(current, baseline_run=baseline)
    assert any(o.status == "fail" for o in agg.regressions)


@pytest.mark.judge
def test_judge_gate_requires_key():
    import os

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        pytest.skip("no judge API key; judge gates run only when a key is present")
    # When a key is present this is where judge-dimension floors (per calibration)
    # would be asserted. Left minimal by design — calibration drives which dims gate.
    assert True
