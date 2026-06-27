"""Simulator/driver (WALK) tests — deterministic, mock backends, no network."""

from __future__ import annotations

import json

import pytest
import soundfile as sf

from voice_evals.cache import ResultCache
from voice_evals.config import Config, GateSpec
from voice_evals.runner import Runner
from voice_evals.simulate.base import (
    MockAgent,
    MockUserSimulator,
    assemble_recording,
    validate_conversation,
)
from voice_evals.simulate.gating import compare_versions
from voice_evals.simulate.orchestrator import run_scenario, run_suite
from voice_evals.simulate.scenario import Persona, Scenario, load_scenarios

pytestmark = pytest.mark.local


def _scn(**kw):
    base = dict(id="s1", goal="get help", persona=Persona(), max_turns=3)
    base.update(kw)
    return Scenario(**base)


def test_scenario_manifest_loads():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    scns = load_scenarios(repo / "scenarios" / "example.yaml")
    assert len(scns) >= 1
    s = scns[0]
    assert s.id and s.persona.adversarial in (True, False) and s.max_turns > 0


def test_mock_user_hangs_up_at_max_turns():
    us = MockUserSimulator()
    scn = _scn(max_turns=2)
    m1 = us.next_turn(scn, [])
    assert m1.speaker == "caller" and not m1.hangup
    m2 = us.next_turn(scn, [m1, MockAgent().greet(scn)])
    assert m2.hangup  # second caller turn hits max_turns


def test_run_scenario_produces_valid_recording(tmp_path):
    call = run_scenario(_scn(max_turns=3), MockAgent(version="v"), MockUserSimulator(), 0, tmp_path)
    assert call.valid and call.wav_path.exists() and call.sidecar_path.exists()
    # dual-channel, agent on ch0
    data, sr = sf.read(str(call.wav_path))
    assert sr == 16000 and data.ndim == 2 and data.shape[1] == 2
    sidecar = json.loads(call.sidecar_path.read_text())
    assert sidecar["channel_map"] == {"agent_channel": 0, "caller_channel": 1}
    assert sidecar["agent_version"] == "v"
    kinds = {e["kind"] for e in sidecar["events"]}
    assert {"user_speech_start", "agent_tts_start", "tts_first_audio", "stt_final"} <= kinds


def test_validate_conversation_flags_degenerate():
    ok, _ = validate_conversation([])
    assert not ok


def test_agent_latency_becomes_exact_event_timing(tmp_path):
    # MockAgent latency should show up as the TTFT the latency scorer reports
    # greet=False so the first agent turn IS the response (latency 1.3) after the caller
    call = run_scenario(_scn(max_turns=2), MockAgent(latency_s=1.3, greet=False),
                        MockUserSimulator(), 0, tmp_path)
    events = json.loads(call.sidecar_path.read_text())["events"]
    stt = [e["t_s"] for e in events if e["kind"] == "stt_final"]
    tts = [e["t_s"] for e in events if e["kind"] == "tts_first_audio"]
    assert tts and stt and abs((tts[0] - stt[0]) - 1.3) < 0.05


def test_kruns_are_distinct(tmp_path):
    import random

    random.seed(0)
    calls = run_suite([_scn(max_turns=2)],
                      lambda: MockAgent(version="v", seed=random.randint(1, 9999)),
                      lambda: MockUserSimulator(seed=random.randint(1, 9999)),
                      k=3, out_dir=tmp_path)
    assert len(calls) == 3
    # distinct audio => distinct content (so k-run aggregation isn't collapsed by dedup).
    # hash the full waveform (leading samples are silence and identical across runs).
    hashes = {sf.read(str(c.wav_path))[0].tobytes() for c in calls}
    assert len(hashes) == 3


def test_no_agent_turn_after_caller_hangup(tmp_path):
    # the conversation must end on the caller's hangup turn, not a phantom agent reply
    call = run_scenario(_scn(max_turns=2), MockAgent(version="v"), MockUserSimulator(), 0, tmp_path)
    assert call.transcript[-1]["speaker"] == "caller"


# ---- gating unit tests (build EvalRuns directly; no audio) ----
from pathlib import Path as _P

from voice_evals.models import AudioClip, ClipMetadata, EvalRun, ScoreResult


def _run(rows):
    """rows: (clip_id, version, scenario, {metric: value})"""
    clips, results = [], []
    for cid, version, scenario, metrics in rows:
        clips.append(AudioClip(cid, _P(f"{cid}.wav"), _P(f"{cid}.wav"), None, None, None,
                               16000, 1.0, 1, ClipMetadata(agent_version=version,
                                                           extra={"scenario_id": scenario})))
        results.append(ScoreResult(cid, "judge", "judge", metrics))
    return EvalRun("r", _P("c"), {}, results, "t0", "t1", clips=clips)


def test_verdict_matches_gate_threshold():
    # max_delta tighter than eps: a small regression must read 'regressed', not '≈same'
    cfg = Config()
    cfg.gates.absolute = {"judge_overall": GateSpec(min=3)}
    cfg.gates.regression.max_delta = {"judge_overall": 0.02}
    run = _run([("a", "cand", "s1", {"judge_overall": 3.97}),
                ("b", "base", "s1", {"judge_overall": 4.00})])
    cmp = compare_versions(run, "cand", "base", cfg)
    d = next(x for x in cmp.deltas if x.metric == "judge_overall")
    assert d.verdict == "regressed" and d.regressed_beyond_gate and not cmp.passed()


def test_empty_version_does_not_pass_vacuously():
    cfg = Config()
    run = _run([("a", "base", "s1", {"judge_overall": 4.0})])
    cmp = compare_versions(run, candidate="cand", baseline="base", config=cfg)  # cand absent
    assert cmp.error is not None and not cmp.passed()


def test_per_scenario_regression_not_masked_by_pooling():
    # candidate regresses badly on s_hard, improves on s_easy; pooled mean would hide it
    cfg = Config()
    cfg.gates.regression.max_delta = {"judge_frustration_handling": 0.5}
    run = _run([
        ("c1", "cand", "s_hard", {"judge_frustration_handling": 1.0}),
        ("c2", "cand", "s_easy", {"judge_frustration_handling": 5.0}),
        ("b1", "base", "s_hard", {"judge_frustration_handling": 4.0}),
        ("b2", "base", "s_easy", {"judge_frustration_handling": 4.0}),
    ])
    cmp = compare_versions(run, "cand", "base", cfg)
    regressed = {(d.scenario, d.metric) for d in cmp.regressions}
    assert ("s_hard", "judge_frustration_handling") in regressed
    assert not cmp.passed()  # one scenario regressing fails the whole comparison


def test_ab_comparison_flags_latency_regression(tmp_path):
    scns = [_scn(id="ac", max_turns=3)]
    import random
    random.seed(1)
    # candidate slow (1.8s), baseline fast (0.5s)
    run_suite(scns, lambda: MockAgent(version="cand", latency_s=1.8, seed=random.randint(1, 9999)),
              lambda: MockUserSimulator(seed=random.randint(1, 9999)), k=2, out_dir=tmp_path)
    run_suite(scns, lambda: MockAgent(version="base", latency_s=0.5, seed=random.randint(1, 9999)),
              lambda: MockUserSimulator(seed=random.randint(1, 9999)), k=2, out_dir=tmp_path)

    cfg = Config()
    cfg.scorers.acoustic = ["dnsmos"]
    cfg.scorers.dynamics = ["latency", "turn_taking"]
    cfg.scorers.judge = []
    cfg.runner.progress = False
    cfg.cache_dir = tmp_path / ".cache"
    cfg.gates.regression.max_delta = {"latency_p50_s": 0.3}
    run = Runner(cfg, cache=ResultCache(tmp_path / ".cache")).run(corpus_dir=tmp_path)

    # exact latency via the event log
    lat = next(r for r in run.results if r.scorer == "latency" and r.ok)
    assert lat.raw["source"] == "events"

    cmp = compare_versions(run, candidate="cand", baseline="base", config=cfg)
    assert "latency_p50_s" in {d.metric for d in cmp.regressions}
    assert not cmp.passed()
