"""Aggregation, gates, regression (design spec §13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_evals.aggregate import (
    Aggregator,
    absolute_gate_cases,
    metric_higher_is_better,
    summarize,
    build_long_df,
)
from voice_evals.config import Config, GateSpec
from voice_evals.models import AudioClip, ClipMetadata, EvalRun, ScoreResult

pytestmark = pytest.mark.local


def _clip(cid, agent_version="v1"):
    return AudioClip(cid, Path(f"{cid}.wav"), Path(f"{cid}.wav"), None, None, None,
                     16000, 1.0, 1, ClipMetadata(agent_version=agent_version))


def _run(run_id, rows, agent_version="v1"):
    """rows: list of (clip_id, scorer, {metric: value})."""
    clips = {}
    results = []
    for cid, scorer, metrics in rows:
        clips.setdefault(cid, _clip(cid, agent_version))
        results.append(ScoreResult(cid, scorer, "acoustic", metrics))
    return EvalRun(run_id, Path("corpus"), {}, results, "t0", "t1", clips=list(clips.values()))


def _cfg(absolute=None, max_delta=None):
    cfg = Config()
    cfg.gates.absolute = {k: GateSpec(**v) for k, v in (absolute or {}).items()}
    cfg.gates.regression.enabled = True
    cfg.gates.regression.max_delta = max_delta or {}
    return cfg


def test_absolute_min_gate_pass_warn_fail():
    cfg = _cfg(absolute={"dnsmos_ovrl": {"min": 2.8, "warn": 3.2}})
    run = _run("r", [("a", "dnsmos", {"dnsmos_ovrl": 3.5}),   # pass
                     ("b", "dnsmos", {"dnsmos_ovrl": 3.0}),   # warn
                     ("c", "dnsmos", {"dnsmos_ovrl": 2.5})])  # fail
    agg = Aggregator(cfg).evaluate(run)
    by_clip = {o.clip_id: o.status for o in agg.outcomes}
    assert by_clip == {"a": "pass", "b": "warn", "c": "fail"}
    assert not agg.passed()
    assert len(agg.failures) == 1


def test_absolute_max_gate():
    cfg = _cfg(absolute={"latency_p95_s": {"max": 3.0, "warn": 2.5}})
    run = _run("r", [("a", "latency", {"latency_p95_s": 2.0}),   # pass
                     ("b", "latency", {"latency_p95_s": 2.8}),   # warn
                     ("c", "latency", {"latency_p95_s": 3.5})])  # fail
    agg = Aggregator(cfg).evaluate(run)
    by_clip = {o.clip_id: o.status for o in agg.outcomes}
    assert by_clip == {"a": "pass", "b": "warn", "c": "fail"}


def test_score_result_passed_rollup():
    cfg = _cfg(absolute={"dnsmos_ovrl": {"min": 2.8}})
    run = _run("r", [("a", "dnsmos", {"dnsmos_ovrl": 2.0})])
    Aggregator(cfg).evaluate(run)
    assert run.results[0].passed is False


def test_regression_fails_on_degradation():
    cfg = _cfg(max_delta={"dnsmos_ovrl": 0.15})
    baseline = _run("base", [("a", "dnsmos", {"dnsmos_ovrl": 3.5})])
    current = _run("cur", [("a", "dnsmos", {"dnsmos_ovrl": 3.2})])  # drop 0.3 > 0.15
    agg = Aggregator(cfg).evaluate(current, baseline_run=baseline)
    assert len(agg.regressions) == 1 and agg.regressions[0].status == "fail"
    assert not agg.passed()


def test_regression_passes_within_delta():
    cfg = _cfg(max_delta={"dnsmos_ovrl": 0.15})
    baseline = _run("base", [("a", "dnsmos", {"dnsmos_ovrl": 3.5})])
    current = _run("cur", [("a", "dnsmos", {"dnsmos_ovrl": 3.45})])  # drop 0.05 < 0.15
    agg = Aggregator(cfg).evaluate(current, baseline_run=baseline)
    assert agg.regressions[0].status == "pass" and agg.passed()


def test_regression_direction_lower_better():
    # latency has a max gate => lower-is-better => a RISE is a regression
    cfg = _cfg(absolute={"latency_p95_s": {"max": 3.0}}, max_delta={"latency_p95_s": 0.2})
    baseline = _run("base", [("a", "latency", {"latency_p95_s": 1.0})])
    current = _run("cur", [("a", "latency", {"latency_p95_s": 1.5})])  # rose 0.5 > 0.2
    agg = Aggregator(cfg).evaluate(current, baseline_run=baseline)
    assert agg.regressions[0].status == "fail"


def test_metric_direction_inference():
    gates = {"dnsmos_ovrl": GateSpec(min=2.8), "latency_p95_s": GateSpec(max=3.0)}
    assert metric_higher_is_better("dnsmos_ovrl", gates) is True
    assert metric_higher_is_better("latency_p95_s", gates) is False
    assert metric_higher_is_better("awkward_gap_count", {}) is False  # name hint
    assert metric_higher_is_better("utmos", {}) is True


def test_summarize_stats():
    run = _run("r", [("a", "dnsmos", {"dnsmos_ovrl": 2.0}),
                     ("b", "dnsmos", {"dnsmos_ovrl": 4.0})])
    df = build_long_df(run)
    summ = summarize(df)
    row = summ[summ["metric"] == "dnsmos_ovrl"].iloc[0]
    assert row["count"] == 2 and row["mean"] == 3.0 and row["min"] == 2.0 and row["max"] == 4.0


def test_errored_result_appears_as_null_metric_row():
    run = EvalRun("r", Path("c"), {}, [ScoreResult("a", "nisqa", "acoustic", {}, error="boom")],
                  "t0", "t1", clips=[_clip("a")])
    df = build_long_df(run)
    assert (df["error"] == "boom").any()
    assert df["metric"].isna().all()


def test_banded_gate_checks_both_bounds():
    # both min and max set: a value above the ceiling must FAIL even though > floor
    cfg = _cfg(absolute={"x": {"min": 1.0, "max": 3.0}})
    run = _run("r", [("a", "s", {"x": 2.0}),   # in band -> pass
                     ("b", "s", {"x": 5.0}),   # above max -> fail
                     ("c", "s", {"x": 0.5})])  # below min -> fail
    agg = Aggregator(cfg).evaluate(run)
    by_clip = {o.clip_id: o.status for o in agg.outcomes}
    assert by_clip == {"a": "pass", "b": "fail", "c": "fail"}


def test_warn_only_respects_direction():
    # warn-only on a lower-better metric (latency) must warn when ABOVE warn
    cfg = _cfg(absolute={"latency_p95_s": {"warn": 2.0}})
    run = _run("r", [("a", "latency", {"latency_p95_s": 3.0}),   # high latency -> warn
                     ("b", "latency", {"latency_p95_s": 1.0})])  # low latency -> pass
    agg = Aggregator(cfg).evaluate(run)
    by_clip = {o.clip_id: o.status for o in agg.outcomes}
    assert by_clip == {"a": "warn", "b": "pass"}


def test_cross_scorer_same_metric_no_collision():
    # two scorers emit the same metric name for one clip: one fails, one passes.
    # The failing row must stay failed (no last-write-wins clobber).
    cfg = _cfg(absolute={"mos": {"min": 3.0}})
    run = _run("r", [("a", "scorerA", {"mos": 4.0}),   # pass
                     ("a", "scorerB", {"mos": 2.0})])  # fail
    agg = Aggregator(cfg).evaluate(run)
    statuses = {(o.scorer): o.status for o in agg.outcomes}
    assert statuses == {"scorerA": "pass", "scorerB": "fail"}
    assert len(agg.failures) == 1
    # the long_df 'passed' annotation must distinguish the two scorers' rows
    df = agg.long_df
    a = df[(df.scorer == "scorerA") & (df.metric == "mos")]["passed"].iloc[0]
    b = df[(df.scorer == "scorerB") & (df.metric == "mos")]["passed"].iloc[0]
    assert bool(a) and not bool(b)  # numpy bool dtype => compare by value, not identity


def test_absolute_gate_cases():
    cfg = _cfg(absolute={"dnsmos_ovrl": {"min": 2.8}, "warnonly": {"warn": 3.0}})
    cases = dict(absolute_gate_cases(cfg))
    assert "dnsmos_ovrl" in cases
    assert "warnonly" not in cases  # warn-only has no hard bound
