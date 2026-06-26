"""Core data-model serialization (design spec §5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voice_evals.models import (
    AudioClip,
    ChannelMap,
    ClipMetadata,
    EvalRun,
    GatewayEvent,
    ScoreResult,
    _jsonable,
)

pytestmark = pytest.mark.local


def _clip(cid: str = "c1", agent_version: str | None = "v1") -> AudioClip:
    return AudioClip(
        clip_id=cid,
        source_path=Path("a.mp3"),
        mono16k_path=Path("a.wav"),
        stereo_path=None,
        agent_only_path=None,
        caller_only_path=None,
        sample_rate=16000,
        duration_s=1.0,
        n_source_channels=1,
        metadata=ClipMetadata(agent_version=agent_version),
    )


def test_score_result_roundtrip():
    r = ScoreResult("c1", "dnsmos", "acoustic", {"dnsmos_ovrl": 3.4}, raw={"x": 1}, scorer_config_hash="h")
    r2 = ScoreResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert r2.clip_id == "c1" and r2.metrics["dnsmos_ovrl"] == 3.4 and r2.scorer_config_hash == "h"
    assert r2.ok


def test_jsonable_handles_numpy():
    payload = {"a": np.float32(3.2), "b": np.array([1, 2, 3]), "c": np.int64(5)}
    out = _jsonable(payload)
    json.dumps(out)  # must not raise
    assert out["a"] == pytest.approx(3.2, abs=1e-5)
    assert out["b"] == [1, 2, 3]
    assert out["c"] == 5


def test_eval_run_roundtrip_and_metric():
    clip = _clip()
    r = ScoreResult("c1", "dnsmos", "acoustic", {"dnsmos_ovrl": 3.4})
    run = EvalRun("run1", Path("corpus"), {"k": "v"}, [r], "t0", "t1", clips=[clip])
    run2 = EvalRun.from_dict(json.loads(json.dumps(run.to_dict())))
    assert run2.metric("c1", "dnsmos_ovrl") == 3.4
    assert run2.metric("c1", "missing") is None
    assert run2.metadata_for("c1").agent_version == "v1"
    assert run2.clip_by_id("c1").sample_rate == 16000


def test_metric_skips_errored_results():
    ok = ScoreResult("c1", "utmos", "acoustic", {"utmos": 4.0})
    bad = ScoreResult("c1", "dnsmos", "acoustic", {}, error="boom")
    run = EvalRun("r", Path("c"), {}, [bad, ok], "t0", "t1")
    assert run.metric("c1", "utmos") == 4.0
    assert run.metric("c1", "dnsmos_ovrl") is None


def test_clip_metadata_from_dict_preserves_unknown_keys():
    meta = ClipMetadata.from_dict({"call_id": "x", "weird_key": 7,
                                   "channel_map": {"agent_channel": 0, "caller_channel": 1}})
    assert meta.call_id == "x"
    assert meta.extra["weird_key"] == 7
    assert isinstance(meta.channel_map, ChannelMap) and meta.channel_map.agent_channel == 0


def test_gateway_event_from_dict():
    e = GatewayEvent.from_dict({"kind": "stt_final", "t_s": "3.4"})
    assert e.kind == "stt_final" and e.t_s == 3.4
