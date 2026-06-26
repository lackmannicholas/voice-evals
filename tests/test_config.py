"""Config loading + defaults (design spec §18)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_evals.config import Config

pytestmark = pytest.mark.local

REPO = Path(__file__).resolve().parents[1]


def test_defaults_are_local_only():
    cfg = Config()
    assert cfg.scorers.judge == []  # judge off by default
    assert cfg.ingest.target_sr == 16000
    assert cfg.judge.judged_channel == "agent_only"
    assert cfg.judge.exclude_if_model_family_matches is True


def test_load_default_yaml():
    cfg = Config.load(REPO / "config" / "default.yaml")
    assert cfg.gates.absolute["dnsmos_ovrl"].min == 2.8
    assert cfg.gates.absolute["latency_p95_s"].max == 3.0
    assert cfg.gates.regression.enabled is True
    assert "agent_version" in cfg.gates.regression.group_by


def test_load_local_profile():
    cfg = Config.load(REPO / "config" / "example.local.yaml")
    assert cfg.scorers.judge == []


def test_snapshot_is_json_able_and_secret_free():
    cfg = Config.load(REPO / "config" / "default.yaml")
    snap = cfg.snapshot()
    s = json.dumps(snap)  # must not raise
    assert "api_key" not in s.lower()


def test_disable_judge():
    cfg = Config()
    cfg.scorers.judge = ["audio_judge"]
    cfg2 = cfg.disable_judge()
    assert cfg2.scorers.judge == []
    assert cfg.scorers.judge == ["audio_judge"]  # original untouched


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        Config.load(Path("/nonexistent/xyz.yaml"))
