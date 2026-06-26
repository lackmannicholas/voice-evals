"""Judge schema, prompt assembly, parsing/retry, guards (judge-prompt spec §3-4).

All mocked — no network. The real API is exercised only under @pytest.mark.judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_evals.config import Config
from voice_evals.models import AudioClip, ClipMetadata
from voice_evals.scorers.base import registered_names
from voice_evals.scorers.judge.base_judge import (
    DIMENSIONS,
    BaseJudgeScorer,
    JudgeOutput,
    build_judge_prompt,
    parse_rubric_version,
)

pytestmark = pytest.mark.local
REPO = Path(__file__).resolve().parents[1]


def _canned(overall=4, unscoreable=False, score=4) -> str:
    d = {dim: {"reasoning": f"heard at 0:03 ({dim})", "score": score} for dim in DIMENSIONS}
    d.update(
        {
            "overall": overall,
            "notable_timestamps": ["0:14 — 2s pause"],
            "summary": "ok",
            "unscoreable": unscoreable,
            "unscoreable_reason": ("silence" if unscoreable else None),
            "rubric_version": "v1",
        }
    )
    return json.dumps(d)


class MockJudge(BaseJudgeScorer):
    def __init__(self, cfg, scripts):
        super().__init__(cfg)
        self.scripts = list(scripts)
        self.calls = 0

    def _call_model(self, system, rubric_block, wav_bytes, mime, nudge):
        out = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return out


@pytest.fixture
def wav_clip(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, (0.1 * np.sin(2 * np.pi * 200 * np.linspace(0, 1, 16000))).astype("float32"), 16000)
    return AudioClip("c1", p, p, None, None, None, 16000, 1.0, 1, ClipMetadata())


def test_rubric_version_and_prompt():
    rubric = (REPO / "config" / "rubric.default.txt").read_text()
    assert parse_rubric_version(rubric) == "v2"
    system, block = build_judge_prompt(rubric, "the AI agent's audio in isolation", "a leasing call")
    assert "ACOUSTIC" in system and "leasing" in system and block == rubric


def test_schema_field_order_reasoning_before_score():
    fields = list(JudgeOutput.model_fields["naturalness"].annotation.model_fields)
    assert fields == ["reasoning", "score"]  # CoT before number


def test_flatten_metrics(config, wav_clip):
    r = MockJudge(config, [_canned(overall=4)]).score(wav_clip)
    assert r.error is None
    assert set(r.metrics) == {f"judge_{d}" for d in DIMENSIONS} | {"judge_overall"}
    assert r.metrics["judge_overall"] == 4.0
    assert "judge_fluency" in r.metrics  # renamed from dysfluency
    assert r.raw["judged_channel_actual"] == "mono"


def test_unscoreable_surfaces_as_error(config, wav_clip):
    r = MockJudge(config, [_canned(unscoreable=True)]).score(wav_clip)
    assert r.error == "silence"
    assert r.metrics == {}


def test_unscoreable_without_dimension_scores_is_salvaged(config, wav_clip):
    # The rubric tells the model to OMIT dimension scores when unscoreable.
    # That must be recorded as unscoreable, NOT misclassified as a parse failure.
    raw = json.dumps({"unscoreable": True, "unscoreable_reason": "no agent speech"})
    r = MockJudge(config, [raw]).score(wav_clip)
    assert r.error == "no agent speech"
    assert r.metrics == {}


def test_parse_retry_recovers(config, wav_clip):
    mj = MockJudge(config, ["NOT JSON", _canned(overall=5)])
    r = mj.score(wav_clip)
    assert r.error is None and mj.calls == 2 and r.metrics["judge_overall"] == 5.0


def test_parse_failure_after_retries(config, wav_clip):
    mj = MockJudge(config, ["garbage", "still garbage"])
    r = mj.score(wav_clip)
    assert r.error is not None and r.error.startswith("judge_parse_failure")


def test_self_enhancement_guard(config, wav_clip):
    same_family = AudioClip("c2", wav_clip.mono16k_path, wav_clip.mono16k_path, None, None, None,
                            16000, 1.0, 1, ClipMetadata(tts_provider="gemini-tts"))
    assert MockJudge(config, [_canned()]).applicable(same_family) is False
    assert MockJudge(config, [_canned()]).applicable(wav_clip) is True


def test_config_hash_sensitive_to_model(config, wav_clip):
    h1 = MockJudge(config, [_canned()]).config_hash()
    cfg2 = Config()
    cfg2.judge.model = "gemini-3-pro-preview"
    h2 = MockJudge(cfg2, [_canned()]).config_hash()
    assert h1 != h2


def test_audio_judge_registered():
    assert "audio_judge" in registered_names()
