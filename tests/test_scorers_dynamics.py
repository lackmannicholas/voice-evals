"""Dynamics layer (design spec §9). Events path is exact; VAD path is gated."""

from __future__ import annotations

import shutil

import pytest

from voice_evals.config import Config
from voice_evals.scorers.dynamics import segmentation
from voice_evals.scorers.dynamics.barge_in import BargeInScorer
from voice_evals.scorers.dynamics.latency import LatencyScorer
from voice_evals.scorers.dynamics.turn_taking import TurnTakingScorer
from voice_evals.scorers.dynamics.turn_taking import _intersect, _merge

pytestmark = pytest.mark.local


# -- exact events path ------------------------------------------------------ #
def test_latency_ttft_exact(event_clip):
    r = LatencyScorer(Config()).score(event_clip)
    assert r.error is None
    assert r.metrics["latency_p50_s"] == pytest.approx(0.7)  # tts_first_audio 4.1 - stt_final 3.4
    assert r.metrics["n_turns"] == 1.0
    assert r.raw["method"] == "ttft_events"
    assert r.raw["estimated"] is False


def test_barge_in_exact(event_clip):
    r = BargeInScorer(Config()).score(event_clip)
    assert r.error is None
    assert r.metrics["bargein_latency_p50_s"] == pytest.approx(0.2)  # 6.25 - 6.05
    assert r.metrics["bargein_success_rate"] == 1.0
    assert r.metrics["bargein_false_alarm_rate"] == 0.0  # unknown utterance != false alarm
    assert r.metrics["n_bargein_events"] == 1.0


def test_barge_in_ignored_is_not_success(event_clip):
    # Agent ignores the barge (no agent_interrupted) and finishes its turn naturally.
    # agent_tts_end must NOT count as a yield => success_rate 0.
    from voice_evals.models import GatewayEvent

    event_clip.metadata.events = [
        GatewayEvent("agent_tts_start", 4.0),
        GatewayEvent("user_speech_start", 5.0),
        GatewayEvent("barge_in_detected", 5.05),
        GatewayEvent("agent_tts_end", 6.25),  # natural end, agent never yielded
    ]
    r = BargeInScorer(Config()).score(event_clip)
    assert r.error is None
    assert r.metrics["n_bargein_events"] == 1.0
    assert r.metrics["bargein_success_rate"] == 0.0


def test_latency_turn_gap_counts_one_per_caller_turn():
    # A single agent reply split into 3 VAD segments must yield ONE latency, not 3.
    from voice_evals.scorers.dynamics.segmentation import Segment, Timeline

    tl = Timeline(
        segments=[
            Segment("caller", 0.0, 3.0),
            Segment("agent", 3.5, 4.0),
            Segment("agent", 4.0, 4.5),
            Segment("agent", 4.5, 5.0),
        ],
        source="channels",
        total_s=6.0,
    )
    lats = LatencyScorer(Config())._turn_gap_latencies(tl)
    assert lats == pytest.approx([0.5])  # single reply, single 0.5s latency


def test_turn_taking_source_events(event_clip):
    r = TurnTakingScorer(Config()).score(event_clip)
    assert r.error is None
    assert r.raw["source"] == "events"
    assert r.raw["estimated"] is False
    assert 0.0 <= r.metrics["agent_talk_ratio"] <= 1.0


def test_barge_in_not_applicable_on_mono(event_clip):
    # an event clip IS applicable (events present)
    assert BargeInScorer(Config()).applicable(event_clip) is True
    # strip events + channels => mono-only => not applicable
    event_clip.metadata.events = []
    assert BargeInScorer(Config()).applicable(event_clip) is False


# -- interval helpers ------------------------------------------------------- #
def test_merge_and_intersect():
    assert _merge([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert _intersect([(0, 2)], [(1, 3)]) == [(1, 2)]
    assert _intersect([(0, 1)], [(2, 3)]) == []


def test_awkward_gap_only_at_speaker_transitions():
    from voice_evals.scorers.dynamics.segmentation import Segment, Timeline

    scorer = TurnTakingScorer(Config())
    # same-speaker pause: NOT an inter-turn gap
    same = Timeline([Segment("agent", 0, 2), Segment("agent", 5, 7)], source="channels", total_s=7)
    assert scorer._gaps(same) == []
    # different-speaker transition: a real inter-turn gap
    diff = Timeline([Segment("caller", 0, 2), Segment("agent", 5, 7)], source="channels", total_s=7)
    gaps = scorer._gaps(diff)
    assert len(gaps) == 1 and gaps[0][1] == pytest.approx(3.0)


# -- VAD channels path (gated on silero) ------------------------------------ #
@pytest.mark.slow
def test_channels_path(config, cache, stereo_conversation, tmp_path):
    pytest.importorskip(config.dynamics.vad_backend.replace("ten", "ten_vad").replace("silero", "silero_vad"))
    from voice_evals.ingest import Ingestor

    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(stereo_conversation, corpus / "conv.wav")
    shutil.copy(stereo_conversation.with_suffix(".json"), corpus / "conv.json")
    segmentation.clear_cache()
    clip = Ingestor(config, cache).ingest_dir(corpus)[0]
    r = LatencyScorer(config).score(clip)
    assert r.error is None
    assert r.raw["source"] == "channels" and r.raw["estimated"] is False
    assert 0.3 <= r.metrics["latency_p50_s"] <= 3.0
