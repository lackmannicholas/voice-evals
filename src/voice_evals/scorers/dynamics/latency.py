"""Latency scorer (design spec §9.2).

Per agent turn, response latency = agent_speech_start − preceding caller_speech_end.
With gateway events we prefer the precise chain ``tts_first_audio − stt_final``
(true time-to-first-token / first-audio). Report percentiles, never just the mean
— the tail is what callers feel.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...config import Config
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register
from .segmentation import Timeline, get_timeline


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


class LatencyScorer(BaseScorer):
    name = "latency"
    layer = "dynamics"
    version = "1"

    def config_dict(self) -> dict[str, Any]:
        return {"vad_backend": self.config.dynamics.vad_backend,
                "vad_threshold": self.config.dynamics.vad_threshold}

    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        tl = get_timeline(clip, self.config)

        latencies, method = self._ttft_from_events(clip)
        if not latencies:
            latencies = self._turn_gap_latencies(tl)
            method = "turn_gap"
        if not latencies:
            raise ScorerError(
                "no agent response latencies derivable (need events or both speakers)",
                raw=tl.to_raw(),
            )

        metrics = {
            "latency_p50_s": _pct(latencies, 50),
            "latency_p95_s": _pct(latencies, 95),
            "latency_p99_s": _pct(latencies, 99),
            "latency_max_s": float(max(latencies)),
            "n_turns": float(len(latencies)),
        }
        raw = {
            "method": method,
            "estimated": tl.estimated,
            "source": tl.source,
            "latencies_s": [round(x, 3) for x in latencies],
        }
        return metrics, raw

    # -- precise TTFT from events -------------------------------------- #
    def _ttft_from_events(self, clip: AudioClip) -> tuple[list[float], str]:
        events = clip.metadata.events
        if not events:
            return [], ""
        stt = sorted((e.t_s for e in events if e.kind == "stt_final"))
        tts = sorted((e.t_s for e in events if e.kind == "tts_first_audio"))
        if not stt or not tts:
            return [], ""
        latencies: list[float] = []
        for tts_t in tts:
            preceding = [s for s in stt if s <= tts_t]
            if preceding:
                latencies.append(tts_t - preceding[-1])
        return [x for x in latencies if x >= 0], "ttft_events"

    # -- turn-gap latency from the timeline ---------------------------- #
    def _turn_gap_latencies(self, tl: Timeline) -> list[float]:
        """One latency per caller turn: the gap to the FIRST agent segment that
        responds to it. VAD often splits a single agent reply into several
        segments; counting each would inflate n_turns and the percentiles."""
        agent = sorted(tl.speaker("agent"), key=lambda s: s.start_s)
        caller = sorted(tl.speaker("caller"), key=lambda s: s.end_s)
        if not agent or not caller:
            return []
        latencies: list[float] = []
        last_caller_end_used = float("-inf")
        for a in agent:
            preceding_ends = [c.end_s for c in caller if c.end_s <= a.start_s]
            if not preceding_ends:
                continue
            ce = max(preceding_ends)
            if ce <= last_caller_end_used:
                continue  # a later sub-segment of the same reply; already counted
            lat = a.start_s - ce
            if lat >= 0:
                latencies.append(lat)
                last_caller_end_used = ce
        return latencies


@register("latency", "dynamics")
def _build(config: Config) -> LatencyScorer:
    return LatencyScorer(config)
