"""Barge-in scorer (design spec §9.4).

Only applicable with gateway events or isolated channels — barge-in cannot be
measured reliably on a mono mix. Definitions (duplex-dialogue literature):
  * barge-in latency: caller-onset-during-agent-TTS → agent TTS stop.
  * success rate: fraction of interruptions where the agent stops within
    ``bargein_success_window_s``.
  * false-alarm rate: agent yields to a sub-100ms backchannel that wasn't a real
    floor-take.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...config import Config
from ...models import AudioClip, GatewayEvent
from ..base import BaseScorer, ScorerError, register
from .segmentation import Segment, Timeline, get_timeline


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


class BargeInScorer(BaseScorer):
    name = "barge_in"
    layer = "dynamics"
    version = "1"

    def config_dict(self) -> dict[str, Any]:
        return {
            "success_window_s": self.config.dynamics.bargein_success_window_s,
            "min_caller_utterance_ms": self.config.dynamics.min_caller_utterance_ms,
            "vad_backend": self.config.dynamics.vad_backend,
            "vad_threshold": self.config.dynamics.vad_threshold,
        }

    def applicable(self, clip: AudioClip) -> bool:
        if clip.metadata.has_events:
            return True
        return bool(clip.agent_only_path and clip.caller_only_path)

    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        window = self.config.dynamics.bargein_success_window_s
        min_utt = self.config.dynamics.min_caller_utterance_ms / 1000.0

        if any(e.kind == "barge_in_detected" for e in clip.metadata.events):
            attempts, source = self._from_events(clip.metadata.events, window), "events"
        else:
            tl = get_timeline(clip, self.config)
            if tl.source == "diarization":
                raise ScorerError(
                    "barge-in not reliable on mono/diarized audio", raw=tl.to_raw()
                )
            attempts, source = self._from_timeline(tl, clip.duration_s), tl.source

        n = len(attempts)
        if n == 0:
            return {"n_bargein_events": 0.0}, {"source": source, "attempts": []}

        # only resolved stops feed the latency percentiles — a failed/ignored barge
        # (no stop) must not inject a synthetic latency that skews p50/p95
        latencies = [a["stop_latency_s"] for a in attempts
                     if a.get("success") and a["stop_latency_s"] is not None]
        successes = sum(1 for a in attempts if a["success"])
        # A false alarm = agent yielded to a confirmed sub-threshold backchannel.
        # If the caller utterance duration is unknown (no end event), we cannot
        # call it a false alarm.
        false_alarms = sum(
            1
            for a in attempts
            if a["caller_utterance_s"] is not None
            and a["caller_utterance_s"] < min_utt
            and a["success"]
        )

        metrics = {
            "bargein_latency_p50_s": _pct(latencies, 50),
            "bargein_latency_p95_s": _pct(latencies, 95),
            "bargein_success_rate": successes / n,
            "bargein_false_alarm_rate": false_alarms / n,
            "n_bargein_events": float(n),
        }
        raw = {"source": source, "attempts": attempts}
        return metrics, raw

    # -- explicit events ----------------------------------------------- #
    def _from_events(self, events: list[GatewayEvent], window: float) -> list[dict[str, Any]]:
        ev = sorted(events, key=lambda e: e.t_s)
        barges = [e.t_s for e in ev if e.kind == "barge_in_detected"]
        # Only an explicit ``agent_interrupted`` is a genuine yield. ``agent_tts_end``
        # fires at the end of EVERY agent turn regardless of barge-in, so counting it
        # as a stop would inflate the success rate when the agent ignored the barge
        # and merely finished its turn.
        stops = [e.t_s for e in ev if e.kind == "agent_interrupted"]
        caller_starts = sorted(e.t_s for e in ev if e.kind == "user_speech_start")
        caller_ends = sorted(e.t_s for e in ev if e.kind == "user_speech_end")
        attempts: list[dict[str, Any]] = []
        for bt in barges:
            stop_after = [s for s in stops if s >= bt]
            stop = stop_after[0] if stop_after else None
            stop_latency = (stop - bt) if stop is not None else float("inf")
            # caller utterance duration around this barge
            cs = [s for s in caller_starts if abs(s - bt) <= 0.5]
            onset = cs[0] if cs else bt
            ce = [e for e in caller_ends if e >= onset]
            utt = (ce[0] - onset) if ce else None  # unknown if no end event
            attempts.append(
                {
                    "onset_s": round(onset, 3),
                    "stop_latency_s": round(stop_latency, 3) if stop_latency != float("inf") else None,
                    "success": stop is not None and stop_latency <= window,
                    "caller_utterance_s": round(utt, 3) if utt is not None else None,
                }
            )
        # normalize None latencies to a large finite number for percentile math
        for a in attempts:
            if a["stop_latency_s"] is None:
                a["stop_latency_s"] = float(window * 10)
        return attempts

    # -- derived from channel timeline --------------------------------- #
    def _from_timeline(self, tl: Timeline, total_s: float) -> list[dict[str, Any]]:
        agent = sorted(tl.speaker("agent"), key=lambda s: s.start_s)
        caller = sorted(tl.speaker("caller"), key=lambda s: s.start_s)
        attempts: list[dict[str, Any]] = []
        for c in caller:
            # caller starts while an agent segment is active => barge-in attempt
            active = next((a for a in agent if a.start_s < c.start_s < a.end_s), None)
            if active is None:
                continue
            stop_latency = active.end_s - c.start_s  # agent kept talking until its end
            attempts.append(
                {
                    "onset_s": round(c.start_s, 3),
                    "stop_latency_s": round(max(0.0, stop_latency), 3),
                    "success": stop_latency <= self.config.dynamics.bargein_success_window_s,
                    "caller_utterance_s": round(c.duration_s, 3),
                }
            )
        return attempts


@register("barge_in", "dynamics")
def _build(config: Config) -> BargeInScorer:
    return BargeInScorer(config)
