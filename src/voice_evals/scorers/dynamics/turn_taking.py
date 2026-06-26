"""Turn-taking quality scorer (design spec §9.3).

  * awkward_gap_count — inter-turn silences exceeding ``gap_threshold_s`` (the
    literal "weird pause"). Located in ``raw`` for a human to jump to.
  * overlap_ratio — fraction of call time agent and caller speak simultaneously
    (excluding very short backchannels).
  * agent_talk_ratio — agent speech time / total speech time.
  * max_gap_s.
"""

from __future__ import annotations

from typing import Any

from ...config import Config
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register
from .segmentation import Segment, Timeline, get_timeline


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    s = sorted(intervals)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect(x: list[tuple[float, float]], y: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(x) and j < len(y):
        lo = max(x[i][0], y[j][0])
        hi = min(x[i][1], y[j][1])
        if hi > lo:
            out.append((lo, hi))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


class TurnTakingScorer(BaseScorer):
    name = "turn_taking"
    layer = "dynamics"
    version = "1"

    def config_dict(self) -> dict[str, Any]:
        return {
            "gap_threshold_s": self.config.dynamics.gap_threshold_s,
            "overlap_min_s": self.config.dynamics.overlap_min_s,
            "vad_backend": self.config.dynamics.vad_backend,
            "vad_threshold": self.config.dynamics.vad_threshold,
        }

    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        tl = get_timeline(clip, self.config)
        if not tl.segments:
            raise ScorerError("no speech segments detected", raw=tl.to_raw())

        gap_thresh = self.config.dynamics.gap_threshold_s
        gaps = self._gaps(tl)
        awkward = [(round(t, 3), round(g, 3)) for (t, g) in gaps if g > gap_thresh]
        max_gap = max((g for _, g in gaps), default=0.0)

        agent_int = _merge([(s.start_s, s.end_s) for s in tl.speaker("agent")])
        caller_int = _merge([(s.start_s, s.end_s) for s in tl.speaker("caller")])
        overlaps = [
            (a, b)
            for (a, b) in _intersect(agent_int, caller_int)
            if (b - a) >= self.config.dynamics.overlap_min_s
        ]
        overlap_time = sum(b - a for a, b in overlaps)
        total = tl.total_s or (max((s.end_s for s in tl.segments), default=0.0))
        overlap_ratio = (overlap_time / total) if total > 0 else 0.0

        agent_time = sum(b - a for a, b in agent_int)
        caller_time = sum(b - a for a, b in caller_int)
        speech = agent_time + caller_time
        agent_talk_ratio = (agent_time / speech) if speech > 0 else 0.0

        metrics = {
            "awkward_gap_count": float(len(awkward)),
            "max_gap_s": float(max_gap),
            "overlap_ratio": float(overlap_ratio),
            "agent_talk_ratio": float(agent_talk_ratio),
        }
        raw = {
            "estimated": tl.estimated,
            "source": tl.source,
            "gap_threshold_s": gap_thresh,
            "awkward_gaps": awkward,  # (start_s, gap_s)
            "n_overlaps": len(overlaps),
        }
        return metrics, raw

    def _gaps(self, tl: Timeline) -> list[tuple[float, float]]:
        """Inter-TURN silences (turn transitions), as (gap_start_s, gap_len_s).

        Merges each speaker's segments separately, then reports a gap only between
        two consecutive turns by DIFFERENT speakers — so a pause within one
        speaker's turn isn't miscounted as the 'weird pause' the metric targets.
        The positive-gap filter drops overlaps (no spurious negative gaps)."""
        turns: list[tuple[str, float, float]] = []
        for who in ("agent", "caller"):
            for a, b in _merge([(s.start_s, s.end_s) for s in tl.speaker(who)]):
                turns.append((who, a, b))
        turns.sort(key=lambda x: (x[1], x[2]))
        gaps: list[tuple[float, float]] = []
        for (sp1, _s1, e1), (sp2, s2, _e2) in zip(turns, turns[1:]):
            if sp1 != sp2:
                gap = s2 - e1
                if gap > 0:
                    gaps.append((e1, gap))
        return gaps


@register("turn_taking", "dynamics")
def _build(config: Config) -> TurnTakingScorer:
    return TurnTakingScorer(config)
