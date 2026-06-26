"""Turn segmentation → a speaker Timeline (design spec §9.1).

Resolution order (most to least accurate):
  1. gateway events present  -> turns derived directly. source="events", exact.
  2. isolated agent/caller channels -> per-channel VAD. source="channels".
  3. mono only -> VAD + diarization to assign speakers. source="diarization",
     estimated=True (barge-in is NOT reliable here).

The timeline is memoized per (clip, config) so the three dynamics scorers don't
each re-run VAD in a single process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from ...config import Config
from ...logging_util import get_logger
from ...models import AudioClip, GatewayEvent

log = get_logger(__name__)

Speaker = Literal["agent", "caller"]
SegSource = Literal["events", "channels", "diarization"]


@dataclass
class Segment:
    speaker: Speaker
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class Timeline:
    segments: list[Segment]
    source: SegSource
    estimated: bool = False
    events: list[GatewayEvent] = field(default_factory=list)
    total_s: float = 0.0

    def speaker(self, who: Speaker) -> list[Segment]:
        return [s for s in self.segments if s.speaker == who]

    def ordered(self) -> list[Segment]:
        return sorted(self.segments, key=lambda s: (s.start_s, s.end_s))

    def speech_time(self, who: Optional[Speaker] = None) -> float:
        segs = self.segments if who is None else self.speaker(who)
        return sum(s.duration_s for s in segs)

    def to_raw(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "estimated": self.estimated,
            "n_segments": len(self.segments),
            "segments": [
                {"speaker": s.speaker, "start_s": round(s.start_s, 3), "end_s": round(s.end_s, 3)}
                for s in self.segments
            ],
        }


# --------------------------------------------------------------------------- #
# memoization
# --------------------------------------------------------------------------- #
_CACHE: dict[tuple, Timeline] = {}
# All dynamics scorers (latency / turn_taking / barge_in) share ONE VAD/diarization
# model. The runner gives each scorer its own single-flight lock keyed by scorer
# name, which does NOT serialize across different scorers — so without this global
# lock they would invoke the shared (non-thread-safe) torch model concurrently and
# crash. Serializing the build here is cheap because the result is memoized.
_BUILD_LOCK = threading.Lock()


def _cache_key(clip: AudioClip, config: Config) -> tuple:
    # clip_id hashes only the decoded audio, not the sidecar events, so the key
    # must fold in the event CONTENT (not just the count) — otherwise two clips
    # with identical audio but different equal-length event lists would collide.
    events = tuple(
        (e.kind, round(e.t_s, 4))
        for e in sorted(clip.metadata.events, key=lambda e: e.t_s)
    )
    return (
        clip.clip_id,
        config.dynamics.vad_backend,
        round(config.dynamics.vad_threshold, 4),
        str(clip.agent_only_path),
        str(clip.caller_only_path),
        str(clip.mono16k_path),
        events,
    )


def get_timeline(clip: AudioClip, config: Config) -> Timeline:
    key = _cache_key(clip, config)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    # double-checked locking: serialize concurrent builds (shared VAD model) but
    # let memo hits run lock-free.
    with _BUILD_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        tl = _build_timeline(clip, config)
        _CACHE[key] = tl
        return tl


def clear_cache() -> None:
    _CACHE.clear()


# --------------------------------------------------------------------------- #
def _build_timeline(clip: AudioClip, config: Config) -> Timeline:
    if clip.metadata.has_events:
        return _from_events(clip.metadata.events, clip.duration_s)
    if clip.agent_only_path and clip.caller_only_path:
        return _from_channels(clip, config)
    return _from_mono(clip, config)


# -- 1. events -------------------------------------------------------------- #
def _from_events(events: list[GatewayEvent], total_s: float) -> Timeline:
    segs: list[Segment] = []
    segs += _pair_events(events, "agent_tts_start", ("agent_tts_end", "agent_interrupted"), "agent")
    segs += _pair_events(events, "user_speech_start", ("user_speech_end",), "caller")
    return Timeline(segments=segs, source="events", estimated=False, events=list(events), total_s=total_s)


def _pair_events(
    events: list[GatewayEvent], start_kind: str, end_kinds: tuple[str, ...], speaker: Speaker
) -> list[Segment]:
    ev = sorted(events, key=lambda e: e.t_s)
    segs: list[Segment] = []
    open_start: Optional[float] = None
    for e in ev:
        if e.kind == start_kind:
            if open_start is not None:
                segs.append(Segment(speaker, open_start, e.t_s))  # back-to-back start: close prior
            open_start = e.t_s
        elif e.kind in end_kinds and open_start is not None:
            segs.append(Segment(speaker, open_start, e.t_s))
            open_start = None
    return [s for s in segs if s.end_s > s.start_s]


# -- 2. channels ------------------------------------------------------------ #
def _from_channels(clip: AudioClip, config: Config) -> Timeline:
    agent = _vad_segments(Path(clip.agent_only_path), config, "agent")
    caller = _vad_segments(Path(clip.caller_only_path), config, "caller")
    return Timeline(
        segments=agent + caller, source="channels", estimated=False, total_s=clip.duration_s
    )


# -- 3. mono + diarization -------------------------------------------------- #
def _from_mono(clip: AudioClip, config: Config) -> Timeline:
    # speech regions from VAD; speaker labels from diarization (best effort)
    try:
        segs = _diarize_mono(Path(clip.mono16k_path), config)
        return Timeline(segments=segs, source="diarization", estimated=True, total_s=clip.duration_s)
    except Exception as e:  # noqa: BLE001
        log.warning("mono diarization unavailable (%s); returning single-speaker estimate", e)
        # fall back: treat all detected speech as 'agent' so latency/turn-taking
        # still produce *something*, but clearly flagged estimated.
        speech = _vad_segments(Path(clip.mono16k_path), config, "agent")
        return Timeline(segments=speech, source="diarization", estimated=True, total_s=clip.duration_s)


# --------------------------------------------------------------------------- #
# VAD backends
# --------------------------------------------------------------------------- #
def _vad_segments(path: Path, config: Config, speaker: Speaker) -> list[Segment]:
    backend = config.dynamics.vad_backend
    if backend == "ten":
        regions = _ten_vad(path, config)
    else:
        regions = _silero_vad(path, config)
    return [Segment(speaker, a, b) for (a, b) in regions if b > a]


def _silero_vad(path: Path, config: Config) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    model = _load_silero()
    wav = read_audio(str(path), sampling_rate=16000)
    ts = get_speech_timestamps(
        wav, model, sampling_rate=16000, threshold=config.dynamics.vad_threshold
    )
    return [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]


_SILERO = None


def _load_silero():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad

        _SILERO = load_silero_vad()
    return _SILERO


_TEN_HOP = 256  # 16 ms frames at 16 kHz


def _ten_vad(path: Path, config: Config) -> list[tuple[float, float]]:
    """Frame-based TEN VAD (TEN-framework/ten-vad). Bundles its own model, so it
    runs offline with no download. ``TenVad.process(int16_frame)`` returns
    ``(probability, flag)`` per 256-sample frame; we group speech frames into
    segments with light hangover/min-duration smoothing to match silero's output."""
    try:
        from ten_vad import TenVad  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"TEN VAD backend selected but `ten_vad` is not importable: {e}. "
            "Install with `pip install ten-vad`, or set dynamics.vad_backend=silero."
        )
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr != 16000:  # canonical inputs are 16 kHz; guard anyway
        from .._audio import _resample_linear

        wav = (_resample_linear(wav.astype("float32"), sr, 16000)).astype(np.int16)
        sr = 16000

    vad = TenVad(_TEN_HOP, config.dynamics.vad_threshold)
    n_frames = len(wav) // _TEN_HOP
    flags: list[int] = []
    for i in range(n_frames):
        frame = wav[i * _TEN_HOP : (i + 1) * _TEN_HOP]
        _prob, flag = vad.process(frame)
        flags.append(int(flag))

    fdur = _TEN_HOP / sr
    raw: list[tuple[float, float]] = []
    start: Optional[int] = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            raw.append((start * fdur, i * fdur))
            start = None
    if start is not None:
        raw.append((start * fdur, len(flags) * fdur))

    return _smooth(raw, merge_gap_s=0.2, min_speech_s=0.1)


def _smooth(
    segs: list[tuple[float, float]], merge_gap_s: float, min_speech_s: float
) -> list[tuple[float, float]]:
    if not segs:
        return []
    merged = [list(segs[0])]
    for a, b in segs[1:]:
        if a - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if (b - a) >= min_speech_s]


def _diarize_mono(path: Path, config: Config) -> list[Segment]:
    """pyannote diarization on a mono mix; speaker labels mapped to agent/caller
    by total speaking time (heuristic). Requires pyannote.audio + an HF token."""
    import os

    from pyannote.audio import Pipeline  # type: ignore

    token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    diar = pipeline(str(path))
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for turn, _, spk in diar.itertracks(yield_label=True):
        by_spk.setdefault(spk, []).append((turn.start, turn.end))
    if not by_spk:
        return []
    # label the speaker with the most total time as 'agent' (support agents talk more)
    totals = {spk: sum(b - a for a, b in segs) for spk, segs in by_spk.items()}
    agent_spk = max(totals, key=totals.get)
    segs: list[Segment] = []
    for spk, regions in by_spk.items():
        who: Speaker = "agent" if spk == agent_spk else "caller"
        segs += [Segment(who, a, b) for a, b in regions]
    return segs
