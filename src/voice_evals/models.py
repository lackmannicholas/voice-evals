"""Core data model for voice-evals.

These dataclasses are the backbone described in the design spec (§5). Every type
here must be JSON-serializable so results can be cached, persisted, and replayed.

Design contract:
  * ``clip_id`` hashes the *decoded* PCM, never the source file, so an mp3 and its
    wav twin collide and the cache is immune to re-encode churn.
  * Scorers are pure functions ``(AudioClip, config) -> ScoreResult``; they never
    set ``passed`` (that is aggregation's job) and never raise (errors go in
    ``ScoreResult.error``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional

Layer = Literal["acoustic", "dynamics", "judge", "semantic"]

GatewayEventKind = Literal[
    "user_speech_start",
    "user_speech_end",
    "agent_tts_start",
    "agent_tts_end",
    "barge_in_detected",
    "agent_interrupted",
    "stt_final",
    "llm_first_token",
    "tts_first_audio",
]


# --------------------------------------------------------------------------- #
# JSON serialization helpers
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    """Recursively coerce a value into something ``json.dumps`` accepts.

    Handles Path, nested dataclasses, and numpy scalars/arrays (which leak in
    through scorer ``raw`` payloads).
    """
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # numpy scalars/arrays (imported lazily; numpy is a base dependency)
    if hasattr(value, "item") and hasattr(value, "dtype") and getattr(value, "ndim", None) == 0:
        return value.item()
    if hasattr(value, "tolist") and hasattr(value, "dtype"):
        return value.tolist()
    return value


# --------------------------------------------------------------------------- #
# Audio + metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelMap:
    """How to interpret channels in a stereo recording (0=left, 1=right)."""

    agent_channel: Optional[int]
    caller_channel: Optional[int]

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> Optional["ChannelMap"]:
        if not d:
            return None
        return cls(
            agent_channel=d.get("agent_channel"),
            caller_channel=d.get("caller_channel"),
        )


@dataclass
class GatewayEvent:
    """Optional precise timing event exported from the AI Voice Gateway."""

    kind: GatewayEventKind
    t_s: float  # seconds from call start
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GatewayEvent":
        return cls(kind=d["kind"], t_s=float(d["t_s"]), payload=d.get("payload", {}))


@dataclass
class ClipMetadata:
    """Parsed from an optional ``<clipname>.json`` sidecar. All fields optional."""

    call_id: Optional[str] = None
    agent_version: Optional[str] = None  # version-tagged regression
    prompt_version: Optional[str] = None
    model_id: Optional[str] = None  # which LLM/TTS produced agent audio
    tts_provider: Optional[str] = None
    channel_map: Optional[ChannelMap] = None
    transcript: Optional[str] = None  # if available, feeds the semantic hook
    domain: Optional[str] = None  # overrides judge default domain when present
    events: list[GatewayEvent] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClipMetadata":
        known = {
            "call_id",
            "agent_version",
            "prompt_version",
            "model_id",
            "tts_provider",
            "channel_map",
            "transcript",
            "domain",
            "events",
            "extra",
        }
        extra = dict(d.get("extra", {}))
        # stash any unrecognized top-level keys so nothing is silently dropped
        for k, v in d.items():
            if k not in known:
                extra[k] = v
        return cls(
            call_id=d.get("call_id"),
            agent_version=d.get("agent_version"),
            prompt_version=d.get("prompt_version"),
            model_id=d.get("model_id"),
            tts_provider=d.get("tts_provider"),
            channel_map=ChannelMap.from_dict(d.get("channel_map")),
            transcript=d.get("transcript"),
            domain=d.get("domain"),
            events=[GatewayEvent.from_dict(e) for e in d.get("events", [])],
            extra=extra,
        )

    @property
    def has_events(self) -> bool:
        return len(self.events) > 0


@dataclass
class AudioClip:
    clip_id: str  # blake2b of decoded mono PCM bytes (NOT the source file)
    source_path: Path
    mono16k_path: Path  # decoded 16kHz mono PCM wav (canonical for MOS)
    stereo_path: Optional[Path]  # original-channel wav if >1 channel present
    agent_only_path: Optional[Path]  # agent channel isolated, if channel_map known
    caller_only_path: Optional[Path]
    sample_rate: int  # of mono16k_path == 16000
    duration_s: float
    n_source_channels: int
    metadata: ClipMetadata

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ScoreResult:
    clip_id: str
    scorer: str  # registry name, e.g. "dnsmos"
    layer: Layer
    metrics: dict[str, float]  # named numeric scores
    raw: dict[str, Any] = field(default_factory=dict)  # raw output for debugging
    passed: Optional[bool] = None  # filled by aggregation; None until then
    error: Optional[str] = None  # populated instead of metrics on failure
    duration_ms: float = 0.0
    cached: bool = False
    scorer_config_hash: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreResult":
        return cls(
            clip_id=d["clip_id"],
            scorer=d["scorer"],
            layer=d["layer"],
            metrics={k: float(v) for k, v in (d.get("metrics") or {}).items()},
            raw=d.get("raw", {}),
            passed=d.get("passed"),
            error=d.get("error"),
            duration_ms=float(d.get("duration_ms", 0.0)),
            cached=bool(d.get("cached", False)),
            scorer_config_hash=d.get("scorer_config_hash", ""),
        )


@dataclass
class EvalRun:
    run_id: str  # timestamp + git sha if available
    corpus_dir: Path
    config_snapshot: dict[str, Any]
    results: list[ScoreResult]
    started_at: str
    finished_at: str
    clips: list[AudioClip] = field(default_factory=list)  # for metadata + report links

    # ------------------------------------------------------------------ #
    # convenience accessors used by aggregation, reporting, and gates
    # ------------------------------------------------------------------ #
    def results_for_clip(self, clip_id: str) -> list[ScoreResult]:
        return [r for r in self.results if r.clip_id == clip_id]

    def clip_by_id(self, clip_id: str) -> Optional[AudioClip]:
        for c in self.clips:
            if c.clip_id == clip_id:
                return c
        return None

    def metadata_for(self, clip_id: str) -> Optional[ClipMetadata]:
        c = self.clip_by_id(clip_id)
        return c.metadata if c else None

    def metric(self, clip_id: str, metric_name: str) -> Optional[float]:
        """Return the named metric for a clip, or None if absent/errored."""
        for r in self.results:
            if r.clip_id == clip_id and r.error is None and metric_name in r.metrics:
                return r.metrics[metric_name]
        return None

    @property
    def clip_ids(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.clip_id not in seen:
                seen.append(r.clip_id)
        return seen

    @property
    def metric_names(self) -> list[str]:
        names: list[str] = []
        for r in self.results:
            for m in r.metrics:
                if m not in names:
                    names.append(m)
        return sorted(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "corpus_dir": str(self.corpus_dir),
            "config_snapshot": _jsonable(self.config_snapshot),
            "results": [r.to_dict() for r in self.results],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalRun":
        return cls(
            run_id=d["run_id"],
            corpus_dir=Path(d["corpus_dir"]),
            config_snapshot=d.get("config_snapshot", {}),
            results=[ScoreResult.from_dict(r) for r in d.get("results", [])],
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            clips=[_clip_from_dict(c) for c in d.get("clips", [])],
        )


def _clip_from_dict(d: dict[str, Any]) -> AudioClip:
    def _p(key: str) -> Optional[Path]:
        v = d.get(key)
        return Path(v) if v else None

    return AudioClip(
        clip_id=d["clip_id"],
        source_path=Path(d["source_path"]),
        mono16k_path=Path(d["mono16k_path"]),
        stereo_path=_p("stereo_path"),
        agent_only_path=_p("agent_only_path"),
        caller_only_path=_p("caller_only_path"),
        sample_rate=int(d["sample_rate"]),
        duration_s=float(d["duration_s"]),
        n_source_channels=int(d["n_source_channels"]),
        metadata=ClipMetadata.from_dict(d.get("metadata", {})),
    )
