"""Scorer protocol, base class, and registry (design spec §7).

Contract:
  * A scorer is a pure function of ``(AudioClip, config) -> ScoreResult``.
  * It NEVER raises out of ``score()`` — failures are returned as
    ``ScoreResult(error=...)``. The runner never sees a scorer exception.
  * Heavy model imports/loads are lazy (inside methods) and process-cached, so
    importing a scorer module is always cheap and a 500-clip run loads weights
    once.
  * ``config_hash()`` folds in everything that affects output, so the cache
    invalidates automatically when config changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from ..config import Config
from ..logging_util import get_logger
from ..models import AudioClip, Layer, ScoreResult

log = get_logger(__name__)


def stable_hash(obj: Any) -> str:
    """Deterministic short hash of a JSON-able object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ScorerError(Exception):
    """Raised inside a scorer's ``_compute`` to signal a clean, recorded failure.

    Use this for expected failure modes (e.g. judge 'unscoreable', missing input
    channel) where you want a specific error string and optional raw payload,
    rather than a generic traceback.
    """

    def __init__(self, message: str, raw: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.raw = raw or {}


@runtime_checkable
class Scorer(Protocol):
    name: str
    layer: Layer

    single_flight: bool

    def config_hash(self) -> str: ...
    def applicable(self, clip: AudioClip) -> bool: ...
    def score(self, clip: AudioClip) -> ScoreResult: ...


class BaseScorer:
    """Convenience base implementing timing, error isolation, and config hashing.

    Subclasses set ``name`` / ``layer`` / ``version`` and implement
    ``_compute(clip) -> (metrics, raw)``. Override ``applicable`` and
    ``config_dict`` as needed.
    """

    name: str = "base"
    layer: Layer = "acoustic"
    version: str = "1"  # bump to invalidate cache on a behavioral change
    # Whether this scorer must run one clip at a time. Default True because the
    # shipped scorers wrap a shared, not-necessarily-thread-safe model (torch /
    # onnx / silero) or shared segmentation state. Set False ONLY for a scorer
    # verified safe to run concurrently across clips (the runner will then let the
    # thread pool parallelize it). Judge scorers ignore this (semaphore-bounded).
    single_flight: bool = True

    def __init__(self, config: Config):
        self.config = config

    # -- config / cache key -------------------------------------------- #
    def config_dict(self) -> dict[str, Any]:
        """Subset of config that affects this scorer's output. Folded into the
        cache key. Override to include only the relevant section."""
        return {}

    def config_hash(self) -> str:
        return stable_hash(
            {"name": self.name, "version": self.version, "config": self.config_dict()}
        )

    # -- applicability -------------------------------------------------- #
    def applicable(self, clip: AudioClip) -> bool:  # noqa: ARG002
        return True

    # -- compute (subclass implements) --------------------------------- #
    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError

    # -- public entry: never raises ------------------------------------ #
    def score(self, clip: AudioClip) -> ScoreResult:
        t0 = time.perf_counter()
        cfg_hash = self.config_hash()
        try:
            metrics, raw = self._compute(clip)
            return ScoreResult(
                clip_id=clip.clip_id,
                scorer=self.name,
                layer=self.layer,
                metrics={k: float(v) for k, v in metrics.items()},
                raw=raw,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                cached=False,
                scorer_config_hash=cfg_hash,
            )
        except ScorerError as e:
            log.warning("%s failed on %s: %s", self.name, clip.clip_id, e.message)
            return ScoreResult(
                clip_id=clip.clip_id,
                scorer=self.name,
                layer=self.layer,
                metrics={},
                raw=e.raw,
                error=e.message,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                cached=False,
                scorer_config_hash=cfg_hash,
            )
        except Exception as e:  # noqa: BLE001 - error isolation is the whole point
            log.exception("%s crashed on %s", self.name, clip.clip_id)
            return ScoreResult(
                clip_id=clip.clip_id,
                scorer=self.name,
                layer=self.layer,
                metrics={},
                raw={},
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                cached=False,
                scorer_config_hash=cfg_hash,
            )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ScorerBuilder = Callable[[Config], Scorer]
_REGISTRY: dict[str, ScorerBuilder] = {}
_LAYER_OF: dict[str, Layer] = {}


def register(name: str, layer: Layer) -> Callable[[ScorerBuilder], ScorerBuilder]:
    """Register a scorer *builder* (a callable ``Config -> Scorer``)."""

    def deco(builder: ScorerBuilder) -> ScorerBuilder:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} already registered")
        _REGISTRY[name] = builder
        _LAYER_OF[name] = layer
        return builder

    return deco


def registered_names() -> list[str]:
    _ensure_imported()
    return sorted(_REGISTRY)


def layer_of(name: str) -> Optional[Layer]:
    _ensure_imported()
    return _LAYER_OF.get(name)


def build_scorer(name: str, config: Config) -> Scorer:
    _ensure_imported()
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config)


def build_selected(config: Config) -> list[Scorer]:
    """Instantiate every scorer named in ``config.scorers`` (acoustic+dynamics+judge)."""
    _ensure_imported()
    out: list[Scorer] = []
    for nm in config.scorers.all_selected():
        out.append(build_scorer(nm, config))
    return out


_IMPORTED = False


def _ensure_imported() -> None:
    """Import all scorer modules so their ``@register`` side effects fire.

    Each scorer module must keep heavy deps lazy (imported inside methods), so
    importing here is always cheap and registration never fails on a missing
    optional dependency.
    """
    global _IMPORTED
    if _IMPORTED:
        return
    _IMPORTED = True
    import importlib

    modules = [
        "voice_evals.scorers.acoustic.dnsmos",
        "voice_evals.scorers.acoustic.nisqa",
        "voice_evals.scorers.acoustic.utmos",
        "voice_evals.scorers.acoustic.squim",
        "voice_evals.scorers.dynamics.turn_taking",
        "voice_evals.scorers.dynamics.latency",
        "voice_evals.scorers.dynamics.barge_in",
        "voice_evals.scorers.judge.base_judge",  # registers "audio_judge"
    ]
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            log.warning("could not import scorer module %s: %s", m, e)
