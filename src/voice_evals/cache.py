"""Content-addressed result cache (design spec §12.2).

Cache key = ``sha256(clip_id + scorer_name + scorer_config_hash)``. Because the
scorer's config hash is part of the key, any config change (including a rubric
version bump) invalidates the relevant entries automatically. Judge calls are
therefore never repeated for an unchanged (clip, config).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .logging_util import get_logger
from .models import ScoreResult

log = get_logger(__name__)


def result_cache_key(clip_id: str, scorer_name: str, config_hash: str) -> str:
    h = hashlib.sha256()
    h.update(clip_id.encode())
    h.update(b"\x00")
    h.update(scorer_name.encode())
    h.update(b"\x00")
    h.update(config_hash.encode())
    return h.hexdigest()


class ResultCache:
    """File-backed cache for ``ScoreResult`` objects and decoded audio artifacts."""

    def __init__(self, cache_dir: Path | str, enabled: bool = True):
        self.root = Path(cache_dir)
        self.results_dir = self.root / "results"
        self.audio_dir = self.root / "audio"
        self.models_dir = self.root / "models"
        # ``enabled`` gates only result get/put (the --no-cache path). Decoded
        # audio + model dirs are always created: decode-once is not optional, and
        # the runner needs the decoded files to score regardless of --no-cache.
        self.enabled = enabled
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # result cache
    # ------------------------------------------------------------------ #
    def _result_path(self, key: str) -> Path:
        return self.results_dir / f"{key}.json"

    def get(self, clip_id: str, scorer_name: str, config_hash: str) -> Optional[ScoreResult]:
        if not self.enabled:
            return None
        key = result_cache_key(clip_id, scorer_name, config_hash)
        p = self._result_path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            result = ScoreResult.from_dict(data)
            result.cached = True
            return result
        except Exception as e:  # corrupted cache entry: treat as a miss
            log.warning("ignoring corrupt cache entry %s: %s", p.name, e)
            return None

    def put(self, result: ScoreResult) -> None:
        if not self.enabled:
            return
        key = result_cache_key(result.clip_id, result.scorer, result.scorer_config_hash)
        p = self._result_path(key)
        to_store = result.to_dict()
        to_store["cached"] = False  # cached flag is set on read, not write
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(to_store, indent=2, ensure_ascii=False))
        tmp.replace(p)  # atomic on POSIX

    # ------------------------------------------------------------------ #
    # decoded-audio cache
    # ------------------------------------------------------------------ #
    def audio_dir_for(self, clip_id: str) -> Path:
        d = self.audio_dir / clip_id
        d.mkdir(parents=True, exist_ok=True)  # always: decoded audio is not gated
        return d

    # ------------------------------------------------------------------ #
    # maintenance
    # ------------------------------------------------------------------ #
    def clear(self, scorer: Optional[str] = None) -> int:
        """Delete cached results. If ``scorer`` is given, only that scorer's results
        whose stored payload names it. Returns the number of files removed."""
        if not self.results_dir.exists():
            return 0
        removed = 0
        for p in self.results_dir.glob("*.json"):
            if scorer is not None:
                try:
                    if json.loads(p.read_text()).get("scorer") != scorer:
                        continue
                except Exception:
                    continue
            p.unlink()
            removed += 1
        return removed
