"""UTMOS — learned naturalness MOS predictor (design spec §8.3).

Best single proxy for "does this sound like natural human speech"; TTS naturalness
regressions show up here first.

Implementation note: UTMOS22 (``tarepan/SpeechMOS``) collides on the module name
``speechmos`` with the Microsoft DNSMOS package, so UTMOS runs in an isolated,
*persistent* subprocess worker (``_utmos_worker``) that loads the model once. See
that module for the rationale. ``utmosv2`` (``sarulab-speech``), which uses a
distinct module name, is used in-process when installed.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ...config import Config
from ...logging_util import get_logger
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register
from .._audio import is_effectively_silent, load_wav, pick_acoustic_path

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# in-process UTMOSv2 (no name collision)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _utmosv2_model():
    import utmosv2  # type: ignore

    return utmosv2.create_model(pretrained=True)


# --------------------------------------------------------------------------- #
# persistent subprocess worker for torch.hub UTMOS22
# --------------------------------------------------------------------------- #
class _UtmosWorker:
    """Process-level singleton wrapping the persistent worker subprocess."""

    _instance: Optional["_UtmosWorker"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.backend = "utmos22_strong"

    @classmethod
    def get(cls) -> "_UtmosWorker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "voice_evals.scorers.acoustic._utmos_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        atexit.register(self._shutdown)
        ready = self._proc.stdout.readline()  # type: ignore[union-attr]
        try:
            msg = json.loads(ready)
        except Exception:
            raise ScorerError(f"UTMOS worker failed to start: {ready!r}")
        if "fatal" in msg:
            raise ScorerError(
                "UTMOS backend unavailable. Install `pip install voice-evals[acoustic]` "
                f"(torch + network for torch.hub). Worker error: {msg['fatal']}"
            )
        self.backend = msg.get("backend", self.backend)

    def _shutdown(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    def score_path(self, path: Path, req: Optional[dict] = None) -> tuple[float, int]:
        with self._lock:
            self._ensure_started()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            payload = {"path": str(path), **(req or {})}
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            resp = self._proc.stdout.readline()
            if not resp:
                self._proc = None
                raise ScorerError("UTMOS worker died mid-request")
            msg = json.loads(resp)
            if "error" in msg:
                raise ScorerError(f"UTMOS worker error: {msg['error']}")
            return float(msg["score"]), int(msg.get("n_windows", 1))


class UtmosScorer(BaseScorer):
    name = "utmos"
    layer = "acoustic"
    version = "1"

    def config_dict(self) -> dict[str, Any]:
        a = self.config.acoustic
        return {
            "target_sr": self.config.ingest.target_sr,
            "window_s": a.window_s,
            "min_window_s": a.min_window_s,
            "aggregate": a.aggregate,
        }

    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        from .._audio import aggregate_values, windows

        path, channel = pick_acoustic_path(clip)
        wav, sr = load_wav(path, target_sr=16000)
        if is_effectively_silent(wav):
            raise ScorerError("clip is effectively silent", raw={"input_channel": channel})
        a = self.config.acoustic

        # prefer in-process utmosv2 if available (windowed for long clips)
        try:
            model = _utmosv2_model()
            import numpy as np

            scores = []
            for w in windows(wav, sr, a.window_s, a.min_window_s):
                if is_effectively_silent(w):
                    continue
                res = model.predict(input_sr=16000, input_audio=np.asarray(w))
                scores.append(res["mos"] if isinstance(res, dict) else float(res))
            if scores:
                return {"utmos": aggregate_values(scores, a.aggregate)}, {
                    "input_channel": channel, "backend": "utmosv2", "n_windows": len(scores)}
        except ScorerError:
            raise
        except Exception:
            pass  # fall back to the subprocess worker

        worker = _UtmosWorker.get()
        score, n_windows = worker.score_path(
            path, {"window_s": a.window_s, "min_window_s": a.min_window_s, "aggregate": a.aggregate}
        )
        return {"utmos": float(score)}, {
            "input_channel": channel, "backend": worker.backend, "n_windows": n_windows}


@register("utmos", "acoustic")
def _build(config: Config) -> UtmosScorer:
    return UtmosScorer(config)
