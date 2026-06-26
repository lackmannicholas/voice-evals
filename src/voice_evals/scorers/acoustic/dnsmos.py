"""DNSMOS — non-intrusive P.835 / P.808 MOS predictor (design spec §8.1).

Outputs SIG (speech), BAK (background-noise), OVRL (overall), plus P.808 overall.
Catches robotic/degraded TTS, background noise, codec artifacts.

Backend: the ``speechmos`` pip package (bundles the DNS-Challenge ONNX models).
Lazy-loaded and process-cached so a large run pays the load once.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ...config import Config
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register
from .._audio import aggregate_values, is_effectively_silent, load_wav, pick_acoustic_path, windows


@lru_cache(maxsize=1)
def _dnsmos_module():
    try:
        from speechmos import dnsmos  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ScorerError(
            "DNSMOS backend unavailable. Install with `pip install voice-evals[acoustic]` "
            f"(needs the `speechmos` package). Import error: {e}"
        )
    return dnsmos


def _f(d: dict[str, Any], *keys: str) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    raise ScorerError(f"DNSMOS output missing any of {keys}; got keys {sorted(d)}")


class DnsmosScorer(BaseScorer):
    name = "dnsmos"
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
        path, channel = pick_acoustic_path(clip)
        wav, sr = load_wav(path, target_sr=16000)
        if is_effectively_silent(wav):
            raise ScorerError("clip is effectively silent", raw={"input_channel": channel})
        dnsmos = _dnsmos_module()
        a = self.config.acoustic
        per: dict[str, list[float]] = {"dnsmos_ovrl": [], "dnsmos_sig": [], "dnsmos_bak": [], "dnsmos_p808": []}
        wins = windows(wav, sr, a.window_s, a.min_window_s)
        for w in wins:
            if is_effectively_silent(w):
                continue
            out = dnsmos.run(w, sr=sr, return_df=False)
            if hasattr(out, "to_dict"):
                out = {k: float(v) for k, v in out.to_dict().items()}
            per["dnsmos_ovrl"].append(_f(out, "ovrl_mos", "OVRL", "ovrl"))
            per["dnsmos_sig"].append(_f(out, "sig_mos", "SIG", "sig"))
            per["dnsmos_bak"].append(_f(out, "bak_mos", "BAK", "bak"))
            per["dnsmos_p808"].append(_f(out, "p808_mos", "P808", "p808"))
        if not per["dnsmos_ovrl"]:
            raise ScorerError("no scoreable (non-silent) windows", raw={"input_channel": channel})
        metrics = {k: aggregate_values(v, a.aggregate) for k, v in per.items()}
        return metrics, {"input_channel": channel, "n_windows": len(per["dnsmos_ovrl"]), "aggregate": a.aggregate}


@register("dnsmos", "acoustic")
def _build(config: Config) -> DnsmosScorer:
    return DnsmosScorer(config)
