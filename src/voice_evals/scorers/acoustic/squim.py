"""SQUIM — TorchAudio non-intrusive objective quality (design spec §8.4).

``SQUIM_OBJECTIVE`` gives reference-free estimates of STOI, PESQ, and SI-SDR —
intelligibility and distortion proxies that complement the MOS predictors.
``SQUIM_SUBJECTIVE`` needs a non-matching clean reference, so we skip it in v1 to
keep everything reference-free (stub left below).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ...config import Config
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register
from .._audio import aggregate_values, is_effectively_silent, load_wav, pick_acoustic_path, windows


@lru_cache(maxsize=1)
def _squim_objective_model():
    try:
        import torchaudio  # type: ignore

        model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        model.eval()
        return model
    except Exception as e:  # noqa: BLE001
        raise ScorerError(
            "SQUIM backend unavailable. Install with `pip install voice-evals[acoustic]` "
            f"(needs torchaudio). Error: {e}"
        )


class SquimScorer(BaseScorer):
    name = "squim"
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

        import torch

        model = _squim_objective_model()
        a = self.config.acoustic
        per: dict[str, list[float]] = {"squim_stoi": [], "squim_pesq": [], "squim_sisdr": []}
        # SQUIM allocates O(input length); windowing keeps memory bounded and the
        # input near the model's training distribution (short utterances).
        for w in windows(wav, sr, a.window_s, a.min_window_s):
            if is_effectively_silent(w):
                continue
            t = torch.from_numpy(w).unsqueeze(0)  # (1, samples) at 16k
            with torch.no_grad():
                stoi, pesq, sisdr = model(t)
            per["squim_stoi"].append(float(stoi.item()))
            per["squim_pesq"].append(float(pesq.item()))
            per["squim_sisdr"].append(float(sisdr.item()))
        if not per["squim_stoi"]:
            raise ScorerError("no scoreable (non-silent) windows", raw={"input_channel": channel})
        metrics = {k: aggregate_values(v, a.aggregate) for k, v in per.items()}
        return metrics, {"input_channel": channel, "n_windows": len(per["squim_stoi"]), "aggregate": a.aggregate}


@register("squim", "acoustic")
def _build(config: Config) -> SquimScorer:
    return SquimScorer(config)
