"""Shared audio-loading helpers for scorers.

Kept dependency-light (numpy + soundfile only) so importing scorer modules never
pulls heavy ML deps. Torch tensors are produced on demand inside scorers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from ..models import AudioClip


def load_wav(path: Path, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Load a wav as a mono float32 array in [-1, 1]. Optionally resample.

    Resampling uses a light linear interpolator (no librosa dependency); it is
    adequate because our canonical inputs are already at the target rate and
    this path is only a safety net.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        data = _resample_linear(data, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(data, dtype=np.float32), sr


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * sr_out / sr_in))
    if n_out <= 1:
        return x[:1].copy()
    xp = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False)
    fp = x
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, fp).astype(np.float32)


def pick_acoustic_path(clip: AudioClip) -> tuple[Path, str]:
    """Prefer the isolated agent channel (we judge the agent's output quality),
    falling back to the mono mix. Returns (path, label) for ``raw["input_channel"]``."""
    if clip.agent_only_path is not None and Path(clip.agent_only_path).exists():
        return Path(clip.agent_only_path), "agent_only"
    return Path(clip.mono16k_path), "mono16k"


def is_effectively_silent(x: np.ndarray, eps: float = 1e-4) -> bool:
    return x.size == 0 or float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) < eps


def windows(wav: np.ndarray, sr: int, window_s: float, min_window_s: float) -> list[np.ndarray]:
    """Split a waveform into consecutive non-overlapping windows. MOS predictors
    are trained on short utterances and some (SQUIM) blow up on very long inputs,
    so long clips are scored window-by-window. A short trailing window is dropped
    (unless it is the only window)."""
    n = int(window_s * sr)
    if n <= 0 or wav.shape[0] <= n:
        return [wav]
    out: list[np.ndarray] = []
    i = 0
    min_n = int(min_window_s * sr)
    while i < wav.shape[0]:
        chunk = wav[i : i + n]
        if chunk.shape[0] >= min_n or not out:
            out.append(chunk)
        i += n
    return out


def aggregate_values(values: list[float], how: str = "median") -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan")
    return float(np.median(vals) if how == "median" else np.mean(vals))
