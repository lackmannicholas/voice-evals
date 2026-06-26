"""Acoustic scorers (design spec §8). Gated on optional [acoustic] deps.

Synthetic audio is unreliable for *strict* MOS ordering (the models are trained on
real speech), so these tests assert structural sanity + plausible ranges rather
than exact values or clean>noisy ordering. The exact, deterministic guarantees
live in the dynamics events tests and the gate suite.
"""

from __future__ import annotations

import math

import pytest

from voice_evals.config import Config
from voice_evals.models import AudioClip, ClipMetadata

pytestmark = [pytest.mark.local, pytest.mark.slow]


def _clip(path):
    return AudioClip("c", path, path, None, None, None, 16000, 2.0, 1, ClipMetadata())


def _finite(x):
    return x is not None and not math.isnan(x)


def test_dnsmos_structural(tone_wav):
    pytest.importorskip("speechmos")
    pytest.importorskip("librosa")
    from voice_evals.scorers.acoustic.dnsmos import DnsmosScorer

    r = DnsmosScorer(Config()).score(_clip(tone_wav))
    assert r.error is None, r.error
    for m in ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_p808"]:
        assert m in r.metrics and 0.0 <= r.metrics[m] <= 5.0
    assert r.raw["input_channel"] == "mono16k"


def test_squim_structural(tone_wav):
    pytest.importorskip("torchaudio")
    from voice_evals.scorers.acoustic.squim import SquimScorer

    r = SquimScorer(Config()).score(_clip(tone_wav))
    assert r.error is None, r.error
    assert 0.0 <= r.metrics["squim_stoi"] <= 1.0
    assert _finite(r.metrics["squim_pesq"]) and _finite(r.metrics["squim_sisdr"])


def test_utmos_structural(tone_wav):
    pytest.importorskip("torch")
    from voice_evals.scorers.acoustic.utmos import UtmosScorer

    r = UtmosScorer(Config()).score(_clip(tone_wav))
    # UTMOS uses torch.hub (network on first run); skip cleanly if unavailable
    if r.error is not None:
        pytest.skip(f"UTMOS backend unavailable: {r.error}")
    assert 1.0 <= r.metrics["utmos"] <= 5.0


def test_nisqa_errors_cleanly_without_weights(tone_wav):
    from voice_evals.scorers.acoustic.nisqa import NisqaScorer

    r = NisqaScorer(Config()).score(_clip(tone_wav))
    # without a vendored repo this must be a clean error, never a crash
    if r.error is None:
        assert "nisqa_mos" in r.metrics  # weights present in this env
    else:
        assert "NISQA" in r.error
