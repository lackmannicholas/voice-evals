"""Shared pytest fixtures: synthetic audio + tiny corpus (design spec §21).

No committed binaries — every fixture is generated on the fly. Heavy-model tests
import their backends lazily and skip when deps are absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_evals.cache import ResultCache
from voice_evals.config import Config
from voice_evals.models import AudioClip, ClipMetadata, GatewayEvent

SR = 16000


# --------------------------------------------------------------------------- #
# audio generators
# --------------------------------------------------------------------------- #
def _modulated_speech(dur: float, f0: float = 150.0) -> np.ndarray:
    """Harmonic stack with amplitude modulation — reliably triggers silero VAD."""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig = sum((0.4 / k) * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3))
    env = np.sin(2 * np.pi * 3 * t) * 0.5 + 0.5
    return (0.3 * sig * env).astype(np.float32)


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(SR * dur), dtype=np.float32)


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    p = tmp_path / "tone.wav"
    sf.write(p, _modulated_speech(2.0, 180), SR)
    return p


@pytest.fixture
def noise_wav(tmp_path: Path) -> Path:
    base = _modulated_speech(2.0, 180)
    noisy = base + 0.25 * np.random.RandomState(0).randn(len(base)).astype(np.float32)
    p = tmp_path / "noisy.wav"
    sf.write(p, noisy, SR)
    return p


@pytest.fixture
def glitch_wav(tmp_path: Path) -> Path:
    """Speech with inserted silence gaps and clicks (a 'choppy' clip)."""
    seg = _modulated_speech(0.6, 180)
    gap = _silence(0.4)
    sig = np.concatenate([seg, gap, seg, gap, seg]).astype(np.float32)
    sig[int(0.6 * SR)] = 0.9  # click
    p = tmp_path / "glitch.wav"
    sf.write(p, sig, SR)
    return p


@pytest.fixture
def stereo_conversation(tmp_path: Path) -> Path:
    """2-channel call: agent (ch0) speaks 3.0-5.0, caller (ch1) speaks 0.0-2.0.
    Known turn boundaries; sidecar declares the channel map."""
    n = int(SR * 6.0)
    agent = np.zeros(n, np.float32)
    caller = np.zeros(n, np.float32)
    caller[: int(2.0 * SR)] = _modulated_speech(2.0, 160)
    agent[int(3.0 * SR) : int(5.0 * SR)] = _modulated_speech(2.0, 120)
    p = tmp_path / "conv.wav"
    sf.write(p, np.stack([agent, caller], axis=1), SR)
    p.with_suffix(".json").write_text(
        json.dumps({"channel_map": {"agent_channel": 0, "caller_channel": 1},
                    "agent_version": "v1"})
    )
    return p


# --------------------------------------------------------------------------- #
# event-log clip (Appendix A) — needs no audio backend
# --------------------------------------------------------------------------- #
APPENDIX_A_EVENTS = [
    ("user_speech_start", 0.0),
    ("user_speech_end", 3.2),
    ("stt_final", 3.4),
    ("llm_first_token", 3.9),
    ("tts_first_audio", 4.1),
    ("agent_tts_start", 4.1),
    ("user_speech_start", 6.0),
    ("barge_in_detected", 6.05),
    ("agent_interrupted", 6.25),
    ("agent_tts_end", 6.25),
]


@pytest.fixture
def event_clip(tmp_path: Path) -> AudioClip:
    wav = tmp_path / "evt.wav"
    sf.write(wav, _modulated_speech(7.0, 150), SR)
    meta = ClipMetadata(
        call_id="abc123",
        agent_version="leasing-2.14.0",
        tts_provider="elevenlabs",
        events=[GatewayEvent(k, t) for k, t in APPENDIX_A_EVENTS],
    )
    return AudioClip(
        clip_id="evt1",
        source_path=wav,
        mono16k_path=wav,
        stereo_path=None,
        agent_only_path=None,
        caller_only_path=None,
        sample_rate=SR,
        duration_s=7.0,
        n_source_channels=1,
        metadata=meta,
    )


# --------------------------------------------------------------------------- #
# config + cache + corpus
# --------------------------------------------------------------------------- #
@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.corpus_dir = tmp_path / "corpus"
    cfg.cache_dir = tmp_path / ".cache"
    cfg.outputs_dir = tmp_path / "outputs"
    cfg.runner.progress = False
    return cfg


@pytest.fixture
def cache(config: Config) -> ResultCache:
    return ResultCache(config.cache_dir)


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    """A small corpus with a mono clip and a stereo+sidecar clip."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    sf.write(corpus / "mono.wav", _modulated_speech(2.0, 180), SR)
    n = int(SR * 6.0)
    agent = np.zeros(n, np.float32)
    caller = np.zeros(n, np.float32)
    caller[: int(2.0 * SR)] = _modulated_speech(2.0, 160)
    agent[int(3.0 * SR) : int(5.0 * SR)] = _modulated_speech(2.0, 120)
    sf.write(corpus / "conv.wav", np.stack([agent, caller], axis=1), SR)
    (corpus / "conv.json").write_text(
        json.dumps({"channel_map": {"agent_channel": 0, "caller_channel": 1},
                    "agent_version": "v1"})
    )
    return corpus


# --------------------------------------------------------------------------- #
# dependency probes (used to skip heavy tests)
# --------------------------------------------------------------------------- #
def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAVE_SPEECHMOS = _have("speechmos")
HAVE_TORCHAUDIO = _have("torchaudio")
HAVE_SILERO = _have("silero_vad")
HAVE_TORCH = _have("torch")

requires_speechmos = pytest.mark.skipif(not HAVE_SPEECHMOS, reason="speechmos not installed")
requires_torchaudio = pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="torchaudio not installed")
requires_silero = pytest.mark.skipif(not HAVE_SILERO, reason="silero-vad not installed")
