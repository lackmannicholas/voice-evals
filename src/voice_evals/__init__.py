"""voice-evals: offline, reference-free batch evaluation of AI voice-call audio.

Three pluggable scorer layers (all cacheable, error-isolated):
  A. acoustic / perceptual MOS predictors (local, no labels, no network)
  B. conversational dynamics (turn-taking, latency, barge-in)
  C. native-audio LLM-as-judge (Gemini, or self-hosted fallback)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_secrets
from .models import (
    AudioClip,
    ChannelMap,
    ClipMetadata,
    EvalRun,
    GatewayEvent,
    ScoreResult,
)

__all__ = [
    "__version__",
    "Config",
    "load_secrets",
    "AudioClip",
    "ChannelMap",
    "ClipMetadata",
    "EvalRun",
    "GatewayEvent",
    "ScoreResult",
]
