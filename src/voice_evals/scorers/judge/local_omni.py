"""Self-hosted open audio-LLM judge backend (design spec §10.5).

Same ``base_judge`` rubric + parsing interface as the Gemini backend, so the two
are interchangeable. A real implementation runs an open audio-LLM (Qwen3-Omni /
Qwen2.5-Omni / Kimi-Audio) locally on a GPU. This is a clean stub: it wires the
interface and errors clearly when weights/GPU are absent. Full inference wiring
is a later phase — the point is that ``gemini.py`` and ``local_omni.py`` are
drop-in interchangeable behind ``base_judge``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ...config import Config
from ...logging_util import get_logger
from ..base import ScorerError
from .base_judge import BaseJudgeScorer

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_local_model(model_id: str):
    """Load the local audio-LLM once per process. Raises a clean ScorerError if
    the required deps / GPU / weights are unavailable."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ScorerError(
            "local_omni backend needs `pip install voice-evals[local-judge]` "
            f"(transformers + a GPU). Import error: {e}"
        )
    # Intentionally not downloading multi-GB weights implicitly. A real build
    # would construct the processor+model here and return them.
    raise ScorerError(
        f"local_omni backend is a stub: wire up {model_id} inference (processor + "
        "generate) before enabling. Use the Gemini backend, or implement here."
    )


class LocalOmniJudge(BaseJudgeScorer):
    name = "audio_judge"
    layer = "judge"

    def _call_model(
        self, system: str, rubric_block: str, wav_bytes: bytes, mime: str, nudge: Optional[str]
    ) -> str:
        model = _load_local_model(self.config.judge.local_model)  # raises until wired
        raise ScorerError("local_omni inference not implemented")  # pragma: no cover
