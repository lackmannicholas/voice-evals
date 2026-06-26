"""Gemini native-audio judge backend (design spec §10.2, judge-prompt spec §4).

Sends the audio bytes (not a transcript) to a native-audio Gemini model with the
shared rubric, forcing structured JSON output via ``response_schema``. Transport
errors (429/5xx) retry with exponential backoff; schema/parse retries are handled
by the base class.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

from ...config import Config, load_secrets
from ...logging_util import get_logger
from ..base import ScorerError
from .base_judge import BaseJudgeScorer, JudgeOutput

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _client(api_key: str):
    try:
        from google import genai
    except Exception as e:  # noqa: BLE001
        raise ScorerError(
            "Gemini backend unavailable. Install with `pip install voice-evals[judge]` "
            f"(needs `google-genai`). Import error: {e}"
        )
    return genai.Client(api_key=api_key)


class GeminiJudge(BaseJudgeScorer):
    name = "audio_judge"
    layer = "judge"

    def _call_model(
        self, system: str, rubric_block: str, wav_bytes: bytes, mime: str, nudge: Optional[str]
    ) -> str:
        key = load_secrets().resolved_gemini_key
        if not key:
            raise ScorerError("no GEMINI_API_KEY/GOOGLE_API_KEY in environment")

        from google.genai import types

        client = _client(key)
        user_text = rubric_block if not nudge else f"{rubric_block}\n\n{nudge}"

        # inline below the threshold; Files API above it
        if len(wav_bytes) <= self.config.judge.upload_threshold_bytes:
            audio_part = types.Part.from_bytes(data=wav_bytes, mime_type=mime)
        else:
            import io

            uploaded = client.files.upload(
                file=io.BytesIO(wav_bytes), config={"mime_type": mime}
            )
            audio_part = uploaded

        gen_config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=JudgeOutput,
            temperature=self.config.judge.temperature,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.config.judge.thinking_budget
            ),
        )

        resp = self._with_backoff(
            lambda: client.models.generate_content(
                model=self.config.judge.model,
                contents=[audio_part, user_text],
                config=gen_config,
            )
        )
        text = getattr(resp, "text", None)
        if not text:
            raise ScorerError("gemini returned empty response")
        return text

    def _with_backoff(self, fn):
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(self.config.judge.max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                retryable = any(c in msg for c in ("429", "500", "503", "rate", "timeout", "unavailable"))
                last = e
                if not retryable or attempt == self.config.judge.max_retries - 1:
                    raise ScorerError(f"gemini call failed: {e}")
                log.warning("gemini transient error (attempt %d), backing off %.1fs: %s", attempt + 1, delay, e)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise ScorerError(f"gemini call failed after retries: {last}")
