"""OpenAI native-audio judge backend.

Same ``base_judge`` rubric + parsing interface as the Gemini backend, so they are
interchangeable. Uses the Chat Completions API with an ``input_audio`` content
part on an audio-capable model (e.g. ``gpt-4o-audio-preview``) and JSON-object
output; the base class handles schema validation, the one-shot retry, and the
``unscoreable`` salvage.

NOTE ON SELF-ENHANCEMENT BIAS: do not trust an OpenAI judge grading audio from an
OpenAI agent (e.g. gpt-realtime) as an absolute verdict — a model judging its own
family inflates scores. Set ``judge.judge_model_family: openai`` and keep
``exclude_if_model_family_matches: true`` (with ``tts_provider: openai`` in the
sidecar) for real runs; it is fine for a smoke test of the pipeline.
"""

from __future__ import annotations

import base64
import time
from functools import lru_cache
from typing import Optional

from ...config import load_secrets
from ...logging_util import get_logger
from ..base import ScorerError
from .base_judge import DIMENSIONS, BaseJudgeScorer

log = get_logger(__name__)

# json_object mode doesn't enforce a schema, so spell out the exact shape. Built
# from DIMENSIONS so it never drifts from JudgeOutput.
_SCHEMA_HINT = (
    "Return ONLY a JSON object with EXACTLY these keys:\n"
    f"  {', '.join(DIMENSIONS)} — each an object "
    '{"reasoning": "<1-2 sentences citing m:ss>", "score": <integer 1-5>};\n'
    '  "overall": <integer 1-5>;\n'
    '  "notable_timestamps": ["m:ss — description", ...];\n'
    '  "summary": "<2-3 sentences>";\n'
    '  "unscoreable": <true|false>; "unscoreable_reason": <string|null>;\n'
    '  "rubric_version": "<the RUBRIC_VERSION from the rubric above>".\n'
    'Write each dimension\'s "reasoning" before its "score". If the audio cannot be'
    " scored (silence/empty/too short), set unscoreable=true and omit the dimension"
    " scores."
)


@lru_cache(maxsize=1)
def _client(api_key: str):
    try:
        from openai import OpenAI
    except Exception as e:  # noqa: BLE001
        raise ScorerError(
            "OpenAI backend unavailable. Install with `pip install voice-evals[judge-openai]` "
            f"(needs `openai`). Import error: {e}"
        )
    return OpenAI(api_key=api_key)


class OpenAIAudioJudge(BaseJudgeScorer):
    name = "audio_judge"
    layer = "judge"

    def _call_model(
        self, system: str, rubric_block: str, wav_bytes: bytes, mime: str, nudge: Optional[str]
    ) -> str:
        key = load_secrets().openai_api_key
        if not key:
            raise ScorerError("no OPENAI_API_KEY in environment")
        client = _client(key)

        fmt = "wav" if "wav" in mime else ("mp3" if "mp3" in mime or "mpeg" in mime else "wav")
        b64 = base64.b64encode(wav_bytes).decode()
        user_text = f"{rubric_block}\n\n{_SCHEMA_HINT}"
        if nudge:
            user_text += f"\n\n{nudge}"

        def _do():
            # NOTE: the gpt-audio family does not accept response_format=json_object,
            # so we instruct JSON-only in the prompt and extract/validate it in the
            # base class (which strips fences and retries once on parse failure).
            return client.chat.completions.create(
                model=self.config.judge.model,
                modalities=["text"],
                temperature=self.config.judge.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                        ],
                    },
                ],
            )

        resp = self._with_backoff(_do)
        text = resp.choices[0].message.content if resp.choices else None
        if not text:
            raise ScorerError("openai returned empty response")
        return text

    def _with_backoff(self, fn):
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(self.config.judge.max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                retryable = any(c in msg for c in ("429", "500", "502", "503", "rate", "timeout", "overloaded"))
                last = e
                if not retryable or attempt == self.config.judge.max_retries - 1:
                    raise ScorerError(f"openai call failed: {e}")
                log.warning("openai transient error (attempt %d), backing off %.1fs: %s", attempt + 1, delay, e)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise ScorerError(f"openai call failed after retries: {last}")
