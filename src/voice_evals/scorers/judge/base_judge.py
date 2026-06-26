"""Shared judge schema, prompt assembly, and base scorer (judge-prompt spec §3–4).

The Gemini and local-Omni backends share this base. Key invariants:
  * field order in ``DimensionScore`` puts ``reasoning`` before ``score`` so the
    model generates chain-of-thought before the number,
  * ``fluency`` (not ``dysfluency``) so 5=best holds across all dimensions,
  * degenerate inputs set ``unscoreable=true`` → surfaced as ScoreResult.error
    with no judge_* metrics,
  * the scorer config hash folds in rubric_version, model, judged_channel, and
    domain so any change invalidates the cache + calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...config import Config
from ...logging_util import get_logger
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register

log = get_logger(__name__)

DIMENSIONS = [
    # spoken-quality dimensions (how the agent sounds)
    "naturalness",
    "prosody",
    "fluency",
    "pace",
    "responsiveness",
    "emotional_alignment",
    "intelligibility",
    # interaction dimensions (require the full mixed conversation to judge)
    "conversational_flow",
    "frustration_handling",
]

CHANNEL_DESC = {
    "agent_only": "the AI agent's audio in isolation (the caller is not present in this clip)",
    "mono": "the full conversation, with the agent and the caller mixed into one channel",
    "caller_only": "the caller's audio only",
}


# --------------------------------------------------------------------------- #
# Output schema (sign-safe revision of design-spec §10.3)
# --------------------------------------------------------------------------- #
class DimensionScore(BaseModel):
    reasoning: str = Field(description="1-2 sentences grounded in the audio; cite m:ss")
    score: int = Field(ge=1, le=5)


class JudgeOutput(BaseModel):
    naturalness: DimensionScore
    prosody: DimensionScore
    fluency: DimensionScore  # renamed from dysfluency; 5 = most fluent
    pace: DimensionScore
    responsiveness: DimensionScore
    emotional_alignment: DimensionScore
    intelligibility: DimensionScore
    conversational_flow: DimensionScore  # interaction-level (needs full conversation)
    frustration_handling: DimensionScore  # interaction-level (needs full conversation)
    overall: int = Field(ge=1, le=5, description="holistic, not a strict average")
    notable_timestamps: list[str] = Field(default_factory=list)
    summary: str = ""
    unscoreable: bool = False
    unscoreable_reason: Optional[str] = None
    rubric_version: str = "unknown"


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #
def build_judge_prompt(rubric_text: str, channel_desc: str, domain: str) -> tuple[str, str]:
    """Returns (system_instruction, rubric_block). Audio is attached separately."""
    system = _SYSTEM_TEMPLATE.format(channel_desc=channel_desc, domain=domain)
    return system, rubric_text


def parse_rubric_version(rubric_text: str) -> str:
    first = rubric_text.strip().splitlines()[0] if rubric_text.strip() else ""
    return first.split(":", 1)[1].strip() if ":" in first else "unknown"


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a model response that may wrap it in prose or
    a ```json fence (backends that can't enforce response_format=json)."""
    t = text.strip()
    if "```" in t:
        # take the content of the first fenced block
        parts = t.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if block.lstrip().lower().startswith("json"):
                block = block.lstrip()[4:]
            t = block.strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


# Fallback system template (used if config/judge_system.txt is missing).
_SYSTEM_TEMPLATE = (
    "You are an expert evaluator of voice AI agents. You assess the ACOUSTIC and\n"
    "CONVERSATIONAL quality of an agent's speech — how it sounds to a caller — not\n"
    "the correctness of what it says.\n\n"
    "What you are hearing: {channel_desc}\n"
    "Call domain: {domain}\n\n"
    "Listen to the entire clip. For each dimension, reason briefly from specific\n"
    "things you heard (cite timestamps as m:ss), then assign an integer score from\n"
    "1 to 5 where 5 is best. Return ONLY a JSON object matching the provided schema.\n"
    "Generate each dimension's reasoning before its score.\n"
)


def _flatten(out: JudgeOutput) -> dict[str, float]:
    metrics = {f"judge_{dim}": float(getattr(out, dim).score) for dim in DIMENSIONS}
    metrics["judge_overall"] = float(out.overall)
    return metrics


def _salvage_unscoreable(raw_text: str, rubric_version: str) -> Optional[JudgeOutput]:
    """If the model declared the clip unscoreable but (per the rubric) omitted the
    dimension scores, build a valid unscoreable JudgeOutput. Returns None if the
    response is not an unscoreable declaration. The filler dimension scores are
    never surfaced — ``_compute`` short-circuits to an error on ``unscoreable``."""
    try:
        d = json.loads(raw_text)
    except Exception:
        return None
    if not (isinstance(d, dict) and d.get("unscoreable")):
        return None
    payload: dict[str, Any] = {dim: {"reasoning": "", "score": 3} for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        if isinstance(d.get(dim), dict):
            payload[dim] = d[dim]
    payload.update(
        {
            "overall": d.get("overall", 3),
            "notable_timestamps": d.get("notable_timestamps", []),
            "summary": d.get("summary", ""),
            "unscoreable": True,
            "unscoreable_reason": d.get("unscoreable_reason") or "unscoreable",
            "rubric_version": d.get("rubric_version") or rubric_version,
        }
    )
    try:
        return JudgeOutput.model_validate(payload)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# base judge scorer
# --------------------------------------------------------------------------- #
class BaseJudgeScorer(BaseScorer):
    name = "audio_judge"
    layer = "judge"
    version = "1"

    def __init__(self, config: Config):
        super().__init__(config)
        self._rubric_text = self._load_text(config.judge.rubric_path, _DEFAULT_RUBRIC_HINT)
        self._system_template = self._load_text(config.judge.system_prompt_path, _SYSTEM_TEMPLATE)
        self.rubric_version = parse_rubric_version(self._rubric_text)

    # -- config / cache key -------------------------------------------- #
    def config_dict(self) -> dict[str, Any]:
        return {
            "backend": self.config.judge.backend,
            "model": self.config.judge.model,
            "judged_channel": self.config.judge.judged_channel,
            "domain": self.config.judge.domain,
            "rubric_version": self.rubric_version,
            "thinking_budget": self.config.judge.thinking_budget,
            "temperature": self.config.judge.temperature,
        }

    # -- self-enhancement guard ---------------------------------------- #
    def applicable(self, clip: AudioClip) -> bool:
        if not self.config.judge.exclude_if_model_family_matches:
            return True
        provider = (clip.metadata.tts_provider or "").lower()
        family = self.config.judge.judge_model_family.lower()
        if provider and family and (family in provider or provider in family):
            log.info(
                "skipping judge on %s: tts_provider %r collides with judge family %r",
                clip.clip_id, provider, family,
            )
            return False
        return True

    def _read_audio_capped(self, path: Path) -> bytes:
        """Return the judged audio as wav bytes, truncated to ``judge.max_audio_s``
        when that is set (>0) — keeps a long call cheap and within model limits."""
        cap = self.config.judge.max_audio_s
        if not cap or cap <= 0:
            return path.read_bytes()
        import io

        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        n = int(cap * sr)
        if data.shape[0] <= n:
            return path.read_bytes()
        buf = io.BytesIO()
        sf.write(buf, data[:n], sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    # -- channel selection --------------------------------------------- #
    def _select_channel(self, clip: AudioClip) -> tuple[Path, str]:
        want = self.config.judge.judged_channel
        if want == "agent_only" and clip.agent_only_path and Path(clip.agent_only_path).exists():
            return Path(clip.agent_only_path), "agent_only"
        if want == "caller_only" and clip.caller_only_path and Path(clip.caller_only_path).exists():
            return Path(clip.caller_only_path), "caller_only"
        return Path(clip.mono16k_path), "mono"

    # -- compute ------------------------------------------------------- #
    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        path, channel_actual = self._select_channel(clip)
        wav_bytes = self._read_audio_capped(path)
        channel_desc = CHANNEL_DESC.get(channel_actual, CHANNEL_DESC["mono"])
        domain = clip.metadata.domain or self.config.judge.domain
        system = self._system_template.format(channel_desc=channel_desc, domain=domain)
        out = self._invoke_with_retry(system, self._rubric_text, wav_bytes, "audio/wav")

        if out.unscoreable:
            raise ScorerError(
                out.unscoreable_reason or "judge marked clip unscoreable",
                raw={"judged_channel_actual": channel_actual, **out.model_dump()},
            )

        metrics = _flatten(out)
        raw = {"judged_channel_actual": channel_actual, **out.model_dump()}
        return metrics, raw

    # -- parse + 1 retry (per judge-prompt spec §4) -------------------- #
    def _invoke_with_retry(
        self, system: str, rubric_block: str, wav_bytes: bytes, mime: str
    ) -> JudgeOutput:
        attempts = self.config.judge.parse_retries + 1
        last_err: Optional[str] = None
        for i in range(attempts):
            nudge = None if i == 0 else "Return ONLY valid JSON matching the schema. No prose."
            raw_text = self._call_model(system, rubric_block, wav_bytes, mime, nudge)
            raw_text = _extract_json(raw_text)
            try:
                out = JudgeOutput.model_validate_json(raw_text)
                if not out.rubric_version or out.rubric_version == "unknown":
                    out.rubric_version = self.rubric_version
                return out
            except Exception as e:  # noqa: BLE001
                # The rubric tells the model to OMIT dimension scores when the clip
                # is unscoreable. The strict schema (which enforces CoT-before-score
                # for real scores) rejects that, so salvage a well-formed unscoreable
                # response rather than misclassifying it as a parse failure.
                salvaged = _salvage_unscoreable(raw_text, self.rubric_version)
                if salvaged is not None:
                    return salvaged
                last_err = str(e)
                log.warning("judge parse failure (attempt %d/%d): %s", i + 1, attempts, e)
        raise ScorerError(f"judge_parse_failure: {last_err}")

    # -- subclasses implement: return raw JSON text -------------------- #
    def _call_model(
        self, system: str, rubric_block: str, wav_bytes: bytes, mime: str, nudge: Optional[str]
    ) -> str:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------- #
    @staticmethod
    def _load_text(path_str: str, fallback: str) -> str:
        try:
            p = Path(path_str)
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists():
                return p.read_text()
        except Exception as e:  # noqa: BLE001
            log.warning("could not load %s: %s; using fallback", path_str, e)
        return fallback


_DEFAULT_RUBRIC_HINT = (
    "RUBRIC_VERSION: v1\nScore seven dimensions (naturalness, prosody, fluency, pace,\n"
    "responsiveness, emotional_alignment, intelligibility) 1-5 where 5 is best, each\n"
    "with brief audio-grounded reasoning before the score, then overall, "
    "notable_timestamps, and summary."
)


@register("audio_judge", "judge")
def _build(config: Config) -> BaseJudgeScorer:
    backend = config.judge.backend
    if backend == "gemini":
        from .gemini import GeminiJudge

        return GeminiJudge(config)
    if backend == "openai":
        from .openai_audio import OpenAIAudioJudge

        return OpenAIAudioJudge(config)
    if backend == "local_omni":
        from .local_omni import LocalOmniJudge

        return LocalOmniJudge(config)
    raise KeyError(f"unknown judge backend {backend!r} (gemini | openai | local_omni)")
