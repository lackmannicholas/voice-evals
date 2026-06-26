# Voice Audio Judge — Prompt & Rubric Spec

**Companion to:** `voice-evals-design-spec.md` (§10 Layer C — Audio LLM-as-judge)
**Purpose:** the exact prompt assets and assembly logic for the native-audio judge. Claude Code should extract the fenced files verbatim into the listed paths.

This file defines:
1. `config/rubric.default.txt` — the scoring rubric (dimensions + behavioral anchors).
2. `config/judge_system.txt` — the system instruction that wraps the rubric.
3. The `JudgeOutput` schema (a small, sign-safe revision of the spec's §10.3).
4. `build_judge_prompt(...)` — how rubric + context + audio + schema assemble into a call.
5. An optional pairwise variant for A/B model comparison.
6. Design rationale + calibration guidance.

---

## 0. Design principles (why the prompt is shaped this way)

- **CoT before score.** Each dimension's `reasoning` is generated *before* its `score`, and the schema field order enforces it. Chain-of-thought before the number measurably raises judge↔human agreement; a bare number is noisy.
- **Behavioral anchors.** Every dimension defines what 5 / 3 / 1 concretely *sound* like. "Rate naturalness 1–5" with no anchor produces drift; anchored scales cut variance and make scores comparable across runs and rubric versions.
- **Audio-grounded, not transcript-grounded.** The judge is told to evaluate *how it sounds*, to listen for acoustic phenomena, and to cite timestamps. It is explicitly told **not** to judge factual correctness or task success — that is the semantic layer's job. Keeping dimensions orthogonal is what makes the scores actionable.
- **Minimal context, no leakage.** The judge receives only the channel description and the call domain. It does **not** receive the transcript, the expected answer, the TTS vendor, or the model name — all of which would bias it (toward content, or toward a model family). Self-enhancement bias is handled in code (don't let a model judge its own family); the prompt stays clean.
- **Higher is always better.** Every dimension is scored so that 5 = best, 1 = worst. See the `fluency` note in §3 — this is a deliberate correction to avoid sign errors in gates.
- **Graceful degenerate handling.** Silence, missing agent speech, or corrupted audio must set `unscoreable=true` with a reason, not produce hallucinated scores.
- **Determinism.** temperature 0; rubric is versioned (`rubric_version`) so a rubric change is a cache-invalidating, calibration-invalidating event.

---

## 1. `config/rubric.default.txt`

Plain text, loaded at runtime and injected into the prompt. Edit dimensions/anchors here without touching code. **Bump `RUBRIC_VERSION` (top line) on any change** — it feeds the scorer config hash and the cache key.

```text
RUBRIC_VERSION: v1

You are scoring SEVEN independent dimensions of a voice agent's SPOKEN output.
Each dimension is scored 1 to 5, where 5 is always best and 1 is always worst.
Score each dimension on its own terms. Do not let a problem in one dimension
drag down the others. For every dimension, first write one or two sentences of
reasoning grounded in what you actually heard (cite timestamps as m:ss), then
give the integer score.

------------------------------------------------------------------------
1. NATURALNESS — does the agent's voice sound like a real human?
   Listen for: synthetic timbre, robotic or "TTS" artifacts, metallic or buzzy
   quality, unnatural breaths or their total absence, audio glitches.
   5 = Indistinguishable from a recorded human; warm, organic.
   3 = Clearly synthetic but acceptable; a listener would tell it is AI.
   1 = Heavily robotic, buzzy, or artifact-ridden; unpleasant to listen to.

2. PROSODY — is the intonation, stress, and rhythm appropriate for the content?
   Listen for: flat/monotone delivery, stress on the wrong words, question
   melody on statements (or vice versa), list items run together, emphasis that
   fights the meaning.
   5 = Expressive and correct; emphasis and melody match the meaning.
   3 = Somewhat flat or generic, but not misleading.
   1 = Monotone or actively wrong intonation that obscures meaning.

3. FLUENCY — is the speech free of unnatural disruptions?
   (Higher = MORE fluent / FEWER problems. A score of 5 means clean speech.)
   Listen for: mid-word cut-offs, clipped first/last syllables, stutters or
   repeated word fragments, audible splice seams, dropouts, and UNNATURAL
   PAUSES — gaps inside or between phrases that a human would not make.
   5 = Perfectly fluent; no cut-offs, no glitchy pauses, clean phrase boundaries.
   3 = One or two minor disruptions or slightly awkward pauses.
   1 = Frequent cut-offs, stutters, dropouts, or jarring mid-utterance gaps.

4. PACE — is the speaking rate appropriate and well-controlled?
   Listen for: rushed delivery, words crammed together, trailing off at phrase
   ends, unnaturally slow or dragging speech, abrupt rushed endings.
   5 = Comfortable, well-modulated rate; speeds up/slows appropriately.
   3 = A little fast or slow overall but followable.
   1 = So rushed or so slow that it harms comprehension or feels unnatural.

5. RESPONSIVENESS — does the agent sound prompt and engaged when responding?
   (Judge the PERCEIVED responsiveness from the audio: does the reply come in
   with natural timing and energy, or does it feel laggy, hesitant, or absent?
   Precise latency numbers are measured separately; here judge the felt sense.)
   5 = Replies land with natural, conversational timing and engaged energy.
   3 = Noticeable but tolerable lag or slightly disengaged delivery.
   1 = Long dead air before responses, or flat/checked-out delivery.

6. EMOTIONAL_ALIGNMENT — does the tone fit the situation?
   Listen for: cheerful tone over a complaint, cold tone over good news, tone
   that ignores apparent caller frustration, tonal whiplash between turns.
   5 = Tone is well-matched and steady for the apparent situation.
   3 = Generic/neutral tone; neither well-matched nor jarring.
   1 = Tone clearly clashes with the situation (e.g. upbeat during distress).

7. INTELLIGIBILITY — how easy is it to understand the agent end to end?
   Listen for: slurring, mumbling, mispronunciations, words lost under noise,
   garbled segments.
   5 = Effortless to understand throughout.
   3 = Mostly clear; a few words need effort or replay.
   1 = Significant portions are hard or impossible to make out.
------------------------------------------------------------------------

Then give:
- OVERALL (1-5): your holistic impression of the agent's spoken quality. This is
  NOT a strict average; weight what would most affect a real caller.
- NOTABLE_TIMESTAMPS: list specific moments worth a human's attention, each as
  "m:ss — short description" (e.g. "0:14 — 2s pause before answering").
- SUMMARY: two or three sentences a reviewer can read without listening.

IMPORTANT — what NOT to do:
- Do NOT judge whether the agent's answer was factually correct, complete, or
  on-policy. Content correctness is evaluated elsewhere. Judge only HOW IT SOUNDS.
- Do NOT assume or reconstruct a transcript. Base everything on the audio.
- Do NOT reward longer responses; length is not quality.
- If you cannot score the audio (silence, no agent speech, corrupted/empty, or
  too short to assess), set unscoreable=true and give the reason; leave scores
  at their schema defaults and do not guess.
```

---

## 2. `config/judge_system.txt`

The system instruction. Kept separate from the rubric so the rubric can be tuned independently. `{channel_desc}` and `{domain}` are filled at assembly time.

```text
You are an expert evaluator of voice AI agents. You assess the ACOUSTIC and
CONVERSATIONAL quality of an agent's speech — how it sounds to a caller — not
the correctness of what it says.

What you are hearing: {channel_desc}
Call domain: {domain}

You will be given an audio clip and a rubric of seven dimensions. Listen to the
entire clip. For each dimension, reason briefly from specific things you heard
(cite timestamps as m:ss), then assign an integer score from 1 to 5 where 5 is
best. Be calibrated and consistent: reserve 5 for genuinely excellent output and
1 for genuinely poor output; most acceptable production audio will land in the
3-4 range.

Return ONLY a JSON object matching the provided schema. Generate each
dimension's reasoning before its score.
```

`{channel_desc}` is one of:
- `agent_only` → `"the AI agent's audio in isolation (the caller is not present in this clip)"`
- `mono` → `"the full conversation, with the agent and the caller mixed into one channel"`
- `caller_only` → `"the caller's audio only"` (rare; used for input-quality checks)

`{domain}` default: `"a property leasing / resident-support phone call"` (config-overridable).

---

## 3. `JudgeOutput` schema (sign-safe revision of spec §10.3)

Two deliberate changes from the spec:
1. **`dysfluency` → `fluency`.** The spec field `dysfluency` is sign-ambiguous (does 5 mean more or less dysfluent?). Rename to `fluency` so 5=best holds uniformly across all dimensions and gate authors never flip a sign. Update the flattened metric to `judge_fluency`.
2. **Add `unscoreable` + `unscoreable_reason` + `rubric_version`** for degenerate inputs and cache/calibration hygiene.

Field order matters: within `DimensionScore`, `reasoning` precedes `score` so CoT is generated first.

```python
from pydantic import BaseModel, Field

class DimensionScore(BaseModel):
    reasoning: str = Field(description="1-2 sentences grounded in the audio; cite m:ss")
    score: int = Field(ge=1, le=5)

class JudgeOutput(BaseModel):
    naturalness: DimensionScore
    prosody: DimensionScore
    fluency: DimensionScore             # renamed from dysfluency; 5 = most fluent
    pace: DimensionScore
    responsiveness: DimensionScore
    emotional_alignment: DimensionScore
    intelligibility: DimensionScore
    overall: int = Field(ge=1, le=5, description="holistic, not a strict average")
    notable_timestamps: list[str] = Field(default_factory=list)
    summary: str
    unscoreable: bool = False
    unscoreable_reason: str | None = None
    rubric_version: str                  # echoed from the rubric file's top line

# Flattened into ScoreResult.metrics:
#   judge_naturalness, judge_prosody, judge_fluency, judge_pace,
#   judge_responsiveness, judge_emotional_alignment, judge_intelligibility,
#   judge_overall
# Full object (reasoning, timestamps, summary) goes into ScoreResult.raw.
# When unscoreable=true, emit no judge_* metrics and set ScoreResult.error
#   = unscoreable_reason so the runner/report surface it cleanly.
```

> Note for the gate config (spec §18): wherever the design doc references `judge_dysfluency`, use `judge_fluency`. All judge gates are `min` floors (higher=better), e.g. `judge_fluency: {min: 3}`.

---

## 4. Prompt assembly (`scorers/judge/base_judge.py`)

Shared by the Gemini and local-Omni backends. The audio bytes are the channel selected by `judge.judged_channel` (default `agent_only`, falling back to `mono16k` when no isolated channel exists — record which in `raw["judged_channel_actual"]`).

```python
def build_judge_prompt(rubric_text: str, channel_desc: str, domain: str) -> tuple[str, str]:
    """Returns (system_instruction, rubric_block). Audio is attached separately."""
    system = JUDGE_SYSTEM_TEMPLATE.format(channel_desc=channel_desc, domain=domain)
    return system, rubric_text

def parse_rubric_version(rubric_text: str) -> str:
    first = rubric_text.strip().splitlines()[0]
    # "RUBRIC_VERSION: v1" -> "v1"
    return first.split(":", 1)[1].strip() if ":" in first else "unknown"
```

Gemini call (backend `gemini.py`), using `google-genai`:

```python
from google import genai
from google.genai import types

system, rubric_block = build_judge_prompt(rubric_text, channel_desc, domain)
audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
# For clips that push request size, upload via client.files.upload(...) instead.

resp = client.models.generate_content(
    model=cfg.model,                          # e.g. "gemini-2.5-pro"
    contents=[audio_part, rubric_block],
    config=types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=JudgeOutput,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=1024),  # enables CoT
    ),
)
result = JudgeOutput.model_validate_json(resp.text)
```

Parsing discipline (both backends): validate against `JudgeOutput`; on validation/parse failure, **retry once** with a terse "return only valid JSON matching the schema" nudge, then on second failure return `ScoreResult(error="judge_parse_failure: ...")`. Never raise out of the scorer. The `scorer_config_hash` must include `rubric_version`, `model`, `judged_channel`, and `domain` so any of them changing invalidates the cache.

---

## 5. Optional pairwise mode (A/B model comparison)

For comparing two TTS/agent versions on the *same* call rather than scoring absolutely. Useful when you're evaluating a model swap and want a direct preference. Build later; interface stub now.

- Input: two clips (A, B) of the comparable content.
- Prompt: present both audios labeled "A" and "B"; ask for a per-dimension preference (`A` / `B` / `tie`) with reasoning, plus an overall preference.
- **Randomize A/B order per call and de-bias** by averaging both orderings (position bias is real in pairwise audio judging).
- Output: `PairwiseJudgeOutput { per_dimension: {dim: "A"|"B"|"tie"}, overall: ..., reasoning, ... }`.
- Keep absolute (pointwise) mode as the default for corpus regression; pairwise is for targeted bake-offs.

---

## 6. Calibration guidance (feeds spec §17)

- The golden set should be rated by a human on the **same 7 dimensions, same anchors**. Put the rubric's anchor text in the labeling instructions so human and judge share definitions — this alone closes much of the gap.
- Compute per-dimension Cohen's κ (treat 1–5 as ordinal; quadratic-weighted κ is appropriate) and Spearman correlation.
- Expect `intelligibility`, `naturalness`, and `fluency` to agree well (concrete, audible). Expect `emotional_alignment` and `prosody` to agree less (more subjective) — treat low-κ dimensions as **advisory**, not CI-gating.
- Only promote a dimension to an enforced CI gate when its weighted κ clears the configured bar (default ≥ 0.6).
- Re-run calibration whenever `rubric_version` or the judge model changes; both invalidate prior agreement numbers.
- Sanity check the anchors before trusting anything: run the judge on a deliberately clean TTS sample and a deliberately glitched one (the §21 fixtures) and confirm the clean clip scores materially higher on `fluency`/`naturalness`. If it doesn't, the anchors or channel selection are wrong, not the agent.

---

## 7. Files to create (summary)

| Path | Content |
|------|---------|
| `config/rubric.default.txt` | §1 verbatim |
| `config/judge_system.txt` | §2 verbatim |
| `scorers/judge/base_judge.py` | `JudgeOutput` (§3), `build_judge_prompt` / `parse_rubric_version` (§4), shared parse-retry logic |
| `scorers/judge/gemini.py` | §4 Gemini call behind the base interface |
| `scorers/judge/local_omni.py` | same interface, local model (stub ok in early phases) |

And one correction to the main spec: rename `dysfluency` → `fluency` everywhere (schema field, flattened metric `judge_fluency`, and the gate config).

---

*End of judge prompt spec.*
