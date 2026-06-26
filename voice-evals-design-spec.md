# Voice Audio Evaluation Framework — Design Spec

**Status:** Draft v1 — build spec for implementation
**Audience:** Claude Code (implementation), engineering review
**Build mode:** Standalone codebase, offline batch eval over a directory of production call recordings. No coupling to production services.

---

## 0. TL;DR for the implementer

Build a standalone Python package, `voice-evals`, that:

1. Ingests a directory of production call recordings (mp3, also wav/m4a/ogg), decodes them to a normalized internal format, and attaches optional sidecar metadata.
2. Runs a battery of **pluggable scorers** organized into three layers:
   - **Layer A — Acoustic / perceptual quality** (reference-free MOS predictors). Fully local, no labels, no network.
   - **Layer B — Conversational dynamics** (turn-taking, latency, barge-in). Local; exact when a gateway event log is present, estimated from VAD otherwise.
   - **Layer C — Audio LLM-as-judge** (native-audio model grades prosody, naturalness, dysfluency, appropriateness). Network (Gemini) by default, with a self-hosted open-model fallback.
3. Caches every scorer result keyed by `(clip content hash, scorer name, scorer config hash)` so reruns are free and judge calls are never repeated.
4. Emits structured results (Parquet + JSON), a human-readable HTML report, and integrates with **pytest** as regression gates.
5. Ships a **judge calibration harness**: a small human-labeled set to measure judge↔human agreement before trusting the judge in CI.

**Critical design principle: everything is reference-free / unsupervised.** There is no per-clip ground-truth labeling step anywhere in the core pipeline. The only human labeling is an optional ~30–50 clip calibration set used once to validate the judge.

Layers A and B must be fully functional with **zero API keys and zero network access**. Layer C is additive.

---

## 1. Goals and non-goals

### 1.1 Goals

- Evaluate the **audio bytes themselves**, not just transcripts. Capture "how it sounded": robotic synthesis, clipping, dropouts, unnatural prosody, awkward pauses, latency, failed barge-in.
- Run over **raw production recordings** dropped into a directory, with no manual labeling required to produce signal.
- Be **deterministic and cacheable** so it can gate CI on every prompt/model/pipeline change.
- Mirror the existing pytest-native behavioral eval framework in shape (parametrized tests, fixtures, threshold gates) so it feels native to the team.
- Be **provider-agnostic and self-hostable** on the judge layer for data-governance reasons (recordings contain resident/tenant PII).

### 1.2 Non-goals (v1)

- Not a live/streaming/online monitor. This is offline batch over recorded files. (Design should not preclude a future streaming adapter, but do not build it.)
- Not a synthetic-call generator / load simulator (that is the value prop of Hamming/Coval/Cekura; out of scope here).
- Not an STT/ASR system. Transcripts, if used at all, come from sidecar metadata or an optional pluggable transcriber; the framework does not own transcription quality.
- Not a labeling tool/UI. The calibration set is a flat file the team fills in by hand.

---

## 2. Background: the three-layer model

Voice quality decomposes into three orthogonal axes. The existing transcript evals only touch the semantic axis and cannot see the other two.

| Layer | Question it answers | Needs labels? | Needs network? | Build priority |
|-------|--------------------|---------------|----------------|----------------|
| A — Acoustic / perceptual | "Did it *sound* good? Artifacts, noise, robotic TTS, glitches?" | No (reference-free) | No | 1st (fastest signal) |
| B — Conversational dynamics | "Weird pauses? Latency? Did barge-in work?" | No (timestamps, not labels) | No | 2nd |
| C — Audio LLM-judge | "Natural prosody? Right tone? Did it hesitate/stumble? Appropriate response *as spoken*?" | No to run; small set to *trust* | Yes (or self-host) | 3rd |
| (Semantic — existing) | "What was said; resolved; correct tool calls?" | varies | varies | integration hook only |

Reference-free MOS predictors (Layer A) require neither a clean source reference nor labels — that is the meaning of "non-intrusive." The judge (Layer C) runs on a rubric and produces scores immediately; calibration only establishes how much to trust it.

---

## 3. High-level architecture

```
                         voice-evals
┌──────────────────────────────────────────────────────────────┐
│  corpus/  (drop .mp3 / .wav / .m4a here, + optional .json)     │
│      │                                                          │
│      ▼                                                          │
│  Ingestion ──► AudioClip  (normalized 16k mono wav + stereo    │
│   - decode      + metadata sidecar + content hash)             │
│   - resample                                                    │
│   - channel split                                               │
│      │                                                          │
│      ▼                                                          │
│  Runner ──► fans clips × scorers, respects applicability,       │
│             reads/writes Cache, collects ScoreResults           │
│      │                                                          │
│      ├── Layer A scorers: DNSMOS, NISQA, UTMOS, SQUIM          │
│      ├── Layer B scorers: TurnTaking, Latency, BargeIn         │
│      └── Layer C scorers: AudioJudge (Gemini | local Omni)     │
│      │                                                          │
│      ▼                                                          │
│  Aggregation ──► EvalRun (all ScoreResults + thresholds)       │
│      │                                                          │
│      ├──► results.parquet  +  run.json                         │
│      ├──► report.html                                          │
│      └──► pytest gates  (assert thresholds / no regression)    │
└──────────────────────────────────────────────────────────────┘

   Side utility:  calibration harness  (golden_set.csv ↔ judge)
```

### Data flow contract

1. Ingestion is the only component that touches raw files. Everything downstream operates on `AudioClip`.
2. Scorers are pure functions of `(AudioClip, config)` → `ScoreResult`. No scorer mutates shared state.
3. The Runner owns concurrency, caching, and error isolation (one scorer failing on one clip must not abort the run).
4. Thresholds/gates are applied at aggregation time, never inside scorers (scorers emit raw metrics; pass/fail is a policy layer).

---

## 4. Repository layout

```
voice-evals/
├── pyproject.toml
├── README.md
├── config/
│   ├── default.yaml            # default thresholds + scorer selection
│   └── example.local.yaml      # local-only (no judge) profile
├── corpus/                     # gitignored; user drops audio here
├── .cache/                     # gitignored; scorer result cache
├── outputs/                    # gitignored; run artifacts
├── calibration/
│   └── golden_set.csv          # human labels for judge validation
├── src/voice_evals/
│   ├── __init__.py
│   ├── models.py               # core dataclasses / pydantic types
│   ├── config.py               # config schema + loader
│   ├── ingest.py               # decode/normalize/hash/sidecar
│   ├── cache.py                # content-addressed result cache
│   ├── runner.py               # orchestration, concurrency, fan-out
│   ├── aggregate.py            # collect results, apply thresholds
│   ├── report.py               # HTML + Parquet/JSON writers
│   ├── cli.py                  # `voice-evals` entrypoint
│   ├── scorers/
│   │   ├── base.py             # Scorer protocol + registry
│   │   ├── acoustic/
│   │   │   ├── dnsmos.py
│   │   │   ├── nisqa.py
│   │   │   ├── utmos.py
│   │   │   └── squim.py
│   │   ├── dynamics/
│   │   │   ├── segmentation.py  # VAD / diarization → turn timeline
│   │   │   ├── turn_taking.py
│   │   │   ├── latency.py
│   │   │   └── barge_in.py
│   │   └── judge/
│   │       ├── base_judge.py    # shared rubric + parsing
│   │       ├── gemini.py        # hosted native-audio judge
│   │       └── local_omni.py    # self-hosted fallback (Qwen-Omni etc.)
│   ├── calibrate.py            # judge↔human agreement harness
│   └── semantic_hook.py        # optional adapter to existing transcript evals
└── tests/
    ├── test_ingest.py
    ├── test_cache.py
    ├── test_scorers_acoustic.py
    ├── test_scorers_dynamics.py
    ├── test_judge_parsing.py
    ├── conftest.py             # fixtures: synthetic clips, tiny corpus
    └── eval_gates/
        └── test_corpus_gates.py  # the actual regression gate suite
```

---

## 5. Core data model (`models.py`)

These types are the backbone. Implement as dataclasses (or pydantic where validation/serialization helps). Everything must be JSON-serializable.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Any

Layer = Literal["acoustic", "dynamics", "judge", "semantic"]

@dataclass(frozen=True)
class ChannelMap:
    """How to interpret channels in a stereo recording."""
    agent_channel: Optional[int]   # 0=left, 1=right, None=unknown/mono
    caller_channel: Optional[int]

@dataclass
class GatewayEvent:
    """Optional precise timing event exported from the AI Voice Gateway."""
    kind: Literal[
        "user_speech_start", "user_speech_end",
        "agent_tts_start", "agent_tts_end",
        "barge_in_detected", "agent_interrupted",
        "stt_final", "llm_first_token", "tts_first_audio",
    ]
    t_s: float                     # seconds from call start
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass
class ClipMetadata:
    """Parsed from optional <clipname>.json sidecar. All fields optional."""
    call_id: Optional[str] = None
    agent_version: Optional[str] = None   # for version-tagged regression
    prompt_version: Optional[str] = None
    model_id: Optional[str] = None        # which LLM/TTS produced agent audio
    tts_provider: Optional[str] = None
    channel_map: Optional[ChannelMap] = None
    transcript: Optional[str] = None      # if available, feeds semantic hook
    events: list[GatewayEvent] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class AudioClip:
    clip_id: str                  # stable: blake2b of decoded-PCM bytes (NOT the mp3)
    source_path: Path
    mono16k_path: Path            # decoded 16kHz mono PCM wav (canonical for MOS)
    stereo_path: Optional[Path]   # original-channel wav if >1 channel present
    agent_only_path: Optional[Path]   # agent channel isolated, if channel_map known
    caller_only_path: Optional[Path]
    sample_rate: int              # of mono16k_path == 16000
    duration_s: float
    n_source_channels: int
    metadata: ClipMetadata

@dataclass
class ScoreResult:
    clip_id: str
    scorer: str                   # registry name, e.g. "dnsmos"
    layer: Layer
    metrics: dict[str, float]     # named numeric scores
    passed: Optional[bool]        # filled by aggregation, None until then
    raw: dict[str, Any]           # raw scorer output for debugging
    error: Optional[str]          # populated instead of metrics on failure
    duration_ms: float
    cached: bool
    scorer_config_hash: str       # so cache invalidates when config changes

@dataclass
class EvalRun:
    run_id: str                   # timestamp + git sha if available
    corpus_dir: Path
    config_snapshot: dict
    results: list[ScoreResult]
    started_at: str
    finished_at: str
```

**`clip_id` must hash the decoded PCM, not the source file.** Two re-encodes of the same call should collide; an mp3 and its wav twin should be the same clip. This makes the cache content-addressed and immune to re-download/re-encode churn.

---

## 6. Ingestion (`ingest.py`)

### Responsibilities

1. Discover audio files in `corpus/` (`.mp3 .wav .m4a .ogg .flac`), recursively, deterministically sorted.
2. For each, locate an optional sidecar `<same-stem>.json` and parse into `ClipMetadata`.
3. Decode to **16 kHz mono 16-bit PCM wav** → `mono16k_path`. This is the canonical input for all MOS models (DNSMOS expects 16 kHz; others tolerate it).
4. If source has ≥2 channels, also write a `stereo_path` preserving channels. If `channel_map` is known (from sidecar), write isolated `agent_only_path` / `caller_only_path`.
5. Compute `clip_id` from decoded mono PCM bytes.
6. Cache decoded artifacts under `.cache/audio/<clip_id>/` so decode happens once.

### Implementation notes

- Use `ffmpeg` (via `imageio-ffmpeg` bundled binary, or `ffmpeg-python`/`subprocess`) for decode; fall back to `soundfile` + `librosa.resample` for wav. ffmpeg is the only reliable mp3/m4a path.
- mp3 is lossy and MOS models were trained largely on PCM, so **absolute** MOS values will be slightly depressed. This is acceptable: the framework is for **relative comparison and regression**, not certifying absolute MOS. Keep the decode path byte-stable so scores are comparable across runs. Document this clearly in README.
- Peak-normalize or leave gain untouched? **Leave gain untouched by default** (clipping/level is signal we want to measure), but expose `ingest.normalize_dbfs: Optional[float]` config for users who want loudness normalization before scoring.
- Validate: warn (don't fail) on clips <0.5 s or >30 min; both break some scorers.

### The channel question (high-leverage)

Layer B accuracy depends entirely on channel separation:

- **Dual-channel (agent L / caller R):** turn-taking and barge-in become near-exact. Strongly preferred. Since the team owns the gateway, request stereo capture if available.
- **Mono mixed:** Layer B must fall back to VAD + diarization to *estimate* who spoke when; barge-in detection on a mixed channel is lossy. Layer A (whole-call MOS) and Layer C (judge handles mixed audio) still work.

Design the dynamics scorers to **prefer**, in order: (1) gateway event log in sidecar → exact; (2) isolated agent/caller channels → accurate; (3) mono + diarization → estimated, flagged `estimated=True` in `raw`.

---

## 7. Scorer interface (`scorers/base.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Scorer(Protocol):
    name: str                     # unique registry key
    layer: Layer

    def config_hash(self) -> str:
        """Stable hash of this scorer's config; part of the cache key."""

    def applicable(self, clip: AudioClip) -> bool:
        """E.g. barge-in scorer returns False when no channel/event data."""

    def score(self, clip: AudioClip) -> ScoreResult:
        """Pure; must catch its own exceptions and return ScoreResult(error=...)."""
```

- A module-level **registry** maps `name -> Scorer factory`. Config selects which scorers run.
- Scorers that load heavy models (NISQA, UTMOS, judge) must lazy-load and **cache the loaded model at the process level** (module singleton or `functools.lru_cache`) so a 500-clip run loads weights once.
- Every scorer wraps its body in try/except and returns a `ScoreResult` with `error` set on failure. The runner never sees an exception from a scorer.

---

## 8. Layer A — Acoustic / perceptual scorers

All reference-free, all local, all run on `mono16k_path` (or `agent_only_path` when available — we care about the *agent's* output quality, not the caller's phone line; prefer agent channel, fall back to mono, record which in `raw["input_channel"]`).

### 8.1 DNSMOS (`scorers/acoustic/dnsmos.py`)

- **What:** Non-intrusive P.835 predictor. Outputs `SIG` (speech quality), `BAK` (background-noise quality), `OVRL` (overall), plus P.808 overall. Correlates ~0.94–0.98 with human MOS.
- **Library:** the `speechmos` pip package (`pip install speechmos`) exposes DNSMOS, or use the ONNX models from `microsoft/DNS-Challenge` directly via `onnxruntime`. Prefer whichever loads cleanly; verify the exact import at build time.
- **Input:** 16 kHz mono. Provide as float array.
- **metrics:** `{"dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"}`.
- **Catches:** robotic/degraded TTS, background noise, codec artifacts.

### 8.2 NISQA (`scorers/acoustic/nisqa.py`)

- **What:** Multidimensional speech-quality CNN-self-attention model. Outputs overall MOS plus four dimensions: **Noisiness, Coloration, Discontinuity, Loudness**.
- **The `Discontinuity` dimension is the primary glitch/dropout/choppy-audio detector** — directly relevant to "weird pause"/artifact complaints.
- **Library:** `gabrielmittag/NISQA` (GitHub, PyTorch, pretrained `nisqa.tar` weights). Vendor the weights into `.cache/models/` on first run (download script) or document manual placement. There may be a pip fork; verify.
- **metrics:** `{"nisqa_mos", "nisqa_noisiness", "nisqa_coloration", "nisqa_discontinuity", "nisqa_loudness"}`.

### 8.3 UTMOS (`scorers/acoustic/utmos.py`)

- **What:** Learned naturalness MOS predictor (VoiceMOS-trained). Best single proxy for "does this sound like natural human speech."
- **Library:** prefer **UTMOSv2** (`sarulab-speech/UTMOSv2`); fall back to UTMOS22 (`speechmos` bundles `utmos22_strong`). Verify import.
- **metrics:** `{"utmos"}`.

### 8.4 SQUIM (`scorers/acoustic/squim.py`)

- **What:** TorchAudio-SQUIM. `SQUIM_OBJECTIVE` gives **reference-free** estimates of STOI, PESQ, SI-SDR — intelligibility and distortion proxies that complement the MOS predictors. (`SQUIM_SUBJECTIVE` needs a non-matching clean reference; **skip it in v1** to keep everything reference-free, but leave a stub.)
- **Library:** built into `torchaudio` (`torchaudio.pipelines.SQUIM_OBJECTIVE`). Expects 16 kHz.
- **metrics:** `{"squim_stoi", "squim_pesq", "squim_sisdr"}`.

### 8.5 Recommended starting thresholds (Layer A)

These are **heuristic starting gates** to be calibrated against the team's own corpus; document them as such. All MOS-style scales are ~1–5.

| Metric | Warn below | Fail below | Notes |
|--------|-----------|-----------|-------|
| `dnsmos_ovrl` | 3.2 | 2.8 | overall perceived quality |
| `dnsmos_sig` | 3.3 | 3.0 | speech signal quality |
| `nisqa_mos` | 3.2 | 2.8 | corroborates DNSMOS |
| `nisqa_discontinuity` | 3.5 | 3.0 | low = choppy/dropouts; **regression-sensitive** |
| `utmos` | 3.3 | 3.0 | naturalness; TTS regressions show here |
| `squim_pesq` | 2.5 | 2.0 | distortion proxy |

Gates should support per-`agent_version` baselines (see §13): the most useful gate is "OVRL dropped >0.15 vs previous version," not just an absolute floor.

---

## 9. Layer B — Conversational dynamics

This is a **timing/instrumentation** problem, not an audio-content problem. "Weird pause" lives here.

### 9.1 Turn segmentation (`scorers/dynamics/segmentation.py`)

Produces a `Timeline`: ordered list of `(speaker: "agent"|"caller", start_s, end_s)` segments. Resolution order:

1. **Gateway events present** → derive turns directly from `agent_tts_start/end`, `user_speech_start/end`. Exact. `source="events"`.
2. **Isolated channels present** → run VAD (Silero, `pip install silero-vad`) per channel; agent VAD on agent channel, caller VAD on caller channel. Accurate. `source="channels"`.
3. **Mono only** → run VAD for speech regions + `pyannote.audio` diarization (free; HF-gated weights, needs free token) to assign speaker labels. Estimated. `source="diarization"`, set `estimated=True`.

This module is a dependency of the three scorers below, not a scorer itself (or expose it as a scorer that emits the timeline in `raw` and segment counts in `metrics`). The team already has deep VAD expertise and a TEN VAD integration — allow TEN VAD as a configurable alternative backend to Silero.

### 9.2 Latency (`scorers/dynamics/latency.py`)

Per agent turn, compute response latency = `agent_speech_start − preceding_caller_speech_end`. With events, prefer the precise chain `tts_first_audio − stt_final` (true TTFT). 

- **metrics:** `{"latency_p50_s", "latency_p95_s", "latency_p99_s", "latency_max_s", "n_turns"}`.
- Report **percentiles, never just mean** — the tail is what users feel. A good mean hides the 1-in-20 sluggish call.
- Starting gates: P50 ≤ ~1.7 s, P95 ≤ ~3.0 s (calibrate to product expectations).

### 9.3 Turn-taking quality (`scorers/dynamics/turn_taking.py`)

- **Awkward pause / gap detection** (the literal "weird pause"): flag inter-turn silences exceeding `gap_threshold_s` (default 1.5 s) that are not explained by a tool call / hold. Count and locate them.
- **Overlap / talk-over:** fraction of time agent and caller speak simultaneously (excluding legitimate backchannels).
- **metrics:** `{"awkward_gap_count", "max_gap_s", "overlap_ratio", "agent_talk_ratio"}`.

### 9.4 Barge-in (`scorers/dynamics/barge_in.py`)

Only `applicable()` when events or isolated channels exist (cannot be measured reliably on mono mix). Definitions (from duplex-dialogue literature):

- **Barge-in latency:** time from caller speech onset (during agent TTS) to agent TTS stop.
- **Success rate:** fraction of interruptions where agent stops within `success_window_s` (default 1.5 s; aspirational target ≤ 200 ms stop latency for "natural").
- **False-alarm rate:** agent stops/yields when caller did not actually take the floor (e.g. backchannel "uh huh"). Exclude caller utterances <100 ms.
- **metrics:** `{"bargein_latency_p50_s", "bargein_latency_p95_s", "bargein_success_rate", "bargein_false_alarm_rate", "n_bargein_events"}`.
- Gates: success_rate ≥ 0.90, latency_p95 ≤ 0.5 s (calibrate).

---

## 10. Layer C — Audio LLM-as-judge

### 10.1 Concept

Send the **audio bytes** (not a transcript) to a native-audio model with a rubric; it reasons over prosody, pauses, tone, and dysfluency directly. Reference frameworks: **AudioJudge** and the newer **TRACE** (cheaper, more paralinguistics-sensitive) — borrow their design choices. Validation literature shows native-audio judge↔human agreement approaching human↔human for speaking-style judgments, *with chain-of-thought before the score*.

### 10.2 Default backend: Gemini (`scorers/judge/gemini.py`)

- **SDK:** `google-genai` (`pip install google-genai`).
- **Model:** configurable; default `gemini-2.5-pro` (strong audio understanding) or `gemini-3-pro-preview`. Allow `gemini-3-flash` / `gemini-2.5-flash` for cheap bulk passes.
- **Input:** prefer `agent_only_path` if isolating agent quality; use full mono mix when judging *interaction* quality (turn-taking feel). Make the judged channel a rubric-level choice.
- **Call shape:**

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=settings.gemini_api_key)

# inline for <~20MB; use client.files.upload(...) for larger
audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")

resp = client.models.generate_content(
    model=cfg.model,
    contents=[audio_part, RUBRIC_PROMPT],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeOutput,        # pydantic model, see below
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=1024),  # CoT
    ),
)
```

- **Structured output:** force JSON via `response_schema`. The model must emit per-dimension reasoning (CoT) *and then* numeric scores. Parse defensively; on parse failure, return `ScoreResult(error=...)` rather than raising.

### 10.3 Rubric (`scorers/judge/base_judge.py`)

Multi-dimensional, each scored 1–5 with a one-sentence rationale. Rubric text is config-driven so the team can tune it without code changes. Default dimensions:

| Dimension | What the judge listens for |
|-----------|----------------------------|
| `naturalness` | Does the agent voice sound human vs robotic/synthetic? |
| `prosody` | Appropriate intonation, stress, rhythm for the content? |
| `dysfluency` | Unnatural pauses, stutters, restarts, clipped/cut-off words? (the "weird pause" the judge can *hear*) |
| `pace` | Too fast/slow; rushed endings; trailing-off? |
| `responsiveness` | Did the agent respond promptly and on-topic *as heard*? |
| `emotional_alignment` | Tone matches the situation (e.g. not chipper at a complaint)? |
| `intelligibility` | Easy to understand end-to-end? |

Output schema:

```python
from pydantic import BaseModel

class DimensionScore(BaseModel):
    score: int          # 1..5
    reasoning: str       # brief CoT, must precede score in generation

class JudgeOutput(BaseModel):
    naturalness: DimensionScore
    prosody: DimensionScore
    dysfluency: DimensionScore
    pace: DimensionScore
    responsiveness: DimensionScore
    emotional_alignment: DimensionScore
    intelligibility: DimensionScore
    overall: int                     # 1..5 holistic
    notable_timestamps: list[str]    # e.g. "0:14 long pause before answer"
    summary: str
```

`metrics` flattens to `{"judge_naturalness", ..., "judge_overall"}`; full reasoning goes in `raw`.

### 10.4 Self-enhancement bias guard

Do **not** let the judge grade audio produced by its own model family if avoidable (Gemini judging Gemini-TTS inflates scores). Expose `judge.exclude_if_model_family_matches` and skip/flag when `clip.metadata.tts_provider` collides with the judge family.

### 10.5 Self-hosted fallback (`scorers/judge/local_omni.py`)

For PII/data-residency or zero-cost bulk: a local open audio-LLM (Qwen3-Omni / Qwen2.5-Omni / Kimi-Audio) behind the **same `base_judge` rubric+parsing interface**. Weaker than Gemini, needs a GPU. Implement the interface and a stub that errors cleanly if weights/GPU absent; full wiring can be a later phase. The point is that `gemini.py` and `local_omni.py` are interchangeable behind `base_judge`.

### 10.6 Cost & governance notes (document in README)

- Audio tokenizes at ~32 tokens/sec; a 60 s clip ≈ ~1,900 audio tokens. At current rates (~$1/M audio input on Flash, ~$1.25/M on 2.5 Pro; output ~$3–10/M), a clip costs ~**$0.003–0.006**. A 200-clip run ≈ **$1–3**. Free tier (~1,000 req/day) covers small runs at $0; Batch API is 50% off for non-urgent runs.
- **Caching is mandatory** (judge calls keyed by clip+config hash) so reruns and CI cost nothing.
- **PII:** recordings contain resident/tenant data. Sending audio to a hosted API has governance implications — document a `local_only` profile that disables all network scorers, and note Vertex-with-DPA or the self-hosted judge as the compliant paths. Make hosted-judge usage an explicit opt-in flag, never the default in CI.

---

## 11. Semantic layer hook (`semantic_hook.py`)

Out of scope to build, but provide a thin adapter interface so the existing transcript/behavioral pytest framework can register results into the same `EvalRun`/report. Define a `SemanticResult -> ScoreResult(layer="semantic")` mapping and a registration function. No implementation beyond the adapter + a documented example.

---

## 12. Runner & caching

### 12.1 Runner (`runner.py`)

- Inputs: corpus dir, resolved config (selected scorers + their configs).
- Builds the clip list via ingestion (cached), builds the scorer list via registry.
- For each `(clip, scorer)` where `scorer.applicable(clip)`: check cache → run on miss → store. 
- **Concurrency:** thread/process pool for CPU MOS scorers; bounded async concurrency + rate limiting for the judge (configurable `judge.max_concurrency`, exponential backoff on 429/5xx). Local model scorers run single-flight (one GPU/model instance).
- **Error isolation:** a scorer error on one clip is recorded and the run continues. Run summary reports error counts per scorer.
- **Progress:** rich/tqdm progress bar; structured logging.

### 12.2 Cache (`cache.py`)

- Content-addressed. Key = `sha256(clip_id + scorer.name + scorer.config_hash())`.
- Store: `.cache/results/<key>.json` (the serialized `ScoreResult` with `cached=False` at write time; loader sets `cached=True`).
- Decoded audio cached under `.cache/audio/<clip_id>/`.
- `--no-cache` flag bypasses; `voice-evals cache clear [--scorer NAME]` utility.
- Cache must invalidate automatically when scorer config changes (that's why `config_hash` is in the key).

---

## 13. Aggregation, thresholds, regression (`aggregate.py`)

- Collect all `ScoreResult`s into a tidy table (one row per clip×scorer×metric).
- Apply gates from config to set `passed`. Two gate types:
  1. **Absolute:** `metric >= floor` / `<= ceiling`.
  2. **Regression:** compare current run to a **baseline run** grouped by `agent_version` / `prompt_version`. Flag when a metric degrades by more than `delta` vs baseline (this is the high-value gate — catches "the new prompt made the voice worse" even when still above absolute floor).
- Baseline storage: persist prior `EvalRun`s under `outputs/runs/<run_id>/`; baseline selectable by tag or "previous run."
- Aggregate stats per metric: mean, median, P95, pass rate, count, error count.

---

## 14. Reporting (`report.py`)

- **`results.parquet`** — tidy long-format table for ad-hoc analysis.
- **`run.json`** — full `EvalRun` serialization (config snapshot + all results).
- **`report.html`** — self-contained single file: summary table (pass rates, percentiles per metric), distribution plots per metric, a sortable per-clip table with links to play the audio and expand judge reasoning + flagged timestamps, and a regression diff vs baseline. No external CDN deps if feasible (inline JS/CSS) so it works offline.
- Worst-offenders view: clips sorted by composite failure, since those are the ones to listen to first.

---

## 15. pytest integration (`tests/eval_gates/`)

Mirror the existing behavioral framework's shape.

- A session-scoped fixture runs (or loads a cached) `EvalRun` over `corpus/` (or a small committed `tests/fixtures/mini_corpus/`).
- Parametrize over clips × gated metrics; each becomes an assertion.
- Two suites:
  - **`test_corpus_gates.py`** — absolute floors (fast, local-only by default; judge gated behind a marker `@pytest.mark.judge` so CI can run MOS+dynamics without API keys).
  - **regression gate** — fails if metrics degrade vs baseline beyond delta.
- Markers: `@pytest.mark.local` (no network), `@pytest.mark.judge` (needs key/GPU). Default CI runs `-m "not judge"`.

```python
@pytest.mark.parametrize("clip, metric, floor", gate_cases())
def test_acoustic_floor(eval_run, clip, metric, floor):
    val = eval_run.metric(clip.clip_id, metric)
    assert val is not None, f"{metric} missing for {clip.clip_id}"
    assert val >= floor, f"{clip.clip_id}: {metric}={val:.2f} < {floor}"
```

---

## 16. CLI (`cli.py`)

Use Typer/argparse. Commands:

- `voice-evals run --corpus corpus/ --config config/default.yaml [--no-judge] [--no-cache] [--baseline TAG]`
- `voice-evals ingest --corpus corpus/` (decode/normalize only; warm the cache)
- `voice-evals report --run outputs/runs/<id>` (regenerate HTML)
- `voice-evals calibrate --golden calibration/golden_set.csv` (see §17)
- `voice-evals cache clear [--scorer NAME]`

`--no-judge` / a `local_only` config profile must produce a complete, useful run (Layers A+B) with no network and no keys.

---

## 17. Judge calibration harness (`calibrate.py`)

The only place humans label, and it's optional + one-time.

- `calibration/golden_set.csv`: columns `clip_id, dimension, human_score(1-5)` for ~30–50 clips a person rated by listening.
- `voice-evals calibrate` runs the judge on those clips and computes agreement vs human: **Cohen's κ** and **Spearman/Pearson correlation** per dimension, plus macro agreement.
- Output a calibration report: which dimensions the judge can be trusted on (high κ) vs which to treat as advisory. Gate recommendation: only enforce judge dimensions in CI where κ clears a configurable bar (e.g. ≥ 0.6).
- Re-runnable when the judge model/rubric changes.

---

## 18. Configuration (`config.py`, `config/default.yaml`)

Pydantic-settings backed; YAML file + env overrides (`GEMINI_API_KEY` from env, never in YAML).

```yaml
ingest:
  target_sr: 16000
  normalize_dbfs: null          # null = leave gain untouched
  accept_extensions: [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

scorers:
  acoustic: [dnsmos, nisqa, utmos, squim]
  dynamics: [latency, turn_taking, barge_in]
  judge: []                     # empty by default => local_only safe

dynamics:
  vad_backend: silero           # silero | ten
  gap_threshold_s: 1.5
  bargein_success_window_s: 1.5

judge:
  backend: gemini               # gemini | local_omni
  model: gemini-2.5-pro
  judged_channel: agent_only    # agent_only | mono | caller_only
  max_concurrency: 4
  exclude_if_model_family_matches: true
  rubric_path: config/rubric.default.txt

gates:
  absolute:
    dnsmos_ovrl: {min: 2.8, warn: 3.2}
    nisqa_discontinuity: {min: 3.0, warn: 3.5}
    utmos: {min: 3.0, warn: 3.3}
    latency_p95_s: {max: 3.0, warn: 2.5}
    bargein_success_rate: {min: 0.90}
  regression:
    enabled: true
    baseline: previous
    group_by: [agent_version]
    max_delta:
      dnsmos_ovrl: 0.15
      utmos: 0.15
```

---

## 19. Dependencies & environment

- Python 3.11+.
- Core: `numpy`, `soundfile`, `librosa`, `ffmpeg` (bundled via `imageio-ffmpeg`), `pydantic`, `pydantic-settings`, `pyyaml`, `pandas`, `pyarrow`, `typer`, `rich`, `tqdm`.
- Acoustic: `torch`, `torchaudio` (SQUIM), `onnxruntime` (DNSMOS), NISQA (vendored), UTMOSv2 (vendored or `speechmos`). Pin and document weight downloads; provide a `scripts/fetch_models.py` that pulls weights into `.cache/models/` from the allowed sources.
- Dynamics: `silero-vad`; optional `pyannote.audio` (HF token), optional TEN VAD backend.
- Judge: `google-genai`; optional local: `transformers`/`vllm` for Omni models (separate extra).
- Packaging extras: `voice-evals[acoustic]`, `[dynamics]`, `[judge]`, `[local-judge]`, `[all]`. The base + `[acoustic]` + `[dynamics]` install must work with **no GPU and no API keys**.

---

## 20. Implementation phases (build in this order)

Each phase ends with something runnable and tested. Do not start a phase before the previous one's acceptance criteria pass.

**Phase 0 — Scaffolding.** Repo layout, `pyproject.toml`, `models.py`, `config.py`, `cache.py`, logging, CLI skeleton, empty scorer registry. *Acceptance:* `voice-evals --help` works; config loads; cache round-trips a `ScoreResult`.

**Phase 1 — Ingestion.** Decode/normalize/hash/sidecar/channel-split. *Acceptance:* dropping mixed mp3/wav into `corpus/` yields `AudioClip`s with correct duration, 16k mono path, stable `clip_id` across re-encode; stereo split works on a 2-channel fixture.

**Phase 2 — Acoustic layer.** DNSMOS, NISQA, UTMOS, SQUIM + registry wiring + runner fan-out + caching. *Acceptance:* full run over a tiny corpus produces all metrics, second run is fully cached (0 model calls), one scorer erroring doesn't abort the run. **This is the first real signal — ship it before moving on.**

**Phase 3 — Aggregation + reporting + pytest gates.** Thresholds, regression vs baseline, Parquet/JSON/HTML, gate suite with `local` marker. *Acceptance:* `pytest -m "not judge"` passes/fails on configured floors; HTML report opens offline and plays clips.

**Phase 4 — Dynamics layer.** Segmentation (events → channels → diarization fallback), latency, turn-taking, barge-in. *Acceptance:* on an event-log fixture, latency/barge-in match hand-computed values exactly; on mono fixture, scorers degrade gracefully with `estimated=True`.

**Phase 5 — Judge layer.** Gemini backend, rubric, structured output, defensive parsing, concurrency+backoff, caching, self-enhancement guard. *Acceptance:* judge runs over a few clips, returns valid `JudgeOutput`, caches correctly, and a `local_only` run with `judge:[]` needs no network.

**Phase 6 — Calibration harness.** Golden-set agreement (κ + correlation), calibration report, CI recommendation. *Acceptance:* `voice-evals calibrate` produces per-dimension agreement on a sample golden set.

**Phase 7 (optional/later).** Local-Omni judge wiring; semantic hook example; streaming adapter exploration.

---

## 21. Testing strategy for the framework itself

- **Synthetic fixtures** (generated in `conftest.py`, not committed binaries where avoidable): pure tones, white noise, a clean TTS sample, a deliberately glitched sample (inserted silence gaps / clicks), a 2-channel constructed conversation with known turn boundaries, and an event-log JSON with known timings.
- Acoustic scorers: assert monotonic sanity (clean speech scores higher than noise-injected version) rather than exact values (model versions drift).
- Dynamics: assert exact latency/barge-in numbers against the constructed event-log fixture.
- Judge: mock the API in unit tests (test parsing/caching/error paths against canned JSON); a separate `@pytest.mark.judge` integration test hits the real API only when a key is present.
- Cache: assert content-addressing, config-hash invalidation, and re-run produces zero model calls.

---

## 22. Open decisions for the team (resolve before/early in build)

1. **Channel capture:** can the gateway emit dual-channel recordings and/or an event log (TTS start/stop, STT final, first-token)? This is the single biggest lever on Layer B accuracy. If yes, define the sidecar JSON schema against the gateway's actual event names and wire `source="events"` first.
2. **Judge data governance:** is sending production audio to Gemini acceptable, or must the judge be Vertex-with-DPA / self-hosted from day one? Determines whether Phase 5 targets `gemini.py` or `local_omni.py`.
3. **Corpus sampling:** how are clips selected into `corpus/`? Recommend seeding from calls the existing transcript evals *fail to catch* (that's where audio-layer signal earns its keep), plus a random production sample for baseline distributions.
4. **Metadata source of truth:** confirm the fields available from the gateway/call store to populate `ClipMetadata` (especially `agent_version`/`prompt_version` for the regression gate, and `tts_provider` for the self-enhancement guard).

---

## 23. Appendix A — sidecar JSON example

`call_2026-06-25_abc123.json` next to `call_2026-06-25_abc123.mp3`:

```json
{
  "call_id": "abc123",
  "agent_version": "leasing-ai-2.14.0",
  "prompt_version": "p-2026-06-20",
  "model_id": "gpt-realtime-...",
  "tts_provider": "elevenlabs",
  "channel_map": {"agent_channel": 0, "caller_channel": 1},
  "transcript": "optional full transcript text...",
  "events": [
    {"kind": "user_speech_start", "t_s": 0.0},
    {"kind": "user_speech_end", "t_s": 3.2},
    {"kind": "stt_final", "t_s": 3.4},
    {"kind": "llm_first_token", "t_s": 3.9},
    {"kind": "tts_first_audio", "t_s": 4.1},
    {"kind": "agent_tts_start", "t_s": 4.1},
    {"kind": "user_speech_start", "t_s": 6.0},
    {"kind": "barge_in_detected", "t_s": 6.05},
    {"kind": "agent_interrupted", "t_s": 6.25},
    {"kind": "agent_tts_end", "t_s": 6.25}
  ]
}
```

From this the latency scorer reads TTFT = `tts_first_audio − stt_final` = 0.7 s, and the barge-in scorer reads stop latency = `agent_interrupted − barge_in_detected` = 0.2 s — both exact, no estimation.

## 24. Appendix B — what each layer catches (for the README)

- Robotic / degraded TTS, noise, codec artifacts → **DNSMOS, UTMOS**.
- Choppy audio, dropouts, glitches → **NISQA discontinuity, SQUIM**.
- Sluggish responses, tail-latency users → **latency percentiles**.
- Awkward silences, talk-over → **turn-taking**.
- Agent talks over the caller / won't yield → **barge-in**.
- Unnatural prosody, hesitation, wrong tone "as heard" → **audio judge**.
- Wrong content / failed task / bad tool call → **existing transcript evals** (semantic hook).

---

*End of spec.*
