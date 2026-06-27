# voice-evals

Offline, **reference-free** batch evaluation of AI voice-call audio quality. Drop a
directory of production call recordings in, get back per-clip scores across three
orthogonal layers, an HTML report, Parquet/JSON, and pytest regression gates.

It evaluates the **audio bytes themselves** — not just transcripts — so it catches
what the semantic evals can't: robotic synthesis, clipping, dropouts, unnatural
prosody, awkward pauses, latency, and failed barge-in.

> **Design principle: everything is reference-free / unsupervised.** There is no
> per-clip ground-truth labeling anywhere in the core pipeline. The only human
> labeling is an optional ~30–50 clip calibration set used once to validate the judge.

---

## The three layers

| Layer | Question | Labels? | Network? | Scorers |
|-------|----------|---------|----------|---------|
| **A — Acoustic / perceptual** | Did it *sound* good? Artifacts, noise, robotic TTS, glitches? | No | No | `dnsmos` `nisqa` `utmos` `squim` |
| **B — Conversational dynamics** | Weird pauses? Latency? Did barge-in work? | No (timestamps) | No | `latency` `turn_taking` `barge_in` |
| **C — Audio LLM-judge** | Natural prosody? Right tone? Hesitation, as *heard*? | No to run; small set to *trust* | Yes (or self-host) | `audio_judge` (Gemini / local) |
| (Semantic — existing) | What was said; resolved; correct tool calls? | varies | varies | `semantic_hook` integration only |

**Layers A and B are fully functional with zero API keys and zero network access.**
Layer C is additive and **off by default** (`scorers.judge: []`).

### What each layer catches
- Robotic / degraded TTS, noise, codec artifacts → **DNSMOS, UTMOS**
- Choppy audio, dropouts, glitches → **NISQA `discontinuity`, SQUIM**
- Sluggish responses, tail-latency users → **latency percentiles**
- Awkward silences, talk-over → **turn-taking**
- Agent talks over the caller / won't yield → **barge-in**
- Unnatural prosody, hesitation, wrong tone "as heard" → **audio judge**
- Wrong content / failed task / bad tool call → **existing transcript evals** (semantic hook)

---

## Install

Python 3.11+ and `ffmpeg` on PATH are required. The base install plus `[acoustic]`
and `[dynamics]` runs with **no GPU and no API keys**.

```bash
pip install -e .                 # base: ingestion, runner, cache, report, gates
pip install -e ".[acoustic]"     # DNSMOS, NISQA, UTMOS, SQUIM
pip install -e ".[dynamics]"     # silero-vad segmentation
pip install -e ".[judge]"        # google-genai (hosted Gemini judge)
pip install -e ".[judge-openai]" # openai (gpt-audio judge)
pip install -e ".[simulate]"     # driver/simulator: user-sim + OpenAI Realtime agent
pip install -e ".[telephony]"    # real Twilio call backend (barge-in over real telephony)
pip install -e ".[all]"          # everything + dev/test deps
```

Model weights self-fetch on first use (UTMOS via torch.hub, SQUIM via torchaudio,
silero on first run, DNSMOS bundled in `speechmos`). **NISQA** has no clean PyPI
package and must be vendored once:

```bash
python scripts/fetch_models.py nisqa     # clones gabrielmittag/NISQA + weights into .cache/models/
```

---

## Quickstart

```bash
# 1. Drop recordings (+ optional <stem>.json sidecars) into corpus/
cp /path/to/*.mp3 corpus/

# 2. Local-only run (Layers A + B; no network, no keys)
voice-evals run --config config/example.local.yaml

# 3. Open the report
open outputs/runs/<run_id>/report.html
```

Other commands:

```bash
voice-evals ingest --corpus corpus/                 # decode/normalize only; warm the cache
voice-evals simulate --scenarios scenarios/example.yaml --mock --baseline-version main  # dev-loop A/B
voice-evals call --scenarios scenarios/example.yaml --to +1XXXXXXXXXX --public-url wss://<tunnel>/caller  # real telephony + barge-in
voice-evals run --no-judge --no-cache               # force local-only / bypass result cache
voice-evals run --baseline previous --strict        # fail (exit 1) on any gate violation
voice-evals report --run outputs/runs/<id>          # regenerate report.html from run.json
voice-evals calibrate --golden calibration/golden_set.csv
voice-evals cache clear [--scorer dnsmos]
voice-evals list-scorers
```

Enable the judge by adding it to your config and exporting a key (never put keys in YAML):

```bash
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
# in your config:  scorers: { judge: [audio_judge] }
```

---

## How it works

```
corpus/  ──►  Ingestion ──►  AudioClip  ──►  Runner ──►  ScoreResults ──►  Aggregation ──►  report.html
 (.mp3 +      decode 16k     (content-      fan clips     (cached by        gates +          results.parquet
  .json)      mono + split   addressed)     × scorers     clip+config       regression       run.json
              channels                      + cache)      hash)             vs baseline      pytest gates
```

- **Ingestion** is the only component that touches raw files. It decodes to 16 kHz
  mono 16-bit PCM (canonical MOS input), splits stereo, isolates agent/caller
  channels when a `channel_map` is known, and computes `clip_id` as the **blake2b of
  the decoded PCM** (not the source file) — so re-encodes collide and the cache is
  immune to re-download churn.
- **Scorers** are pure `(AudioClip, config) → ScoreResult`. They never raise
  (errors are returned in `ScoreResult.error`), lazy-load + process-cache heavy
  models, and one scorer failing on one clip never aborts the run.
- **Cache** keys every result on `sha256(clip_id + scorer + config_hash)`, so reruns
  and CI cost nothing and judge calls are never repeated. A config change (including
  a rubric version bump) invalidates exactly the affected entries.
- **Aggregation** applies gates (policy lives here, never in scorers): absolute
  floors/ceilings and **regression vs a baseline run** grouped by `agent_version`.

### ⚠️ mp3 / absolute-MOS caveat
mp3 is lossy and MOS models were trained largely on PCM, so **absolute** MOS values
are slightly depressed for mp3 sources. This framework is built for **relative
comparison and regression**, not certifying absolute MOS. The decode path is
byte-stable so scores are comparable across runs.

---

## Dev-loop evaluation: driving scenarios through the agent (the simulator)

The pieces above are the **evaluator** — they score a recording. To get pre-merge
signal on whether an agent change *improved the experience*, you also need a
**driver** that produces recordings by running scenarios through the agent. That's
`voice_evals.simulate`.

The field has converged (EVA-Bench, τ²-Bench, Coval) on **live simulation, not
audio replay**: you can't replay a recorded caller because a changed agent makes
the conversation diverge (the Waymo "don't replay sensor logs" problem). So a
**persona-conditioned user-simulator** holds a live multi-turn conversation with
the agent, seeded from a bad production call:

```
scenario (mined from a bad call)         voice_evals.simulate            voice-evals evaluator
  goal + persona + emotional arc  ──►  UserSimulator ⇄ Agent  ──►  recording + event log  ──►  scores + A/B regression
                                       (multi-turn, live audio)    (dual-channel + sidecar)    (by agent_version)
```

The simulator's only job is to produce a dual-channel recording + event log +
transcript. Scoring is the **unchanged** evaluator — and the event log gives
**exact** Layer-B timing. **Scope:** this repo scores the *audio experience* only;
task/tool-call/information correctness is owned by your text/OTel evals. A
scenario's `goal` exists to make the simulated caller behave realistically (and
get frustrated when mishandled), **not** to score completion here.

```bash
# Try the harness with no API (mock backends):
voice-evals simulate --scenarios scenarios/example.yaml --mock --baseline-version main

# Real run against your agent (A/B candidate vs main), OPENAI_API_KEY in .env:
voice-evals simulate --scenarios scenarios/example.yaml \
  --agent-version pr-1234   --agent-instructions path/to/agent.txt \
  --baseline-version main   --baseline-instructions path/to/main_agent.txt \
  -k 5            # runs per scenario; sims are stochastic, so gate on aggregates
```

Key design choices (all research-grounded):
- **Multi-turn from the start** — voice failures live in the flow, not one turn.
- **k runs per scenario, aggregate gating** — trial stochasticity dominates variance;
  one pass/fail flaps. Use a stochastic simulator (`user_sim_temperature > 0`) or
  identical runs dedupe by content hash.
- **Adversarial personas** (`persona.adversarial: true`) — friendly simulators give a
  falsely optimistic signal; you mined *bad* calls, so model impatience/tangents.
- **A/B candidate-vs-baseline** rather than absolute sim scores — simulated users are
  miscalibrated against humans in absolute terms; relative comparison is reliable.
  Reuses the `agent_version` regression machinery.
- **Consistency-check-and-regenerate** — drop and re-run degenerate simulations.
- The agent backend is **pluggable** (`AgentBackend`): the reference impl drives an
  OpenAI Realtime session; point it at your agent's prompt, or implement the protocol
  against your production agent (SIP / your own service).

There are two transports for the agent under test:
- **OpenAI Realtime (websocket)** — turn-based/half-duplex. Fast local iteration;
  captures latency, flow, naturalness, prosody, pace, frustration-handling. Does
  **not** reproduce real-time barge-in (no overlap).
- **Telephony (real Twilio call)** — full-duplex over real μ-law/8kHz. Reproduces
  and **measures barge-in**, with production-realistic codec/jitter/latency. ↓

Keep a human spot-check; an LLM judge of an agent in its own model family inflates scores.

### Real telephony + barge-in (`voice-evals call`)

A phone call is inherently full-duplex, so the simulated caller can talk **over**
the agent and we measure whether/how fast it yields — real barge-in, the thing
half-duplex can't see. The mechanism is one REST call + a bundled media-stream
websocket (you set up **no** Twilio infra):

```bash
# 1. Run your dev agent locally behind a Twilio number + ngrok (Programmable Voice).
# 2. Expose THIS command's media server via a tunnel; set it as telephony.public_url.
# 3. Credentials in .env:  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN  (+ OPENAI_API_KEY)
voice-evals call --scenarios scenarios/example.yaml \
  --agent-version pr-1234 --to +1XXXXXXXXXX --public-url wss://<tunnel>/caller
```

`voice-evals call` dials the number, runs the persona caller-bot full-duplex over
the call's μ-law audio (it barges in per the scenario's `barge_in` policy), records
both legs to a dual-channel recording + event log, and scores it — the **barge-in
scorer** then reports stop-latency p50/p95, **success rate**, and false-alarm rate.
A scenario opts into interruptions with:

```yaml
barge_in: { enabled: true, after_agent_s: 2.0, max_barge_ins: 2 }
```

A/B across agent versions is across invocations (run `call --agent-version main`,
change your agent, run `call --agent-version pr-1234`) compared via the regression
baseline. **v1 notes:** the caller follows its persona/goal arc rather than STT-ing
the agent's exact words (fine for barge-in timing; content-coherence via STT is the
next increment); the Twilio dial + media server are verified at the session level
offline (codec, turn-taking, barge-in timing, recording) and validated by your live call.

## Conversational dynamics: the channel question

Layer B accuracy depends on channel separation. The dynamics scorers prefer, in order:

1. **Gateway event log** in the sidecar → exact (`source="events"`).
2. **Isolated agent/caller channels** → accurate VAD (`source="channels"`).
3. **Mono mix** → VAD + diarization estimate (`source="diarization"`, `estimated=True`);
   barge-in is **not** measured on mono.

If your gateway can emit dual-channel recordings and/or an event log (TTS start/stop,
STT final, first-token), wire it up — it is the single biggest lever on Layer B
accuracy.

### Sidecar JSON (`<stem>.json` next to the recording)

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
    {"kind": "user_speech_end",   "t_s": 3.2},
    {"kind": "stt_final",         "t_s": 3.4},
    {"kind": "llm_first_token",   "t_s": 3.9},
    {"kind": "tts_first_audio",   "t_s": 4.1},
    {"kind": "agent_tts_start",   "t_s": 4.1},
    {"kind": "user_speech_start", "t_s": 6.0},
    {"kind": "barge_in_detected", "t_s": 6.05},
    {"kind": "agent_interrupted", "t_s": 6.25},
    {"kind": "agent_tts_end",     "t_s": 6.25}
  ]
}
```

From this, latency reads TTFT = `tts_first_audio − stt_final` = 0.7 s and barge-in
reads stop latency = `agent_interrupted − barge_in_detected` = 0.2 s — both exact.
All fields are optional; with none, Layer A still works and Layer B falls back to VAD.

---

## The audio judge (Layer C)

The judge sends the **audio bytes** (not a transcript) to a native-audio model with a
versioned rubric and scores seven dimensions 1–5 (5 = best), with chain-of-thought
*before* each score:

`naturalness · prosody · fluency · pace · responsiveness · emotional_alignment · intelligibility`

plus a holistic `overall`, `notable_timestamps`, and a `summary`. Flattened metrics are
`judge_naturalness … judge_overall`. The rubric and system prompt are plain text in
`config/` — tune them without touching code, but **bump `RUBRIC_VERSION`** on any
change (it feeds the cache key and invalidates calibration).

> **Naming note:** the rubric uses **`fluency`** (not `dysfluency`) so that 5 = best
> holds uniformly across every dimension and gate authors never flip a sign. All judge
> gates are `min` floors, e.g. `judge_fluency: {min: 3}`.

**Guards.** Self-enhancement bias is handled in code: the judge won't grade audio from
its own model family (`exclude_if_model_family_matches`, matched against
`tts_provider`). Degenerate inputs (silence, no agent speech) set `unscoreable=true`
and surface as an error with no scores — never hallucinated numbers.

### Cost & governance
- Audio tokenizes at ~32 tokens/s; a 60 s clip ≈ ~1,900 tokens ≈ **$0.003–0.006**.
  A 200-clip run ≈ **$1–3**; caching makes reruns free. Batch API is 50% off.
- **PII:** recordings contain resident/tenant data. Sending audio to a hosted API has
  governance implications. Use the `local_only` profile (judge disabled) for CI, and
  Vertex-with-DPA or the self-hosted judge (`backend: local_omni`) for compliant paths.
  Hosted-judge usage is **explicit opt-in**, never the default.

---

## Calibration (validate the judge before trusting it)

The only place humans label — optional and one-time. Fill `calibration/golden_set.csv`
(`clip_id, dimension, human_score`) by listening to ~30–50 clips using the **same
anchors** as the rubric, then:

```bash
voice-evals calibrate --golden calibration/golden_set.csv
```

It runs the judge on those clips and reports per-dimension **quadratic-weighted Cohen's
κ** + Spearman/Pearson correlation, and recommends which dimensions to enforce in CI
(κ ≥ `calibration.kappa_bar`, default 0.6). Expect `intelligibility`, `naturalness`,
`fluency` to agree well; treat low-κ dimensions (`emotional_alignment`, `prosody`) as
advisory. Re-run whenever the rubric version or judge model changes.

---

## pytest regression gates

```bash
pytest -m "not judge"     # default CI: Layers A + B, no network, no keys
pytest -m judge           # judge gates (needs GEMINI_API_KEY or a local GPU model)
pytest -m "local"         # explicitly network-free
```

`tests/eval_gates/test_corpus_gates.py` parametrizes one assertion per (clip, gated
metric) and includes a regression gate that fails when a metric degrades beyond
`max_delta` vs the baseline run.

---

## Configuration

`config/default.yaml` (judge on opt-in) and `config/example.local.yaml` (local-only).
Gates support **absolute** floors/ceilings (with soft `warn` levels) and **regression**
deltas grouped by `agent_version`. The most valuable gate is usually *"OVRL dropped
>0.15 vs the previous version,"* not just an absolute floor. See `src/voice_evals/config.py`
for the full schema. Thresholds are heuristic starting points — calibrate them to your
own corpus.

---

## Repository layout

```
voice-evals/
├── config/            default + local YAML, rubric.default.txt, judge_system.txt
├── corpus/            (gitignored) drop audio + optional .json sidecars here
├── .cache/            (gitignored) decoded audio + scorer result cache
├── outputs/           (gitignored) run artifacts (report.html, results.parquet, run.json)
├── calibration/       golden_set.csv (human labels for judge validation)
├── scenarios/         scenario manifests for the simulator (mined from prod calls)
├── scripts/           fetch_models.py (vendor NISQA)
├── src/voice_evals/   models, config, ingest, cache, runner, aggregate, report, cli,
│                      calibrate, semantic_hook, scorers/{acoustic,dynamics,judge},
│                      simulate/ (driver: scenario, orchestrator, backends, gating,
│                      telephony — Twilio Media Streams caller + barge-in)
└── tests/             unit tests + eval_gates/ regression suite
```

## Non-goals (v1)
Not a live/streaming monitor, not a synthetic-call generator, not an STT system, and
not a labeling UI. This is offline batch over recorded files.
