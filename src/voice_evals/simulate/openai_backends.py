"""OpenAI backends: a persona user-simulator (chat + TTS) and the Realtime agent.

The user simulator is high-confidence and cheap to verify (chat next-line + TTS).
The Realtime agent connector drives an OpenAI Realtime (speech-to-speech) session
turn-based; point it at YOUR agent's instructions/voice/model. Audio is resampled
between our 16 kHz pipeline and Realtime's 24 kHz pcm16.

All imports are lazy so the simulate package loads without the openai SDK.
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Optional

import numpy as np
import soundfile as sf

from ..config import Config, load_secrets
from ..logging_util import get_logger
from .base import SR, Move
from .scenario import Scenario

log = get_logger(__name__)

_REALTIME_SR = 24000


def _client():
    from openai import OpenAI

    key = load_secrets().openai_api_key
    if not key:
        raise RuntimeError("no OPENAI_API_KEY in environment/.env")
    return OpenAI(api_key=key)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from ..scorers._audio import _resample_linear

    return _resample_linear(x.astype(np.float32), sr_in, sr_out)


def _f32_to_pcm16_b64(x: np.ndarray) -> str:
    pcm = np.clip(x, -1.0, 1.0)
    return base64.b64encode((pcm * 32767.0).astype("<i2").tobytes()).decode()


def _pcm16_b64_to_f32(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


# --------------------------------------------------------------------------- #
# User simulator: persona-conditioned caller (chat -> text, TTS -> audio)
# --------------------------------------------------------------------------- #
class OpenAIUserSimulator:
    def __init__(self, config: Config):
        self.cfg = config.simulate
        self._client = None

    def _c(self):
        if self._client is None:
            self._client = _client()
        return self._client

    def _system_prompt(self, scenario: Scenario) -> str:
        p = scenario.persona
        return (
            f"You are role-playing a CALLER on {scenario.domain}. Stay in character.\n"
            f"Your goal: {scenario.goal or 'get help with your issue'}.\n"
            f"Persona: {p.name}; speaking style: {p.style}; patience: {p.patience}.\n"
            f"Emotional arc: {p.emotional_arc}.\n"
            + ("You are a NON-COLLABORATIVE caller: be impatient, interrupt-prone, give "
               "incomplete answers, go on tangents when frustrated.\n" if p.adversarial else "")
            + f"Hang up when: {scenario.hangup_when}.\n"
            "Speak ONE short, natural spoken turn at a time (1-2 sentences, as a real person "
            "on a phone would). Do not narrate actions or stage directions.\n"
            'Return ONLY JSON: {"utterance": "<what you say>", "emotion": "<e.g. neutral, '
            'frustrated, angry>", "hangup": <true|false>}.'
        )

    def next_turn(self, scenario: Scenario, history) -> Move:
        convo = "\n".join(f"{m.speaker}: {m.text}" for m in history) or "(call just connected)"
        n_caller = sum(1 for m in history if m.speaker == "caller")
        client = self._c()
        resp = client.chat.completions.create(
            model=self.cfg.user_sim_model,
            temperature=self.cfg.user_sim_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt(scenario)},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nYour next caller turn as JSON:"},
            ],
        )
        try:
            d = json.loads(resp.choices[0].message.content)
            utterance = str(d.get("utterance", "")).strip() or "Hello? Are you there?"
            emotion = str(d.get("emotion", "neutral"))
            hangup = bool(d.get("hangup", False))
        except Exception:
            utterance, emotion, hangup = "Hello? Are you still there?", "frustrated", False
        # force a stop at max_turns regardless of the model's hangup flag
        if (n_caller + 1) >= scenario.max_turns:
            hangup = True
        audio = self._tts(utterance, emotion, scenario.persona.voice)
        return Move("caller", utterance, audio, hangup=hangup)

    def _tts(self, text: str, emotion: str, voice: str) -> np.ndarray:
        client = self._c()
        resp = client.audio.speech.create(
            model=self.cfg.tts_model,
            voice=voice,
            input=text,
            instructions=f"Speak as a caller who sounds {emotion}, natural phone cadence.",
            response_format="wav",
        )
        data, sr = sf.read(io.BytesIO(resp.read()), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != SR:
            data = _resample(data, sr, SR)
        return np.ascontiguousarray(data, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Transcriber: STT for the agent's turns (so the telephony caller can react)
# --------------------------------------------------------------------------- #
class OpenAITranscriber:
    """Transcribes a completed agent turn (μ-law-decoded PCM) to text. The telephony
    caller has no text channel to a real phone agent — only its audio — so the
    session STTs each agent turn and feeds the words back into the caller's context."""

    def __init__(self, config: Config):
        self.cfg = config.simulate
        self._client = None

    def _c(self):
        if self._client is None:
            self._client = _client()
        return self._client

    def transcribe(self, pcm16: np.ndarray, sr: int) -> str:
        buf = io.BytesIO()
        sf.write(buf, np.asarray(pcm16), sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        buf.name = "agent_turn.wav"  # the SDK keys off the file extension
        try:
            resp = self._c().audio.transcriptions.create(
                model=self.cfg.stt_model, file=buf, language="en",
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:  # noqa: BLE001 - a failed STT must not kill the call
            log.warning("agent-turn STT failed: %s", e)
            return ""


# --------------------------------------------------------------------------- #
# Agent under test: OpenAI Realtime (speech-to-speech), turn-based
# --------------------------------------------------------------------------- #
class OpenAIRealtimeAgent:
    """Drives an OpenAI Realtime session one turn at a time. Manual turn-taking
    (server VAD off): append caller audio -> commit -> request response -> collect
    the agent's audio + transcript, measuring time-to-first-audio as latency.

    NOTE: this is the live integration point. Verify with a smoke run and point
    ``agent_instructions`` at your real agent's system prompt. If your production
    agent is reachable another way (SIP/your own service), implement AgentBackend
    against that instead — the orchestrator/evaluator don't change.
    """

    def __init__(self, config: Config, agent_version: str, instructions: str):
        self.cfg = config.simulate
        self._version = agent_version
        self._instructions = instructions
        self._conn = None
        self._cm = None

    @property
    def agent_version(self) -> str:
        return self._version

    def start(self, scenario: Scenario) -> None:
        client = _client()
        self._cm = client.realtime.connect(model=self.cfg.agent_model)
        self._conn = self._cm.__enter__()
        try:
            self._configure()
        except Exception:
            self.close()  # tear down the entered CM if session setup fails
            raise

    def _configure(self) -> None:
        # manual turn control: no server VAD; audio in/out as pcm16
        self._conn.session.update(session={
            "type": "realtime",
            "instructions": self._instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": _REALTIME_SR},
                          "turn_detection": None},
                "output": {"format": {"type": "audio/pcm", "rate": _REALTIME_SR},
                           "voice": self.cfg.agent_voice},
            },
        })

    def greet(self, scenario: Scenario) -> Optional[Move]:
        return None  # caller opens (scenario.opening); agent-first greeting is a later option

    def respond(self, user_audio: np.ndarray) -> Move:
        conn = self._conn
        up = _resample(user_audio, SR, _REALTIME_SR)
        # the Realtime API rejects a commit with <100ms of audio; pad short turns
        min_len = int(0.15 * _REALTIME_SR)
        if up.shape[0] < min_len:
            up = np.concatenate([up, np.zeros(min_len - up.shape[0], np.float32)])
        conn.input_audio_buffer.append(audio=_f32_to_pcm16_b64(up))
        conn.input_audio_buffer.commit()
        t0 = time.perf_counter()
        conn.response.create()

        chunks: list[np.ndarray] = []
        transcript_parts: list[str] = []
        latency: Optional[float] = None
        deadline = t0 + self.cfg.turn_timeout_s
        timed_out = False
        for event in conn:
            etype = getattr(event, "type", "")
            if "audio" in etype and etype.endswith("delta") and "transcript" not in etype:
                if latency is None:
                    latency = time.perf_counter() - t0
                chunks.append(_pcm16_b64_to_f32(event.delta))
            elif "transcript" in etype and etype.endswith("delta"):
                transcript_parts.append(getattr(event, "delta", "") or "")
            elif etype == "error":
                raise RuntimeError(f"realtime error: {getattr(event, 'error', event)}")
            # terminal: clean done OR any non-clean response close
            elif etype.startswith("response.") and etype.rsplit(".", 1)[-1] in (
                "done", "failed", "incomplete", "cancelled"
            ):
                break
            if time.perf_counter() > deadline:
                timed_out = True
                log.warning("realtime turn exceeded %.0fs; returning partial audio", self.cfg.turn_timeout_s)
                break

        if chunks:
            agent_audio = _resample(np.concatenate(chunks), _REALTIME_SR, SR)
        else:
            # no audio produced — don't report a falsely-perfect 0.0s latency
            agent_audio = np.zeros(int(0.2 * SR), np.float32)
            if latency is None:
                latency = time.perf_counter() - t0
                log.warning("realtime turn produced no audio; latency set to full wall-time %.2fs", latency)
        return Move("agent", "".join(transcript_parts).strip() or "(no transcript)",
                    agent_audio, latency_s=latency or 0.0)

    def close(self) -> None:
        try:
            if self._cm is not None:
                self._cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001 - teardown failures shouldn't crash the suite
            log.warning("realtime session teardown failed: %s", e)
        finally:
            self._conn = None
            self._cm = None
