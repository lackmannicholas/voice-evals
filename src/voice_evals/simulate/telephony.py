"""Twilio Media Streams caller — places a real call to the agent's number and
runs the persona caller-bot full-duplex over the call audio.

You do NOT set up any Twilio infra: ``voice-evals call`` embeds the media-stream
websocket and places the call. The dev runs their agent locally (Twilio number +
ngrok, Programmable Voice) and points a tunnel at our server's port; we dial the
number with TwiML ``<Connect><Stream url=wss://.../caller>``.

Why telephony: a phone call is inherently full-duplex, so the caller can talk
OVER the agent and we measure whether/how fast it yields — real barge-in, over
real μ-law/8kHz telephony, not a pristine socket.

This module's protocol/codec/recording/barge-in logic is unit-tested by driving
``MediaStreamSession`` with scripted Twilio frames + a mock caller. The live REST
dial + websocket serving are thin shells around it (see ``call_cli``).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import soundfile as sf

from ..logging_util import get_logger
from .base import SimulatedCall
from .scenario import Scenario

log = get_logger(__name__)

SR8 = 8000  # Twilio media streams are μ-law (PCMU) 8 kHz mono
FRAME_MS = 20
FRAME_SAMPLES = SR8 * FRAME_MS // 1000  # 160 samples / 20 ms frame


# --------------------------------------------------------------------------- #
# G.711 μ-law codec (vectorized; no audioop dependency — removed in py3.13)
# --------------------------------------------------------------------------- #
_MU = 255
_BIAS = 0x84
_CLIP = 32635


def mulaw_encode(pcm16: np.ndarray) -> bytes:
    x = np.clip(pcm16.astype(np.int32), -_CLIP, _CLIP)
    sign = (x < 0).astype(np.int32) * 0x80
    mag = np.abs(x) + _BIAS
    exponent = np.zeros_like(mag)
    for e in range(8):  # ascending so the LARGEST qualifying segment wins
        exponent = np.where(mag >= (1 << (e + 7)), e, exponent)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    ulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw.astype(np.uint8).tobytes()


def mulaw_decode(data: bytes) -> np.ndarray:
    u = (~np.frombuffer(data, dtype=np.uint8).astype(np.int32)) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + _BIAS) << exponent
    sample -= _BIAS
    return np.where(sign != 0, -sample, sample).astype(np.int16)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from ..scorers._audio import _resample_linear

    return _resample_linear(x.astype(np.float32), sr_in, sr_out)


def f32_16k_to_mulaw8k(wav16k: np.ndarray) -> bytes:
    """Caller TTS (16 kHz float) → μ-law 8 kHz bytes for the media stream."""
    pcm = np.clip(_resample(wav16k, 16000, SR8), -1.0, 1.0)
    return mulaw_encode((pcm * 32767.0).astype(np.int16))


def _rms(pcm16: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pcm16.astype(np.float64) / 32768.0) ** 2))) if pcm16.size else 0.0


# --------------------------------------------------------------------------- #
# caller-bot
# --------------------------------------------------------------------------- #
class CallerBot(Protocol):
    """Produces the caller's next utterance as μ-law 8 kHz bytes + text + hangup."""

    def next_turn(self, scenario: Scenario, turn_index: int, history: list[dict]) -> tuple[bytes, str, bool]: ...


class OpenAICascadeCaller:
    """Persona caller for telephony: reuses the OpenAI user-simulator for content,
    then encodes to μ-law 8 kHz for the call. NOTE (v1): the caller does not STT the
    agent, so it follows its persona/goal arc rather than reacting to the agent's
    exact words — fine for barge-in timing; content-coherence (STT) is a later step."""

    def __init__(self, config):
        from .openai_backends import OpenAIUserSimulator

        self._us = OpenAIUserSimulator(config)

    def next_turn(self, scenario: Scenario, turn_index: int, history: list[dict]) -> tuple[bytes, str, bool]:
        from types import SimpleNamespace

        moves = [SimpleNamespace(speaker=h["speaker"], text=h["text"]) for h in history]
        move = self._us.next_turn(scenario, moves)  # Move(caller, text, audio@16k, hangup)
        return f32_16k_to_mulaw8k(move.audio), move.text, move.hangup


# --------------------------------------------------------------------------- #
# Media Streams session (the testable core)
# --------------------------------------------------------------------------- #
@dataclass
class _Placed:
    speaker: str
    start_ms: int
    pcm8k: np.ndarray  # int16 @ 8k


class MediaStreamSession:
    """Drives one call. Fed Twilio media-stream messages (dicts); emits outbound
    ``media`` messages (caller audio) and records both legs + barge-in events.

    Turn-taking is VAD-driven on the AGENT's inbound audio: the caller replies
    after the agent goes silent; on a barge-in turn it starts talking once the
    agent has spoken continuously for ``after_agent_s`` — and we time the agent's
    yield from the inbound audio falling silent after the caller's onset.
    """

    def __init__(
        self,
        scenario: Scenario,
        caller: CallerBot,
        agent_version: str,
        prompt_version: Optional[str] = None,
        vad_rms: float = 0.02,
        end_silence_ms: int = 600,
        success_window_s: float = 1.5,
    ):
        self.scenario = scenario
        self.caller = caller
        self.agent_version = agent_version
        self.prompt_version = prompt_version
        self.vad_rms = vad_rms
        self.end_silence_ms = end_silence_ms
        self.success_window_ms = int(success_window_s * 1000)

        self.placed: list[_Placed] = []
        self.events: list[dict] = []
        self.transcript: list[dict] = []
        self._clock_ms = 0  # advanced by inbound frames (20 ms each)
        self._voice_gap_tol_ms = 200  # inter-word gaps shorter than this don't break a voice run
        self._yield_grace_ms = 500  # agent silence this soon after the caller's utterance = a yield
        # agent VAD state
        self._agent_speaking = False
        self._agent_in_turn = False  # between agent_tts_start and agent_tts_end
        self._voice_run_start_ms: Optional[int] = None  # start of the CURRENT continuous voice run
        self._last_agent_voice_ms: Optional[int] = None
        # caller streaming state
        self._caller_queue: list[np.ndarray] = []  # remaining 20 ms frames to emit
        self._caller_turns = 0
        self._barge_ins = 0
        self.should_hangup = False  # transport ends the call when the caller is done
        # barge-in tracking: {onset_ms, caller_end_ms}
        self._active_barge: Optional[dict] = None

    # -- public: handle one inbound Twilio message; return outbound messages ---- #
    def handle(self, msg: dict) -> list[dict]:
        event = msg.get("event")
        if event == "start":
            return []
        if event == "stop":
            return []
        if event != "media":
            return []
        media = msg.get("media", {})
        # inbound media is the AGENT (the far party on the call)
        if media.get("track", "inbound") == "inbound":
            self._on_agent_frame(base64.b64decode(media["payload"]))
        out = self._pump_caller()
        self._clock_ms += FRAME_MS
        return out

    # -- agent audio + VAD + turn logic ---------------------------------------- #
    def _on_agent_frame(self, ulaw: bytes) -> None:
        pcm = mulaw_decode(ulaw)
        self.placed.append(_Placed("agent", self._clock_ms, pcm))
        voiced = _rms(pcm) >= self.vad_rms
        if voiced:
            # start a new continuous voice run if this is the first voice after a
            # real gap (> tolerance). Short inter-word gaps don't reset the run.
            if (self._voice_run_start_ms is None or self._last_agent_voice_ms is None
                    or self._clock_ms - self._last_agent_voice_ms > self._voice_gap_tol_ms):
                self._voice_run_start_ms = self._clock_ms
            if not self._agent_in_turn:
                # agent starts a turn — emit the events latency/turn-taking need
                self._agent_in_turn = True
                self.events.append({"kind": "agent_tts_start", "t_s": round(self._clock_ms / 1000, 3)})
                self.events.append({"kind": "tts_first_audio", "t_s": round(self._clock_ms / 1000, 3)})
            self._agent_speaking = True
            self._last_agent_voice_ms = self._clock_ms
        elif self._agent_speaking and self._last_agent_voice_ms is not None and \
                self._clock_ms - self._last_agent_voice_ms >= self.end_silence_ms:
            # speaking → silence: the agent finished a turn; the caller responds
            self._agent_speaking = False
            self._end_agent_turn(self._last_agent_voice_ms)
            if not self._caller_queue and not self._active_barge:
                self._start_caller_turn(barge_in=False)

        self._resolve_barge_in(voiced)

        # barge-in: caller cuts in while the agent has been CONTINUOUSLY voiced for
        # >= after_agent_s (intra-word gaps tolerated; real pauses reset the run)
        pol = self.scenario.barge_in
        if (pol.enabled and voiced and self._active_barge is None
                and not self._caller_queue and self._barge_ins < pol.max_barge_ins
                and self._voice_run_start_ms is not None
                and self._clock_ms - self._voice_run_start_ms >= int(pol.after_agent_s * 1000)):
            self._start_caller_turn(barge_in=True)

    def _resolve_barge_in(self, voiced: bool) -> None:
        """Decide a pending barge: a YIELD = the agent goes silent while/just-after
        the caller is talking over it; an agent that keeps talking PAST the caller's
        utterance (+grace) did NOT yield → failed barge-in (no agent_interrupted)."""
        b = self._active_barge
        if b is None:
            return
        deadline = b["caller_end_ms"] + self._yield_grace_ms
        if voiced and self._clock_ms > deadline:
            # agent talked straight through the barge → failed; don't emit a yield
            self._active_barge = None
            return
        if (not voiced and self._last_agent_voice_ms is not None
                and self._last_agent_voice_ms >= b["onset_ms"]
                and self._last_agent_voice_ms <= deadline
                and self._clock_ms - self._last_agent_voice_ms >= 200):
            self._record_agent_interrupted(self._last_agent_voice_ms)

    # -- caller turns ----------------------------------------------------------- #
    def _start_caller_turn(self, barge_in: bool) -> None:
        if self._caller_turns >= self.scenario.max_turns:
            return
        ulaw, text, hangup = self.caller.next_turn(self.scenario, self._caller_turns, self.transcript)
        pcm = mulaw_decode(ulaw)
        self._caller_queue = [pcm[i:i + FRAME_SAMPLES] for i in range(0, len(pcm), FRAME_SAMPLES)]
        self._caller_turns += 1
        self.transcript.append({"speaker": "caller", "text": text})
        onset = self._clock_ms
        self.events.append({"kind": "user_speech_start", "t_s": round(onset / 1000, 3)})
        if barge_in:
            self._barge_ins += 1
            caller_dur_ms = int(len(pcm) / SR8 * 1000)
            self._active_barge = {"onset_ms": onset, "caller_end_ms": onset + caller_dur_ms}
            self.events.append({"kind": "barge_in_detected", "t_s": round(onset / 1000, 3)})
            # a subsequent barge must wait for a fresh after_agent_s of continuous voice
            self._voice_run_start_ms = self._clock_ms
        if hangup or self._caller_turns >= self.scenario.max_turns:
            self.should_hangup = True

    def _pump_caller(self) -> list[dict]:
        if not self._caller_queue:
            return []
        frame = self._caller_queue.pop(0)
        self.placed.append(_Placed("caller", self._clock_ms, frame))
        if not self._caller_queue:  # caller turn just ended
            self.events.append({"kind": "user_speech_end", "t_s": round((self._clock_ms + FRAME_MS) / 1000, 3)})
        payload = base64.b64encode(mulaw_encode(frame)).decode()
        return [{"event": "media", "media": {"payload": payload}}]

    def _end_agent_turn(self, t_ms: int) -> None:
        """Close an agent turn (emit agent_tts_end once) — used by both the natural
        turn-end and the barge-yield paths so dynamics see complete agent turns."""
        if self._agent_in_turn:
            self.events.append({"kind": "agent_tts_end", "t_s": round(t_ms / 1000, 3)})
            self._agent_in_turn = False

    def _record_agent_interrupted(self, t_ms: int) -> None:
        self.events.append({"kind": "agent_interrupted", "t_s": round(t_ms / 1000, 3)})
        self._end_agent_turn(t_ms)  # the yield also ends the agent's turn
        self._active_barge = None

    # -- finalize → recording the evaluator scores ----------------------------- #
    def finalize(self, out_dir: Path) -> SimulatedCall:
        import json

        # close any agent turn still open at hangup so its segment isn't dropped
        if self._agent_in_turn and self._last_agent_voice_ms is not None:
            self._end_agent_turn(self._last_agent_voice_ms)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total_ms = max((p.start_ms + len(p.pcm8k) * 1000 // SR8 for p in self.placed), default=0)
        n = int((total_ms / 1000 + 1) * SR8)
        agent_ch = np.zeros(n, np.int16)
        caller_ch = np.zeros(n, np.int16)
        for p in self.placed:
            i = int(p.start_ms / 1000 * SR8)
            ch = agent_ch if p.speaker == "agent" else caller_ch
            j = min(i + len(p.pcm8k), n)
            ch[i:j] = p.pcm8k[: j - i]
        stereo = np.stack([agent_ch, caller_ch], axis=1)
        stem = f"{self.scenario.id}__{self.agent_version}__call"
        wav_path = out_dir / f"{stem}.wav"
        sf.write(wav_path, stereo, SR8, subtype="PCM_16")  # native 8k; ingest resamples to 16k

        sidecar = {
            "call_id": stem,
            "agent_version": self.agent_version,
            "prompt_version": self.prompt_version,
            "tts_provider": "telephony-sim",
            "channel_map": {"agent_channel": 0, "caller_channel": 1},
            "domain": self.scenario.domain,
            "transcript": "\n".join(f"{x['speaker']}: {x['text']}" for x in self.transcript),
            "events": sorted(self.events, key=lambda e: e["t_s"]),
            "extra": {**self.scenario.metadata_extra(), "transport": "twilio"},
        }
        sidecar_path = out_dir / f"{stem}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        return SimulatedCall(self.scenario.id, self.agent_version, 0, wav_path, sidecar_path,
                             self.transcript, sidecar["events"], total_ms / 1000)
