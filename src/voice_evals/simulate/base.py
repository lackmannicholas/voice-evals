"""Driver building blocks: backend protocols, mocks, and recording assembly.

The orchestrator drives a turn-based (half-duplex) conversation between a
``UserSimulator`` and an ``AgentBackend``, then assembles the turns into a
dual-channel recording (agent=ch0, caller=ch1) + an event-log sidecar. That
artifact is exactly what the existing voice-evals evaluator ingests — the event
log gives EXACT Layer-B timing (source="events").

Half-duplex limitation: turns don't overlap, so real-time barge-in/talk-over
isn't reproduced here (that needs the full-duplex streaming increment). Latency,
flow, naturalness, prosody, pace, and frustration-handling are all captured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import soundfile as sf

from ..logging_util import get_logger
from .scenario import Scenario

log = get_logger(__name__)

SR = 16000
_THINK_TIME_S = 0.3  # caller "think time" after the agent finishes, before next caller turn


# --------------------------------------------------------------------------- #
@dataclass
class Move:
    speaker: str  # "agent" | "caller"
    text: str
    audio: np.ndarray  # mono float32 @ 16k
    latency_s: float = 0.0  # agent turns: measured time-to-first-audio after caller stop
    hangup: bool = False  # caller turns: caller ends the call after this


@dataclass
class SimulatedCall:
    scenario_id: str
    agent_version: str
    run_index: int
    wav_path: Path
    sidecar_path: Path
    transcript: list[dict]
    events: list[dict]
    duration_s: float
    valid: bool = True
    invalid_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# backend protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class UserSimulator(Protocol):
    """Persona-conditioned caller. Generates the next caller utterance (text +
    audio) given the conversation so far, and decides when to hang up."""

    def next_turn(self, scenario: Scenario, history: list[Move]) -> Move: ...


@runtime_checkable
class AgentBackend(Protocol):
    """The agent under test. Turn-based: receives caller audio, returns the
    agent's spoken response + measured response latency."""

    def start(self, scenario: Scenario) -> None: ...
    def greet(self, scenario: Scenario) -> Optional[Move]: ...  # None => caller opens
    def respond(self, user_audio: np.ndarray) -> Move: ...
    def close(self) -> None: ...
    @property
    def agent_version(self) -> str: ...


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def speech_like(duration_s: float, f0: float = 150.0, seed: int = 0) -> np.ndarray:
    """Modulated harmonic tone that reliably trips VAD — for mock turns so the
    full dynamics/judge pipeline exercises end-to-end without real TTS."""
    n = int(SR * max(0.2, duration_s))
    t = np.linspace(0, duration_s, n, endpoint=False)
    rng = np.random.RandomState(seed)
    f0 = f0 * (1.0 + 0.02 * rng.randn())
    sig = sum((0.4 / k) * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3))
    env = np.sin(2 * np.pi * 3 * t) * 0.5 + 0.5
    return (0.3 * sig * env).astype(np.float32)


# --------------------------------------------------------------------------- #
# recording assembly
# --------------------------------------------------------------------------- #
def assemble_recording(
    moves: list[Move],
    scenario: Scenario,
    agent_version: str,
    run_index: int,
    out_dir: Path,
    prompt_version: Optional[str] = None,
) -> SimulatedCall:
    """Lay the alternating turns onto a 2-channel timeline (agent=0, caller=1),
    insert the agent's measured latency as a real gap, and emit the event log."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    placed: list[tuple[str, float, float, np.ndarray]] = []  # (speaker, start, end, audio)
    events: list[dict] = []
    transcript: list[dict] = []
    t = 0.0
    for mv in moves:
        dur = len(mv.audio) / SR
        if mv.speaker == "caller":
            start = t
            end = start + dur
            events.append({"kind": "user_speech_start", "t_s": round(start, 3)})
            events.append({"kind": "user_speech_end", "t_s": round(end, 3)})
            events.append({"kind": "stt_final", "t_s": round(end, 3)})
            t = end + _THINK_TIME_S * 0  # caller end; agent latency handled below
        else:  # agent
            start = t + max(0.0, mv.latency_s)  # real response latency becomes a gap
            end = start + dur
            events.append({"kind": "tts_first_audio", "t_s": round(start, 3)})
            events.append({"kind": "agent_tts_start", "t_s": round(start, 3)})
            events.append({"kind": "agent_tts_end", "t_s": round(end, 3)})
            t = end + _THINK_TIME_S  # caller think-time before the next caller turn
        placed.append((mv.speaker, start, end, mv.audio))
        transcript.append({"speaker": mv.speaker, "text": mv.text})

    total = max((e for _, _, e, _ in placed), default=0.0)
    n = int(np.ceil(total * SR)) + SR  # +1s tail padding
    agent_ch = np.zeros(n, np.float32)
    caller_ch = np.zeros(n, np.float32)
    for speaker, start, _end, audio in placed:
        i = int(start * SR)
        ch = agent_ch if speaker == "agent" else caller_ch
        j = min(i + len(audio), n)
        ch[i:j] += audio[: j - i]

    stereo = np.stack([agent_ch, caller_ch], axis=1)
    stem = f"{scenario.id}__{agent_version}__run{run_index}"
    wav_path = out_dir / f"{stem}.wav"
    sf.write(wav_path, stereo, SR, subtype="PCM_16")

    sidecar = {
        "call_id": stem,
        "agent_version": agent_version,
        "prompt_version": prompt_version,
        "tts_provider": "simulated",
        "channel_map": {"agent_channel": 0, "caller_channel": 1},
        "domain": scenario.domain,
        "transcript": "\n".join(f"{x['speaker']}: {x['text']}" for x in transcript),
        "events": events,
        "extra": {**scenario.metadata_extra(), "run_index": run_index},
    }
    sidecar_path = out_dir / f"{stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return SimulatedCall(
        scenario_id=scenario.id,
        agent_version=agent_version,
        run_index=run_index,
        wav_path=wav_path,
        sidecar_path=sidecar_path,
        transcript=transcript,
        events=events,
        duration_s=total,
    )


def validate_conversation(moves: list[Move]) -> tuple[bool, Optional[str]]:
    """Lightweight simulation-validity check (EVA-Bench 'regenerate on simulator
    error' pattern). Catches degenerate runs before they pollute scores. A real
    LLM persona-consistency check is a later increment."""
    caller = [m for m in moves if m.speaker == "caller"]
    agent = [m for m in moves if m.speaker == "agent"]
    if not caller:
        return False, "no caller turns produced"
    if not agent:
        return False, "agent never responded"
    if any(len(m.audio) < int(0.1 * SR) for m in moves):
        return False, "a turn produced empty/too-short audio"
    if any(not m.text.strip() for m in moves):  # caller OR agent
        return False, "a turn had empty text"
    return True, None


# --------------------------------------------------------------------------- #
# mock backends (deterministic; no network) — for testing the orchestration
# --------------------------------------------------------------------------- #
class MockUserSimulator:
    """Scripted caller: escalates tone, hangs up at max_turns or on a trigger.
    ``seed`` perturbs the audio so distinct runs differ (real stochastic sims do)."""

    def __init__(self, lines: Optional[list[str]] = None, seed: int = 0):
        self._lines = lines
        self._seed = seed

    def next_turn(self, scenario: Scenario, history: list[Move]) -> Move:
        n_caller = sum(1 for m in history if m.speaker == "caller")
        if self._lines and n_caller < len(self._lines):
            text = self._lines[n_caller]
        elif n_caller == 0:
            text = scenario.opening or f"Hi, {scenario.goal or 'I need some help'}."
        else:
            text = "That still doesn't answer my question."
        hangup = (n_caller + 1) >= scenario.max_turns
        # escalating-frustration personas talk a touch longer as they repeat
        dur = 1.2 + 0.3 * min(n_caller, 4)
        return Move("caller", text, speech_like(dur, f0=170, seed=n_caller + 1000 * self._seed),
                    hangup=hangup)


class MockAgent:
    """Canned agent. ``latency_s`` / ``flat`` let tests model a worse agent
    (slower, more monotone) to exercise A/B regression detection."""

    def __init__(self, version: str = "mock", latency_s: float = 0.8, reply_dur: float = 1.8,
                 f0: float = 120.0, greet: bool = True, seed: int = 0):
        self._version = version
        self._latency = latency_s
        self._reply_dur = reply_dur
        self._f0 = f0
        self._greet = greet
        self._seed = seed
        self._n = 0

    @property
    def agent_version(self) -> str:
        return self._version

    def start(self, scenario: Scenario) -> None:
        self._n = 0

    def greet(self, scenario: Scenario) -> Optional[Move]:
        if not self._greet:
            return None
        return Move("agent", "Thanks for calling, how can I help?",
                    speech_like(self._reply_dur, self._f0, seed=99), latency_s=0.2)

    def respond(self, user_audio: np.ndarray) -> Move:
        self._n += 1
        return Move("agent", f"Let me look into that (turn {self._n}).",
                    speech_like(self._reply_dur, self._f0, seed=self._n + 10 + 1000 * self._seed),
                    latency_s=self._latency)

    def close(self) -> None:
        pass
