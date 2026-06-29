"""Twilio Media Streams caller core — codec + session barge-in logic.

Driven by scripted Twilio frames + a mock caller (no live call). Verifies the
μ-law codec, full-duplex turn-taking, barge-in detection/timing, the recorded
dual-channel artifact, and that the existing barge-in scorer reads the events.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
import soundfile as sf

from voice_evals.config import Config
from voice_evals.scorers.dynamics.barge_in import BargeInScorer
from voice_evals.simulate.scenario import BargeInPolicy, Persona, Scenario
from voice_evals.simulate.telephony import (
    FRAME_SAMPLES,
    SR8,
    MediaStreamSession,
    mulaw_decode,
    mulaw_encode,
)

pytestmark = pytest.mark.local


def test_mulaw_roundtrip():
    # μ-law is lossy but monotone/structure-preserving; check bounded error on a ramp+tone
    t = np.linspace(0, 1, SR8, endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
    back = mulaw_decode(mulaw_encode(pcm))
    assert back.shape == pcm.shape
    # μ-law guarantees ~< 1% segment error; allow generous bound on normalized RMS error
    err = np.sqrt(np.mean(((back.astype(float) - pcm.astype(float)) / 32768) ** 2))
    assert err < 0.02


def _voiced(amp=0.25) -> bytes:
    t = np.linspace(0, FRAME_SAMPLES / SR8, FRAME_SAMPLES, endpoint=False)
    pcm = (amp * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16)
    return mulaw_encode(pcm)


def _silence() -> bytes:
    return mulaw_encode(np.zeros(FRAME_SAMPLES, np.int16))


def _media(ulaw: bytes) -> dict:
    return {"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(ulaw).decode()}}


class _MockCaller:
    """Returns ~0.5s of voiced μ-law per turn; hangs up at max_turns."""

    def next_turn(self, scenario, turn_index, history):
        n = int(0.5 * SR8)
        t = np.linspace(0, 0.5, n, endpoint=False)
        pcm = (0.3 * np.sin(2 * np.pi * 160 * t) * 32767).astype(np.int16)
        hangup = (turn_index + 1) >= scenario.max_turns
        return mulaw_encode(pcm), f"caller turn {turn_index}", hangup


def _feed(session, ulaw, n_frames):
    out = []
    for _ in range(n_frames):
        out += session.handle(_media(ulaw))
    return out


def test_caller_responds_after_agent_turn():
    scn = Scenario(id="t1", max_turns=3, persona=Persona())
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    _feed(s, _voiced(), 50)          # agent greets ~1s
    out = _feed(s, _silence(), 80)   # silence -> agent turn end -> caller replies (~0.5s)
    assert any(m["event"] == "media" for m in out)  # caller audio was emitted
    kinds = [e["kind"] for e in s.events]
    assert "user_speech_start" in kinds and "user_speech_end" in kinds


def test_barge_in_detected_and_agent_yields():
    scn = Scenario(id="t2", max_turns=3, persona=Persona(adversarial=True),
                   barge_in=BargeInPolicy(enabled=True, after_agent_s=1.0, max_barge_ins=1))
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    # agent greets; enough silence for the caller's first reply (~0.5s) to finish
    _feed(s, _voiced(), 30)
    _feed(s, _silence(), 70)
    # agent launches a long response; caller barges in after 1.0s of agent speech,
    # the agent talks a touch longer, then yields (goes silent)
    _feed(s, _voiced(), 60)           # barge fires ~1.0s in; agent voiced a bit past it
    _feed(s, _silence(), 50)          # agent silent -> yielded
    kinds = [e["kind"] for e in s.events]
    assert "barge_in_detected" in kinds, kinds
    assert "agent_interrupted" in kinds, kinds
    bd = next(e["t_s"] for e in s.events if e["kind"] == "barge_in_detected")
    ai = next(e["t_s"] for e in s.events if e["kind"] == "agent_interrupted")
    stop_latency = ai - bd
    assert 0.0 < stop_latency < 1.5  # agent yielded within the window


def _score_barge(events) -> dict:
    from voice_evals.models import AudioClip, ClipMetadata, GatewayEvent
    meta = ClipMetadata(events=[GatewayEvent(e["kind"], e["t_s"]) for e in events])
    p = __import__("pathlib").Path("x")
    clip = AudioClip("c", p, p, None, None, None, SR8, 5.0, 2, meta)
    return BargeInScorer(Config()).score(clip).metrics


def test_agent_talks_through_barge_then_ends_turn_is_a_FAILURE():
    # THE case the team most wants to catch: caller barges in, the agent ignores it,
    # keeps talking PAST the caller's utterance, then ends its turn naturally.
    # That trailing silence must NOT be mis-scored as a yield/success.
    scn = Scenario(id="noyield", max_turns=3, persona=Persona(adversarial=True),
                   barge_in=BargeInPolicy(enabled=True, after_agent_s=1.0, max_barge_ins=1))
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    _feed(s, _voiced(), 30); _feed(s, _silence(), 70)   # greet + caller1
    _feed(s, _voiced(), 130)  # barge fires ~1.0s in; agent keeps talking well PAST the caller
    _feed(s, _silence(), 50)  # then the agent ends its turn naturally (trailing silence)
    kinds = [e["kind"] for e in s.events]
    assert "barge_in_detected" in kinds
    assert "agent_interrupted" not in kinds       # agent talked through -> no real yield
    m = _score_barge(s.events)
    assert m["n_bargein_events"] == 1.0 and m["bargein_success_rate"] == 0.0


def test_agent_yields_while_caller_talking_is_a_success():
    scn = Scenario(id="yield", max_turns=3, persona=Persona(adversarial=True),
                   barge_in=BargeInPolicy(enabled=True, after_agent_s=1.0, max_barge_ins=1))
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    _feed(s, _voiced(), 30); _feed(s, _silence(), 70)
    _feed(s, _voiced(), 55)   # barge fires ~1.0s in (caller still talking)
    _feed(s, _silence(), 40)  # agent stops quickly while/just-after the caller -> yield
    kinds = [e["kind"] for e in s.events]
    assert "barge_in_detected" in kinds and "agent_interrupted" in kinds
    m = _score_barge(s.events)
    assert m["bargein_success_rate"] == 1.0


def test_stream_twiml_normalizes_https_to_wss_and_escapes():
    from voice_evals.simulate.telephony_live import stream_twiml
    # ngrok shows https:// — Twilio <Stream> needs wss://; must auto-convert
    t = stream_twiml("https://abc-123.ngrok-free.app/caller", "broken_ac")
    assert "url=\"wss://abc-123.ngrok-free.app/caller\"" in t
    assert "https://" not in t
    assert "broken_ac" in t
    # ampersands / quotes in values must be escaped, not break the XML
    t2 = stream_twiml("wss://h/caller?a=1&b=2", "x&y")
    assert "&amp;" in t2 and "<Stream" in t2


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_real_live_server_handles_a_fake_twilio_stream(tmp_path):
    """Drive the REAL telephony_live websocket server (connection_handler) with a
    local fake-Twilio client replaying the media-stream protocol — exercises the
    transport (json over ws, start/media/stop, outbound media, one-call guard,
    finalize) that scripted-frame unit tests skip, without a live PSTN call."""
    import asyncio

    asyncio.run(_run_fake_twilio(tmp_path))


async def _run_fake_twilio(tmp_path):
    import asyncio
    import json

    import websockets

    from voice_evals.config import Config
    from voice_evals.simulate.telephony_live import connection_handler

    cfg = Config()
    tc = cfg.telephony
    port = _free_port()
    scn = Scenario(id="h", max_turns=2, persona=Persona(adversarial=True),
                   barge_in=BargeInPolicy(enabled=True, after_agent_s=1.0, max_barge_ins=1))
    loop = asyncio.get_event_loop()
    done = loop.create_future()
    holder: dict = {}

    def factory():
        return MediaStreamSession(scn, _MockCaller(), "v", None,
                                  vad_rms=tc.vad_rms, end_silence_ms=tc.end_silence_ms)

    async def handler(ws):
        await connection_handler(ws, factory, done, holder)

    received: list[str] = []
    async with websockets.serve(handler, "127.0.0.1", port):
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            async def reader():
                try:
                    async for m in client:
                        received.append(m)
                except Exception:
                    pass

            rt = asyncio.create_task(reader())
            try:
                await client.send(json.dumps({"event": "connected"}))
                await client.send(json.dumps({"event": "start", "start": {"streamSid": "MZ123"}}))
                for ulaw, n in [(_voiced(), 30), (_silence(), 70), (_voiced(), 130), (_silence(), 50)]:
                    for _ in range(n):
                        await client.send(json.dumps(_media(ulaw)))
                        await asyncio.sleep(0)  # let the server process + stream audio back
                await client.send(json.dumps({"event": "stop"}))
            except websockets.exceptions.ConnectionClosed:
                pass  # server hung up when the caller reached max_turns — expected
            await asyncio.wait_for(done, timeout=10)
            rt.cancel()

    session = holder["session"]
    call = session.finalize(tmp_path)
    assert call.wav_path.exists()
    kinds = {e["kind"] for e in session.events}
    assert "barge_in_detected" in kinds and "user_speech_start" in kinds
    # the server actually streamed caller audio back over the real socket
    assert any('"media"' in m for m in received)


def test_recording_and_scorer_integration(tmp_path):
    scn = Scenario(id="t3", max_turns=2, persona=Persona(adversarial=True),
                   barge_in=BargeInPolicy(enabled=True, after_agent_s=1.0))
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    _feed(s, _voiced(), 30); _feed(s, _silence(), 40)
    _feed(s, _voiced(), 60); _feed(s, _silence(), 40)
    call = s.finalize(tmp_path)
    # dual-channel 8k recording written
    data, sr = sf.read(str(call.wav_path))
    assert sr == SR8 and data.ndim == 2 and data.shape[1] == 2
    # the barge-in SCORER reads the event log and computes a stop latency + success
    from voice_evals.models import AudioClip, ClipMetadata, GatewayEvent
    meta = ClipMetadata(events=[GatewayEvent(e["kind"], e["t_s"]) for e in call.events])
    clip = AudioClip("c", call.wav_path, call.wav_path, None, None, None, SR8, call.duration_s, 2, meta)
    r = BargeInScorer(Config()).score(clip)
    assert r.error is None
    assert r.metrics["n_bargein_events"] >= 1.0
    assert "bargein_success_rate" in r.metrics


def test_agent_turn_events_make_latency_and_turntaking_work(tmp_path):
    # regression for the live-call bug: the session must emit agent_tts_start/end so
    # the event-log dynamics (latency, turn-taking) see the agent's turns, not just barges.
    from voice_evals.models import AudioClip, ClipMetadata, GatewayEvent
    from voice_evals.scorers.dynamics.latency import LatencyScorer
    from voice_evals.scorers.dynamics.turn_taking import TurnTakingScorer

    scn = Scenario(id="dyn", max_turns=3, persona=Persona())
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    _feed(s, _voiced(), 40); _feed(s, _silence(), 60)   # agent greets, then caller replies
    _feed(s, _voiced(), 40); _feed(s, _silence(), 60)   # agent responds again, caller replies
    call = s.finalize(tmp_path)
    kinds = [e["kind"] for e in call.events]
    assert "agent_tts_start" in kinds and "agent_tts_end" in kinds

    meta = ClipMetadata(events=[GatewayEvent(e["kind"], e["t_s"]) for e in call.events])
    p = call.wav_path
    clip = AudioClip("c", p, p, None, None, None, SR8, call.duration_s, 2, meta)
    lat = LatencyScorer(Config()).score(clip)
    tt = TurnTakingScorer(Config()).score(clip)
    assert lat.error is None and lat.metrics["n_turns"] >= 1.0     # was: errored
    assert tt.error is None and tt.metrics["agent_talk_ratio"] > 0  # was: 0.0


class _RecordingCaller:
    """Like _MockCaller, but captures the history handed to it each turn."""

    def __init__(self):
        self.seen_histories: list[list[dict]] = []

    def next_turn(self, scenario, turn_index, history):
        self.seen_histories.append([dict(h) for h in history])
        n = int(0.5 * SR8)
        t = np.linspace(0, 0.5, n, endpoint=False)
        pcm = (0.3 * np.sin(2 * np.pi * 160 * t) * 32767).astype(np.int16)
        hangup = (turn_index + 1) >= scenario.max_turns
        return mulaw_encode(pcm), f"caller turn {turn_index}", hangup


class _FakeTranscriber:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, pcm16, sr):
        self.calls += 1
        assert sr == SR8 and pcm16.size > 0
        return self.text


def test_agent_turn_transcribed_and_caller_reacts():
    # the reactive-caller fix: the agent's turn is STT'd into the transcript so the
    # caller-bot's NEXT turn is conditioned on what the agent actually said.
    scn = Scenario(id="react", max_turns=3, persona=Persona())
    caller = _RecordingCaller()
    stt = _FakeTranscriber("we can schedule a technician for tomorrow")
    s = MediaStreamSession(scn, caller, agent_version="v", transcriber=stt)
    _feed(s, _voiced(), 40)     # agent turn ~0.8s
    _feed(s, _silence(), 60)    # silence -> agent turn ends -> STT -> caller replies
    assert stt.calls >= 1
    assert {"speaker": "agent", "text": "we can schedule a technician for tomorrow"} in s.transcript
    assert "stt_final" in [e["kind"] for e in s.events]
    # the caller actually SAW the agent's line when generating its reply
    assert any(any(h["speaker"] == "agent" for h in hist) for hist in caller.seen_histories)


def test_no_transcriber_keeps_caller_deaf():
    # without a transcriber (scripted/unit path) the agent is NOT transcribed; the
    # transcript holds only caller turns, preserving prior behavior.
    scn = Scenario(id="deaf", max_turns=3, persona=Persona())
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v")
    _feed(s, _voiced(), 40); _feed(s, _silence(), 60)
    assert s.transcript and all(t["speaker"] == "caller" for t in s.transcript)
    assert "stt_final" not in [e["kind"] for e in s.events]


def test_build_background_bed_loops_and_attenuates():
    from voice_evals.simulate.telephony import build_background_bed
    seg = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.8, int(0.8 * 16000), endpoint=False))
           ).astype(np.float32)
    bed = build_background_bed([seg], total_s=4.0, gain_db=-18.0)
    assert bed is not None and bed.dtype == np.int16
    assert abs(len(bed) - int(4.0 * SR8)) <= SR8           # ~4s @ 8k (loop-filled)
    assert 0 < np.max(np.abs(bed)) < 0.5 * 32767           # attenuated well below full scale
    assert build_background_bed([], 4.0, -18.0) is None    # nothing to build from


def test_background_mixes_onto_caller_channel_continuously():
    # a synthetic always-on bed lets us assert continuous emission deterministically
    bed = (0.2 * np.sin(2 * np.pi * 200 * np.linspace(0, 1.0, SR8, endpoint=False)) * 32767
           ).astype(np.int16)
    scn = Scenario(id="noisy", max_turns=3, persona=Persona())
    s = MediaStreamSession(scn, _MockCaller(), agent_version="v", background=bed)
    out_during_agent = _feed(s, _voiced(), 30)             # agent talking; caller not speaking yet
    # the background streams on the caller channel even while the caller is silent
    assert any(m["event"] == "media" for m in out_during_agent)
    _feed(s, _silence(), 60)                               # agent ends -> caller speaks (mixed w/ bg)
    kinds = [e["kind"] for e in s.events]
    assert "user_speech_start" in kinds and "user_speech_end" in kinds
    # caller channel carries audio throughout the call, not only during the spoken turn
    caller_frames = [p for p in s.placed if p.speaker == "caller"]
    assert len(caller_frames) > 50
