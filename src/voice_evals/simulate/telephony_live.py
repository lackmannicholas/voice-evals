"""Live Twilio transport: dial the agent's number + serve the media stream.

Thin async shell around the tested ``MediaStreamSession`` core. Placing the call
and bridging audio is standard Twilio Media Streams boilerplate:
  * REST ``calls.create`` with TwiML ``<Connect><Stream url=wss://.../caller>``,
  * a websocket that receives the agent's μ-law frames and sends the caller's back.

Requires (env/.env): TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN. Run the dev agent
locally (Twilio number + ngrok), expose THIS server via a tunnel, set
telephony.public_url to that wss URL. Cannot be unit-tested without a live call;
the conversation/barge-in logic it drives is tested in test_telephony.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Optional

from ..config import Config, load_secrets
from ..logging_util import get_logger
from .base import SimulatedCall
from .scenario import Scenario
from .telephony import CallerBot, MediaStreamSession

log = get_logger(__name__)


def _twilio_client(config: Config):
    from twilio.rest import Client

    sec = load_secrets()
    tc = config.telephony
    if not (sec.twilio_account_sid and sec.twilio_auth_token):
        raise RuntimeError("set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in env/.env")
    if not (tc.from_number and tc.agent_number and tc.public_url):
        raise RuntimeError("set telephony.from_number, agent_number, and public_url in config")
    return Client(sec.twilio_account_sid, sec.twilio_auth_token)


def place_call(config: Config, scenario_id: str) -> str:
    """Place the outbound call; Twilio connects its media stream to public_url."""
    from xml.sax.saxutils import quoteattr  # proper XML attribute escaping

    tc = config.telephony
    client = _twilio_client(config)
    twiml = (
        f"<Response><Connect><Stream url={quoteattr(tc.public_url)}>"
        f"<Parameter name='scenario' value={quoteattr(scenario_id)}/>"
        "</Stream></Connect></Response>"
    )
    call = client.calls.create(to=tc.agent_number, from_=tc.from_number, twiml=twiml)
    log.info("placed call %s -> %s (scenario %s)", call.sid, tc.agent_number, scenario_id)
    return call.sid


def _hangup(config: Config, call_sid: Optional[str]) -> None:
    if not call_sid:
        return
    try:
        _twilio_client(config).calls(call_sid).update(status="completed")
    except Exception as e:  # noqa: BLE001
        log.warning("failed to hang up call %s: %s", call_sid, e)


async def connection_handler(ws, session_factory, done: asyncio.Future, holder: dict) -> None:
    """The Twilio Media Streams websocket loop (extracted so it can be driven by a
    fake-Twilio client in tests, exercising the real transport without a live call).
    Translates Twilio JSON ↔ MediaStreamSession and surfaces any error via ``done``."""
    if "session" in holder:  # one call per server — ignore stray/duplicate connections
        await ws.close()
        return
    session = session_factory()
    holder["session"] = session
    stream_sid = None
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue  # one malformed frame shouldn't kill the call
            if msg.get("event") == "start":
                stream_sid = (msg.get("start") or {}).get("streamSid")
            for out in session.handle(msg):
                if stream_sid is None:
                    continue  # don't emit media before Twilio's 'start'
                out["streamSid"] = stream_sid
                await ws.send(json.dumps(out))
            if msg.get("event") == "stop" or session.should_hangup:
                break
        if not done.done():
            done.set_result(True)
    except Exception as e:  # noqa: BLE001 - surface it instead of scoring garbage
        if not done.done():
            done.set_exception(e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def _serve_one_call(
    config: Config, scenario: Scenario, caller: CallerBot, agent_version: str,
    prompt_version: Optional[str], out_dir: Path, placer=place_call, hangup=_hangup,
) -> SimulatedCall:
    """Start a one-shot media-stream server, place the call, run the session, finalize.

    NOTE (live-path): the caller-bot's OpenAI calls are synchronous; a future
    increment should pre-generate/prefetch turns off-loop so generation never
    blocks inbound media. The recorded timeline stays correct regardless (clock is
    frame-driven), but live outbound pacing can stall during generation."""
    import websockets

    tc = config.telephony
    loop = asyncio.get_event_loop()
    done: asyncio.Future = loop.create_future()
    holder: dict = {}

    def factory():
        return MediaStreamSession(scenario, caller, agent_version, prompt_version,
                                  vad_rms=tc.vad_rms, end_silence_ms=tc.end_silence_ms)

    async def handler(ws):
        await connection_handler(ws, factory, done, holder)

    call_sid: Optional[str] = None
    async with websockets.serve(handler, tc.host, tc.port):
        call_sid = await loop.run_in_executor(None, placer, config, scenario.id)
        try:
            await asyncio.wait_for(done, timeout=tc.max_call_s)
        except asyncio.TimeoutError:
            log.warning("call %s exceeded %.0fs; finalizing partial", scenario.id, tc.max_call_s)
        finally:
            await loop.run_in_executor(None, hangup, config, call_sid)

    session = holder.get("session")
    if session is None:
        raise RuntimeError("Twilio never connected the media stream (check public_url / tunnel)")
    return session.finalize(out_dir)


def run_calls(
    config: Config, scenarios: list[Scenario], caller_factory: Callable[[], CallerBot],
    agent_version: str, out_dir: Path, prompt_version: Optional[str] = None,
) -> list[SimulatedCall]:
    """Place one real call per scenario (sequentially) and return the recordings."""
    calls: list[SimulatedCall] = []
    for scenario in scenarios:
        call = asyncio.run(
            _serve_one_call(config, scenario, caller_factory(), agent_version, prompt_version, Path(out_dir))
        )
        log.info("call done: %s [%s] %.1fs, %d events", scenario.id, agent_version,
                 call.duration_s, len(call.events))
        calls.append(call)
    return calls
