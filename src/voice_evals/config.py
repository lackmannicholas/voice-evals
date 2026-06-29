"""Configuration schema and loader (design spec §18).

YAML on disk + environment overrides for secrets. Secrets (the Gemini API key)
are read from the environment only, never from YAML. The resolved, secret-free
config is snapshotted into every ``run.json`` for reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Secrets — environment only
# --------------------------------------------------------------------------- #
class Secrets(BaseSettings):
    """Resolved from the environment at runtime; never serialized into snapshots."""

    # Reads process env first, then a gitignored .env file (handy for local keys).
    model_config = SettingsConfigDict(env_prefix="", extra="ignore", env_file=".env")

    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    twilio_account_sid: Optional[str] = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    # telephony connection params — not secret, but convenient to keep in .env
    twilio_from_number: Optional[str] = Field(default=None, alias="TWILIO_FROM_NUMBER")
    twilio_agent_number: Optional[str] = Field(default=None, alias="TWILIO_AGENT_NUMBER")
    public_stream_url: Optional[str] = Field(default=None, alias="PUBLIC_STREAM_URL")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    @property
    def resolved_gemini_key(self) -> Optional[str]:
        return self.gemini_api_key or self.google_api_key


# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #
class IngestConfig(BaseModel):
    target_sr: int = 16000
    normalize_dbfs: Optional[float] = None  # null = leave gain untouched
    accept_extensions: list[str] = Field(
        default_factory=lambda: [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    )
    min_duration_s: float = 0.5  # warn (don't fail) below this
    max_duration_s: float = 1800.0  # warn (don't fail) above this (30 min)
    # Production default: clip_id = content hash (re-encode-stable, dedups re-uploads).
    # Simulated corpora set this True so each (scenario, version, run) stays a DISTINCT
    # clip even when audio coincides — otherwise content-dedup collapses k-run samples.
    identity_from_call_id: bool = False


class AcousticConfig(BaseModel):
    # MOS predictors are trained on short utterances and SQUIM blows up on very
    # long inputs, so long clips are scored in windows and aggregated per metric.
    window_s: float = 20.0
    min_window_s: float = 3.0  # drop a trailing window shorter than this
    aggregate: Literal["median", "mean"] = "median"


class ScorersConfig(BaseModel):
    acoustic: list[str] = Field(default_factory=lambda: ["dnsmos", "nisqa", "utmos", "squim"])
    dynamics: list[str] = Field(default_factory=lambda: ["latency", "turn_taking", "barge_in"])
    judge: list[str] = Field(default_factory=list)  # empty => local_only safe

    def all_selected(self) -> list[str]:
        return [*self.acoustic, *self.dynamics, *self.judge]


class DynamicsConfig(BaseModel):
    vad_backend: Literal["silero", "ten"] = "ten"
    gap_threshold_s: float = 1.5  # awkward inter-turn silence
    bargein_success_window_s: float = 1.5
    bargein_stop_target_s: float = 0.2  # aspirational "natural" stop latency
    min_caller_utterance_ms: float = 100.0  # below this is a backchannel, not a turn
    overlap_min_s: float = 0.1  # ignore overlaps shorter than this
    vad_threshold: float = 0.5  # speech-probability threshold for VAD backends


class JudgeConfig(BaseModel):
    backend: Literal["gemini", "openai", "local_omni"] = "gemini"
    model: str = "gemini-2.5-pro"  # set an audio model matching the backend
    judged_channel: Literal["agent_only", "mono", "caller_only"] = "agent_only"
    domain: str = "a property leasing / resident-support phone call"
    max_concurrency: int = 4
    exclude_if_model_family_matches: bool = True
    judge_model_family: str = "gemini"  # used by the self-enhancement guard
    rubric_path: str = "config/rubric.default.txt"
    system_prompt_path: str = "config/judge_system.txt"
    thinking_budget: int = 1024
    temperature: float = 0.0
    max_audio_s: float = 0.0  # >0 => judge only the first N seconds (cost/length cap)
    max_retries: int = 5  # transport retries (429/5xx) with backoff
    parse_retries: int = 1  # schema/parse-failure nudges before giving up
    upload_threshold_bytes: int = 18_000_000  # inline below, Files API above
    local_model: str = "Qwen/Qwen2.5-Omni-7B"  # used by local_omni backend


class GateSpec(BaseModel):
    """A single absolute gate. ``min`` is a floor, ``max`` a ceiling, ``warn`` a soft level."""

    min: Optional[float] = None
    max: Optional[float] = None
    warn: Optional[float] = None


class RegressionConfig(BaseModel):
    enabled: bool = True
    baseline: str = "previous"  # tag of a prior run, or "previous"
    group_by: list[str] = Field(default_factory=lambda: ["agent_version"])
    max_delta: dict[str, float] = Field(default_factory=dict)


class GatesConfig(BaseModel):
    absolute: dict[str, GateSpec] = Field(default_factory=dict)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)


class RunnerConfig(BaseModel):
    cpu_workers: int = 0  # 0 => auto (cpu_count - 1, min 1)
    progress: bool = True


class CalibrationConfig(BaseModel):
    kappa_bar: float = 0.6  # min weighted kappa to recommend a judge dim for CI gating
    golden_path: str = "calibration/golden_set.csv"


class SimulateConfig(BaseModel):
    """Driver/simulator (WALK) settings — produces recordings the evaluator scores."""

    k_runs: int = 3  # runs per scenario per agent version (sims are stochastic)
    max_turns: int = 8
    max_regen: int = 2  # regenerate a run this many times if simulation is invalid
    # user simulator (persona-conditioned caller)
    user_sim_model: str = "gpt-4o"  # text model that generates the caller's lines
    user_sim_temperature: float = 0.8  # >0 so k runs actually differ (avoid dedup collapse)
    tts_model: str = "gpt-4o-mini-tts"
    # agent under test (OpenAI Realtime by default; swap for your own AgentBackend)
    agent_model: str = "gpt-realtime"
    agent_voice: str = "alloy"
    agent_instructions_path: Optional[str] = None  # your agent's system prompt
    turn_timeout_s: float = 60.0  # wall-clock cap on one agent response (avoids hangs)


class TelephonyConfig(BaseModel):
    """Twilio Media Streams caller — places a real outbound call to the agent's
    number and runs the caller-bot over the call's μ-law/8kHz audio. The websocket
    is embedded; you only run the command + a tunnel to its port (Twilio creds and
    SIDs come from the environment/.env)."""

    from_number: Optional[str] = None  # your Twilio dev number (caller ID)
    agent_number: Optional[str] = None  # the agent's number to dial
    public_url: Optional[str] = None  # wss://<tunnel>/caller — Twilio connects here
    host: str = "127.0.0.1"  # bind local-only; the tunnel (ngrok) forwards to it
    port: int = 8080
    vad_rms: float = 0.02  # agent-speech energy threshold (on the 8k call audio)
    end_silence_ms: int = 600  # silence after agent speech that marks a turn end
    max_call_s: float = 180.0  # hard cap per call


class Config(BaseModel):
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    acoustic: AcousticConfig = Field(default_factory=AcousticConfig)
    scorers: ScorersConfig = Field(default_factory=ScorersConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    simulate: SimulateConfig = Field(default_factory=SimulateConfig)
    telephony: TelephonyConfig = Field(default_factory=TelephonyConfig)

    # Filesystem layout (relative to project root unless absolute).
    corpus_dir: Path = Path("corpus")
    cache_dir: Path = Path(".cache")
    outputs_dir: Path = Path("outputs")
    # Real-call recordings accumulate here, one dated subdir per `call` invocation
    # (a kept library — never auto-wiped; delete entries by hand as needed).
    recordings_dir: Path = Path("recordings")

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "Config":
        """Load config from a YAML file, falling back to all defaults."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
        return cls.model_validate(data)

    def snapshot(self) -> dict[str, Any]:
        """Secret-free, JSON-able snapshot for embedding in run.json."""
        return self.model_dump(mode="json")

    def disable_judge(self) -> "Config":
        """Return a copy with judge scorers stripped (the ``--no-judge`` path)."""
        clone = self.model_copy(deep=True)
        clone.scorers.judge = []
        return clone

    def resolve_path(self, p: Path | str) -> Path:
        """Resolve a possibly-relative config path against the current working dir."""
        pp = Path(p)
        return pp if pp.is_absolute() else (Path.cwd() / pp)


def load_secrets() -> Secrets:
    """Read secrets from the environment (and .env if present)."""
    return Secrets()  # type: ignore[call-arg]


def default_config_path() -> Optional[Path]:
    """Locate ``config/default.yaml`` relative to CWD if it exists."""
    candidate = Path.cwd() / "config" / "default.yaml"
    return candidate if candidate.exists() else None
