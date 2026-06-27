"""Scenario manifest — the reusable test unit mined from a bad production call.

A scenario captures *how a hard caller behaves*, not what the right answer is.
The ``goal`` and ``persona`` drive the user-simulator so the conversation is
realistic and gets adversarial when mishandled; correctness/tool-call success is
evaluated elsewhere. Schema follows the goal + persona (+ optional goal-shift)
pattern that survived the research (AgentChangeBench / τ²-Bench).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str = "caller"
    style: str = "a typical caller"  # e.g. "terse, frustrated renter; short sentences"
    patience: Literal["low", "medium", "high"] = "medium"
    # how the caller's emotion should evolve; the lever that surfaces mishandling
    emotional_arc: str = "stays neutral if helped; mild frustration if stalled"
    adversarial: bool = False  # impatient/tangential/incomplete — needed for bad-UX sets
    traits: list[str] = Field(default_factory=list)
    voice: str = "alloy"  # TTS voice (or a clone reference id)


class BargeInPolicy(BaseModel):
    """When the simulated caller talks OVER the agent (needs a full-duplex transport,
    i.e. telephony). The agent's yield latency/success is then measured."""

    enabled: bool = False
    # interrupt once the agent has been speaking continuously for this long (seconds)
    after_agent_s: float = 2.5
    # how many of the caller's turns should be barge-ins (from the first eligible turn)
    max_barge_ins: int = 1
    # (backchannel interjections that should NOT make the agent yield — and the
    #  false-alarm metric they exercise — are a follow-up increment, not yet wired)


class Scenario(BaseModel):
    id: str
    source_call_id: Optional[str] = None  # provenance: which prod call seeded this
    domain: str = "a property leasing / resident-support phone call"
    persona: Persona = Field(default_factory=Persona)
    goal: str = ""  # what the caller wants — drives sim behavior, NOT scored here
    opening: Optional[str] = None  # first caller line; None => the agent greets first
    goal_shifts: list[str] = Field(default_factory=list)  # mid-call goal changes (optional)
    max_turns: int = 8  # caller turns before forced stop
    hangup_when: str = "you get what you need, or you give up in frustration"
    barge_in: BargeInPolicy = Field(default_factory=BargeInPolicy)
    tags: list[str] = Field(default_factory=list)

    # tagged onto produced recordings so voice-evals can group/gate by them
    def metadata_extra(self) -> dict:
        return {
            "scenario_id": self.id,
            "source_call_id": self.source_call_id,
            "persona": self.persona.name,
            "adversarial": self.persona.adversarial,
            "tags": self.tags,
        }


def load_scenarios(path: Path | str) -> list[Scenario]:
    """Load a YAML manifest: either a top-level list, or {scenarios: [...]}"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scenario manifest not found: {p}")
    data = yaml.safe_load(p.read_text()) or []
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    if not isinstance(data, list):
        raise ValueError("scenario manifest must be a list (or {scenarios: [...]})")
    return [Scenario.model_validate(s) for s in data]
