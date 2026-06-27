"""Simulation/driver half (WALK): run scenarios through the agent under test.

A persona-conditioned user simulator holds a multi-turn conversation with the
agent and produces a recording + event log + transcript. Those artifacts are then
scored by the existing voice-evals *evaluator* (audio-experience only — task /
tool-call correctness is owned by the team's text/OTel evals, by design).

Pattern sources (deep-research, 2024-2026): live bot-to-bot simulation rather
than audio replay (divergence / Dec-POMDP), persona conditioning, adversarial
non-collaborative users, consistency-check-and-regenerate, and k-run aggregate
gating because trial stochasticity dominates variance.
"""

from __future__ import annotations

from .scenario import Persona, Scenario, load_scenarios

__all__ = ["Persona", "Scenario", "load_scenarios"]
