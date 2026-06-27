"""Drive scenarios through the agent and emit recordings for the evaluator.

run_scenario holds one multi-turn conversation (with regenerate-on-invalid);
run_suite fans that over k runs × scenarios for one agent version. Live sims are
stochastic, so k>1 + aggregate gating is the point (research: trial stochasticity
dominates variance, single-pass scoring overstates quality).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..logging_util import get_logger
from .base import (
    AgentBackend,
    Move,
    SimulatedCall,
    UserSimulator,
    assemble_recording,
    validate_conversation,
)
from .scenario import Scenario

log = get_logger(__name__)


def run_scenario(
    scenario: Scenario,
    agent: AgentBackend,
    user_sim: UserSimulator,
    run_index: int,
    out_dir: Path,
    max_regen: int = 2,
    prompt_version: str | None = None,
) -> SimulatedCall:
    last_moves: list[Move] = []
    reason = None
    for attempt in range(max_regen + 1):
        moves = _converse(scenario, agent, user_sim)
        ok, reason = validate_conversation(moves)
        last_moves = moves
        if ok:
            return assemble_recording(
                moves, scenario, agent.agent_version, run_index, out_dir, prompt_version
            )
        log.warning(
            "scenario %s run %d invalid (attempt %d/%d): %s; regenerating",
            scenario.id, run_index, attempt + 1, max_regen + 1, reason,
        )
    # couldn't get a valid run — emit anyway but flag it (never silently drop)
    call = assemble_recording(
        last_moves, scenario, agent.agent_version, run_index, out_dir, prompt_version
    )
    call.valid = False
    call.invalid_reason = reason
    return call


def _converse(scenario: Scenario, agent: AgentBackend, user_sim: UserSimulator) -> list[Move]:
    agent.start(scenario)
    moves: list[Move] = []
    try:
        greeting = agent.greet(scenario)
        if greeting is not None:
            moves.append(greeting)
        while True:
            user_move = user_sim.next_turn(scenario, moves)
            moves.append(user_move)
            # caller hung up: the call is over — the agent never speaks to a dead line
            if user_move.hangup:
                break
            agent_move = agent.respond(user_move.audio)
            moves.append(agent_move)
            if sum(1 for m in moves if m.speaker == "caller") >= scenario.max_turns:
                break
    finally:
        agent.close()
    return moves


def run_suite(
    scenarios: list[Scenario],
    make_agent: Callable[[], AgentBackend],
    make_user_sim: Callable[[], UserSimulator],
    k: int,
    out_dir: Path,
    prompt_version: str | None = None,
) -> list[SimulatedCall]:
    """Run every scenario k times for one agent version. Fresh backend + simulator
    per run so stochastic state never bleeds across runs."""
    out_dir = Path(out_dir)
    calls: list[SimulatedCall] = []
    for scenario in scenarios:
        for run_index in range(k):
            call = run_scenario(
                scenario, make_agent(), make_user_sim(), run_index, out_dir, prompt_version=prompt_version
            )
            status = "ok" if call.valid else f"INVALID({call.invalid_reason})"
            log.info("simulated %s [%s] run %d -> %s (%.1fs)",
                     scenario.id, call.agent_version, run_index, status, call.duration_s)
            calls.append(call)
    return calls
