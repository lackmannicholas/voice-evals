"""Semantic-layer adapter (design spec §11).

Out of scope to *build* the semantic evals here — that lives in the team's
existing transcript/behavioral pytest framework. This module is the thin adapter
so those results can register into the same ``EvalRun`` / report under
``layer="semantic"`` and flow through the same gates and HTML.

Usage from the existing framework:

    from voice_evals.semantic_hook import semantic_result, register_semantic

    sr = semantic_result(
        clip_id=clip.clip_id,
        metrics={"task_success": 1.0, "tool_call_correct": 1.0},
        raw={"transcript": "...", "expected": "..."},
    )
    eval_run.results.append(sr)            # or:
    register_semantic(eval_run, [sr, ...]) # bulk-merge into a run
"""

from __future__ import annotations

from typing import Any, Optional

from .models import EvalRun, ScoreResult


def semantic_result(
    clip_id: str,
    metrics: dict[str, float],
    raw: Optional[dict[str, Any]] = None,
    scorer: str = "semantic",
    error: Optional[str] = None,
    scorer_config_hash: str = "external",
) -> ScoreResult:
    """Map an external semantic/behavioral eval outcome into a ``ScoreResult``.

    Metric names are the caller's responsibility; prefix them (e.g. ``sem_*``)
    if you want them visually grouped in the report. Gates in ``config`` apply to
    them exactly like acoustic/dynamics metrics.
    """
    return ScoreResult(
        clip_id=clip_id,
        scorer=scorer,
        layer="semantic",
        metrics={k: float(v) for k, v in metrics.items()},
        raw=raw or {},
        error=error,
        scorer_config_hash=scorer_config_hash,
    )


def register_semantic(run: EvalRun, results: list[ScoreResult]) -> EvalRun:
    """Merge externally-produced semantic results into an existing run (in place)."""
    for r in results:
        if r.layer != "semantic":
            raise ValueError(f"expected layer='semantic', got {r.layer!r} for {r.scorer}")
        run.results.append(r)
    return run
