"""Aggregation, thresholds, and regression detection (design spec §13).

Gates are policy applied at aggregation time — never inside scorers. Two kinds:
  1. Absolute: ``metric >= floor`` (min) or ``metric <= ceiling`` (max), with an
     optional soft ``warn`` level.
  2. Regression: compare the current run to a baseline run, grouped by
     ``agent_version`` / ``prompt_version``; flag a metric that degrades by more
     than ``max_delta`` vs baseline. This catches "the new prompt made the voice
     worse" even when the value is still above the absolute floor.

Metric direction (higher-better vs lower-better) is inferred from the absolute
gate: a ``min`` gate => higher is better; a ``max`` gate => lower is better.
Metrics that only appear in regression are assumed higher-better unless they
match a known lower-better name pattern (latency/gap/overlap/false_alarm).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from .config import Config, GateSpec
from .logging_util import get_logger
from .models import EvalRun, ScoreResult

log = get_logger(__name__)

GateStatus = Literal["pass", "warn", "fail"]
_LOWER_BETTER_HINTS = ("latency", "gap", "overlap", "false_alarm", "_delay", "talk_ratio")


def metric_higher_is_better(metric: str, gates: dict[str, GateSpec]) -> bool:
    spec = gates.get(metric)
    if spec is not None:
        if spec.max is not None and spec.min is None:
            return False
        if spec.min is not None:
            return True
    return not any(h in metric for h in _LOWER_BETTER_HINTS)


# --------------------------------------------------------------------------- #
@dataclass
class GateOutcome:
    clip_id: str
    metric: str
    value: Optional[float]
    kind: Literal["absolute", "regression"]
    status: GateStatus
    threshold: Optional[float]
    message: str
    scorer: str = ""  # which scorer produced the metric (disambiguates name collisions)
    agent_version: Optional[str] = None
    prompt_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "scorer": self.scorer,
            "metric": self.metric,
            "value": self.value,
            "kind": self.kind,
            "status": self.status,
            "threshold": self.threshold,
            "message": self.message,
            "agent_version": self.agent_version,
            "prompt_version": self.prompt_version,
        }


@dataclass
class AggregationResult:
    long_df: pd.DataFrame
    summary_df: pd.DataFrame
    outcomes: list[GateOutcome] = field(default_factory=list)  # absolute, per clip×metric
    regressions: list[GateOutcome] = field(default_factory=list)  # vs baseline, per group×metric

    @property
    def failures(self) -> list[GateOutcome]:
        return [o for o in (self.outcomes + self.regressions) if o.status == "fail"]

    @property
    def warnings(self) -> list[GateOutcome]:
        return [o for o in (self.outcomes + self.regressions) if o.status == "warn"]

    def passed(self) -> bool:
        return len(self.failures) == 0


# --------------------------------------------------------------------------- #
def build_long_df(run: EvalRun) -> pd.DataFrame:
    """One row per (clip, scorer, metric). Errored scorers contribute a single
    row with metric=None and the error string, so they surface in the report."""
    rows: list[dict[str, Any]] = []
    for r in run.results:
        meta = run.metadata_for(r.clip_id)
        base = {
            "clip_id": r.clip_id,
            "scorer": r.scorer,
            "layer": r.layer,
            "cached": r.cached,
            "duration_ms": r.duration_ms,
            "error": r.error,
            "agent_version": getattr(meta, "agent_version", None),
            "prompt_version": getattr(meta, "prompt_version", None),
            "model_id": getattr(meta, "model_id", None),
            "tts_provider": getattr(meta, "tts_provider", None),
        }
        if r.error is not None or not r.metrics:
            rows.append({**base, "metric": None, "value": None})
            continue
        for metric, value in r.metrics.items():
            rows.append({**base, "metric": metric, "value": float(value)})
    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric aggregate stats: mean, median, p95, count, error_count, pass_rate."""
    if df.empty or "metric" not in df.columns:
        return pd.DataFrame()
    valued = df[df["metric"].notna() & df["value"].notna()]
    if valued.empty:
        return pd.DataFrame()

    def _p95(s: pd.Series) -> float:
        return float(s.quantile(0.95)) if len(s) else float("nan")

    g = valued.groupby("metric")["value"]
    summary = pd.DataFrame(
        {
            "count": g.count(),
            "mean": g.mean(),
            "median": g.median(),
            "p95": g.apply(_p95),
            "min": g.min(),
            "max": g.max(),
        }
    )
    # pass_rate filled later by gate evaluation if a 'passed' column exists
    if "passed" in df.columns:
        pr = (
            valued[valued["passed"].notna()]
            .groupby("metric")["passed"]
            .apply(lambda s: float(s.mean()) if len(s) else float("nan"))
        )
        summary["pass_rate"] = pr
    # error counts per scorer's metric scope: count error rows per metric is N/A,
    # so report error rows by scorer separately via the long df; keep a global tally
    return summary.reset_index()


# --------------------------------------------------------------------------- #
class Aggregator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(
        self, run: EvalRun, baseline_run: Optional[EvalRun] = None
    ) -> AggregationResult:
        df = build_long_df(run)
        outcomes = self._absolute_gates(run, df)
        regressions = (
            self._regression_gates(run, baseline_run)
            if (self.config.gates.regression.enabled and baseline_run is not None)
            else []
        )
        # mark a 'passed' column on the long df for summary pass_rate
        self._annotate_passed(df, outcomes)
        summary = summarize(df)
        return AggregationResult(
            long_df=df, summary_df=summary, outcomes=outcomes, regressions=regressions
        )

    # -- absolute ------------------------------------------------------- #
    def _absolute_gates(self, run: EvalRun, df: pd.DataFrame) -> list[GateOutcome]:
        gates = self.config.gates.absolute
        outcomes: list[GateOutcome] = []
        # roll up pass/fail to each ScoreResult.passed
        per_result_status: dict[tuple[str, str], list[GateStatus]] = {}
        for r in run.results:
            if r.error is not None:
                continue
            meta = run.metadata_for(r.clip_id)
            for metric, value in r.metrics.items():
                spec = gates.get(metric)
                if spec is None:
                    continue
                status, threshold, msg = _eval_absolute(
                    metric, float(value), spec, metric_higher_is_better(metric, gates)
                )
                outcomes.append(
                    GateOutcome(
                        clip_id=r.clip_id,
                        metric=metric,
                        value=float(value),
                        kind="absolute",
                        status=status,
                        threshold=threshold,
                        message=msg,
                        scorer=r.scorer,
                        agent_version=getattr(meta, "agent_version", None),
                        prompt_version=getattr(meta, "prompt_version", None),
                    )
                )
                per_result_status.setdefault((r.clip_id, r.scorer), []).append(status)
        # set ScoreResult.passed: False if any gated metric fails, True if all pass/warn, else None
        for r in run.results:
            statuses = per_result_status.get((r.clip_id, r.scorer))
            if not statuses:
                r.passed = None
            else:
                r.passed = all(s != "fail" for s in statuses)
        return outcomes

    # -- regression ----------------------------------------------------- #
    def _regression_gates(
        self, run: EvalRun, baseline: EvalRun
    ) -> list[GateOutcome]:
        reg = self.config.gates.regression
        if not reg.max_delta:
            return []
        cur = build_long_df(run)
        base = build_long_df(baseline)
        # group only by columns present in BOTH runs, so a key can actually match
        group_cols = [c for c in reg.group_by if c in cur.columns and c in base.columns]
        outcomes: list[GateOutcome] = []
        for metric, delta in reg.max_delta.items():
            higher_better = metric_higher_is_better(metric, self.config.gates.absolute)
            cur_m = cur[(cur["metric"] == metric) & cur["value"].notna()]
            base_m = base[(base["metric"] == metric) & base["value"].notna()]
            if cur_m.empty or base_m.empty:
                continue
            cur_g = _grouped_mean(cur_m, group_cols)
            base_g = _grouped_mean(base_m, group_cols)
            for key, cur_val in cur_g.items():
                if key not in base_g:
                    # don't silently hide a regression when a version label was lost
                    log.warning(
                        "regression: no baseline group %s for metric %s; skipping",
                        _key_str(key, group_cols), metric,
                    )
                    continue
                base_val = base_g[key]
                if higher_better:
                    drop = base_val - cur_val
                    failed = drop > delta
                    msg = (
                        f"{metric} {cur_val:.3f} vs baseline {base_val:.3f} "
                        f"(drop {drop:+.3f}, max {delta})"
                    )
                else:
                    rise = cur_val - base_val
                    failed = rise > delta
                    msg = (
                        f"{metric} {cur_val:.3f} vs baseline {base_val:.3f} "
                        f"(rise {rise:+.3f}, max {delta})"
                    )
                outcomes.append(
                    GateOutcome(
                        clip_id=f"group:{_key_str(key, group_cols)}",
                        metric=metric,
                        value=float(cur_val),
                        kind="regression",
                        status="fail" if failed else "pass",
                        threshold=float(delta),
                        message=msg,
                        agent_version=_key_field(key, group_cols, "agent_version"),
                        prompt_version=_key_field(key, group_cols, "prompt_version"),
                    )
                )
        return outcomes

    @staticmethod
    def _annotate_passed(df: pd.DataFrame, outcomes: list[GateOutcome]) -> None:
        if df.empty:
            return
        # key on (clip, scorer, metric) so two scorers emitting the same metric
        # name for one clip don't clobber each other (last-write-wins could hide a fail).
        status_by: dict[tuple[str, str, str], GateStatus] = {
            (o.clip_id, o.scorer, o.metric): o.status for o in outcomes
        }
        df["passed"] = [
            (None if (row.clip_id, row.scorer, row.metric) not in status_by
             else status_by[(row.clip_id, row.scorer, row.metric)] != "fail")
            for row in df.itertuples()
        ]


# --------------------------------------------------------------------------- #
def _eval_absolute(
    metric: str, value: float, spec: GateSpec, higher_better: bool
) -> tuple[GateStatus, Optional[float], str]:
    # Hard bounds first — both may apply (a banded min+max gate).
    if spec.min is not None and value < spec.min:
        return "fail", spec.min, f"{metric}={value:.3f} < min {spec.min}"
    if spec.max is not None and value > spec.max:
        return "fail", spec.max, f"{metric}={value:.3f} > max {spec.max}"
    # Warn level. Direction: a min gate warns below; a max gate warns above; a
    # warn-only gate uses the metric's inferred direction.
    if spec.warn is not None:
        warn_below = spec.min is not None or (spec.max is None and higher_better)
        if warn_below and value < spec.warn:
            return "warn", spec.warn, f"{metric}={value:.3f} < warn {spec.warn}"
        if not warn_below and value > spec.warn:
            return "warn", spec.warn, f"{metric}={value:.3f} > warn {spec.warn}"
    threshold = spec.min if spec.min is not None else spec.max
    return "pass", threshold, f"{metric}={value:.3f} ok"


def _grouped_mean(df: pd.DataFrame, group_cols: list[str]) -> dict:
    if not group_cols:
        return {"__all__": float(df["value"].mean())}
    out: dict = {}
    for key, sub in df.groupby(group_cols, dropna=False):
        out[key if isinstance(key, tuple) else (key,)] = float(sub["value"].mean())
    return out


def _key_str(key, group_cols: list[str]) -> str:
    if not group_cols:
        return "all"
    vals = key if isinstance(key, tuple) else (key,)
    return ",".join(f"{c}={v}" for c, v in zip(group_cols, vals))


def _key_field(key, group_cols: list[str], field_name: str):
    if field_name not in group_cols:
        return None
    vals = key if isinstance(key, tuple) else (key,)
    idx = group_cols.index(field_name)
    return vals[idx] if idx < len(vals) else None


# --------------------------------------------------------------------------- #
# baseline storage / selection
# --------------------------------------------------------------------------- #
def runs_dir(config: Config) -> Path:
    return config.resolve_path(config.outputs_dir) / "runs"


def persist_run(run: EvalRun, config: Config) -> Path:
    """Write run.json under outputs/runs/<run_id>/ and return that directory."""
    out = runs_dir(config) / run.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False))
    return out


def load_baseline(config: Config, current_run_id: Optional[str] = None) -> Optional[EvalRun]:
    """Resolve the baseline run per ``gates.regression.baseline`` ('previous' or a run_id)."""
    base_sel = config.gates.regression.baseline
    base_root = runs_dir(config)
    if not base_root.exists():
        return None
    run_dirs = sorted([d for d in base_root.iterdir() if (d / "run.json").exists()])
    if base_sel == "previous":
        candidates = [d for d in run_dirs if d.name != current_run_id]
        if not candidates:
            return None
        target = candidates[-1]
    else:
        match = [d for d in run_dirs if d.name == base_sel or d.name.startswith(base_sel)]
        if not match:
            log.warning("baseline run %r not found under %s", base_sel, base_root)
            return None
        target = match[-1]
    try:
        return EvalRun.from_dict(json.loads((target / "run.json").read_text()))
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load baseline %s: %s", target, e)
        return None


# --------------------------------------------------------------------------- #
# pytest gate-case enumeration (used by tests/eval_gates)
# --------------------------------------------------------------------------- #
def absolute_gate_cases(config: Config) -> list[tuple[str, GateSpec]]:
    """Return (metric, spec) for every configured absolute gate with a hard bound."""
    return [
        (m, spec)
        for m, spec in config.gates.absolute.items()
        if spec.min is not None or spec.max is not None
    ]
