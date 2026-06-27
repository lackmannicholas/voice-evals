"""A/B version comparison + k-run aggregation for the dev loop.

Given ONE EvalRun that contains recordings from a candidate and a baseline agent
version (each scenario run k times), compute per-(scenario, metric) deltas and a
verdict. Research-grounded choices:
  * aggregate over k runs (median) before comparing — single-pass is unreliable,
  * compare PER SCENARIO so a regression on one hard scenario isn't masked by
    improvements on easy ones (then fail if ANY scenario regresses),
  * relative candidate-vs-baseline comparison rather than absolute sim scores
    (simulated users are miscalibrated against humans in absolute terms).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..aggregate import metric_higher_is_better
from ..config import Config
from ..logging_util import get_logger
from ..models import EvalRun

log = get_logger(__name__)


@dataclass
class MetricDelta:
    scenario: str
    metric: str
    baseline: float
    candidate: float
    delta: float  # candidate - baseline (raw)
    higher_better: bool
    verdict: str  # "improved" | "regressed" | "≈same"
    regressed_beyond_gate: bool


@dataclass
class VersionComparison:
    candidate: str
    baseline: str
    n_runs_candidate: int
    n_runs_baseline: int
    deltas: list[MetricDelta]
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed_beyond_gate]

    @property
    def improved(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.verdict == "improved"]

    def passed(self) -> bool:
        return self.error is None and len(self.regressions) == 0

    def render_text(self) -> str:
        if self.error:
            return f"A/B comparison FAILED: {self.error}"
        head = (
            f"A/B: candidate '{self.candidate}' ({self.n_runs_candidate} runs) "
            f"vs baseline '{self.baseline}' ({self.n_runs_baseline} runs)\n"
            f"{'scenario':22}{'metric':24}{'base':>9}{'cand':>9}{'Δ':>8}  verdict\n" + "-" * 88
        )
        lines = [head]
        for d in sorted(self.deltas, key=lambda x: (x.verdict != "regressed", x.scenario, x.metric)):
            flag = "  ⚠REGRESSION" if d.regressed_beyond_gate else ""
            lines.append(
                f"{d.scenario[:21]:22}{d.metric[:23]:24}{d.baseline:>9.3f}{d.candidate:>9.3f}"
                f"{d.delta:>+8.3f}  {d.verdict}{flag}"
            )
        lines.append("-" * 88)
        for w in self.warnings:
            lines.append(f"  ! {w}")
        verdict = "PASS — no regressions beyond gate" if self.passed() else (
            f"FAIL — {len(self.regressions)} (scenario,metric) regressed beyond gate"
        )
        lines.append(f"{len(self.improved)} improved · {verdict}")
        return "\n".join(lines)


def _medians(run: EvalRun) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, int]]:
    """(version, scenario) -> metric -> median value; plus runs-per-version count.
    Reads scenario_id from sidecar metadata.extra so per-scenario A/B is possible."""
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    runs_per_version: dict[str, set[str]] = defaultdict(set)
    for r in run.results:
        if r.error or not r.metrics:
            continue
        meta = run.metadata_for(r.clip_id)
        version = getattr(meta, "agent_version", None)
        if not version:
            continue
        scenario = (getattr(meta, "extra", {}) or {}).get("scenario_id", "_all")
        runs_per_version[version].add(r.clip_id)
        for m, v in r.metrics.items():
            acc[(version, scenario)][m].append(float(v))
    medians = {k: {m: statistics.median(vs) for m, vs in d.items()} for k, d in acc.items()}
    return medians, {v: len(s) for v, s in runs_per_version.items()}


def compare_versions(
    run: EvalRun, candidate: str, baseline: str, config: Config, eps: float = 0.05
) -> VersionComparison:
    medians, n_runs = _medians(run)
    n_cand, n_base = n_runs.get(candidate, 0), n_runs.get(baseline, 0)
    if n_cand == 0 or n_base == 0:
        missing = candidate if n_cand == 0 else baseline
        return VersionComparison(
            candidate, baseline, n_cand, n_base, [],
            error=f"no scored runs for version '{missing}' "
                  f"(found versions: {sorted({v for v, _ in medians})})",
        )

    max_delta = config.gates.regression.max_delta
    gates = config.gates.absolute
    cand_scenarios = {s for (v, s) in medians if v == candidate}
    base_scenarios = {s for (v, s) in medians if v == baseline}
    scenarios = cand_scenarios & base_scenarios
    warnings: list[str] = []
    for s in sorted(cand_scenarios ^ base_scenarios):
        warnings.append(f"scenario '{s}' present in only one version; excluded from A/B")

    deltas: list[MetricDelta] = []
    for scenario in sorted(scenarios):
        cand_m = medians[(candidate, scenario)]
        base_m = medians[(baseline, scenario)]
        only_one = set(cand_m) ^ set(base_m)
        for m in sorted(only_one):
            warnings.append(f"metric '{m}' in scenario '{scenario}' present in only one version; skipped")
        for metric in sorted(set(cand_m) & set(base_m)):
            b, c = base_m[metric], cand_m[metric]
            raw = c - b
            higher_better = metric_higher_is_better(metric, gates)
            improvement = raw if higher_better else -raw  # positive = better
            thresh = max_delta.get(metric, eps)  # ONE threshold for gate AND verdict
            # banded (min AND max) gates are non-monotone — no single direction is
            # "better", so don't flag a regression from a monotone delta (advisory).
            spec = gates.get(metric)
            banded = spec is not None and spec.min is not None and spec.max is not None
            if banded:
                verdict = "≈same"
            elif improvement > thresh:
                verdict = "improved"
            elif improvement < -thresh:
                verdict = "regressed"
            else:
                verdict = "≈same"
            deltas.append(MetricDelta(scenario, metric, b, c, raw, higher_better,
                                      verdict, regressed_beyond_gate=(verdict == "regressed")))
    return VersionComparison(candidate, baseline, n_cand, n_base, deltas, warnings=warnings)
