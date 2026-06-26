"""Judge↔human calibration harness (design spec §17, judge-prompt spec §6).

The only place humans label, and it's optional + one-time. Reads a small
human-labeled golden set, runs the judge on those clips, and reports per-dimension
agreement: quadratic-weighted Cohen's κ (treating 1–5 as ordinal) plus Spearman
and Pearson correlation, with a CI gate recommendation (enforce a judge dimension
only where κ clears the configured bar).

golden_set.csv columns: ``clip_id, dimension, human_score``. ``clip_id`` may be a
content-hash clip_id OR a source filename/stem (resolved against the corpus).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cache import ResultCache
from .config import Config
from .ingest import Ingestor
from .logging_util import get_logger
from .models import AudioClip
from .scorers.base import build_scorer
from .scorers.judge.base_judge import DIMENSIONS

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# statistics (numpy-free; small N)
# --------------------------------------------------------------------------- #
def quadratic_weighted_kappa(human: list[int], judge: list[int], k: int = 5) -> float:
    """Cohen's quadratic-weighted κ over ordinal ratings 1..k."""
    n = len(human)
    if n == 0:
        return float("nan")
    O = [[0.0] * k for _ in range(k)]
    for h, j in zip(human, judge):
        O[h - 1][j - 1] += 1.0
    hist_h = [sum(O[i]) for i in range(k)]
    hist_j = [sum(O[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * O[i][j] for i in range(k) for j in range(k))
    E = [[hist_h[i] * hist_j[j] / n for j in range(k)] for i in range(k)]
    den = sum(w[i][j] * E[i][j] for i in range(k) for j in range(k))
    if den == 0:
        # no expected disagreement (one side constant); perfect iff observed agrees
        return 1.0 if num == 0 else 0.0
    return 1.0 - num / den


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


# --------------------------------------------------------------------------- #
@dataclass
class DimensionAgreement:
    dimension: str
    n: int
    kappa: float
    spearman: float
    pearson: float
    mean_abs_error: float
    recommend_gate: bool


@dataclass
class CalibrationReport:
    per_dimension: list[DimensionAgreement]
    macro_kappa: float
    kappa_bar: float
    model: str
    rubric_version: str
    n_clips: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "rubric_version": self.rubric_version,
            "n_clips": self.n_clips,
            "kappa_bar": self.kappa_bar,
            "macro_kappa": self.macro_kappa,
            "per_dimension": [vars(d) for d in self.per_dimension],
            "errors": self.errors,
        }

    def render_text(self) -> str:
        lines = [
            f"Judge calibration — model={self.model}  rubric={self.rubric_version}  "
            f"clips={self.n_clips}  kappa_bar={self.kappa_bar}",
            f"{'dimension':<22}{'n':>4}{'kappa':>9}{'spearman':>10}{'pearson':>9}{'MAE':>7}  gate",
            "-" * 72,
        ]
        for d in self.per_dimension:
            lines.append(
                f"{d.dimension:<22}{d.n:>4}{d.kappa:>9.3f}{d.spearman:>10.3f}"
                f"{d.pearson:>9.3f}{d.mean_abs_error:>7.2f}  "
                f"{'ENFORCE' if d.recommend_gate else 'advisory'}"
            )
        lines.append("-" * 72)
        lines.append(f"macro kappa = {self.macro_kappa:.3f}")
        gated = [d.dimension for d in self.per_dimension if d.recommend_gate]
        lines.append(
            "Recommended CI gates (kappa >= bar): "
            + (", ".join(f"judge_{d}" for d in gated) if gated else "none — treat all as advisory")
        )
        if self.errors:
            lines.append(f"({len(self.errors)} judge error(s); see calibration_report.json)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _load_golden(path: Path) -> dict[str, dict[str, int]]:
    """Return {clip_key: {dimension: human_score}}; clip_key is whatever the CSV holds."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("clip_id") or "").strip()
            dim = (row.get("dimension") or "").strip().lower()
            raw = (row.get("human_score") or "").strip()
            if not cid or not dim or not raw or cid.startswith("#"):
                continue
            try:
                score = int(round(float(raw)))
            except ValueError:
                continue
            if 1 <= score <= 5 and dim in DIMENSIONS:
                out[cid][dim] = score
    return out


def _resolve_clip(key: str, clips: list[AudioClip]) -> Optional[AudioClip]:
    for c in clips:
        if c.clip_id == key:
            return c
    for c in clips:
        if c.source_path.name == key or c.source_path.stem == key:
            return c
    # prefix match on clip_id (humans often paste a short hash)
    for c in clips:
        if c.clip_id.startswith(key):
            return c
    return None


def run_calibration(
    config: Config, golden_path: Path | str, out_dir: Optional[Path] = None
) -> CalibrationReport:
    golden_path = Path(golden_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"golden set not found: {golden_path}")
    golden = _load_golden(golden_path)

    cache = ResultCache(config.cache_dir)
    clips = Ingestor(config, cache).ingest_dir(config.corpus_dir)
    judge = build_scorer("audio_judge", config)

    # collect paired (human, judge) per dimension
    paired: dict[str, list[tuple[int, int]]] = defaultdict(list)
    errors: list[str] = []
    n_clips = 0
    for key, dims in golden.items():
        clip = _resolve_clip(key, clips)
        if clip is None:
            errors.append(f"golden clip {key!r} not found in corpus")
            continue
        cached = cache.get(clip.clip_id, judge.name, judge.config_hash())
        result = cached or judge.score(clip)
        if cached is None and result.error is None:
            cache.put(result)
        if result.error is not None:
            errors.append(f"{key}: judge error: {result.error}")
            continue
        n_clips += 1
        for dim, human in dims.items():
            jval = result.metrics.get(f"judge_{dim}")
            if jval is not None:
                paired[dim].append((human, int(round(jval))))

    per_dim: list[DimensionAgreement] = []
    for dim in DIMENSIONS:
        pairs = paired.get(dim, [])
        if not pairs:
            continue
        h = [p[0] for p in pairs]
        j = [p[1] for p in pairs]
        kappa = quadratic_weighted_kappa(h, j)
        mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
        per_dim.append(
            DimensionAgreement(
                dimension=dim,
                n=len(pairs),
                kappa=kappa,
                spearman=spearman([float(x) for x in h], [float(x) for x in j]),
                pearson=pearson([float(x) for x in h], [float(x) for x in j]),
                mean_abs_error=mae,
                recommend_gate=(not math.isnan(kappa)) and kappa >= config.calibration.kappa_bar,
            )
        )
    valid_kappas = [d.kappa for d in per_dim if not math.isnan(d.kappa)]
    macro = sum(valid_kappas) / len(valid_kappas) if valid_kappas else float("nan")

    report = CalibrationReport(
        per_dimension=per_dim,
        macro_kappa=macro,
        kappa_bar=config.calibration.kappa_bar,
        model=config.judge.model,
        rubric_version=getattr(judge, "rubric_version", "unknown"),
        n_clips=n_clips,
        errors=errors,
    )

    out = Path(out_dir) if out_dir else (config.resolve_path(config.outputs_dir) / "calibration")
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    (out / "calibration_report.txt").write_text(report.render_text())
    log.info("wrote calibration report to %s", out)
    return report
