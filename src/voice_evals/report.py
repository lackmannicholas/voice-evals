"""Reporting: Parquet + JSON + self-contained offline HTML (design spec §14).

Artifacts written to ``outputs/runs/<run_id>/``:
  * results.parquet — tidy long-format table for ad-hoc analysis
  * run.json        — full EvalRun (config snapshot + all results + clip metadata)
  * report.html     — single self-contained file (inline CSS/JS, no CDN):
        a Key-findings callout (critical issues first, including low judge
        dimensions even when ungated), color-coded audio-judge scorecards,
        per-metric stats with inline histograms, gate + regression diffs, a
        worst-offenders view, and a sortable per-clip table with audio players.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Optional

from .aggregate import AggregationResult
from .config import Config
from .logging_util import get_logger
from .models import EvalRun

log = get_logger(__name__)

try:
    from .scorers.judge.base_judge import DIMENSIONS as _JUDGE_DIMENSIONS
except Exception:  # pragma: no cover - keep report importable if judge deps shift
    _JUDGE_DIMENSIONS = [
        "naturalness", "prosody", "fluency", "pace", "responsiveness",
        "emotional_alignment", "intelligibility", "conversational_flow",
        "frustration_handling",
    ]


def write_outputs(
    run: EvalRun, agg: AggregationResult, config: Config, out_dir: Optional[Path] = None
) -> Path:
    out_dir = Path(out_dir) if out_dir else (config.resolve_path(config.outputs_dir) / "runs" / run.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run.json").write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False))

    df = agg.long_df
    try:
        df.to_parquet(out_dir / "results.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        log.warning("parquet write failed (%s); writing results.csv instead", e)
        df.to_csv(out_dir / "results.csv", index=False)

    (out_dir / "report.html").write_text(render_html(run, agg, config))
    log.info("wrote report to %s", out_dir / "report.html")
    return out_dir


# --------------------------------------------------------------------------- #
def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _score_class(score: Any) -> str:
    """Severity class for a 1-5 judge score (1-2 bad, 3 fair, 4-5 good)."""
    try:
        s = int(round(float(score)))
    except (TypeError, ValueError):
        return ""
    if s <= 2:
        return "sev-bad"
    if s == 3:
        return "sev-warn"
    return "sev-good"


def _svg_histogram(values: list[float], width: int = 150, height: int = 32, bins: int = 12) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return '<span class="muted">—</span>'
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1e-9
    counts = [0] * bins
    for v in vals:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[idx] += 1
    cmax = max(counts) or 1
    bw = width / bins
    bars = []
    for i, c in enumerate(counts):
        bh = (c / cmax) * (height - 2)
        bars.append(
            f'<rect x="{i * bw:.1f}" y="{height - bh:.1f}" width="{bw - 1:.1f}" '
            f'height="{bh:.1f}" class="hbar"><title>{c}</title></rect>'
        )
    return f'<svg width="{width}" height="{height}" class="hist" viewBox="0 0 {width} {height}">{"".join(bars)}</svg>'


def render_html(run: EvalRun, agg: AggregationResult, config: Config) -> str:
    df = agg.long_df
    judge_on = bool(config.scorers.judge)
    n_clips, n_results = len(run.clip_ids), len(run.results)
    n_errors = sum(1 for r in run.results if r.error)
    fails, warns = agg.failures, agg.warnings

    # metric -> worst gate status, and metric -> layer (for coloring/grouping)
    metric_status: dict[str, str] = {}
    for o in warns:
        metric_status[o.metric] = "warn"
    for o in fails:
        metric_status[o.metric] = "bad"
    metric_layer: dict[str, str] = {}
    if not df.empty and "metric" in df.columns:
        for row in df[df["metric"].notna()].itertuples():
            metric_layer.setdefault(row.metric, getattr(row, "layer", ""))

    # judge raw per clip
    judge_by_clip: dict[str, dict] = {}
    for r in run.results:
        if r.layer == "judge" and not r.error and r.raw:
            judge_by_clip[r.clip_id] = r.raw

    # ---- Key findings (critical first; surfaces low judge dims even when ungated) ----
    findings: list[tuple[int, str, str, str, str]] = []  # (sortkey, badge, clip, kind, detail)
    for o in fails:
        findings.append((-1, _esc(o.metric), o.clip_id, "gate fail", _esc(o.message)))
    for cid, jb in judge_by_clip.items():
        ov = jb.get("overall")
        if isinstance(ov, (int, float)) and ov <= 2:
            findings.append((int(ov), f"overall {int(ov)}/5", cid, "judge", _esc(jb.get("summary", ""))))
        for dim in _JUDGE_DIMENSIONS:
            d = jb.get(dim)
            if isinstance(d, dict) and isinstance(d.get("score"), (int, float)) and d["score"] <= 2:
                findings.append((int(d["score"]), f"{dim} {int(d['score'])}/5", cid,
                                 "judge", _esc(d.get("reasoning", ""))))
    findings.sort(key=lambda f: f[0])
    if findings:
        items = "".join(
            f"<li><span class='tag sev-bad'>{badge}</span>"
            f"<span class='mono muted'>{_esc(cid[:12])}</span>"
            f"<span class='fkind'>{_esc(kind)}</span><div class='fwhy'>{detail}</div></li>"
            for _s, badge, cid, kind, detail in findings[:15]
        )
        headline = (
            "<div class='callout bad'><div class='callout-h'>⚠ Key findings — "
            f"{len(findings)} issue(s) need attention</div><ul class='findings'>{items}</ul></div>"
        )
    else:
        headline = "<div class='callout good'><div class='callout-h'>✓ No critical findings</div></div>"

    # ---- Audio-judge scorecards (color-coded, always visible) ----
    if judge_by_clip:
        cards = ""
        for cid, jb in judge_by_clip.items():
            clip = run.clip_by_id(cid)
            tiles = ""
            for dim in _JUDGE_DIMENSIONS:
                d = jb.get(dim) or {}
                tiles += (
                    f"<div class='tile {_score_class(d.get('score'))}' "
                    f"title='{_esc(d.get('reasoning', ''))}'>"
                    f"<div class='tnum'>{_esc(d.get('score'))}</div>"
                    f"<div class='tlbl'>{_esc(dim)}</div></div>"
                )
            ov = jb.get("overall")
            overall_tile = (
                f"<div class='tile big {_score_class(ov)}'><div class='tnum'>{_esc(ov)}</div>"
                "<div class='tlbl'>overall</div></div>"
            )
            ts = jb.get("notable_timestamps") or []
            ts_html = "".join(f"<li>{_esc(t)}</li>" for t in ts)
            audio = ""
            if clip is not None:
                p = clip.source_path if Path(clip.source_path).exists() else clip.mono16k_path
                audio = f"<audio controls preload='none' src='file://{_esc(str(Path(p).resolve()))}'></audio>"
            reasoning = "".join(
                f"<li><b>{_esc(dim)}</b> {_esc((jb.get(dim) or {}).get('score'))}/5 — "
                f"{_esc((jb.get(dim) or {}).get('reasoning'))}</li>"
                for dim in _JUDGE_DIMENSIONS if isinstance(jb.get(dim), dict)
            )
            cards += (
                "<div class='jcard'>"
                f"<div class='jcard-h'><span class='mono'>{_esc(cid[:12])}</span>{overall_tile}</div>"
                f"<div class='tiles'>{tiles}</div>"
                f"<p class='jsum'>{_esc(jb.get('summary', ''))}</p>"
                + (f"<div class='jts'><b>Notable moments</b><ul>{ts_html}</ul></div>" if ts else "")
                + (f"<div class='jaudio'>{audio}</div>" if audio else "")
                + f"<details class='jreason'><summary>per-dimension reasoning</summary>"
                f"<ul>{reasoning}</ul></details></div>"
            )
        judge_section = (
            "<h2>Audio judge <span class='muted'>(hover a tile for the reasoning)</span></h2>"
            "<div class='legend'>"
            "<span class='swatch sev-bad'></span>1–2 poor"
            "<span class='swatch sev-warn'></span>3 fair"
            "<span class='swatch sev-good'></span>4–5 good</div>"
            f"{cards}"
        )
    else:
        judge_section = (
            "<h2>Audio judge</h2><p class='muted'>Judge layer not run (local-only profile). "
            "Enable with <code>scorers.judge: [audio_judge]</code> + an API key.</p>"
        )

    # ---- per-metric summary (colored mean) ----
    metric_rows = []
    if not df.empty and "metric" in df.columns:
        valued = df[df["metric"].notna() & df["value"].notna()]
        summary = agg.summary_df
        for _, row in (summary.iterrows() if not summary.empty else []):
            metric = row["metric"]
            vals = valued[valued["metric"] == metric]["value"].tolist()
            pr = row.get("pass_rate")
            status = metric_status.get(metric)
            if not status and str(metric).startswith("judge_"):
                status = {"sev-bad": "bad", "sev-warn": "warn", "sev-good": "good"}.get(
                    _score_class(row["mean"]), ""
                )
            mean_cls = {"bad": "bad", "warn": "warn"}.get(status, "")
            layer = metric_layer.get(metric, "")
            metric_rows.append(
                "<tr>"
                f"<td class='mono'>{_esc(metric)}</td>"
                f"<td><span class='tag layer-{_esc(layer)}'>{_esc(layer)}</span></td>"
                f"<td>{int(row['count'])}</td>"
                f"<td class='{mean_cls}'><b>{_fmt(row['mean'])}</b></td>"
                f"<td>{_fmt(row['median'])}</td>"
                f"<td>{_fmt(row['p95'])}</td>"
                f"<td>{_fmt(row['min'])}</td>"
                f"<td>{_fmt(row['max'])}</td>"
                f"<td>{'—' if pr is None or (isinstance(pr, float) and math.isnan(pr)) else f'{pr * 100:.0f}%'}</td>"
                f"<td>{_svg_histogram(vals)}</td>"
                "</tr>"
            )

    def _outcome_row(o) -> str:
        cls = {"fail": "bad", "warn": "warn", "pass": "good"}[o.status]
        return (
            f"<tr class='{cls}'><td>{_esc(o.kind)}</td><td class='mono'>{_esc(o.clip_id)}</td>"
            f"<td class='mono'>{_esc(o.metric)}</td><td>{_fmt(o.value)}</td>"
            f"<td>{_esc(o.status.upper())}</td><td>{_esc(o.message)}</td></tr>"
        )

    gate_rows = "".join(_outcome_row(o) for o in (fails + warns)) or (
        "<tr><td colspan='6' class='muted'>No gate violations.</td></tr>"
    )
    reg_rows = "".join(_outcome_row(o) for o in agg.regressions) or (
        "<tr><td colspan='6' class='muted'>No baseline comparison (no baseline run found "
        "or regression gates disabled).</td></tr>"
    )

    # ---- worst offenders (gate fails/warns + low judge dims) ----
    score_by_clip: dict[str, float] = {}
    for o in fails:
        score_by_clip[o.clip_id] = score_by_clip.get(o.clip_id, 0) + 2.0
    for o in warns:
        score_by_clip[o.clip_id] = score_by_clip.get(o.clip_id, 0) + 0.5
    for cid, jb in judge_by_clip.items():
        for dim in _JUDGE_DIMENSIONS:
            d = jb.get(dim)
            if isinstance(d, dict) and isinstance(d.get("score"), (int, float)) and d["score"] <= 3:
                score_by_clip[cid] = score_by_clip.get(cid, 0) + (3 - d["score"])
    worst = sorted(score_by_clip.items(), key=lambda kv: -kv[1])[:15]
    worst_rows = "".join(
        f"<tr><td class='mono'>{_esc(cid)}</td><td><b>{score:.1f}</b></td></tr>" for cid, score in worst
    ) or "<tr><td colspan='2' class='muted'>No flagged clips.</td></tr>"

    # ---- per-clip table ----
    clip_rows = []
    for cid in run.clip_ids:
        clip = run.clip_by_id(cid)
        meta = clip.metadata if clip else None
        results = run.results_for_clip(cid)
        metrics: dict[str, float] = {}
        errs = []
        for r in results:
            metrics.update(r.metrics)
            if r.error:
                errs.append(f"{r.scorer}: {r.error}")
        clip_fail = [o for o in fails if o.clip_id == cid]
        clip_warn = [o for o in warns if o.clip_id == cid]
        status_cls = "bad" if clip_fail else ("warn" if clip_warn else "good")
        audio_src = ""
        if clip is not None:
            p = clip.source_path if Path(clip.source_path).exists() else clip.mono16k_path
            audio_src = f"<audio controls preload='none' src='file://{_esc(str(Path(p).resolve()))}'></audio>"
        key_metrics_html = "".join(
            f"<span class='chip'>{_esc(k)}={_fmt(v, 2)}</span>" for k, v in sorted(metrics.items())
        ) or "<span class='muted'>no metrics</span>"
        jb = judge_by_clip.get(cid)
        judge_overall = f"{int(jb['overall'])}/5" if jb and isinstance(jb.get("overall"), (int, float)) else "—"
        err_html = f"<div class='err'>{'<br>'.join(_esc(e) for e in errs)}</div>" if errs else ""
        dur = f"{clip.duration_s:.1f}s" if clip else "—"
        clip_rows.append(
            f"<tr class='{status_cls}' data-score='{len(clip_fail) * 10 + len(clip_warn)}'>"
            f"<td><details><summary class='mono'>{_esc(cid[:12])}</summary>"
            f"<div class='detail'>{audio_src}{err_html}"
            f"<div class='muted mono'>{_esc(clip.source_path.name if clip else '')}</div></div></details></td>"
            f"<td>{_esc(getattr(meta, 'agent_version', None))}</td><td>{dur}</td>"
            f"<td>{judge_overall}</td>"
            f"<td>{key_metrics_html}</td>"
            f"<td>{len(clip_fail)} / {len(clip_warn)}</td></tr>"
        )

    metric_table = "".join(metric_rows) or "<tr><td colspan='10' class='muted'>No metrics.</td></tr>"
    clip_table = "".join(clip_rows) or "<tr><td colspan='6' class='muted'>No clips.</td></tr>"
    overall = "PASS" if agg.passed() else "FAIL"
    overall_cls = "good" if agg.passed() else "bad"

    return _TEMPLATE.format(
        run_id=_esc(run.run_id),
        corpus=_esc(run.corpus_dir),
        started=_esc(run.started_at),
        finished=_esc(run.finished_at),
        judge=("enabled (" + _esc(config.judge.model) + ")") if judge_on else "disabled (local-only)",
        scorers=_esc(", ".join(config.scorers.all_selected())),
        n_clips=n_clips,
        n_results=n_results,
        n_errors=n_errors,
        n_fail=len(fails),
        n_warn=len(warns),
        overall=overall,
        overall_cls=overall_cls,
        headline=headline,
        judge_section=judge_section,
        metric_table=metric_table,
        gate_rows=gate_rows,
        reg_rows=reg_rows,
        worst_rows=worst_rows,
        clip_table=clip_table,
    )


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>voice-evals report — {run_id}</title>
<style>
:root {{ --fg:#1b1f24; --muted:#6b7280; --good:#137333; --warn:#a06400; --bad:#b3261e;
        --line:#e5e7eb; --bg:#fff; --chip:#f1f3f5;
        --bad-bg:#fdecea; --warn-bg:#fff7e6; --good-bg:#e6f4ea; }}
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:var(--fg);
        margin:0 auto; max-width:1100px; padding:24px; background:var(--bg); line-height:1.45; }}
h1 {{ font-size:21px; margin:0 0 4px; }}
h2 {{ font-size:15px; margin:28px 0 10px; padding-bottom:4px; border-bottom:2px solid var(--line); }}
.muted {{ color:var(--muted); }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
code {{ background:var(--chip); padding:1px 5px; border-radius:4px; font-size:12px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:14px 0; }}
.card {{ border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:104px; }}
.card .n {{ font-size:24px; font-weight:650; }}
.card .l {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin-bottom:8px; }}
th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ cursor:pointer; user-select:none; background:#fafafa; position:sticky; top:0; font-size:12px; }}
tr.bad td {{ background:var(--bad-bg); }} tr.warn td {{ background:var(--warn-bg); }}
.good {{ color:var(--good); }} .bad {{ color:var(--bad); }} .warn {{ color:var(--warn); }}
.chip {{ display:inline-block; background:var(--chip); border-radius:6px; padding:1px 6px; margin:1px;
         font-size:11px; font-family:ui-monospace,monospace; }}
.tag {{ display:inline-block; border-radius:6px; padding:1px 7px; font-size:11px; font-weight:600; margin-right:6px; }}
.layer-acoustic {{ background:#eef2ff; color:#3b4ea0; }}
.layer-dynamics {{ background:#ecfdf5; color:#0f7a52; }}
.layer-judge {{ background:#fef3f2; color:#9a2c22; }}
.layer-semantic {{ background:#f5f3ff; color:#6b3fa0; }}
.hist .hbar {{ fill:#5b8def; }}
.pill {{ display:inline-block; padding:2px 12px; border-radius:99px; font-weight:650; font-size:12px; }}
.pill.good {{ background:var(--good-bg); color:var(--good); }} .pill.bad {{ background:var(--bad-bg); color:var(--bad); }}
/* severity */
.sev-bad {{ background:var(--bad-bg); color:var(--bad); }}
.sev-warn {{ background:var(--warn-bg); color:var(--warn); }}
.sev-good {{ background:var(--good-bg); color:var(--good); }}
/* callout */
.callout {{ border-radius:10px; padding:14px 18px; margin:14px 0; border:1px solid var(--line); }}
.callout.bad {{ background:var(--bad-bg); border-color:#f3c2bd; }}
.callout.good {{ background:var(--good-bg); border-color:#bfe3cb; }}
.callout-h {{ font-weight:650; margin-bottom:6px; }}
.findings {{ list-style:none; margin:0; padding:0; }}
.findings li {{ padding:6px 0; border-top:1px solid rgba(0,0,0,.06); }}
.findings li:first-child {{ border-top:none; }}
.fkind {{ font-size:11px; color:var(--muted); text-transform:uppercase; margin-left:6px; }}
.fwhy {{ margin-top:3px; font-size:13px; }}
/* judge scorecards */
.legend {{ font-size:12px; color:var(--muted); margin:6px 0 12px; }}
.swatch {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin:0 5px 0 14px; vertical-align:-1px; }}
.jcard {{ border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:12px 0; }}
.jcard-h {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(108px,1fr)); gap:8px; }}
.tile {{ border-radius:9px; padding:9px 6px; text-align:center; border:1px solid rgba(0,0,0,.05); }}
.tile.big {{ min-width:84px; padding:6px 14px; }}
.tnum {{ font-size:22px; font-weight:700; line-height:1.1; }}
.tile.big .tnum {{ font-size:26px; }}
.tlbl {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.02em; margin-top:2px; opacity:.85; }}
.jsum {{ margin:12px 0 8px; font-size:13.5px; }}
.jts ul {{ margin:4px 0 8px; padding-left:18px; font-size:13px; }}
.jaudio {{ margin:6px 0; }}
.jreason {{ font-size:12.5px; margin-top:6px; }} .jreason ul {{ padding-left:18px; }}
details summary {{ cursor:pointer; }} .detail {{ padding:8px 0; }} .err {{ color:var(--bad); font-size:12px; }}
audio {{ height:32px; vertical-align:middle; max-width:420px; }}
</style></head><body>
<h1>voice-evals report <span class="pill {overall_cls}">{overall}</span></h1>
<div class="muted mono">run {run_id} · corpus {corpus}</div>
<div class="muted mono">started {started} · finished {finished}</div>
<div class="muted">judge: {judge} · scorers: {scorers}</div>

<div class="cards">
  <div class="card"><div class="n">{n_clips}</div><div class="l">clips</div></div>
  <div class="card"><div class="n">{n_results}</div><div class="l">scorer results</div></div>
  <div class="card"><div class="n bad">{n_fail}</div><div class="l">gate fails</div></div>
  <div class="card"><div class="n warn">{n_warn}</div><div class="l">gate warns</div></div>
  <div class="card"><div class="n">{n_errors}</div><div class="l">scorer errors</div></div>
</div>

{headline}

{judge_section}

<h2>Per-metric summary</h2>
<table id="metrics"><thead><tr>
  <th>metric</th><th>layer</th><th>count</th><th>mean</th><th>median</th><th>p95</th>
  <th>min</th><th>max</th><th>pass rate</th><th>distribution</th></tr></thead>
<tbody>{metric_table}</tbody></table>

<h2>Gate violations</h2>
<table><thead><tr><th>kind</th><th>clip</th><th>metric</th><th>value</th><th>status</th><th>detail</th></tr></thead>
<tbody>{gate_rows}</tbody></table>

<h2>Regression vs baseline</h2>
<table><thead><tr><th>kind</th><th>group</th><th>metric</th><th>value</th><th>status</th><th>detail</th></tr></thead>
<tbody>{reg_rows}</tbody></table>

<h2>Worst offenders <span class="muted">(gate violations + low judge dimensions)</span></h2>
<table><thead><tr><th>clip</th><th>severity score</th></tr></thead>
<tbody>{worst_rows}</tbody></table>

<h2>Per-clip detail <span class="muted">(click a clip id to expand · click headers to sort)</span></h2>
<table id="clips"><thead><tr>
  <th>clip</th><th>agent_version</th><th>duration</th><th>judge overall</th>
  <th>metrics</th><th>fail / warn</th></tr></thead>
<tbody>{clip_table}</tbody></table>

<script>
document.querySelectorAll('table th').forEach(function(th){{
  th.addEventListener('click', function(){{
    var table = th.closest('table'); var idx = Array.from(th.parentNode.children).indexOf(th);
    var tbody = table.querySelector('tbody'); var rows = Array.from(tbody.querySelectorAll('tr'));
    var asc = !(th.dataset.asc === 'true'); th.dataset.asc = asc;
    rows.sort(function(a,b){{
      var x=(a.children[idx]||{{}}).innerText||'', y=(b.children[idx]||{{}}).innerText||'';
      var nx=parseFloat(x), ny=parseFloat(y);
      if(!isNaN(nx)&&!isNaN(ny)) return asc? nx-ny : ny-nx;
      return asc? x.localeCompare(y) : y.localeCompare(x);
    }});
    rows.forEach(function(r){{ tbody.appendChild(r); }});
  }});
}});
</script>
</body></html>
"""
