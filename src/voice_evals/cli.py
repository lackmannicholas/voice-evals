"""``voice-evals`` command-line interface (design spec §16)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .aggregate import Aggregator, load_baseline, persist_run
from .cache import ResultCache
from .config import Config, default_config_path
from .logging_util import setup_logging
from .models import EvalRun
from .report import render_html, write_outputs
from .runner import Runner

app = typer.Typer(add_completion=False, help="Offline voice-call audio quality evaluation.")
cache_app = typer.Typer(help="Cache maintenance.")
app.add_typer(cache_app, name="cache")
console = Console()


def _load_config(config: Optional[Path]) -> Config:
    path = config or default_config_path()
    cfg = Config.load(path) if path else Config()
    return cfg


@app.command()
def run(
    corpus: Optional[Path] = typer.Option(None, help="Corpus directory of recordings."),
    config: Optional[Path] = typer.Option(None, help="Config YAML (defaults to config/default.yaml)."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Disable the judge layer (force local-only)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the result cache."),
    baseline: Optional[str] = typer.Option(None, help="Baseline run id/tag, or 'previous'."),
    out: Optional[Path] = typer.Option(None, help="Output dir (default outputs/runs/<run_id>)."),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if any gate fails."),
) -> None:
    """Ingest, score, aggregate, gate, and report over a corpus."""
    setup_logging()
    cfg = _load_config(config)
    if corpus is not None:
        cfg.corpus_dir = corpus
    if no_judge:
        cfg = cfg.disable_judge()
    if baseline is not None:
        cfg.gates.regression.baseline = baseline

    runner = Runner(cfg, no_cache=no_cache)
    eval_run = runner.run(corpus_dir=cfg.corpus_dir)

    base = load_baseline(cfg, current_run_id=eval_run.run_id)
    agg = Aggregator(cfg).evaluate(eval_run, baseline_run=base)
    # Always persist the canonical run.json under outputs/runs/<run_id>/ so this
    # run is discoverable as a future baseline, even when --out redirects the report.
    persist_run(eval_run, cfg)
    out_dir = write_outputs(eval_run, agg, cfg, out_dir=out)

    _print_summary(eval_run, agg, out_dir)
    if strict and not agg.passed():
        raise typer.Exit(code=1)


@app.command()
def ingest(
    corpus: Optional[Path] = typer.Option(None, help="Corpus directory."),
    config: Optional[Path] = typer.Option(None, help="Config YAML."),
) -> None:
    """Decode/normalize/hash/sidecar only — warm the decode cache."""
    setup_logging()
    from .ingest import Ingestor

    cfg = _load_config(config)
    if corpus is not None:
        cfg.corpus_dir = corpus
    cache = ResultCache(cfg.cache_dir)
    clips = Ingestor(cfg, cache).ingest_dir(cfg.corpus_dir)
    console.print(f"[green]Ingested {len(clips)} clip(s).[/green]")
    for c in clips:
        ch = "stereo" if c.n_source_channels >= 2 else "mono"
        console.print(f"  {c.clip_id[:12]}  {c.duration_s:6.1f}s  {ch}  {c.source_path.name}")


@app.command()
def report(
    run_dir: Path = typer.Option(..., "--run", help="A run dir containing run.json."),
    config: Optional[Path] = typer.Option(None, help="Config YAML (for gate thresholds)."),
) -> None:
    """Regenerate report.html from a persisted run.json."""
    setup_logging()
    run_json = run_dir / "run.json"
    if not run_json.exists():
        console.print(f"[red]run.json not found in {run_dir}[/red]")
        raise typer.Exit(code=2)
    eval_run = EvalRun.from_dict(json.loads(run_json.read_text()))
    # Faithfully reflect what the run actually used: rebuild config from its
    # snapshot unless the user explicitly passes a --config to re-gate.
    if config is not None:
        cfg = _load_config(config)
    else:
        try:
            cfg = Config.model_validate(eval_run.config_snapshot)
        except Exception:
            cfg = _load_config(None)
    base = load_baseline(cfg, current_run_id=eval_run.run_id)
    agg = Aggregator(cfg).evaluate(eval_run, baseline_run=base)
    (run_dir / "report.html").write_text(render_html(eval_run, agg, cfg))
    console.print(f"[green]Wrote {run_dir / 'report.html'}[/green]")


@app.command()
def calibrate(
    golden: Path = typer.Option("calibration/golden_set.csv", help="Golden-set CSV."),
    config: Optional[Path] = typer.Option(None, help="Config YAML."),
    corpus: Optional[Path] = typer.Option(None, help="Corpus directory."),
    out: Optional[Path] = typer.Option(None, help="Output dir for the calibration report."),
) -> None:
    """Measure judge<->human agreement on a human-labeled golden set."""
    setup_logging()
    from .calibrate import run_calibration

    cfg = _load_config(config)
    if corpus is not None:
        cfg.corpus_dir = corpus
    report = run_calibration(cfg, golden, out_dir=out)
    console.print(report.render_text())


@app.command()
def simulate(
    scenarios: Path = typer.Option(..., help="Scenario manifest (YAML)."),
    config: Optional[Path] = typer.Option(None, help="Config YAML."),
    agent_version: str = typer.Option("candidate", help="Version tag for the agent under test."),
    agent_instructions: Optional[Path] = typer.Option(None, help="Agent system-prompt file."),
    baseline_version: Optional[str] = typer.Option(None, help="Baseline version tag (enables A/B)."),
    baseline_instructions: Optional[Path] = typer.Option(None, help="Baseline agent system-prompt file."),
    runs: Optional[int] = typer.Option(None, "-k", "--runs", help="Runs per scenario (overrides config)."),
    out: Optional[Path] = typer.Option(None, help="Output dir."),
    mock: bool = typer.Option(False, "--mock", help="Use mock backends (no API) to try the harness."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the judge when scoring."),
    strict: bool = typer.Option(False, "--strict", help="Also fail (exit 1) on absolute-gate failures, not just A/B regressions."),
) -> None:
    """Drive scenarios through the agent (multi-turn), then score the recordings."""
    setup_logging()
    import hashlib
    import shutil

    from .simulate.gating import compare_versions
    from .simulate.orchestrator import run_suite
    from .simulate.scenario import load_scenarios

    cfg = _load_config(config)
    if no_judge:
        cfg = cfg.disable_judge()
    if runs is not None:
        cfg.simulate.k_runs = runs
    # each (scenario, version, run) must stay a distinct clip — don't let content
    # dedup collapse repeated/identical simulated runs or the two versions together.
    cfg.ingest.identity_from_call_id = True
    scns = load_scenarios(scenarios)
    out_dir = out or (cfg.resolve_path(cfg.outputs_dir) / "sim")
    corpus = Path(out_dir) / "corpus"
    # start from a clean corpus so stale recordings from a prior run never contaminate
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True, exist_ok=True)

    def _pv(text: str) -> str:
        return "p-" + hashlib.sha256(text.encode()).hexdigest()[:8]  # prompt identity

    def _instructions(p: Optional[Path], default_for: str) -> str:
        path = p or (Path(cfg.simulate.agent_instructions_path) if cfg.simulate.agent_instructions_path else None)
        if path and Path(path).exists():
            return Path(path).read_text()
        return "You are a helpful property leasing / resident-support phone agent. Keep replies brief."

    def make_user():
        if mock:
            from .simulate.base import MockUserSimulator
            import random
            return MockUserSimulator(seed=random.randint(1, 10_000))  # vary runs
        from .simulate.openai_backends import OpenAIUserSimulator
        return OpenAIUserSimulator(cfg)

    def make_agent_factory(version: str, instr: str):
        if mock:
            from .simulate.base import MockAgent
            import random
            return lambda: MockAgent(version=version, seed=random.randint(1, 10_000))
        from .simulate.openai_backends import OpenAIRealtimeAgent
        return lambda: OpenAIRealtimeAgent(cfg, version, instr)

    console.print(f"[bold]Simulating[/bold] {len(scns)} scenario(s) × {cfg.simulate.k_runs} run(s) "
                  f"→ agent '{agent_version}'" + (f" vs baseline '{baseline_version}'" if baseline_version else ""))
    cand_instr = _instructions(agent_instructions, "candidate")
    run_suite(scns, make_agent_factory(agent_version, cand_instr),
              make_user, cfg.simulate.k_runs, corpus, prompt_version=_pv(cand_instr))
    if baseline_version:
        base_instr = _instructions(baseline_instructions, "baseline")
        run_suite(scns, make_agent_factory(baseline_version, base_instr),
                  make_user, cfg.simulate.k_runs, corpus, prompt_version=_pv(base_instr))

    # score everything with the existing evaluator
    eval_run = Runner(cfg).run(corpus_dir=corpus)
    agg = Aggregator(cfg).evaluate(eval_run)
    out_run = write_outputs(eval_run, agg, cfg, out_dir=Path(out_dir) / "report")
    _print_summary(eval_run, agg, out_run)

    fail = False
    if baseline_version:
        cmp = compare_versions(eval_run, candidate=agent_version, baseline=baseline_version, config=cfg)
        console.print("\n" + cmp.render_text())
        fail = not cmp.passed()
    if strict and not agg.passed():
        console.print("[red]--strict: absolute gates failed[/red]")
        fail = True
    if fail:
        raise typer.Exit(code=1)


@app.command()
def call(
    scenarios: Path = typer.Option(..., help="Scenario manifest (YAML)."),
    config: Optional[Path] = typer.Option(None, help="Config YAML."),
    agent_version: str = typer.Option("candidate", help="Version tag for the agent you're calling."),
    prompt_version: Optional[str] = typer.Option(None, help="Optional prompt-version tag."),
    to: Optional[str] = typer.Option(None, help="Agent phone number to dial (overrides config)."),
    public_url: Optional[str] = typer.Option(None, help="wss URL Twilio streams to (your tunnel)."),
    out: Optional[Path] = typer.Option(None, help="Output dir."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the judge when scoring."),
    strict: bool = typer.Option(False, "--strict", help="Exit 1 if absolute gates fail."),
) -> None:
    """Place a REAL Twilio call to the agent's number and score the audio (incl. barge-in).

    Run your dev agent locally (Twilio number + ngrok), expose this command's media
    server via a tunnel, and set telephony.public_url to that wss URL.
    """
    setup_logging()
    import shutil

    from .simulate.scenario import load_scenarios
    from .simulate.telephony import OpenAICascadeCaller
    from .simulate.telephony_live import run_calls

    from .config import load_secrets

    cfg = _load_config(config)
    if no_judge:
        cfg = cfg.disable_judge()
    # resolve telephony connection params: CLI flag > config YAML > .env/environment
    sec = load_secrets()
    cfg.telephony.agent_number = to or cfg.telephony.agent_number or sec.twilio_agent_number
    cfg.telephony.public_url = public_url or cfg.telephony.public_url or sec.public_stream_url
    cfg.telephony.from_number = cfg.telephony.from_number or sec.twilio_from_number
    cfg.ingest.identity_from_call_id = True  # keep each call a distinct clip

    # fail clearly (not with a mid-call traceback) if anything required is missing
    missing = []
    if not (sec.twilio_account_sid and sec.twilio_auth_token):
        missing.append("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN")
    if not cfg.telephony.from_number:
        missing.append("TWILIO_FROM_NUMBER (your Twilio caller-ID number)")
    if not cfg.telephony.agent_number:
        missing.append("TWILIO_AGENT_NUMBER or --to (the agent's number to dial)")
    if not cfg.telephony.public_url:
        missing.append("PUBLIC_STREAM_URL or --public-url (wss tunnel to this server)")
    if missing:
        console.print("[red]Missing telephony config[/red] (set in .env — see .env.example):")
        for m in missing:
            console.print(f"  • {m}")
        raise typer.Exit(code=2)
    scns = load_scenarios(scenarios)
    out_dir = out or (cfg.resolve_path(cfg.outputs_dir) / "calls")
    corpus = Path(out_dir) / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Calling[/bold] {cfg.telephony.agent_number} for {len(scns)} scenario(s) "
                  f"as agent '{agent_version}' (real Twilio call)")
    run_calls(cfg, scns, lambda: OpenAICascadeCaller(cfg), agent_version, corpus, prompt_version)

    eval_run = Runner(cfg).run(corpus_dir=corpus)
    base = load_baseline(cfg, current_run_id=eval_run.run_id)
    agg = Aggregator(cfg).evaluate(eval_run, baseline_run=base)
    persist_run(eval_run, cfg)
    out_run = write_outputs(eval_run, agg, cfg, out_dir=Path(out_dir) / "report")
    _print_summary(eval_run, agg, out_run)
    if strict and not agg.passed():
        raise typer.Exit(code=1)


@app.command("list-scorers")
def list_scorers() -> None:
    """List registered scorers and their layers."""
    from .scorers.base import layer_of, registered_names

    table = Table("scorer", "layer")
    for name in registered_names():
        table.add_row(name, str(layer_of(name)))
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    scorer: Optional[str] = typer.Option(None, help="Only clear this scorer's cached results."),
    config: Optional[Path] = typer.Option(None, help="Config YAML."),
) -> None:
    """Delete cached results (optionally for a single scorer)."""
    cfg = _load_config(config)
    n = ResultCache(cfg.cache_dir).clear(scorer=scorer)
    console.print(f"[green]Cleared {n} cached result(s){f' for {scorer}' if scorer else ''}.[/green]")


def _print_summary(run: EvalRun, agg, out_dir: Path) -> None:
    table = Table(title=f"Run {run.run_id}")
    table.add_column("metric")
    table.add_column("count", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("pass rate", justify="right")
    if not agg.summary_df.empty:
        for _, r in agg.summary_df.iterrows():
            pr = r.get("pass_rate")
            pr_s = "—" if pr is None or (isinstance(pr, float) and pr != pr) else f"{pr*100:.0f}%"
            table.add_row(
                str(r["metric"]),
                str(int(r["count"])),
                f"{r['mean']:.3f}",
                f"{r['p95']:.3f}",
                pr_s,
            )
    console.print(table)
    n_err = sum(1 for x in run.results if x.error)
    color = "green" if agg.passed() else "red"
    console.print(
        f"[{color}]{'PASS' if agg.passed() else 'FAIL'}[/{color}] · "
        f"{len(agg.failures)} fail · {len(agg.warnings)} warn · {n_err} scorer error(s)"
    )
    console.print(f"Report: {out_dir / 'report.html'}")


if __name__ == "__main__":
    app()
