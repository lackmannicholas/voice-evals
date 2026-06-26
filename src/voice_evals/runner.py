"""Runner: orchestration, concurrency, caching, error isolation (design spec §12.1).

The runner is the only place that owns concurrency. Scorers stay pure. Rules:
  * one task per (clip, scorer) where ``scorer.applicable(clip)``,
  * cache hit  -> reuse (0 model calls),
  * cache miss -> run, and cache only successful results (errors retry next run),
  * model-backed (``single_flight``) acoustic/dynamics scorers run SERIALLY on the
    MAIN thread: native model libs (libtorch, silero, onnx) can crash when driven
    from worker threads, and they are GIL-bound so little parallelism is lost,
  * the network-bound judge runs in a thread pool bounded by
    ``judge.max_concurrency`` (transport retry/backoff lives inside the backend),
    overlapping with the main-thread model work; a scorer marked
    ``single_flight=False`` may also run in the pool,
  * a scorer error on one clip is recorded; the run never aborts.
"""

from __future__ import annotations

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cache import ResultCache
from .config import Config
from .ingest import Ingestor
from .logging_util import get_logger
from .models import AudioClip, EvalRun, ScoreResult
from .scorers.base import Scorer, build_selected

log = get_logger(__name__)


def _git_sha() -> Optional[str]:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        sha = cp.stdout.strip()
        return sha or None
    except Exception:
        return None


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_sha()
    return f"{ts}-{sha}" if sha else ts


def _auto_workers(configured: int) -> int:
    if configured and configured > 0:
        return configured
    return max(1, (os.cpu_count() or 2) - 1)


class Runner:
    def __init__(self, config: Config, cache: Optional[ResultCache] = None, no_cache: bool = False):
        self.config = config
        self.cache = cache or ResultCache(config.cache_dir, enabled=not no_cache)
        self.no_cache = no_cache
        # Network concurrency for the judge layer. Model-backed scorers don't use
        # threads at all (run on the main thread) — see run().
        self._judge_sema = threading.Semaphore(max(1, config.judge.max_concurrency))

    # ------------------------------------------------------------------ #
    def run(
        self, corpus_dir: Optional[Path] = None, clips: Optional[list[AudioClip]] = None
    ) -> EvalRun:
        corpus_dir = Path(corpus_dir or self.config.corpus_dir)
        started_at = datetime.now(timezone.utc).isoformat()

        if clips is None:
            log.info("ingesting corpus: %s", corpus_dir)
            clips = Ingestor(self.config, self.cache).ingest_dir(corpus_dir)
        log.info("ingested %d clip(s)", len(clips))

        scorers = build_selected(self.config)
        log.info("running %d scorer(s): %s", len(scorers), [s.name for s in scorers])

        # build the (clip, scorer) work list, honoring applicability
        tasks: list[tuple[AudioClip, Scorer]] = []
        for clip in clips:
            for scorer in scorers:
                try:
                    applicable = scorer.applicable(clip)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s.applicable crashed on %s: %s", scorer.name, clip.clip_id, e)
                    applicable = False
                if applicable:
                    tasks.append((clip, scorer))

        # Native model libraries (libtorch SQUIM/UTMOS, silero VAD, onnx DNSMOS)
        # crash when driven from worker threads on some platforms, so model-backed
        # (single_flight) scorers run serially on the MAIN thread — they are
        # GIL-bound anyway, so little parallelism is lost. The thread pool is used
        # only for the network-bound judge (and any scorer marked parallel-safe),
        # which overlaps with the main-thread model work.
        results: list[ScoreResult] = []
        main_tasks = [(c, s) for (c, s) in tasks
                      if s.layer != "judge" and getattr(s, "single_flight", True)]
        pool_tasks = [(c, s) for (c, s) in tasks if (c, s) not in main_tasks]

        progress = self._progress(total=len(tasks)) if self.config.runner.progress else None
        try:
            if pool_tasks:
                max_workers = max(1, self.config.judge.max_concurrency)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(self._run_one, c, s): (c, s) for c, s in pool_tasks}
                    for clip, scorer in main_tasks:  # run models here while judge runs in pool
                        results.append(self._run_one(clip, scorer))
                        if progress is not None:
                            progress()
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        if progress is not None:
                            progress()
            else:
                for clip, scorer in main_tasks:
                    results.append(self._run_one(clip, scorer))
                    if progress is not None:
                        progress()
        finally:
            if progress is not None:
                progress(close=True)

        finished_at = datetime.now(timezone.utc).isoformat()
        self._log_summary(results)
        return EvalRun(
            run_id=_make_run_id(),
            corpus_dir=corpus_dir,
            config_snapshot=self.config.snapshot(),
            results=results,
            started_at=started_at,
            finished_at=finished_at,
            clips=list(clips),
        )

    # ------------------------------------------------------------------ #
    def _run_one(self, clip: AudioClip, scorer: Scorer) -> ScoreResult:
        cfg_hash = scorer.config_hash()
        cached = self.cache.get(clip.clip_id, scorer.name, cfg_hash)
        if cached is not None:
            return cached

        if scorer.layer == "judge":
            with self._judge_sema:  # bound network concurrency
                result = scorer.score(clip)
        else:
            # single_flight model scorers are invoked only from the main thread
            # (serial); parallel-safe scorers run free in the pool. Either way no
            # lock is needed here.
            result = scorer.score(clip)

        # cache only successful results; let errors (often missing deps) retry
        if result.ok:
            self.cache.put(result)
        return result

    # ------------------------------------------------------------------ #
    def _progress(self, total: int):
        try:
            from tqdm import tqdm

            bar = tqdm(total=total, desc="scoring", unit="task")

            def _tick(close: bool = False):
                if close:
                    bar.close()
                else:
                    bar.update(1)

            return _tick
        except Exception:
            return None

    def _log_summary(self, results: list[ScoreResult]) -> None:
        by_scorer: dict[str, list[ScoreResult]] = {}
        for r in results:
            by_scorer.setdefault(r.scorer, []).append(r)
        for name, rs in sorted(by_scorer.items()):
            errs = sum(1 for r in rs if r.error)
            cached = sum(1 for r in rs if r.cached)
            log.info("%-14s: %3d results, %d cached, %d error(s)", name, len(rs), cached, errs)


def run_eval(
    config: Config,
    corpus_dir: Optional[Path] = None,
    no_cache: bool = False,
) -> EvalRun:
    """Convenience entry used by the CLI and pytest fixtures."""
    runner = Runner(config, no_cache=no_cache)
    return runner.run(corpus_dir=corpus_dir)
