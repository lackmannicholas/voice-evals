"""Runner fan-out, caching, error isolation (design spec §12.1).

Uses dummy in-process scorers (monkeypatched registry) so it needs no model deps.
"""

from __future__ import annotations

import pytest

from voice_evals import runner as runner_mod
from voice_evals.models import AudioClip
from voice_evals.runner import Runner
from voice_evals.scorers.base import BaseScorer

pytestmark = pytest.mark.local

CALLS = {"count": 0, "boom": 0}


class CountScorer(BaseScorer):
    name = "count"
    layer = "acoustic"

    def _compute(self, clip: AudioClip):
        CALLS["count"] += 1
        return {"x": 1.0}, {}


class BoomScorer(BaseScorer):
    name = "boom"
    layer = "acoustic"

    def _compute(self, clip: AudioClip):
        CALLS["boom"] += 1
        raise RuntimeError("kaboom")


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS["count"] = 0
    CALLS["boom"] = 0


def test_fan_out_error_isolation_and_caching(config, mini_corpus, monkeypatch):
    config.corpus_dir = mini_corpus
    monkeypatch.setattr(runner_mod, "build_selected",
                        lambda cfg: [CountScorer(cfg), BoomScorer(cfg)])

    r = Runner(config)
    run1 = r.run(corpus_dir=mini_corpus)
    n_clips = len(run1.clip_ids)
    assert n_clips == 2
    assert len(run1.results) == 2 * n_clips  # 2 scorers x clips

    ok = [x for x in run1.results if x.scorer == "count"]
    boom = [x for x in run1.results if x.scorer == "boom"]
    assert all(x.error is None for x in ok)         # count succeeded
    assert all(x.error is not None for x in boom)   # boom errored, run not aborted
    assert CALLS["count"] == n_clips and CALLS["boom"] == n_clips

    # second run: count cached (no recompute); boom retried (errors not cached)
    run2 = r.run(corpus_dir=mini_corpus)
    assert CALLS["count"] == n_clips, "cached results must not recompute"
    assert CALLS["boom"] == 2 * n_clips, "errored results must retry"
    assert all(x.cached for x in run2.results if x.scorer == "count")
    assert all(not x.cached for x in run2.results if x.scorer == "boom")


def test_no_cache_bypasses(config, mini_corpus, monkeypatch):
    config.corpus_dir = mini_corpus
    monkeypatch.setattr(runner_mod, "build_selected", lambda cfg: [CountScorer(cfg)])
    r = Runner(config, no_cache=True)
    r.run(corpus_dir=mini_corpus)
    r.run(corpus_dir=mini_corpus)
    n_clips = 2
    assert CALLS["count"] == 2 * n_clips  # recomputed both runs (cache bypassed)
