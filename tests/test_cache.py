"""Content-addressed cache (design spec §12.2)."""

from __future__ import annotations

import pytest

from voice_evals.cache import ResultCache, result_cache_key
from voice_evals.models import ScoreResult

pytestmark = pytest.mark.local


def _r(cid="c1", scorer="dnsmos", h="h1", metrics=None):
    return ScoreResult(cid, scorer, "acoustic", metrics or {"dnsmos_ovrl": 3.4}, scorer_config_hash=h)


def test_roundtrip_sets_cached_flag(tmp_path):
    c = ResultCache(tmp_path / ".cache")
    assert c.get("c1", "dnsmos", "h1") is None
    c.put(_r())
    got = c.get("c1", "dnsmos", "h1")
    assert got is not None and got.cached is True and got.metrics["dnsmos_ovrl"] == 3.4


def test_config_hash_invalidation(tmp_path):
    c = ResultCache(tmp_path / ".cache")
    c.put(_r(h="h1"))
    assert c.get("c1", "dnsmos", "h1") is not None
    assert c.get("c1", "dnsmos", "h2") is None  # changed config => miss


def test_key_is_deterministic_and_distinct():
    k1 = result_cache_key("c1", "dnsmos", "h1")
    assert k1 == result_cache_key("c1", "dnsmos", "h1")
    assert k1 != result_cache_key("c1", "dnsmos", "h2")
    assert k1 != result_cache_key("c2", "dnsmos", "h1")


def test_clear_all_and_by_scorer(tmp_path):
    c = ResultCache(tmp_path / ".cache")
    c.put(_r(scorer="dnsmos", h="h"))
    c.put(_r(scorer="utmos", h="h", metrics={"utmos": 4.0}))
    assert c.clear(scorer="dnsmos") == 1
    assert c.get("c1", "dnsmos", "h") is None
    assert c.get("c1", "utmos", "h") is not None
    assert c.clear() == 1  # remaining utmos


def test_disabled_cache_is_noop(tmp_path):
    c = ResultCache(tmp_path / ".cache", enabled=False)
    c.put(_r())
    assert c.get("c1", "dnsmos", "h1") is None
