"""Unit tests for the central perf cache (podcast_scraper.perf_cache)."""

from __future__ import annotations

import pytest

from podcast_scraper import perf_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_hit_miss_and_token_invalidation():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    # First call at token 1.0 → miss + compute.
    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) == 1
    assert calls["n"] == 1
    # Same token → hit, no recompute.
    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) == 1
    assert calls["n"] == 1
    # New token → miss, recompute.
    assert perf_cache.get_or_compute("ns", "k", 2.0, compute) == 2
    assert calls["n"] == 2

    s = perf_cache.stats()["ns"]
    assert s["hits"] == 1 and s["misses"] == 2 and s["entries"] == 1
    assert s["hit_rate_pct"] == pytest.approx(33.3, abs=0.1)


def test_distinct_keys_and_namespaces_isolated():
    perf_cache.get_or_compute("a", "k1", 1.0, lambda: "v1")
    perf_cache.get_or_compute("a", "k2", 1.0, lambda: "v2")
    perf_cache.get_or_compute("b", "k1", 1.0, lambda: "v3")
    stats = perf_cache.stats()
    assert stats["a"]["entries"] == 2
    assert stats["b"]["entries"] == 1


def test_clear_namespace_and_all():
    perf_cache.get_or_compute("a", "k", 1.0, lambda: 1)
    perf_cache.get_or_compute("b", "k", 1.0, lambda: 1)
    perf_cache.clear("a")
    assert "a" not in perf_cache.stats()
    assert "b" in perf_cache.stats()
    perf_cache.clear()
    assert perf_cache.stats() == {}


def test_caches_none_result():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return None

    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) is None
    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) is None
    assert calls["n"] == 1  # None is a valid cached value


def test_lance_mtime_and_corpus_mtime(tmp_path):
    # Absent paths → -1.0 sentinel, never raises.
    assert perf_cache.lance_mtime(tmp_path / "nope") == -1.0
    assert perf_cache.corpus_mtime(tmp_path / "nope") == -1.0
    # corpus_mtime prefers corpus_run_summary.json over the dir mtime.
    (tmp_path / "corpus_run_summary.json").write_text("{}", encoding="utf-8")
    import os

    assert perf_cache.corpus_mtime(tmp_path) == pytest.approx(
        os.path.getmtime(tmp_path / "corpus_run_summary.json")
    )
