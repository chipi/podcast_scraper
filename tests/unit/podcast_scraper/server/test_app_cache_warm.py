"""Unit tests for the best-effort cache warmer.

The warmer is pure optimization, so the contract is: it warms once, re-warms ONLY when the corpus
mtime moves (trigger B), and never raises — a failing step must not abort the others or kill the
loop (the anti-fragility guarantee).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import app_cache_warm

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clean():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_warm_loop_rewarms_only_when_the_corpus_mtime_moves(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(app_cache_warm, "warm_caches", lambda root: calls.append(1))

    stop = threading.Event()
    # tokens: 1.0 (warm), 1.0 (unchanged → skip), 2.0 (warm), then stop the loop.
    tokens = iter([1.0, 1.0, 2.0])

    def _mtime(_root):
        try:
            return next(tokens)
        except StopIteration:
            stop.set()
            return 2.0

    monkeypatch.setattr(app_cache_warm.perf_cache, "corpus_mtime", _mtime)
    app_cache_warm._warm_loop(Path("/x"), interval_s=0.0, stop=stop)

    assert calls == [
        1,
        1,
    ], "warmed on first-seen + on the mtime change, but not on the unchanged tick"


def test_warm_loop_survives_a_transient_error(monkeypatch) -> None:
    stop = threading.Event()
    seen: list[float] = []

    def _mtime(_root):
        seen.append(0.0)
        if len(seen) == 1:
            raise RuntimeError("stat blip")
        stop.set()
        return 1.0

    monkeypatch.setattr(app_cache_warm.perf_cache, "corpus_mtime", _mtime)
    # Must not propagate — a transient error on one tick can't kill the warmer.
    app_cache_warm._warm_loop(Path("/x"), interval_s=0.0, stop=stop)
    assert len(seen) >= 2


def test_warm_caches_is_best_effort_one_failure_does_not_skip_the_rest(
    monkeypatch, tmp_path
) -> None:
    from podcast_scraper.server import app_catalog_cache, app_kg_index

    monkeypatch.setattr(
        app_catalog_cache,
        "cached_catalog",
        lambda root: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    kg_calls: list[int] = []
    monkeypatch.setattr(app_kg_index, "get_kg_index", lambda root: kg_calls.append(1))

    # Must not raise even though the catalog step throws, and later steps still run.
    app_cache_warm.warm_caches(tmp_path)
    assert kg_calls == [1], "a failing warm step aborted the rest"
