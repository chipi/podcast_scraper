"""perf_cache.get_or_compute is single-flight: concurrent misses share ONE build.

Prod 2026-10-03: the post-deploy smoke retried the corpus digest 12 times against a cold api, each
retry started its own cold build, and the duplicates starved each other — 131 s for the first
answer against ~17 s for one build alone, so the probe never passed.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from podcast_scraper import perf_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_concurrent_misses_run_one_build_and_all_get_its_value() -> None:
    calls = []
    lock = threading.Lock()

    def build():
        with lock:
            calls.append(1)
        time.sleep(0.3)
        return "digest"

    with ThreadPoolExecutor(max_workers=10) as ex:
        out = list(ex.map(lambda _: perf_cache.get_or_compute("ns", "k", 1.0, build), range(10)))

    assert out == ["digest"] * 10
    assert len(calls) == 1


def test_a_failed_build_does_not_strand_its_waiters() -> None:
    attempts = []
    lock = threading.Lock()

    def build():
        with lock:
            attempts.append(1)
            first = len(attempts) == 1
        time.sleep(0.2)
        if first:
            raise RuntimeError("cold build failed")
        return "ok"

    def call():
        try:
            return perf_cache.get_or_compute("ns", "k", 1.0, build)
        except RuntimeError:
            return "raised"

    with ThreadPoolExecutor(max_workers=5) as ex:
        out = list(ex.map(lambda _: call(), range(5)))

    assert out.count("raised") == 1
    assert out.count("ok") == 4
    assert len(attempts) == 2


def test_a_new_token_is_a_new_build() -> None:
    assert perf_cache.get_or_compute("ns", "k", 1.0, lambda: "old") == "old"
    assert perf_cache.get_or_compute("ns", "k", 2.0, lambda: "new") == "new"
    assert perf_cache.get_or_compute("ns", "k", 2.0, lambda: "unused") == "new"
