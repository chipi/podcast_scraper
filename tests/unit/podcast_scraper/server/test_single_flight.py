"""A heavy listing is computed once for every caller that asks while it runs (2026-10-15).

The post-deploy smoke retried /api/artifacts every 15 s against a 10 s timeout; each retry started
another full listing on a saturated process, and calls piled up to ~600 s.
"""

from __future__ import annotations

import threading
import time

import pytest

from podcast_scraper.server.single_flight import single_flight

pytestmark = [pytest.mark.unit]


def _hold(calls: list, release: threading.Event, value):
    def compute():
        calls.append(1)
        release.wait(5)
        return value

    return compute


def _run_concurrently(n, fn):
    out, errors = [None] * n, [None] * n

    def worker(i):
        try:
            out[i] = fn()
        except BaseException as exc:  # noqa: BLE001 -- collected for the assertion
            errors[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    return threads, out, errors


def test_callers_that_arrive_while_it_runs_share_one_computation() -> None:
    calls: list = []
    release = threading.Event()
    result = object()
    threads, out, errors = _run_concurrently(
        8, lambda: single_flight(("k", 1), _hold(calls, release, result))
    )
    time.sleep(0.2)  # every caller is waiting on the one computation
    release.set()
    for t in threads:
        t.join(5)
    assert calls == [1]
    assert all(r is result for r in out) and errors == [None] * 8


def test_an_error_reaches_every_waiter_and_the_next_call_retries() -> None:
    release = threading.Event()

    def boom():
        release.wait(5)
        raise RuntimeError("listing failed")

    threads, out, errors = _run_concurrently(4, lambda: single_flight(("k", 2), boom))
    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(5)
    assert all(isinstance(e, RuntimeError) for e in errors)
    assert single_flight(("k", 2), lambda: "fresh") == "fresh"


def test_nothing_is_kept_after_it_finishes() -> None:
    calls: list = []

    def compute():
        calls.append(1)
        return len(calls)

    assert single_flight(("k", 3), compute) == 1
    assert single_flight(("k", 3), compute) == 2


def test_different_keys_do_not_wait_for_each_other() -> None:
    release = threading.Event()
    threads, _, _ = _run_concurrently(1, lambda: single_flight(("k", 4), _hold([], release, 1)))
    time.sleep(0.1)
    start = time.time()
    assert single_flight(("k", 5), lambda: "other") == "other"
    assert time.time() - start < 1
    release.set()
    for t in threads:
        t.join(5)
