"""The memory watchdog reports above its threshold, at each further step, and names live work."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from podcast_scraper.server import mem_watchdog as mw
from podcast_scraper.utils import access_log as al

pytestmark = pytest.mark.unit


def _run(samples: list[int], threshold: int, caplog) -> list[str]:
    it = iter(samples)
    stop = threading.Event()

    def sample() -> int:
        try:
            return next(it)
        except StopIteration:
            stop.set()
            return 0

    with caplog.at_level(logging.WARNING, logger=mw.__name__):
        mw.watch(threshold, stop, interval_s=0.0, sample=sample)
    return [r.getMessage() for r in caplog.records if r.name == mw.__name__]


def test_reports_once_above_threshold_then_each_step(caplog) -> None:
    msgs = _run([900, 2600, 2700, 3200, 3300], 2500, caplog)
    assert [m.split()[2] for m in msgs] == ["rss=2600MB", "rss=3200MB"]


def test_rearms_after_dropping_below(caplog) -> None:
    msgs = _run([2600, 1000, 2600], 2500, caplog)
    assert len(msgs) == 2


def test_quiet_below_threshold(caplog) -> None:
    assert _run([100, 2499, 2000], 2500, caplog) == []


def test_report_names_in_flight_requests_and_project_frames() -> None:
    with al._INFLIGHT_LOCK:
        al._INFLIGHT[1] = ("GET", "/api/app/episodes/x/related", time.monotonic() - 3.0)
    try:
        text = mw.report(3000)
    finally:
        with al._INFLIGHT_LOCK:
            al._INFLIGHT.pop(1, None)
    assert "rss=3000MB" in text
    assert "GET /api/app/episodes/x/related 3." in text


def test_off_unless_threshold_set(monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_MEM_WATCHDOG_MB", raising=False)
    assert mw.start_mem_watchdog() is None
    monkeypatch.setenv("PODCAST_MEM_WATCHDOG_MB", "nope")
    assert mw.start_mem_watchdog() is None


def test_access_log_tracks_and_clears_in_flight() -> None:
    import asyncio

    seen: list[list] = []

    async def app(scope, receive, send):
        seen.append(al.inflight_requests())
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        return None

    mw_app = al.AccessLogMiddleware(app)
    asyncio.run(mw_app({"type": "http", "path": "/api/x", "method": "GET"}, receive, send))
    assert [(m, p) for m, p, _ in seen[0]] == [("GET", "/api/x")]
    assert al.inflight_requests() == []
