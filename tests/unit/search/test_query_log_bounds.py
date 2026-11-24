"""The activity log is size-bounded and the dashboard read touches only its window (#2527)."""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from podcast_scraper.search import query_log
from podcast_scraper.search.query_log import (
    append_query_event,
    query_log_path,
    read_query_activity,
)

pytestmark = pytest.mark.unit

TODAY = date(2026, 6, 30)


def _line(day: date, hour: int = 12) -> str:
    ts = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).isoformat()
    return (
        f'{{"ts": "{ts}", "schema": 1, "event_type": "search_query", "query_type": "semantic"}}\n'
    )


def test_log_is_bounded_by_rotation(tmp_path, monkeypatch):
    monkeypatch.setattr(query_log, "MAX_LOG_BYTES", 2000, raising=False)
    now = datetime(2026, 6, 30, 12, tzinfo=timezone.utc)
    for _ in range(500):
        append_query_event(tmp_path, "semantic", now=now)
    active = query_log_path(tmp_path)
    backup = active.with_name(active.name + ".1")
    assert active.stat().st_size <= 2000 + 300
    assert backup.exists()
    assert not active.with_name(active.name + ".2").exists()


def test_read_loads_only_the_window(tmp_path, monkeypatch):
    path = query_log_path(tmp_path)
    path.parent.mkdir(parents=True)
    old = "".join(_line(date(2026, 1, 1)) for _ in range(20000))
    recent = "".join(_line(TODAY) for _ in range(5))
    path.write_text(old + recent, encoding="utf-8")
    total_size = path.stat().st_size

    read_bytes = {"n": 0}
    real_open = Path.open

    class Counting(io.BufferedReader):
        def read(self, size=-1):
            data = super().read(size)
            read_bytes["n"] += len(data)
            return data

    def patched(self, mode="r", *a, **kw):
        if self == path and "b" in mode:
            return Counting(real_open(self, mode, *a, **kw))
        return real_open(self, mode, *a, **kw)

    monkeypatch.setattr(Path, "open", patched)
    out = read_query_activity(tmp_path, days=30, today=TODAY)
    assert out["total"] == 5
    assert total_size > 1_000_000
    assert 0 < read_bytes["n"] < total_size // 10


def test_read_spans_the_rotated_backup(tmp_path):
    path = query_log_path(tmp_path)
    path.parent.mkdir(parents=True)
    d1, d2 = TODAY - timedelta(days=5), TODAY
    backup = path.with_name(path.name + ".1")
    backup.write_text(_line(date(2026, 1, 1)) + _line(d1) * 3, encoding="utf-8")
    path.write_text(_line(d2) * 2, encoding="utf-8")
    out = read_query_activity(tmp_path, days=30, today=TODAY)
    by = {b["date"]: b["count"] for b in out["buckets"]}
    assert by[d1.isoformat()] == 3 and by[d2.isoformat()] == 2
    assert out["total"] == 5


def test_oversized_legacy_log_is_rotated_on_first_append(tmp_path, monkeypatch):
    monkeypatch.setattr(query_log, "MAX_LOG_BYTES", 1000, raising=False)
    path = query_log_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(_line(TODAY) * 50, encoding="utf-8")
    append_query_event(tmp_path, "semantic", now=datetime(2026, 6, 30, 13, tzinfo=timezone.utc))
    assert path.stat().st_size < 400
    assert read_query_activity(tmp_path, days=1, today=TODAY)["total"] == 51
