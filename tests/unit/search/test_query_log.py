"""Unit tests for the search-activity log (PRD-033 FR6.2, #888 follow-up)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from podcast_scraper.search.query_log import (
    append_query_event,
    query_log_path,
    read_query_activity,
)

pytestmark = pytest.mark.unit


def test_append_then_read_buckets_by_day(tmp_path):
    d = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
    append_query_event(tmp_path, "semantic", now=d)
    append_query_event(tmp_path, "entity_lookup", now=d)
    append_query_event(tmp_path, "semantic", now=datetime(2026, 6, 5, 9, 0, tzinfo=timezone.utc))

    out = read_query_activity(tmp_path, days=5, today=date(2026, 6, 5))
    assert out["total"] == 3
    by_date = {b["date"]: b["count"] for b in out["buckets"]}
    assert by_date["2026-06-03"] == 2
    assert by_date["2026-06-05"] == 1
    assert by_date["2026-06-04"] == 0


def test_read_zero_fills_window_oldest_to_newest(tmp_path):
    out = read_query_activity(tmp_path, days=3, today=date(2026, 6, 5))
    assert [b["date"] for b in out["buckets"]] == ["2026-06-03", "2026-06-04", "2026-06-05"]
    assert out["total"] == 0


def test_events_outside_window_excluded(tmp_path):
    append_query_event(tmp_path, "semantic", now=datetime(2026, 5, 1, tzinfo=timezone.utc))
    out = read_query_activity(tmp_path, days=3, today=date(2026, 6, 5))
    assert out["total"] == 0


def test_append_writes_under_search_dir(tmp_path):
    append_query_event(tmp_path, "semantic", now=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert query_log_path(tmp_path) == tmp_path / "search" / "query_log.jsonl"
    assert query_log_path(tmp_path).exists()


def test_log_omits_raw_query_text(tmp_path):
    append_query_event(tmp_path, "semantic", now=datetime(2026, 6, 1, tzinfo=timezone.utc))
    content = query_log_path(tmp_path).read_text(encoding="utf-8")
    rec = json.loads(content.strip())
    # Privacy: only the timestamp + the intent label are stored (plus the canonical
    # envelope). There is NO raw-query field — the exact key set is the guarantee.
    assert rec["query_type"] == "semantic"
    assert "ts" in rec
    assert "query" not in rec  # no raw-query field (event_type "search_query" is a label)
    assert set(rec) <= {"ts", "schema", "event_type", "query_type"}


def test_corrupt_lines_are_skipped(tmp_path):
    path = query_log_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "not json",
                '{"ts": "2026-06-05T01:00:00+00:00", "query_type": "semantic"}',
                '{"no_ts": true}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = read_query_activity(tmp_path, days=2, today=date(2026, 6, 5))
    assert out["total"] == 1


def test_read_missing_log_is_all_zero(tmp_path):
    out = read_query_activity(tmp_path, days=1, today=date(2026, 6, 5))
    assert out == {"total": 0, "buckets": [{"date": "2026-06-05", "count": 0}]}
