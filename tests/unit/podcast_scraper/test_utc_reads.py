"""Stored times are compared as UTC whatever form they were stored in (public-utc, step 3)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest

from podcast_scraper.archive.cli_handlers import _select
from podcast_scraper.utils.utc import parse_stored, utc_day, utc_sort_key


@pytest.mark.parametrize(
    "raw, day, key",
    [
        ("2026-09-16T23:05:00", "2026-09-16", "2026-09-16T23:05:00Z"),
        ("2026-09-16T23:05:00Z", "2026-09-16", "2026-09-16T23:05:00Z"),
        ("2026-09-17T01:05:00+02:00", "2026-09-16", "2026-09-16T23:05:00Z"),
        ("2026-09-16T20:05:00-05:00", "2026-09-17", "2026-09-17T01:05:00Z"),
        ("2026-09-16", "2026-09-16", "2026-09-16"),
        ("garbage text here", None, ""),
        (None, None, ""),
    ],
)
def test_a_stored_time_has_one_utc_day_and_one_sort_key(raw: object, day: object, key: str) -> None:
    assert utc_day(raw) == day
    assert utc_sort_key(raw) == key


def test_a_datetime_is_read_as_utc() -> None:
    assert parse_stored(datetime(2026, 9, 16, 23, 5)) == datetime(
        2026, 9, 16, 23, 5, tzinfo=timezone.utc
    )


def test_mixed_forms_sort_in_time_order() -> None:
    stored = ["2026-09-17T01:05:00+02:00", "2026-09-16T23:30:00", "2026-09-16T23:10:00Z"]
    # As text the +02:00 value sorts last; as UTC it is the earliest (23:05Z).
    assert sorted(stored, key=utc_sort_key) == [
        "2026-09-17T01:05:00+02:00",
        "2026-09-16T23:10:00Z",
        "2026-09-16T23:30:00",
    ]


def test_the_archive_since_filter_compares_the_utc_day() -> None:
    episodes = [
        {
            "feed_title": "",
            "feed_id": "",
            "episode_id": "a",
            "guid": "a",
            "published_date": "2026-09-17T01:05:00+02:00",
        },
        {
            "feed_title": "",
            "feed_id": "",
            "episode_id": "b",
            "guid": "b",
            "published_date": "2026-09-17T08:00:00Z",
        },
    ]
    picked = _select(episodes, argparse.Namespace(feed=None, episode=None, since="2026-09-17"))
    assert [e["episode_id"] for e in picked] == ["b"]
