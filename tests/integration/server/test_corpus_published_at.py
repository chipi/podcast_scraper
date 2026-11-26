"""The server serves the full UTC publish instant beside the date (operator, 2026-11-26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.corpus_catalog import (
    _parse_publish_date_str,
    _parse_published_at,
    build_catalog_rows,
)
from podcast_scraper.server.corpus_digest import episode_in_utc_window, utc_bounds_for_window

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_the_24h_digest_window_holds_an_episode_from_yesterday_evening(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    _write(tmp_path, "evening", "2026-11-25T20:00:00Z")
    _write(tmp_path, "old", "2026-11-24T20:00:00Z")
    rows = {r.episode_id: r for r in build_catalog_rows(tmp_path)}
    start, end = utc_bounds_for_window(
        "24h", since=None, now_utc=datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
    )
    assert episode_in_utc_window(rows["evening"], start, end)
    assert not episode_in_utc_window(rows["old"], start, end)


@pytest.mark.parametrize(
    "stored, published_at, publish_date",
    [
        # A naive value is UTC (-0000 parses naive): the same day either way.
        ("2026-09-16T23:05:00", "2026-09-16T23:05:00Z", "2026-09-16"),
        ("2026-09-16T23:05:00Z", "2026-09-16T23:05:00Z", "2026-09-16"),
        # The publisher's own offset: the UTC day, not the day in that offset.
        ("2026-09-17T01:05:00+02:00", "2026-09-16T23:05:00Z", "2026-09-16"),
        ("2026-09-16T20:05:00-05:00", "2026-09-17T01:05:00Z", "2026-09-17"),
        # A date with no time has no instant.
        ("2026-09-16", None, "2026-09-16"),
        ("not a date", None, None),
        (None, None, None),
    ],
)
def test_the_instant_is_utc_and_the_date_is_its_utc_day(
    stored: object, published_at: object, publish_date: object
) -> None:
    assert _parse_published_at(stored) == published_at
    assert _parse_publish_date_str(stored) == publish_date


def _write(root: Path, stem: str, published: str) -> None:
    meta = root / "metadata"
    meta.mkdir(exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "Show"},
        "episode": {"episode_id": stem, "title": stem, "published_date": published},
    }
    (meta / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def test_the_catalog_row_and_the_episode_routes_carry_published_at(tmp_path: Path) -> None:
    _write(tmp_path, "e1", "2026-09-17T01:05:00+02:00")
    [row] = build_catalog_rows(tmp_path)
    assert (row.publish_date, row.published_at) == ("2026-09-16", "2026-09-16T23:05:00Z")

    client = TestClient(create_app(tmp_path, static_dir=False))
    listed = client.get("/api/corpus/episodes", params={"path": str(tmp_path)}).json()["items"]
    assert [(i["publish_date"], i["published_at"]) for i in listed] == [
        ("2026-09-16", "2026-09-16T23:05:00Z")
    ]
    detail = client.get(
        "/api/corpus/episodes/detail",
        params={"path": str(tmp_path), "metadata_relpath": "metadata/e1.metadata.json"},
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["published_at"] == "2026-09-16T23:05:00Z"
