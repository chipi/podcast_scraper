"""Tests for the generic feed-metadata backfill (BS.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.workflow.backfill import refresh_feed_metadata

pytestmark = [pytest.mark.unit]

_RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Show</title>
    <description>A fresh blurb</description>
    <itunes:category text="Business"/>
  </channel>
</rss>"""


def _write(path: Path, feed_id: str, url: str | None, category: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feed: dict = {"feed_id": feed_id, "title": "Show"}
    if url is not None:
        feed["url"] = url
    if category is not None:
        feed["category"] = category
    path.write_text(
        json.dumps({"feed": feed, "episode": {"episode_id": path.stem}}), encoding="utf-8"
    )


def _feed_block(path: Path) -> dict:
    return dict(json.loads(path.read_text())["feed"])


def test_refresh_patches_every_file_of_a_feed(tmp_path: Path) -> None:
    a = tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json"
    b = tmp_path / "feeds/p1/run_1/metadata/e2.metadata.json"
    _write(a, "p1", "http://x/rss")
    _write(b, "p1", "http://x/rss")

    result = refresh_feed_metadata(tmp_path, fetch=lambda url: _RSS)

    assert result.feeds_seen == 1 and result.feeds_refreshed == 1 and result.files_updated == 2
    for p in (a, b):
        fb = _feed_block(p)
        assert fb["category"] == "Business"
        assert fb["description"] == "A fresh blurb"
        # Preserves identity fields.
        assert fb["feed_id"] == "p1" and fb["title"] == "Show"


def test_refresh_is_idempotent(tmp_path: Path) -> None:
    a = tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json"
    _write(a, "p1", "http://x/rss")
    refresh_feed_metadata(tmp_path, fetch=lambda url: _RSS)
    second = refresh_feed_metadata(tmp_path, fetch=lambda url: _RSS)
    assert second.files_updated == 0  # already current


def test_refresh_skips_a_feed_with_no_url(tmp_path: Path) -> None:
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", None)
    result = refresh_feed_metadata(tmp_path, fetch=lambda url: _RSS)
    assert result.skipped == ["p1"] and result.files_updated == 0


def test_refresh_skips_on_failed_fetch(tmp_path: Path) -> None:
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", "http://x/rss")
    result = refresh_feed_metadata(tmp_path, fetch=lambda url: None)
    assert result.skipped == ["p1"] and result.files_updated == 0


def test_feed_id_filter_only_touches_that_feed(tmp_path: Path) -> None:
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", "http://x/rss")
    _write(tmp_path / "feeds/p2/run_1/metadata/e1.metadata.json", "p2", "http://y/rss")
    result = refresh_feed_metadata(tmp_path, feed_id="p1", fetch=lambda url: _RSS)
    assert result.feeds_seen == 1 and result.files_updated == 1
    # p2 untouched.
    assert "category" not in _feed_block(tmp_path / "feeds/p2/run_1/metadata/e1.metadata.json")
