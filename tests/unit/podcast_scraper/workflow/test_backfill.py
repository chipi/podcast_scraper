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

    result = refresh_feed_metadata(tmp_path, fetch=lambda url: (_RSS, url))

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
    refresh_feed_metadata(tmp_path, fetch=lambda url: (_RSS, url))
    second = refresh_feed_metadata(tmp_path, fetch=lambda url: (_RSS, url))
    assert second.files_updated == 0  # already current


def test_idempotent_across_last_updated_zulu_vs_offset(tmp_path: Path) -> None:
    # advisor M2: the pipeline serializes `…Z`, our derive gives `…+00:00`. A naive `==` would
    # rewrite (and format-flip) forever. Pre-seed a file already current except last_updated's
    # format; a refresh must be a no-op.
    dated_rss = (
        b'<?xml version="1.0"?>'
        b'<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">'
        b"<channel><title>Show</title><description>A fresh blurb</description>"
        b'<itunes:category text="Business"/>'
        b"<lastBuildDate>Mon, 01 Jan 2024 00:00:00 GMT</lastBuildDate>"
        b"</channel></rss>"
    )
    path = tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "feed": {
                    "feed_id": "p1",
                    "title": "Show",
                    "url": "http://x/rss",
                    "description": "A fresh blurb",
                    "category": "Business",
                    "last_updated": "2024-01-01T00:00:00Z",  # pipeline's Zulu form
                },
                "episode": {"episode_id": "e1"},
            }
        ),
        encoding="utf-8",
    )
    result = refresh_feed_metadata(tmp_path, fetch=lambda url: (dated_rss, url))
    assert result.files_updated == 0  # same instant, just a different string form → no rewrite


def test_refresh_skips_a_feed_with_no_url(tmp_path: Path) -> None:
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", None)
    result = refresh_feed_metadata(tmp_path, fetch=lambda url: (_RSS, url))
    assert result.skipped == ["p1"] and result.files_updated == 0


def test_refresh_skips_on_failed_fetch(tmp_path: Path) -> None:
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", "http://x/rss")
    result = refresh_feed_metadata(tmp_path, fetch=lambda url: None)
    assert result.skipped == ["p1"] and result.files_updated == 0


def test_feed_id_filter_only_touches_that_feed(tmp_path: Path) -> None:
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", "http://x/rss")
    _write(tmp_path / "feeds/p2/run_1/metadata/e1.metadata.json", "p2", "http://y/rss")
    result = refresh_feed_metadata(tmp_path, feed_id="p1", fetch=lambda url: (_RSS, url))
    assert result.feeds_seen == 1 and result.files_updated == 1
    # p2 untouched.
    assert "category" not in _feed_block(tmp_path / "feeds/p2/run_1/metadata/e1.metadata.json")


def test_patches_a_yaml_corpus_and_keeps_it_yaml(tmp_path: Path) -> None:
    # advisor L7: a metadata_format=yaml corpus must not be invisible to the backfill.
    import yaml

    path = tmp_path / "feeds/p1/run_1/metadata/e1.metadata.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"feed": {"feed_id": "p1", "title": "Show", "url": "http://x/rss"}}),
        encoding="utf-8",
    )
    result = refresh_feed_metadata(tmp_path, fetch=lambda url: (_RSS, url))
    assert result.files_updated == 1
    reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))  # still valid YAML
    assert reloaded["feed"]["category"] == "Business"


def test_relative_image_href_resolves_against_the_final_url(tmp_path: Path) -> None:
    # advisor L5: a relative <image> href must resolve against the post-redirect final URL (what the
    # fetch returns), not the stored feed.url.
    rss = (
        b'<?xml version="1.0"?><rss version="2.0"><channel><title>S</title>'
        b"<image><url>/img/cover.png</url></image></channel></rss>"
    )
    _write(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json", "p1", "http://stored/rss")
    # Fetch reports a DIFFERENT final host (a redirect).
    refresh_feed_metadata(tmp_path, fetch=lambda url: (rss, "https://final.example/feed.xml"))
    fb = _feed_block(tmp_path / "feeds/p1/run_1/metadata/e1.metadata.json")
    assert fb["image_url"] == "https://final.example/img/cover.png"
