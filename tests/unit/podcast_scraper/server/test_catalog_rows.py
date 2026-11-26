"""The catalog from corpus rows (server.catalog_rows): each field from its stored value."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.server.catalog_rows import catalog_row

RELPATH = "feeds/rss_a/run_1/metadata/0001 - One.metadata.json"


def _item(**fields: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "raw_feed_id": "rss_a",
        "external_item_id": "ep-1",
        "title": "One",
        "published_at": datetime(2026, 9, 17, 1, 5, tzinfo=timezone(timedelta(hours=2))),
        "duration_s": 600,
        "language": "en",
        "has_gi": True,
        "has_kg": False,
        "type_attrs": {
            "metadata_relative_path": RELPATH,
            "summary_title": "A lede",
            "summary_bullets": ["First.", "Second."],
            "summary_text": "Body.",
            "has_bridge": True,
            "episode_image_url": "https://img.example/e.jpg",
            "episode_description": "About one.",
            "episode_number": 1,
            "episode_licence_json": '{"spdx": "CC-BY-4.0"}',
        },
        "source_attrs": {
            "title": "A show",
            "category": "News",
            "language": "en",
            "licence": '{"spdx": "CC-BY-4.0"}',
            "type_attrs": {
                "feed_rss_url": "https://feed.example/rss",
                "feed_description": "A show about things.",
                "feed_authors": ["Host One"],
                "feed_last_updated": "2026-09-16T23:05:00Z",
                "feed_image_url": "https://img.example/f.jpg",
            },
        },
    }
    item.update(fields)
    return item


def test_a_row_carries_every_field_from_its_stored_value(tmp_path: Path) -> None:
    row = catalog_row(tmp_path, _item())
    assert row is not None
    assert (row.metadata_relative_path, row.feed_id, row.episode_id) == (RELPATH, "rss_a", "ep-1")
    assert (row.feed_title, row.episode_title, row.publish_date) == ("A show", "One", "2026-09-16")
    assert row.summary_bullets == ("First.", "Second.")
    assert (row.summary_title, row.summary_text) == ("A lede", "Body.")
    assert row.gi_relative_path.endswith("0001 - One.gi.json")
    assert row.kg_relative_path.endswith("0001 - One.kg.json")
    assert (row.has_gi, row.has_kg, row.has_bridge) == (True, False, True)
    assert (row.duration_seconds, row.episode_number) == (600, 1)
    assert row.feed_authors == ("Host One",)
    assert (row.feed_rss_url, row.feed_last_updated) == (
        "https://feed.example/rss",
        "2026-09-16T23:05:00Z",
    )
    assert (row.feed_category, row.feed_language, row.episode_language) == ("News", "en", "en")
    assert row.feed_licence_json == row.episode_licence_json == '{"spdx": "CC-BY-4.0"}'


def test_an_item_without_a_metadata_path_has_no_row(tmp_path: Path) -> None:
    assert catalog_row(tmp_path, _item(type_attrs={})) is None


def test_artwork_shows_only_when_its_file_is_on_disk(tmp_path: Path) -> None:
    attrs = {
        **_item()["type_attrs"],
        "episode_image_local_relpath": ".podcast_scraper/corpus-art/sha256/ab/e.jpg",
        "feed_image_local_relpath": ".podcast_scraper/corpus-art/sha256/ab/f.jpg",
    }
    assert catalog_row(tmp_path, _item(type_attrs=attrs)).episode_image_local_relpath is None
    art = tmp_path / ".podcast_scraper" / "corpus-art" / "sha256" / "ab"
    art.mkdir(parents=True)
    (art / "e.jpg").write_bytes(b"jpg")
    row = catalog_row(tmp_path, _item(type_attrs=attrs))
    assert row is not None and row.episode_image_local_relpath is not None
    assert row.feed_image_local_relpath is None
