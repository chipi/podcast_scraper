"""``sources`` and ``content_items`` column values from an item's metadata (CRFC-005 §3.1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from podcast_scraper.db.corpus_keys import KeyNotDerivable
from podcast_scraper.db.corpus_rows import (
    item_row,
    NOT_SERVED,
    SERVED,
    served_for,
    source_row,
    source_row_for,
)
from podcast_scraper.publishers import Publisher, Registry

_FEED_ID = "sha256:" + "ab" * 32
_REGISTRY = Registry(
    publishers=(
        Publisher(id="org:example-times", name="The Example Times", domains=("example.com",)),
    )
)


def _episode(**episode: Any) -> dict[str, Any]:
    block = {
        "title": "An episode",
        "episode_id": "ep-1",
        "guid": "guid-1",
        "published_date": "2026-11-20T08:00:00+00:00",
        "duration_seconds": 1800,
        "language": "en",
    }
    block.update(episode)
    return {
        "feed": {
            "feed_id": _FEED_ID,
            "title": "A show",
            "language": "en-us",
            "category": "Technology",
            "licence": {"status": "declared"},
        },
        "episode": block,
        "content": {"media_url": "https://cdn.example/ep1.mp3", "media_type": "audio/mpeg"},
        "processing": {"processing_timestamp": "2026-11-21T02:00:00.5"},
    }


def _news(**fields: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "content_type": "news_article",
        "item_id": "item:news_article:0001",
        "source_id": "source:0001",
        "url": "https://www.example.com/a/1",
        "title": "A headline",
        "published_at": "2026-11-22T01:00:09Z",
        "ingested_at": "2026-11-24T06:00:13Z",
        "language": "en",
        "serveable": "summary",
        "news_article": {
            "editorial_type": "analysis",
            "byline": ["A. Writer"],
            "text_status": "full",
        },
    }
    doc.update(fields)
    return doc


def test_an_episode_row_carries_the_catalogs_values() -> None:
    row = item_row(_episode())
    assert row["natural_id"] == "ep-1"
    assert row["external_item_id"] == "guid-1"
    assert row["content_type"] == "podcast_episode"
    assert row["title"] == "An episode"
    assert row["duration_s"] == 1800
    assert row["language"] == "en"
    assert row["published_at"] == datetime(2026, 11, 20, 8, tzinfo=timezone.utc)
    assert row["served"] == SERVED
    assert (row["withheld"], row["unusable"]) == (False, False)
    assert row["editorial_type"] is None
    assert row["type_attrs"]["media_url"] == "https://cdn.example/ep1.mp3"
    assert "published_at_raw" not in row["type_attrs"]


def test_an_episode_with_no_title_or_language_falls_back_as_the_catalog_does() -> None:
    doc = _episode(language=None)
    del doc["episode"]["title"]
    row = item_row(doc)
    assert row["title"] == "(untitled)"
    assert row["language"] == "en"


def test_a_time_with_no_offset_is_utc() -> None:
    row = item_row(_episode(published_date="2026-09-16T23:05:00"))
    assert row["published_at"] == datetime(2026, 9, 16, 23, 5, tzinfo=timezone.utc)
    assert row["ingested_at"] == datetime(2026, 11, 21, 2, 0, 0, 500000, tzinfo=timezone.utc)
    assert "published_at_raw" not in row["type_attrs"]


def test_a_time_with_an_offset_is_stored_as_utc() -> None:
    published = item_row(_episode(published_date="2026-11-20T21:30:00-05:00"))["published_at"]
    assert published == datetime(2026, 11, 21, 2, 30, tzinfo=timezone.utc)
    assert published.utcoffset() is not None and not published.utcoffset()


def test_an_unreadable_time_stays_null() -> None:
    assert item_row(_episode(published_date="soon"))["published_at"] is None
    assert item_row(_episode(published_date=None))["published_at"] is None


def test_an_unusable_episode_is_not_served() -> None:
    row = item_row(_episode(unusable=True, unusable_reason="translation_incomplete"))
    assert row["served"] == NOT_SERVED
    assert row["unusable"] is True


def test_a_news_row_carries_its_own_fields() -> None:
    row = item_row(_news())
    assert row["natural_id"] == "item:news_article:0001"
    assert row["content_type"] == "news_article"
    assert row["editorial_type"] == "analysis"
    assert row["duration_s"] is None
    assert row["published_at"] == datetime(2026, 11, 22, 1, 0, 9, tzinfo=timezone.utc)
    assert row["served"] == SERVED
    assert row["type_attrs"] == {
        "serveable": "summary",
        "url": "https://www.example.com/a/1",
        "byline": ["A. Writer"],
        "text_status": "full",
    }


def test_a_news_item_shown_to_no_one_is_withheld_and_not_served() -> None:
    row = item_row(_news(serveable="none"))
    assert (row["served"], row["withheld"]) == (NOT_SERVED, True)


def test_a_teaser_without_a_kept_class_is_not_served_but_not_withheld() -> None:
    doc = _news()
    doc["news_article"]["text_status"] = "teaser"
    row = item_row(doc)
    assert (row["served"], row["withheld"], row["unusable"]) == (NOT_SERVED, False, False)


def test_an_unknown_editorial_type_is_dropped() -> None:
    doc = _news()
    doc["news_article"]["editorial_type"] = "gossip"
    assert item_row(doc)["editorial_type"] is None


def test_served_for_agrees_with_the_row() -> None:
    for doc in (_episode(), _episode(unusable=True), _news(), _news(serveable="none")):
        assert served_for(doc) == item_row(doc)["served"]


def test_a_missing_key_raises() -> None:
    doc = _episode()
    del doc["episode"]["episode_id"]
    with pytest.raises(KeyNotDerivable):
        item_row(doc)


def test_a_podcast_source_row() -> None:
    row = source_row(_episode())
    assert row["raw_feed_id"] == _FEED_ID
    assert row["natural_id"].startswith("podcast:")
    assert row["content_type"] == "podcast_episode"
    assert row["title"] == "A show"
    assert row["language"] == "en"
    assert row["category"] == "Technology"
    assert json.loads(row["licence"]) == {"status": "declared"}


def test_a_news_source_is_named_by_its_publisher_else_its_host() -> None:
    assert source_row(_news(), registry=_REGISTRY)["title"] == "The Example Times"
    other = _news(url="https://www.elsewhere.org/a/2")
    row = source_row(other, registry=_REGISTRY)
    assert row["title"] == "elsewhere.org"
    assert (row["raw_feed_id"], row["language"], row["licence"]) == (None, None, None)


def test_a_news_source_needs_the_registry() -> None:
    with pytest.raises(ValueError, match="registry"):
        source_row(_news())


def test_a_show_takes_each_column_from_its_newest_item_that_states_it() -> None:
    old = _episode(episode_id="ep-old", published_date="2026-01-01T00:00:00+00:00")
    new = _episode(episode_id="ep-new", published_date="2026-11-01T00:00:00+00:00")
    new["feed"]["category"] = None
    new["feed"]["licence"] = {"status": "no_explicit_licence_found"}
    old["feed"]["category"] = "Science"
    row = source_row_for([old, new])
    assert json.loads(row["licence"]) == {"status": "no_explicit_licence_found"}
    assert row["category"] == "Science"


def test_an_undated_item_counts_as_oldest() -> None:
    undated = _episode(episode_id="ep-x", published_date=None)
    undated["feed"]["title"] = "Old name"
    dated = _episode(episode_id="ep-y")
    assert source_row_for([undated, dated])["title"] == "A show"


def test_a_show_of_items_from_two_sources_is_refused() -> None:
    other = _episode(episode_id="ep-2")
    other["feed"]["feed_id"] = "sha256:" + "cd" * 32
    with pytest.raises(ValueError, match="different sources"):
        source_row_for([_episode(), other])
    with pytest.raises(ValueError, match="at least one"):
        source_row_for([])
