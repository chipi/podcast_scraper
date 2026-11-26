"""The pipeline writes every time as UTC with ``Z`` (operator, 2026-11-26; utils.utc)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import pytest

from podcast_scraper.rss.parser import extract_episode_published_date
from podcast_scraper.utils.utc import as_utc, rfc2822_from_iso, utc_iso, utc_now
from podcast_scraper.workflow.metadata_generation import (
    EpisodeMetadata,
    FeedMetadata,
    generate_episode_id,
    ProcessingMetadata,
)
from podcast_scraper.workflow.stages.scraping import _synthesize_feed_item

PLUS_TWO = timezone(timedelta(hours=2))


def test_a_naive_time_is_utc_and_an_offset_is_converted() -> None:
    assert utc_iso(datetime(2026, 9, 16, 23, 5)) == "2026-09-16T23:05:00Z"
    assert utc_iso(datetime(2026, 9, 17, 1, 5, tzinfo=PLUS_TWO)) == "2026-09-16T23:05:00Z"
    assert as_utc(datetime(2026, 9, 17, 1, 5, tzinfo=PLUS_TWO)).tzinfo == timezone.utc
    assert utc_now().tzinfo == timezone.utc


@pytest.mark.parametrize(
    "parsed",
    [
        parsedate_to_datetime("Wed, 16 Sep 2026 23:05:00 -0000"),
        parsedate_to_datetime("Thu, 17 Sep 2026 01:05:00 +0200"),
        parsedate_to_datetime("Wed, 16 Sep 2026 23:05:00 GMT"),
    ],
)
def test_the_stored_publish_and_feed_dates_are_utc_with_z(parsed: datetime) -> None:
    episode = EpisodeMetadata(title="E", episode_id="e", published_date=parsed).model_dump(
        mode="json"
    )
    feed = FeedMetadata(title="F", url="u", feed_id="f", last_updated=parsed).model_dump(
        mode="json"
    )
    assert episode["published_date"] == "2026-09-16T23:05:00Z"
    assert feed["last_updated"] == "2026-09-16T23:05:00Z"


def test_a_processing_timestamp_is_written_in_utc() -> None:
    doc = ProcessingMetadata(processing_timestamp=utc_now(), output_directory="o").model_dump(
        mode="json"
    )
    assert doc["processing_timestamp"].endswith("Z")


def test_an_episode_id_without_a_guid_still_hashes_the_date_as_parsed() -> None:
    # The id of a GUID-less episode hashes published_date.isoformat(): it must not change with
    # the stored form, or the next scrape would make a second episode.
    naive = parsedate_to_datetime("Wed, 16 Sep 2026 23:05:00 -0000")
    expected = hashlib.sha256(
        "https://feed.example/rss|an episode|2026-09-16T23:05:00".encode("utf-8")
    ).hexdigest()
    got = generate_episode_id(
        feed_url="https://feed.example/rss",
        episode_title="An episode",
        episode_guid=None,
        published_date=naive,
    )
    assert got == f"sha256:{expected}"


@pytest.mark.parametrize(
    "stored, parsed",
    [
        ("2026-09-16T23:05:00", datetime(2026, 9, 16, 23, 5)),
        ("2026-09-16T23:05:00Z", datetime(2026, 9, 16, 23, 5, tzinfo=timezone.utc)),
        ("2026-09-17T01:05:00+02:00", datetime(2026, 9, 17, 1, 5, tzinfo=PLUS_TWO)),
    ],
)
def test_a_reprocessed_episode_keeps_its_stored_publish_date(stored: str, parsed: datetime) -> None:
    item = _synthesize_feed_item("guid-1", {"title": "E", "published_date": stored})
    got = extract_episode_published_date(item)
    assert got == parsed
    assert (got.tzinfo is None) == (parsed.tzinfo is None)
    assert rfc2822_from_iso(stored) == item.findtext("pubDate")


def test_an_unreadable_stored_publish_date_is_left_out_of_the_synthesized_item() -> None:
    item = _synthesize_feed_item("guid-1", {"title": "E", "published_date": "last Tuesday"})
    assert item.find("pubDate") is None
