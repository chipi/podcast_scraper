"""Unit tests for podcast_scraper.workflow.stages.scraping module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest
from defusedxml.ElementTree import fromstring as safe_item_fromstring

from podcast_scraper.config import Config
from podcast_scraper.models import RssFeed
from podcast_scraper.rss.parser import published_date_for_episode_filter
from podcast_scraper.workflow.stages.scraping import (
    extract_feed_metadata_for_generation,
    prepare_episodes_from_feed,
)
from podcast_scraper.workflow.types import FeedMetadata


def _scraping_cfg(**kwargs):
    """Build a MagicMock cfg with episode-selection defaults (GitHub #521)."""
    c = MagicMock()
    c.max_episodes = None
    c.episode_order = "newest"
    c.episode_offset = 0
    c.episode_since = None
    c.episode_until = None
    for key, val in kwargs.items():
        setattr(c, key, val)
    return c


def _rss_item(title: str, pub_date: str | None = None):
    inner = f"<title>{title}</title>"
    if pub_date is not None:
        inner += f"<pubDate>{pub_date}</pubDate>"
    return safe_item_fromstring(f"<item>{inner}</item>")


@pytest.mark.unit
class TestExtractFeedMetadataForGeneration:
    """Tests for extract_feed_metadata_for_generation."""

    def test_disabled_generate_metadata_returns_empty(self):
        """When generate_metadata is False, returns empty FeedMetadata."""
        cfg = MagicMock()
        cfg.generate_metadata = False
        feed = MagicMock()
        feed.base_url = "https://example.com"
        result = extract_feed_metadata_for_generation(cfg, feed, b"")
        assert result == FeedMetadata(None, None, None)

    def test_empty_rss_bytes_returns_empty(self):
        """When rss_bytes is empty, returns empty FeedMetadata."""
        cfg = MagicMock()
        cfg.generate_metadata = True
        feed = MagicMock()
        result = extract_feed_metadata_for_generation(cfg, feed, b"")
        assert result == FeedMetadata(None, None, None)


@pytest.mark.unit
class TestPrepareEpisodesFromFeed:
    """Tests for prepare_episodes_from_feed."""

    def test_empty_feed_returns_empty_list(self):
        """Feed with no items returns empty episode list."""
        feed = RssFeed(title="T", items=[], base_url="https://example.com", authors=[])
        cfg = _scraping_cfg()
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert episodes == []

    def test_respects_max_episodes(self):
        """When max_episodes is set, only that many items are processed."""
        feed = RssFeed(
            title="T",
            items=[_rss_item("A"), _rss_item("B")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(max_episodes=1)
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert len(episodes) == 1
        assert episodes[0].title == "A"

    def test_episode_order_oldest_reverses_document_order(self):
        """oldest order processes items from the end of the feed first."""
        feed = RssFeed(
            title="T",
            items=[_rss_item("newest"), _rss_item("mid"), _rss_item("oldest")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(episode_order="oldest")
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert [e.title for e in episodes] == ["oldest", "mid", "newest"]

    def test_episode_offset_after_order(self):
        """Offset skips leading items after order (and date filter)."""
        feed = RssFeed(
            title="T",
            items=[_rss_item("a"), _rss_item("b"), _rss_item("c")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(episode_offset=1, max_episodes=10)
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert [e.title for e in episodes] == ["b", "c"]

    def test_date_filter_since_until(self):
        """episode_since / episode_until keep only matching pubDates."""
        feed = RssFeed(
            title="T",
            items=[
                _rss_item("early", "Mon, 01 Jan 2024 12:00:00 GMT"),
                _rss_item("mid", "Mon, 15 Jul 2024 12:00:00 GMT"),
                _rss_item("late", "Mon, 30 Dec 2024 12:00:00 GMT"),
            ],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(
            episode_since=date(2024, 6, 1),
            episode_until=date(2024, 8, 1),
        )
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert len(episodes) == 1
        assert episodes[0].title == "mid"

    def test_missing_pubdate_kept_when_date_filter_on(self, caplog):
        """Items without pubDate stay in the set when date filters are active."""
        feed = RssFeed(
            title="T",
            items=[
                _rss_item("no_date"),
                _rss_item("dated", "Mon, 15 Jul 2024 12:00:00 GMT"),
            ],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(episode_since=date(2024, 1, 1))
        with caplog.at_level("WARNING"):
            episodes = prepare_episodes_from_feed(feed, cfg)
        assert len(episodes) == 2
        titles = {e.title for e in episodes}
        assert titles == {"no_date", "dated"}
        assert "no parseable pubDate" in caplog.text

    def test_prepare_with_real_config_defaults(self):
        """Real Config uses defaults compatible with prepare_episodes_from_feed."""
        feed = RssFeed(
            title="T",
            items=[_rss_item("x"), _rss_item("y")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = Config(rss_url="https://example.com/feed.xml", max_episodes=1)
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert len(episodes) == 1
        assert episodes[0].title == "x"


@pytest.mark.unit
class TestPublishedDateForEpisodeFilter:
    """Tests for published_date_for_episode_filter (GitHub #521)."""

    def test_utc_aware_uses_utc_date(self):
        """Timezone-aware pubDate maps to UTC calendar date."""
        item = safe_item_fromstring(
            "<item><pubDate>Wed, 01 Jan 2025 23:00:00 +0000</pubDate></item>"
        )
        assert published_date_for_episode_filter(item) == date(2025, 1, 1)
