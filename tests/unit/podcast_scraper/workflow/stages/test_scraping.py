"""Unit tests for podcast_scraper.workflow.stages.scraping module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.workflow.stages.scraping import (
    extract_feed_metadata_for_generation,
    prepare_episodes_from_feed,
)
from podcast_scraper.workflow.types import FeedMetadata


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
        feed = MagicMock()
        feed.items = []
        feed.base_url = "https://example.com"
        cfg = MagicMock()
        cfg.max_episodes = None
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert episodes == []

    def test_respects_max_episodes(self):
        """When max_episodes is set, only that many items are passed to create_episode."""
        from unittest.mock import patch

        from podcast_scraper.models import Episode

        mock_ep1 = MagicMock(spec=Episode)
        mock_ep1.title = "E1"
        with patch(
            "podcast_scraper.workflow.stages.scraping.create_episode_from_item",
            side_effect=[mock_ep1],
        ):
            feed = MagicMock()
            feed.items = [MagicMock(), MagicMock()]
            feed.base_url = "https://example.com"
            cfg = MagicMock()
            cfg.max_episodes = 1
            episodes = prepare_episodes_from_feed(feed, cfg)
            assert len(episodes) == 1
            assert episodes[0].title == "E1"
