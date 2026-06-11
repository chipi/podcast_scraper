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
    collect_existing_guids,
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
    # #876: default OFF so a bare MagicMock cfg doesn't accidentally trip the
    # existing-only migration branch (MagicMock attrs are otherwise truthy).
    c.reprocess_existing_only = False
    c.output_dir = None
    for key, val in kwargs.items():
        setattr(c, key, val)
    return c


def _rss_item(title: str, pub_date: str | None = None, guid: str | None = None):
    inner = f"<title>{title}</title>"
    if pub_date is not None:
        inner += f"<pubDate>{pub_date}</pubDate>"
    if guid is not None:
        inner += f"<guid>{guid}</guid>"
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


def _write_meta(path, guid):
    """Write a minimal episode.metadata.json with the given guid."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"episode": {"guid": guid}}), encoding="utf-8")


@pytest.mark.unit
class TestCollectExistingGuids:
    """Tests for collect_existing_guids (#876)."""

    def test_dedupes_across_run_dirs(self, tmp_path):
        _write_meta(tmp_path / "run_A" / "metadata" / "a.metadata.json", "g1")
        _write_meta(tmp_path / "run_B" / "metadata" / "b.metadata.json", "g2")
        _write_meta(tmp_path / "run_B" / "metadata" / "c.metadata.json", "g1")  # dup
        assert collect_existing_guids(str(tmp_path)) == {"g1", "g2"}

    def test_skips_corrupt_and_guidless(self, tmp_path):
        _write_meta(tmp_path / "run_A" / "metadata" / "good.metadata.json", "g1")
        bad = tmp_path / "run_A" / "metadata" / "bad.metadata.json"
        bad.write_text("{not json", encoding="utf-8")
        _write_meta(tmp_path / "run_A" / "metadata" / "empty.metadata.json", "")
        (tmp_path / "run_A" / "metadata" / "noguid.metadata.json").write_text(
            '{"episode": {}}', encoding="utf-8"
        )
        assert collect_existing_guids(str(tmp_path)) == {"g1"}

    def test_empty_dir_returns_empty_set(self, tmp_path):
        assert collect_existing_guids(str(tmp_path)) == set()


@pytest.mark.unit
class TestPrepareEpisodesExistingOnly:
    """#876 strict existing-only migration selection."""

    def _corpus(self, tmp_path, *guids):
        for i, g in enumerate(guids):
            _write_meta(tmp_path / "run_A" / "metadata" / f"{i}.metadata.json", g)
        return tmp_path

    def test_keeps_only_existing_drops_new_and_guidless(self, tmp_path):
        self._corpus(tmp_path, "g1", "g2")
        feed = RssFeed(
            title="T",
            items=[
                _rss_item("keep1", guid="g1"),
                _rss_item("new", guid="g3"),
                _rss_item("keep2", guid="g2"),
                _rss_item("noguid"),
            ],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(reprocess_existing_only=True, output_dir=str(tmp_path))
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert sorted(e.title for e in episodes) == ["keep1", "keep2"]

    def test_ignores_max_episodes_cap(self, tmp_path):
        self._corpus(tmp_path, "g1", "g2")
        feed = RssFeed(
            title="T",
            items=[_rss_item("a", guid="g1"), _rss_item("b", guid="g2")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(reprocess_existing_only=True, output_dir=str(tmp_path), max_episodes=1)
        episodes = prepare_episodes_from_feed(feed, cfg)
        assert len(episodes) == 2  # cap ignored in migration mode

    def test_rolled_off_existing_is_logged(self, tmp_path, caplog):
        import logging

        self._corpus(tmp_path, "g1", "g2", "g_rolled")
        feed = RssFeed(
            title="T",
            items=[_rss_item("a", guid="g1"), _rss_item("b", guid="g2")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(reprocess_existing_only=True, output_dir=str(tmp_path))
        with caplog.at_level(logging.INFO):
            prepare_episodes_from_feed(feed, cfg)
        assert "1 on-disk GUID(s) not in the live feed" in caplog.text

    def test_empty_corpus_aborts_loud(self, tmp_path):
        feed = RssFeed(
            title="T",
            items=[_rss_item("a", guid="g1")],
            base_url="https://example.com",
            authors=[],
        )
        cfg = _scraping_cfg(reprocess_existing_only=True, output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="no on-disk episode GUIDs"):
            prepare_episodes_from_feed(feed, cfg)
