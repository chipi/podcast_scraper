"""Tests for optional RSS feed XML disk cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.rss import feed_cache


def test_cache_dir_from_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(feed_cache.ENV_RSS_CACHE_DIR, raising=False)
    assert feed_cache.cache_dir_from_env() is None


def test_cache_dir_from_env_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(feed_cache.ENV_RSS_CACHE_DIR, str(tmp_path))
    got = feed_cache.cache_dir_from_env()
    assert got is not None
    assert got == tmp_path.resolve()


def test_read_write_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(feed_cache.ENV_RSS_CACHE_DIR, str(tmp_path))
    url = "https://example.com/podcast.xml"
    body = b"<rss><channel><title>t</title></channel></rss>"
    assert feed_cache.read_cached_rss(url) is None
    feed_cache.write_cached_rss(url, body)
    assert feed_cache.read_cached_rss(url) == body


def test_cache_path_stable_for_same_url(tmp_path: Path) -> None:
    """Same URL must always resolve to the same cache filename."""
    url = "https://example.com/feed.xml"
    p1 = feed_cache.cache_path_for_url(tmp_path, url)
    p2 = feed_cache.cache_path_for_url(tmp_path, url)
    assert p1 == p2
    assert p1.name.startswith("rss_")
    assert p1.suffix == ".xml"
