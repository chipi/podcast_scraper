"""Unit tests for ``feed_list_text``."""

from __future__ import annotations

from podcast_scraper.feed_list_text import parse_rss_list_file_lines


def test_parse_skips_blank_and_hash_comments() -> None:
    text = "\n  https://a/rss  \n# skip\n\nhttps://b/rss\n"
    assert parse_rss_list_file_lines(text) == ["https://a/rss", "https://b/rss"]


def test_parse_preserves_duplicates() -> None:
    text = "https://x/rss\nhttps://x/rss\n"
    assert parse_rss_list_file_lines(text) == ["https://x/rss", "https://x/rss"]
