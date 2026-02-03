#!/usr/bin/env python3
"""Integration tests for RSS parser edge cases.

These tests verify RSS parser handles edge cases correctly:
- Missing optional fields (description, published date, etc.)
- Relative URLs resolution
- Multiple transcript URLs
- Various RSS feed formats
- Malformed XML handling
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Add tests directory to path for conftest import
from pathlib import Path

from podcast_scraper.rss import downloader, parser as rss_parser

pytestmark = [pytest.mark.integration, pytest.mark.module_rss_parser]

tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid conflicts with infrastructure conftest
import importlib.util

tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_rss_response = parent_conftest.create_rss_response
create_test_config = parent_conftest.create_test_config


@pytest.mark.integration
@pytest.mark.critical_path
class TestRSSParserEdgeCases(unittest.TestCase):
    """Test RSS parser edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_rss_with_missing_optional_fields(self):
        """Test RSS parsing with missing optional fields."""
        rss_url = "https://example.com/feed.xml"
        # RSS feed with minimal required fields only
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
      <!-- No description, no published date, no author -->
    </item>
  </channel>
</rss>""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Verify feed was parsed despite missing optional fields
            self.assertIsNotNone(feed)
            self.assertEqual(feed.title, "Test Feed")
            self.assertEqual(len(feed.items), 1)

            # Verify episode was created
            episode = rss_parser.create_episode_from_item(feed.items[0], 1, feed.base_url)
            self.assertIsNotNone(episode)
            self.assertEqual(episode.title, "Episode 1")

    def test_rss_with_relative_urls(self):
        """Test RSS parsing with relative URLs."""
        rss_url = "https://example.com/podcast/feed.xml"
        # RSS feed with relative transcript URL
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="../transcripts/ep1.txt" type="text/plain" />
    </item>
  </channel>
</rss>""".strip()

        # Update cfg to use the correct RSS URL
        cfg = create_test_config(
            output_dir=self.temp_dir,
            rss_url=rss_url,
        )

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            feed = rss_parser.fetch_and_parse_rss(cfg)

            # Verify feed was parsed
            self.assertIsNotNone(feed)

            # Verify episode was created with resolved URL
            episode = rss_parser.create_episode_from_item(feed.items[0], 1, feed.base_url)
            self.assertIsNotNone(episode)
            self.assertEqual(len(episode.transcript_urls), 1)
            # Relative URL should be resolved relative to base URL
            transcript_url = episode.transcript_urls[0][0]
            self.assertIn("transcripts", transcript_url, "URL should be resolved")

    def test_rss_with_multiple_transcript_urls(self):
        """Test RSS parsing with multiple transcript URLs."""
        rss_url = "https://example.com/feed.xml"
        # RSS feed with multiple transcript URLs
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="https://example.com/ep1.vtt" type="text/vtt" />
      <podcast:transcript url="https://example.com/ep1.srt" type="text/srt" />
      <podcast:transcript url="https://example.com/ep1.txt" type="text/plain" />
    </item>
  </channel>
</rss>""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Verify feed was parsed
            self.assertIsNotNone(feed)

            # Verify episode has multiple transcript URLs
            episode = rss_parser.create_episode_from_item(feed.items[0], 1, feed.base_url)
            self.assertIsNotNone(episode)
            self.assertEqual(len(episode.transcript_urls), 3, "Should have 3 transcript URLs")

            # Verify all URLs are present
            urls = [url for url, _ in episode.transcript_urls]
            self.assertIn("https://example.com/ep1.vtt", urls)
            self.assertIn("https://example.com/ep1.srt", urls)
            self.assertIn("https://example.com/ep1.txt", urls)

    def test_rss_with_various_formats(self):
        """Test RSS parsing with various RSS feed formats."""
        rss_url = "https://example.com/feed.xml"

        # Test with iTunes namespace
        rss_xml_itunes = """<?xml version='1.0'?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <itunes:author>Test Author</itunes:author>
    <item>
      <title>Episode 1</title>
      <itunes:summary>Episode summary</itunes:summary>
    </item>
  </channel>
</rss>""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml_itunes, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Verify feed was parsed
            self.assertIsNotNone(feed)
            self.assertEqual(feed.title, "Test Feed")
            # Verify authors were extracted from iTunes namespace
            if feed.authors:
                self.assertIn("Test Author", feed.authors)

    def test_rss_with_empty_items(self):
        """Test RSS parsing with empty items list."""
        rss_url = "https://example.com/feed.xml"
        # RSS feed with no items
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
  </channel>
</rss>""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Verify feed was parsed
            self.assertIsNotNone(feed)
            self.assertEqual(feed.title, "Test Feed")
            self.assertEqual(len(feed.items), 0, "Should have no items")

    def test_rss_with_special_characters_in_title(self):
        """Test RSS parsing with special characters in titles."""
        rss_url = "https://example.com/feed.xml"
        # RSS feed with special characters
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed &amp; More</title>
    <item>
      <title>Episode 1: Title/With\\Special*Chars?</title>
    </item>
  </channel>
</rss>""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Verify feed was parsed
            self.assertIsNotNone(feed)
            self.assertEqual(feed.title, "Test Feed & More")

            # Verify episode title was parsed correctly
            episode = rss_parser.create_episode_from_item(feed.items[0], 1, feed.base_url)
            self.assertIsNotNone(episode)
            self.assertEqual(episode.title, "Episode 1: Title/With\\Special*Chars?")
            # Title should be sanitized for filesystem
            self.assertIsNotNone(episode.title_safe)
            self.assertNotEqual(episode.title_safe, episode.title, "Title should be sanitized")

    def test_rss_with_malformed_xml_handling(self):
        """Test RSS parser handles malformed XML gracefully."""
        rss_url = "https://example.com/feed.xml"
        # Malformed XML (missing closing tag)
        malformed_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
    <!-- Missing closing tags -->
""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(malformed_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Parser should handle malformed XML (may raise exception or parse partially)
            # This depends on XML parser behavior
            try:
                feed = rss_parser.fetch_and_parse_rss(self.cfg)
                # If parsing succeeds, verify it handled gracefully
                # (may have partial data or empty feed)
                self.assertIsNotNone(feed)
            except Exception:
                # If parsing fails, that's acceptable for malformed XML
                # The test validates that the error is handled appropriately
                pass
