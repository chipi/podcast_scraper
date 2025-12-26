#!/usr/bin/env python3
"""Tests for RSS parsing functionality."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import sys
import unittest

# Bandit: tests construct safe XML elements
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path

from podcast_scraper import rss_parser

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: F401, E402
    build_rss_xml_with_media,
    build_rss_xml_with_speakers,
    build_rss_xml_with_transcript,
    create_media_response,
    create_mock_spacy_model,
    create_rss_response,
    create_test_args,
    create_test_config,
    create_test_episode,
    create_test_feed,
    create_transcript_response,
    MockHTTPResponse,
    TEST_BASE_URL,
    TEST_CONTENT_TYPE_SRT,
    TEST_CONTENT_TYPE_VTT,
    TEST_CUSTOM_OUTPUT_DIR,
    TEST_EPISODE_TITLE,
    TEST_EPISODE_TITLE_SPECIAL,
    TEST_FEED_TITLE,
    TEST_FEED_URL,
    TEST_FULL_URL,
    TEST_MEDIA_TYPE_M4A,
    TEST_MEDIA_TYPE_MP3,
    TEST_MEDIA_URL,
    TEST_OUTPUT_DIR,
    TEST_PATH,
    TEST_RELATIVE_MEDIA,
    TEST_RELATIVE_TRANSCRIPT,
    TEST_RUN_ID,
    TEST_TRANSCRIPT_TYPE_SRT,
    TEST_TRANSCRIPT_TYPE_VTT,
    TEST_TRANSCRIPT_URL,
    TEST_TRANSCRIPT_URL_SRT,
)


class TestParseRSSItems(unittest.TestCase):
    """Tests for parse_rss_items function."""

    def test_parse_simple_rss(self):
        """Test parsing a simple RSS feed."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <item>
                    <title>Episode 1</title>
                </item>
                <item>
                    <title>Episode 2</title>
                </item>
            </channel>
        </rss>""".encode()
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        self.assertEqual(len(items), 2)

    def test_parse_rss_with_namespace(self):
        """Test parsing RSS feed with namespace."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <item>
                    <title>Episode 1</title>
                </item>
            </channel>
        </rss>""".encode()
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        self.assertEqual(len(authors), 0)
        self.assertEqual(len(items), 1)

    def test_parse_rss_with_author_tags(self):
        """Test parsing RSS feed with author tags."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <author>John Doe</author>
                <itunes:author>Bob Johnson</itunes:author>
                <itunes:owner>
                    <itunes:name>Jane Smith</itunes:name>
                    <itunes:email>jane@example.com</itunes:email>
                </itunes:owner>
                <item>
                    <title>Episode 1</title>
                </item>
            </channel>
        </rss>""".encode()
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        self.assertEqual(len(authors), 3)
        self.assertIn("John Doe", authors)
        self.assertIn("Bob Johnson", authors)
        self.assertIn("Jane Smith", authors)
        self.assertEqual(len(items), 1)

    def test_parse_rss_author_channel_level_only(self):
        """Test that author tags are only extracted from channel level, not item level."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <author>Channel Author</author>
                <item>
                    <title>Episode 1</title>
                    <author>Item Author</author>
                </item>
            </channel>
        </rss>""".encode()
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        # Should only have channel-level author, not item-level
        self.assertEqual(len(authors), 1)
        self.assertIn("Channel Author", authors)
        self.assertNotIn("Item Author", authors)
        self.assertEqual(len(items), 1)

    def test_parse_rss_with_author_email(self):
        """Test parsing RSS feed with author tag containing email."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <author>John Doe &lt;john@example.com&gt;</author>
                <item>
                    <title>Episode 1</title>
                </item>
            </channel>
        </rss>""".encode()
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        # Email should be extracted but we'll handle cleaning in detect_hosts_from_feed
        self.assertEqual(len(authors), 1)
        self.assertEqual(len(items), 1)


class TestFindTranscriptURLs(unittest.TestCase):
    """Tests for find_transcript_urls function."""

    def test_find_podcast_transcript(self):
        """Test finding podcast:transcript URLs."""
        item = ET.Element("item")
        transcript = ET.SubElement(item, "podcast:transcript")
        transcript.set("url", TEST_TRANSCRIPT_URL)
        transcript.set("type", TEST_TRANSCRIPT_TYPE_VTT)

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], TEST_TRANSCRIPT_URL)
        self.assertEqual(result[0][1], TEST_TRANSCRIPT_TYPE_VTT)

    def test_find_relative_transcript_url(self):
        """Test finding relative transcript URLs."""
        item = ET.Element("item")
        transcript = ET.SubElement(item, "transcript")
        transcript.text = TEST_RELATIVE_TRANSCRIPT

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], f"{TEST_BASE_URL}/{TEST_RELATIVE_TRANSCRIPT}")

    def test_find_multiple_transcripts(self):
        """Test finding multiple transcript URLs."""
        item = ET.Element("item")
        transcript1 = ET.SubElement(item, "podcast:transcript")
        transcript1.set("url", TEST_TRANSCRIPT_URL)
        transcript2 = ET.SubElement(item, "podcast:transcript")
        transcript2.set("url", TEST_TRANSCRIPT_URL_SRT)

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        self.assertEqual(len(result), 2)


class TestFindEnclosureMedia(unittest.TestCase):
    """Tests for find_enclosure_media function."""

    def test_find_enclosure(self):
        """Test finding enclosure media."""
        item = ET.Element("item")
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", TEST_MEDIA_URL)
        enclosure.set("type", TEST_MEDIA_TYPE_MP3)

        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], TEST_MEDIA_URL)
        self.assertEqual(result[1], TEST_MEDIA_TYPE_MP3)

    def test_find_relative_enclosure_url(self):
        """Test finding relative enclosure URLs."""
        item = ET.Element("item")
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", TEST_RELATIVE_MEDIA)
        enclosure.set("type", TEST_MEDIA_TYPE_MP3)

        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], f"{TEST_BASE_URL}/{TEST_RELATIVE_MEDIA}")

    def test_no_enclosure(self):
        """Test when no enclosure is found."""
        item = ET.Element("item")
        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        self.assertIsNone(result)


class TestChooseTranscriptURL(unittest.TestCase):
    """Tests for choose_transcript_url function."""

    def test_choose_first_when_no_preference(self):
        """Test choosing first URL when no preference specified."""
        candidates = [
            (TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT),
            (TEST_TRANSCRIPT_URL_SRT, TEST_TRANSCRIPT_TYPE_SRT),
        ]
        result = rss_parser.choose_transcript_url(candidates, [])
        self.assertEqual(result, candidates[0])

    def test_choose_by_preference(self):
        """Test choosing URL by preferred type."""
        candidates = [
            (TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT),
            (TEST_TRANSCRIPT_URL_SRT, TEST_TRANSCRIPT_TYPE_SRT),
        ]
        result = rss_parser.choose_transcript_url(candidates, [TEST_TRANSCRIPT_TYPE_SRT])
        self.assertEqual(result, candidates[1])

    def test_empty_candidates(self):
        """Test with empty candidates list."""
        result = rss_parser.choose_transcript_url([], [])
        self.assertIsNone(result)


class TestExtractEpisodeTitle(unittest.TestCase):
    """Tests for extract_episode_title function."""

    def test_extract_title(self):
        """Test extracting episode title."""
        item = ET.Element("item")
        title = ET.SubElement(item, "title")
        title.text = TEST_EPISODE_TITLE

        result_title, result_safe = rss_parser.extract_episode_title(item, 1)
        self.assertEqual(result_title, TEST_EPISODE_TITLE)
        # sanitize_filename preserves spaces, only replaces special chars
        self.assertEqual(result_safe, TEST_EPISODE_TITLE)

    def test_fallback_to_episode_number(self):
        """Test fallback to episode number when no title."""
        item = ET.Element("item")
        result_title, result_safe = rss_parser.extract_episode_title(item, 5)
        self.assertEqual(result_title, "episode_5")
        self.assertEqual(result_safe, "episode_5")
