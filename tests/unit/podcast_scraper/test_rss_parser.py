#!/usr/bin/env python3
"""Tests for RSS parsing functionality."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import sys
import unittest

# Bandit: tests construct safe XML elements
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path

from podcast_scraper import rss_parser

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

# Import helper functions from parent conftest
build_rss_xml_with_media = parent_conftest.build_rss_xml_with_media
build_rss_xml_with_speakers = parent_conftest.build_rss_xml_with_speakers
build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_media_response = parent_conftest.create_media_response
create_mock_spacy_model = parent_conftest.create_mock_spacy_model
create_rss_response = parent_conftest.create_rss_response
create_test_args = parent_conftest.create_test_args
create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
create_transcript_response = parent_conftest.create_transcript_response
MockHTTPResponse = parent_conftest.MockHTTPResponse
TEST_BASE_URL = parent_conftest.TEST_BASE_URL
TEST_CONTENT_TYPE_SRT = parent_conftest.TEST_CONTENT_TYPE_SRT
TEST_CONTENT_TYPE_VTT = parent_conftest.TEST_CONTENT_TYPE_VTT
TEST_CUSTOM_OUTPUT_DIR = parent_conftest.TEST_CUSTOM_OUTPUT_DIR
TEST_EPISODE_TITLE = parent_conftest.TEST_EPISODE_TITLE
TEST_EPISODE_TITLE_SPECIAL = parent_conftest.TEST_EPISODE_TITLE_SPECIAL
TEST_FEED_TITLE = parent_conftest.TEST_FEED_TITLE
TEST_FEED_URL = parent_conftest.TEST_FEED_URL
TEST_FULL_URL = parent_conftest.TEST_FULL_URL
TEST_MEDIA_TYPE_M4A = parent_conftest.TEST_MEDIA_TYPE_M4A
TEST_MEDIA_TYPE_MP3 = parent_conftest.TEST_MEDIA_TYPE_MP3
TEST_MEDIA_URL = parent_conftest.TEST_MEDIA_URL
TEST_OUTPUT_DIR = parent_conftest.TEST_OUTPUT_DIR
TEST_PATH = parent_conftest.TEST_PATH
TEST_RELATIVE_MEDIA = parent_conftest.TEST_RELATIVE_MEDIA
TEST_RELATIVE_TRANSCRIPT = parent_conftest.TEST_RELATIVE_TRANSCRIPT
TEST_RUN_ID = parent_conftest.TEST_RUN_ID
TEST_TRANSCRIPT_TYPE_SRT = parent_conftest.TEST_TRANSCRIPT_TYPE_SRT
TEST_TRANSCRIPT_TYPE_VTT = parent_conftest.TEST_TRANSCRIPT_TYPE_VTT
TEST_TRANSCRIPT_URL = parent_conftest.TEST_TRANSCRIPT_URL
TEST_TRANSCRIPT_URL_SRT = parent_conftest.TEST_TRANSCRIPT_URL_SRT


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


class TestExtractFeedMetadata(unittest.TestCase):
    """Tests for extract_feed_metadata function."""

    def test_extract_feed_metadata_success(self):
        """Test extracting feed metadata successfully."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <description>Feed description</description>
                <image>
                    <url>https://example.com/image.jpg</url>
                </image>
                <lastBuildDate>Mon, 01 Jan 2024 12:00:00 GMT</lastBuildDate>
            </channel>
        </rss>""".encode()
        description, image_url, last_updated = rss_parser.extract_feed_metadata(
            xml_bytes, TEST_FEED_URL
        )
        self.assertEqual(description, "Feed description")
        self.assertIsNotNone(image_url)
        self.assertIsNotNone(last_updated)

    def test_extract_feed_metadata_no_channel(self):
        """Test extracting feed metadata when no channel element."""
        xml_bytes = """<?xml version="1.0"?>
        <rss version="2.0">
        </rss>""".encode()
        description, image_url, last_updated = rss_parser.extract_feed_metadata(
            xml_bytes, TEST_FEED_URL
        )
        self.assertIsNone(description)
        self.assertIsNone(image_url)
        self.assertIsNone(last_updated)

    def test_extract_feed_metadata_itunes_image(self):
        """Test extracting iTunes image URL."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <itunes:image href="https://example.com/itunes-image.jpg"/>
            </channel>
        </rss>""".encode()
        description, image_url, last_updated = rss_parser.extract_feed_metadata(
            xml_bytes, TEST_FEED_URL
        )
        self.assertIn("itunes-image.jpg", image_url)

    def test_extract_feed_metadata_invalid_date(self):
        """Test extracting feed metadata with invalid date format."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <lastBuildDate>Invalid Date Format</lastBuildDate>
            </channel>
        </rss>""".encode()
        description, image_url, last_updated = rss_parser.extract_feed_metadata(
            xml_bytes, TEST_FEED_URL
        )
        # Should handle invalid date gracefully
        self.assertIsNone(last_updated)

    def test_extract_feed_metadata_atom_updated(self):
        """Test extracting Atom updated date."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <atom:updated>2024-01-01T12:00:00Z</atom:updated>
            </channel>
        </rss>""".encode()
        description, image_url, last_updated = rss_parser.extract_feed_metadata(
            xml_bytes, TEST_FEED_URL
        )
        self.assertIsNotNone(last_updated)


class TestExtractDurationSeconds(unittest.TestCase):
    """Tests for _extract_duration_seconds function."""

    def test_extract_duration_hhmmss(self):
        """Test extracting duration in HH:MM:SS format."""
        item = ET.Element("item")
        duration = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
        duration.text = "01:30:45"
        result = rss_parser._extract_duration_seconds(item)
        self.assertEqual(result, 5445)  # 1*3600 + 30*60 + 45

    def test_extract_duration_mmss(self):
        """Test extracting duration in MM:SS format."""
        item = ET.Element("item")
        duration = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
        duration.text = "30:45"
        result = rss_parser._extract_duration_seconds(item)
        self.assertEqual(result, 1845)  # 30*60 + 45

    def test_extract_duration_ss(self):
        """Test extracting duration in SS format."""
        item = ET.Element("item")
        duration = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
        duration.text = "45"
        result = rss_parser._extract_duration_seconds(item)
        self.assertEqual(result, 45)

    def test_extract_duration_invalid_format(self):
        """Test extracting duration with invalid format."""
        item = ET.Element("item")
        duration = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
        duration.text = "invalid"
        result = rss_parser._extract_duration_seconds(item)
        self.assertIsNone(result)

    def test_extract_duration_missing(self):
        """Test extracting duration when missing."""
        item = ET.Element("item")
        result = rss_parser._extract_duration_seconds(item)
        self.assertIsNone(result)


class TestExtractEpisodeNumber(unittest.TestCase):
    """Tests for _extract_episode_number function."""

    def test_extract_episode_number_success(self):
        """Test extracting episode number successfully."""
        item = ET.Element("item")
        episode = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}episode")
        episode.text = "42"
        result = rss_parser._extract_episode_number(item)
        self.assertEqual(result, 42)

    def test_extract_episode_number_invalid(self):
        """Test extracting episode number with invalid value."""
        item = ET.Element("item")
        episode = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}episode")
        episode.text = "invalid"
        result = rss_parser._extract_episode_number(item)
        self.assertIsNone(result)

    def test_extract_episode_number_missing(self):
        """Test extracting episode number when missing."""
        item = ET.Element("item")
        result = rss_parser._extract_episode_number(item)
        self.assertIsNone(result)


class TestExtractImageUrl(unittest.TestCase):
    """Tests for _extract_image_url function."""

    def test_extract_image_url_success(self):
        """Test extracting image URL successfully."""
        item = ET.Element("item")
        image = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        image.set("href", "https://example.com/image.jpg")
        result = rss_parser._extract_image_url(item, TEST_FEED_URL)
        self.assertEqual(result, "https://example.com/image.jpg")

    def test_extract_image_url_relative(self):
        """Test extracting relative image URL."""
        item = ET.Element("item")
        image = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        image.set("href", "image.jpg")
        result = rss_parser._extract_image_url(item, TEST_FEED_URL)
        self.assertIsNotNone(result)
        self.assertIn("image.jpg", result)

    def test_extract_image_url_missing(self):
        """Test extracting image URL when missing."""
        item = ET.Element("item")
        result = rss_parser._extract_image_url(item, TEST_FEED_URL)
        self.assertIsNone(result)


class TestExtractEpisodeMetadata(unittest.TestCase):
    """Tests for extract_episode_metadata function."""

    def test_extract_episode_metadata_success(self):
        """Test extracting episode metadata successfully."""
        item = ET.Element("item")
        title = ET.SubElement(item, "title")
        title.text = TEST_EPISODE_TITLE
        description = ET.SubElement(item, "description")
        description.text = "Episode description"
        guid = ET.SubElement(item, "guid")
        guid.text = "episode-1"
        link = ET.SubElement(item, "link")
        link.text = "https://example.com/episode-1"
        duration = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
        duration.text = "30:00"
        episode = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}episode")
        episode.text = "1"
        image = ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        image.set("href", "https://example.com/episode-image.jpg")

        result = rss_parser.extract_episode_metadata(item, TEST_FEED_URL)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], "Episode description")  # description
        self.assertEqual(result[1], "episode-1")  # guid
        self.assertEqual(result[2], "https://example.com/episode-1")  # link
        self.assertEqual(result[3], 1800)  # duration_seconds
        self.assertEqual(result[4], 1)  # episode_number
        self.assertIn("episode-image.jpg", result[5])  # image_url

    def test_extract_episode_metadata_fallback_to_summary(self):
        """Test extracting episode metadata falls back to summary."""
        item = ET.Element("item")
        title = ET.SubElement(item, "title")
        title.text = TEST_EPISODE_TITLE
        summary = ET.SubElement(item, "summary")
        summary.text = "Episode summary"

        result = rss_parser.extract_episode_metadata(item, TEST_FEED_URL)
        self.assertEqual(result[0], "Episode summary")  # description from summary

    def test_extract_episode_metadata_empty(self):
        """Test extracting episode metadata when empty."""
        item = ET.Element("item")
        result = rss_parser.extract_episode_metadata(item, TEST_FEED_URL)
        self.assertEqual(len(result), 6)
        # All should be None
        for value in result:
            self.assertIsNone(value)


class TestExtractEpisodePublishedDate(unittest.TestCase):
    """Tests for extract_episode_published_date function."""

    def test_extract_published_date_success(self):
        """Test extracting published date successfully."""
        item = ET.Element("item")
        pub_date = ET.SubElement(item, "pubDate")
        pub_date.text = "Mon, 01 Jan 2024 12:00:00 GMT"

        result = rss_parser.extract_episode_published_date(item)
        self.assertIsNotNone(result)

    def test_extract_published_date_invalid(self):
        """Test extracting published date with invalid format."""
        item = ET.Element("item")
        pub_date = ET.SubElement(item, "pubDate")
        pub_date.text = "Invalid Date"

        result = rss_parser.extract_episode_published_date(item)
        self.assertIsNone(result)

    def test_extract_published_date_missing(self):
        """Test extracting published date when missing."""
        item = ET.Element("item")
        result = rss_parser.extract_episode_published_date(item)
        self.assertIsNone(result)


class TestExtractEpisodeDescription(unittest.TestCase):
    """Tests for extract_episode_description function."""

    def test_extract_description_success(self):
        """Test extracting episode description successfully."""
        item = ET.Element("item")
        description = ET.SubElement(item, "description")
        description.text = "<p>Episode description</p>"

        result = rss_parser.extract_episode_description(item)
        self.assertEqual(result, "Episode description")  # HTML stripped

    def test_extract_description_with_html(self):
        """Test extracting description with HTML tags."""
        item = ET.Element("item")
        description = ET.SubElement(item, "description")
        description.text = "<p>Episode description</p>"

        result = rss_parser.extract_episode_description(item)
        # HTML should be stripped
        self.assertIn("Episode description", result)

    def test_extract_description_missing(self):
        """Test extracting description when missing."""
        item = ET.Element("item")
        result = rss_parser.extract_episode_description(item)
        self.assertIsNone(result)


class TestMalformedRSS(unittest.TestCase):
    """Tests for handling malformed RSS feeds."""

    def test_parse_rss_items_malformed_xml(self):
        """Test parsing malformed XML."""
        xml_bytes = (
            b"<?xml version='1.0'?><rss><channel><title>Test</title>"
            b"<item><title>Episode</title></item></channel></rss>"
        )
        # Should not raise, but may return empty or partial results
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertIsInstance(title, str)
        self.assertIsInstance(authors, list)
        self.assertIsInstance(items, list)

    def test_parse_rss_items_no_channel(self):
        """Test parsing RSS with no channel element."""
        xml_bytes = b"<?xml version='1.0'?><rss version='2.0'></rss>"
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, "")
        self.assertEqual(len(authors), 0)
        self.assertIsInstance(items, list)

    def test_parse_rss_items_no_items(self):
        """Test parsing RSS with no items."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
            </channel>
        </rss>""".encode()
        title, authors, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        self.assertEqual(len(items), 0)

    def test_find_transcript_urls_malformed(self):
        """Test finding transcript URLs in malformed item."""
        item = ET.Element("item")
        # Missing url attribute
        transcript = ET.SubElement(item, "podcast:transcript")
        transcript.set("type", TEST_TRANSCRIPT_TYPE_VTT)

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        # Should handle gracefully
        self.assertIsInstance(result, list)

    def test_find_enclosure_media_malformed(self):
        """Test finding enclosure media in malformed item."""
        item = ET.Element("item")
        # Missing url attribute
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("type", TEST_MEDIA_TYPE_MP3)

        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        # Should return None when url is missing
        self.assertIsNone(result)
