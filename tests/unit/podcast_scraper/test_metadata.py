#!/usr/bin/env python3
"""Tests for metadata generation functionality."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import tempfile
import unittest

# Bandit: tests construct safe XML elements
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock ML dependencies before importing modules that require them
# Unit tests run without ML dependencies installed
with patch.dict("sys.modules", {"spacy": MagicMock()}):
    from podcast_scraper import metadata, rss_parser

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


class TestMetadataIDGeneration(unittest.TestCase):
    """Tests for metadata ID generation functions."""

    def test_generate_feed_id(self):
        """Test feed ID generation from URL."""
        feed_url = "https://example.com/feed.xml"
        feed_id = metadata.generate_feed_id(feed_url)
        self.assertTrue(feed_id.startswith("sha256:"))
        self.assertEqual(len(feed_id), 71)  # "sha256:" + 64 hex chars

        # Same URL should generate same ID
        feed_id2 = metadata.generate_feed_id(feed_url)
        self.assertEqual(feed_id, feed_id2)

        # Different URL should generate different ID
        feed_id3 = metadata.generate_feed_id("https://example.com/other.xml")
        self.assertNotEqual(feed_id, feed_id3)

    def test_generate_feed_id_normalizes_url(self):
        """Test that feed ID generation normalizes URLs."""
        url1 = "https://example.com/feed.xml"
        url2 = "https://example.com/feed.xml/"
        url3 = "https://example.com/feed.xml?param=value"
        url4 = "HTTPS://EXAMPLE.COM/feed.xml"

        id1 = metadata.generate_feed_id(url1)
        id2 = metadata.generate_feed_id(url2)
        id3 = metadata.generate_feed_id(url3)
        id4 = metadata.generate_feed_id(url4)

        # All should generate same ID (normalized)
        self.assertEqual(id1, id2)
        self.assertEqual(id1, id3)
        self.assertEqual(id1, id4)

    def test_generate_episode_id_with_guid(self):
        """Test episode ID generation with RSS GUID."""
        guid = "https://example.com/episode/123"
        episode_id = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=guid,
        )
        self.assertEqual(episode_id, guid)

    def test_generate_episode_id_without_guid(self):
        """Test episode ID generation without GUID (hash-based)."""
        episode_id = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
        )
        self.assertTrue(episode_id.startswith("sha256:"))
        self.assertEqual(len(episode_id), 71)

        # Same inputs should generate same ID
        episode_id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
        )
        self.assertEqual(episode_id, episode_id2)

    def test_generate_episode_id_with_published_date(self):
        """Test episode ID generation includes published date."""
        from datetime import datetime

        published_date = datetime(2024, 1, 15, 10, 30, 0)
        episode_id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            published_date=published_date,
        )
        episode_id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            published_date=None,
        )
        # Different IDs when date is included vs not
        self.assertNotEqual(episode_id1, episode_id2)

    def test_generate_content_id(self):
        """Test content ID generation from URL."""
        content_url = "https://example.com/transcript.vtt"
        content_id = metadata.generate_content_id(content_url)
        self.assertTrue(content_id.startswith("sha256:"))
        self.assertEqual(len(content_id), 71)

        # Same URL should generate same ID
        content_id2 = metadata.generate_content_id(content_url)
        self.assertEqual(content_id, content_id2)

        # Different URL should generate different ID
        content_id3 = metadata.generate_content_id("https://example.com/other.vtt")
        self.assertNotEqual(content_id, content_id3)


class TestMetadataGeneration(unittest.TestCase):
    """Tests for metadata document generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.feed = create_test_feed()
        self.episode = create_test_episode()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # test_generate_metadata_json moved to tests/integration/test_metadata_integration.py
    # (performs real file I/O)

    # test_generate_metadata_yaml moved to tests/integration/test_metadata_integration.py
    # (performs real file I/O)

    def test_generate_metadata_disabled(self):
        """Test that metadata generation is skipped when disabled."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=False,
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
        )

        self.assertIsNone(metadata_path)

    def test_generate_metadata_dry_run(self):
        """Test metadata generation in dry-run mode."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            dry_run=True,
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
        )

        self.assertIsNotNone(metadata_path)
        # File should not exist in dry-run mode
        self.assertFalse(os.path.exists(metadata_path))

    # test_generate_metadata_skip_existing moved to
    # tests/integration/test_metadata_integration.py (performs real file I/O)

    # test_generate_metadata_with_subdirectory moved to
    # tests/integration/test_metadata_integration.py (performs real file I/O)

    # test_generate_metadata_with_whisper_transcription moved to
    # tests/integration/test_metadata_integration.py (performs real file I/O)

    # test_generate_metadata_with_run_suffix moved to tests/integration/test_metadata_integration.py
    # (performs real file I/O)

    # test_generate_metadata_minimal_data moved to tests/integration/test_metadata_integration.py
    # (performs real file I/O)

    # test_generate_metadata_transcript_urls moved to tests/integration/test_metadata_integration.py
    # (performs real file I/O)


class TestMetadataRSSExtraction(unittest.TestCase):
    """Tests for RSS metadata extraction functions."""

    def test_extract_feed_metadata(self):
        """Test feed metadata extraction from RSS XML."""
        xml_bytes = b"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <description>Test feed description</description>
                <image>
                    <url>https://example.com/image.png</url>
                </image>
                <lastBuildDate>Mon, 15 Jan 2024 10:30:00 GMT</lastBuildDate>
            </channel>
        </rss>"""

        description, image_url, last_updated = rss_parser.extract_feed_metadata(
            xml_bytes, "https://example.com"
        )

        self.assertEqual(description, "Test feed description")
        self.assertEqual(image_url, "https://example.com/image.png")
        self.assertIsNotNone(last_updated)

    def test_extract_episode_metadata(self):
        """Test episode metadata extraction from RSS item."""
        item = ET.Element("item")
        title_elem = ET.SubElement(item, "title")
        title_elem.text = "Test Episode"
        desc_elem = ET.SubElement(item, "description")
        desc_elem.text = "Episode description"
        guid_elem = ET.SubElement(item, "guid")
        guid_elem.text = "episode-123"
        link_elem = ET.SubElement(item, "link")
        link_elem.text = "https://example.com/episode/123"

        (
            description,
            guid,
            link,
            duration_seconds,
            episode_number,
            image_url,
        ) = rss_parser.extract_episode_metadata(item, "https://example.com")

        self.assertEqual(description, "Episode description")
        self.assertEqual(guid, "episode-123")
        self.assertEqual(link, "https://example.com/episode/123")

    def test_extract_episode_published_date(self):
        """Test episode published date extraction."""
        item = ET.Element("item")
        pub_date_elem = ET.SubElement(item, "pubDate")
        pub_date_elem.text = "Mon, 15 Jan 2024 10:30:00 GMT"

        published_date = rss_parser.extract_episode_published_date(item)

        self.assertIsNotNone(published_date)
        self.assertEqual(published_date.year, 2024)
        self.assertEqual(published_date.month, 1)
        self.assertEqual(published_date.day, 15)


class TestIDGenerationStability(unittest.TestCase):
    """Additional tests for ID generation stability and uniqueness."""

    def test_feed_id_stability_across_calls(self):
        """Test that feed IDs are stable across multiple calls."""
        url = "https://example.com/feed.xml"
        ids = [metadata.generate_feed_id(url) for _ in range(10)]
        # All IDs should be identical
        self.assertEqual(len(set(ids)), 1)

    def test_episode_id_stability_with_same_inputs(self):
        """Test that episode IDs are stable with same inputs."""
        from datetime import datetime

        published_date = datetime(2024, 1, 15, 10, 30, 0)
        ids = [
            metadata.generate_episode_id(
                feed_url=TEST_FEED_URL,
                episode_title="Test Episode",
                episode_guid=None,
                published_date=published_date,
            )
            for _ in range(10)
        ]
        # All IDs should be identical
        self.assertEqual(len(set(ids)), 1)

    def test_episode_id_uniqueness_different_titles(self):
        """Test that different episode titles generate different IDs."""
        id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Episode 1",
            episode_guid=None,
        )
        id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Episode 2",
            episode_guid=None,
        )
        self.assertNotEqual(id1, id2)

    def test_episode_id_uniqueness_different_feeds(self):
        """Test that same episode in different feeds generates different IDs."""
        id1 = metadata.generate_episode_id(
            feed_url="https://example.com/feed1.xml",
            episode_title="Test Episode",
            episode_guid=None,
        )
        id2 = metadata.generate_episode_id(
            feed_url="https://example.com/feed2.xml",
            episode_title="Test Episode",
            episode_guid=None,
        )
        self.assertNotEqual(id1, id2)

    def test_episode_id_guid_priority(self):
        """Test that GUID takes priority over hash-based generation."""
        guid = "unique-guid-12345"
        hash_id = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
        )
        guid_id = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=guid,
        )
        # GUID ID should be the GUID itself, not a hash
        self.assertEqual(guid_id, guid)
        self.assertNotEqual(guid_id, hash_id)

    def test_content_id_stability(self):
        """Test that content IDs are stable across multiple calls."""
        url = "https://example.com/transcript.vtt"
        ids = [metadata.generate_content_id(url) for _ in range(10)]
        # All IDs should be identical
        self.assertEqual(len(set(ids)), 1)

    def test_content_id_uniqueness(self):
        """Test that different content URLs generate different IDs."""
        id1 = metadata.generate_content_id("https://example.com/transcript1.vtt")
        id2 = metadata.generate_content_id("https://example.com/transcript2.vtt")
        self.assertNotEqual(id1, id2)

    def test_episode_id_with_link(self):
        """Test that episode link affects ID generation."""
        id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_link="https://example.com/ep1",
        )
        id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_link="https://example.com/ep2",
        )
        self.assertNotEqual(id1, id2)

    def test_episode_id_link_normalization(self):
        """Test that episode link normalization produces same ID."""
        id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_link="https://example.com/episode",
        )
        id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_link="https://example.com/episode/",
        )
        # Should be same after normalization
        self.assertEqual(id1, id2)


# TestMetadataGenerationComprehensive moved to tests/integration/test_metadata_integration.py
# These tests perform real file I/O and are better suited as integration tests.
