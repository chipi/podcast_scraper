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
from unittest.mock import MagicMock, Mock, patch

# Mock ML dependencies before importing modules that require them
# Unit tests run without ML dependencies installed
with patch.dict("sys.modules", {"spacy": MagicMock()}):
    from podcast_scraper import config, metadata, rss_parser

# Import from parent conftest explicitly to avoid conflicts
import importlib.util

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
        # Should be different when published_date is included
        self.assertNotEqual(episode_id1, episode_id2)

    def test_generate_episode_id_with_episode_link(self):
        """Test episode ID generation includes episode link."""
        episode_id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_link="https://example.com/episode1",
        )
        episode_id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_link=None,
        )
        # Should be different when episode_link is included
        self.assertNotEqual(episode_id1, episode_id2)

    def test_generate_episode_id_normalizes_guid_whitespace(self):
        """Test episode ID generation strips whitespace from GUID."""
        guid = "  https://example.com/episode/123  "
        episode_id = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=guid,
        )
        self.assertEqual(episode_id, "https://example.com/episode/123")

    def test_generate_episode_id_with_episode_number(self):
        """Test episode ID generation (episode_number is not used in ID generation)."""
        episode_id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_number=1,
        )
        episode_id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
            episode_number=2,
        )
        # Note: episode_number is not included in ID generation, so IDs are the same
        self.assertEqual(episode_id1, episode_id2)

    def test_generate_content_id(self):
        """Test content ID generation from URL."""
        content_url = "https://example.com/content.mp3"
        content_id = metadata.generate_content_id(content_url)
        self.assertTrue(content_id.startswith("sha256:"))
        self.assertEqual(len(content_id), 71)  # "sha256:" + 64 hex chars

        # Same URL should generate same ID
        content_id2 = metadata.generate_content_id(content_url)
        self.assertEqual(content_id, content_id2)

        # Different URL should generate different ID
        content_id3 = metadata.generate_content_id("https://example.com/other.mp3")
        self.assertNotEqual(content_id, content_id3)

    def test_generate_content_id_normalizes_url(self):
        """Test that content ID generation normalizes URLs."""
        url1 = "https://example.com/content.mp3"
        url2 = "https://example.com/content.mp3/"
        url3 = "https://example.com/content.mp3?param=value"
        url4 = "HTTPS://EXAMPLE.COM/content.mp3"

        id1 = metadata.generate_content_id(url1)
        id2 = metadata.generate_content_id(url2)
        id3 = metadata.generate_content_id(url3)
        id4 = metadata.generate_content_id(url4)

        # All should generate same ID (normalized)
        self.assertEqual(id1, id2)
        self.assertEqual(id1, id3)
        self.assertEqual(id1, id4)

    def test_generate_episode_id_empty_title(self):
        """Test episode ID generation with empty title."""
        episode_id = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="",
            episode_guid=None,
        )
        # Should still generate a valid ID
        self.assertTrue(episode_id.startswith("sha256:"))
        self.assertEqual(len(episode_id), 71)

    def test_generate_episode_id_whitespace_title(self):
        """Test episode ID generation normalizes whitespace in title."""
        episode_id1 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="  Test Episode  ",
            episode_guid=None,
        )
        episode_id2 = metadata.generate_episode_id(
            feed_url=TEST_FEED_URL,
            episode_title="Test Episode",
            episode_guid=None,
        )
        # Should generate same ID (whitespace normalized)
        self.assertEqual(episode_id1, episode_id2)


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


class TestBuildFeedMetadata(unittest.TestCase):
    """Tests for _build_feed_metadata function."""

    def test_build_feed_metadata_basic(self):
        """Test building feed metadata with all fields."""
        feed = create_test_feed()
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        cfg = create_test_config(language="en")
        feed_description = "Feed description"
        feed_image_url = "https://example.com/image.jpg"
        from datetime import datetime

        feed_last_updated = datetime.now()

        result = metadata._build_feed_metadata(
            feed=feed,
            feed_url=TEST_FEED_URL,
            feed_id=feed_id,
            cfg=cfg,
            feed_description=feed_description,
            feed_image_url=feed_image_url,
            feed_last_updated=feed_last_updated,
        )

        self.assertEqual(result.title, feed.title)
        self.assertEqual(result.url, TEST_FEED_URL)
        self.assertEqual(result.feed_id, feed_id)
        self.assertEqual(result.description, feed_description)
        self.assertEqual(result.language, "en")
        self.assertEqual(result.image_url, feed_image_url)
        self.assertEqual(result.last_updated, feed_last_updated)

    def test_build_feed_metadata_with_none_values(self):
        """Test building feed metadata with None values."""
        feed = create_test_feed()
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        cfg = create_test_config()

        result = metadata._build_feed_metadata(
            feed=feed,
            feed_url=TEST_FEED_URL,
            feed_id=feed_id,
            cfg=cfg,
            feed_description=None,
            feed_image_url=None,
            feed_last_updated=None,
        )

        self.assertIsNone(result.description)
        self.assertIsNone(result.image_url)
        self.assertIsNone(result.last_updated)

    def test_build_feed_metadata_with_authors(self):
        """Test building feed metadata with authors."""
        feed = create_test_feed()
        feed.authors = ["Author 1", "Author 2"]
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        cfg = create_test_config()

        result = metadata._build_feed_metadata(
            feed=feed,
            feed_url=TEST_FEED_URL,
            feed_id=feed_id,
            cfg=cfg,
            feed_description=None,
            feed_image_url=None,
            feed_last_updated=None,
        )

        self.assertEqual(result.authors, ["Author 1", "Author 2"])

    def test_build_feed_metadata_without_authors(self):
        """Test building feed metadata without authors."""
        feed = create_test_feed()
        feed.authors = None
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        cfg = create_test_config()

        result = metadata._build_feed_metadata(
            feed=feed,
            feed_url=TEST_FEED_URL,
            feed_id=feed_id,
            cfg=cfg,
            feed_description=None,
            feed_image_url=None,
            feed_last_updated=None,
        )

        self.assertEqual(result.authors, [])


class TestBuildEpisodeMetadata(unittest.TestCase):
    """Tests for _build_episode_metadata function."""

    def test_build_episode_metadata_basic(self):
        """Test building episode metadata with all fields."""
        episode = create_test_episode(idx=1, title="Test Episode")
        episode_id = metadata.generate_episode_id(TEST_FEED_URL, "Test Episode")
        from datetime import datetime

        published_date = datetime.now()

        result = metadata._build_episode_metadata(
            episode=episode,
            episode_id=episode_id,
            episode_description="Episode description",
            episode_published_date=published_date,
            episode_guid="guid-123",
            episode_link="https://example.com/episode",
            episode_duration_seconds=3600,
            episode_number=1,
            episode_image_url="https://example.com/episode.jpg",
        )

        self.assertEqual(result.title, episode.title)
        self.assertEqual(result.episode_id, episode_id)
        self.assertEqual(result.description, "Episode description")
        self.assertEqual(result.published_date, published_date)
        self.assertEqual(result.guid, "guid-123")
        self.assertEqual(result.link, "https://example.com/episode")
        self.assertEqual(result.duration_seconds, 3600)
        self.assertEqual(result.episode_number, 1)
        self.assertEqual(result.image_url, "https://example.com/episode.jpg")

    def test_build_episode_metadata_with_none_values(self):
        """Test building episode metadata with None values."""
        episode = create_test_episode(idx=1, title="Test Episode")
        episode_id = metadata.generate_episode_id(TEST_FEED_URL, "Test Episode")

        result = metadata._build_episode_metadata(
            episode=episode,
            episode_id=episode_id,
            episode_description=None,
            episode_published_date=None,
            episode_guid=None,
            episode_link=None,
            episode_duration_seconds=None,
            episode_number=None,
            episode_image_url=None,
        )

        self.assertIsNone(result.description)
        self.assertIsNone(result.published_date)
        self.assertIsNone(result.guid)
        self.assertIsNone(result.link)
        self.assertIsNone(result.duration_seconds)
        self.assertIsNone(result.episode_number)
        self.assertIsNone(result.image_url)


class TestBuildContentMetadata(unittest.TestCase):
    """Tests for _build_content_metadata function."""

    def test_build_content_metadata_basic(self):
        """Test building content metadata with all fields."""
        episode = create_test_episode(
            idx=1,
            title="Test Episode",
            media_url=TEST_MEDIA_URL,
            transcript_urls=[(TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT)],
        )
        transcript_infos = [
            metadata.TranscriptInfo(
                url=TEST_TRANSCRIPT_URL,
                transcript_id=metadata.generate_content_id(TEST_TRANSCRIPT_URL),
                type=TEST_TRANSCRIPT_TYPE_VTT,
                language="en",
            )
        ]
        media_id = metadata.generate_content_id(TEST_MEDIA_URL)

        result = metadata._build_content_metadata(
            episode=episode,
            transcript_infos=transcript_infos,
            media_id=media_id,
            transcript_file_path="transcript.txt",
            transcript_source="direct_download",
            whisper_model=None,
            detected_hosts=["Host"],
            detected_guests=["Guest"],
        )

        self.assertEqual(result.transcript_urls, transcript_infos)
        self.assertEqual(result.media_url, TEST_MEDIA_URL)
        self.assertEqual(result.media_id, media_id)
        self.assertEqual(result.transcript_file_path, "transcript.txt")
        self.assertEqual(result.transcript_source, "direct_download")
        self.assertEqual(result.detected_hosts, ["Host"])
        self.assertEqual(result.detected_guests, ["Guest"])

    def test_build_content_metadata_with_none_values(self):
        """Test building content metadata with None values."""
        episode = create_test_episode(idx=1, title="Test Episode")
        transcript_infos = []

        result = metadata._build_content_metadata(
            episode=episode,
            transcript_infos=transcript_infos,
            media_id=None,
            transcript_file_path=None,
            transcript_source=None,
            whisper_model=None,
            detected_hosts=None,
            detected_guests=None,
        )

        self.assertEqual(result.transcript_urls, [])
        self.assertIsNone(result.media_id)
        self.assertIsNone(result.transcript_file_path)
        self.assertIsNone(result.transcript_source)
        self.assertEqual(result.detected_hosts, [])
        self.assertEqual(result.detected_guests, [])


class TestBuildProcessingMetadata(unittest.TestCase):
    """Tests for _build_processing_metadata function."""

    def test_build_processing_metadata_basic(self):
        """Test building processing metadata."""
        cfg = create_test_config(
            language="en",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""),
            transcribe_missing=True,
            auto_speakers=True,
            screenplay=True,
            max_episodes=10,
        )

        result = metadata._build_processing_metadata(cfg, "/output")

        self.assertEqual(result.output_directory, "/output")
        self.assertEqual(result.config_snapshot["language"], "en")
        self.assertEqual(result.config_snapshot["auto_speakers"], True)
        self.assertEqual(result.config_snapshot["screenplay"], True)
        self.assertEqual(result.config_snapshot["max_episodes"], 10)
        self.assertIsNotNone(result.processing_timestamp)

        # Check ml_providers structure
        self.assertIn("ml_providers", result.config_snapshot)
        ml_providers = result.config_snapshot["ml_providers"]
        self.assertIn("transcription", ml_providers)
        self.assertEqual(ml_providers["transcription"]["provider"], "whisper")
        self.assertEqual(
            ml_providers["transcription"]["whisper_model"],
            config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""),
        )

    def test_build_processing_metadata_transcribe_disabled(self):
        """Test building processing metadata when transcription is disabled."""
        cfg = create_test_config(transcribe_missing=False)

        result = metadata._build_processing_metadata(cfg, "/output")

        # When transcribe_missing is False, whisper_model should not be in ml_providers
        if "ml_providers" in result.config_snapshot:
            ml_providers = result.config_snapshot["ml_providers"]
            if "transcription" in ml_providers:
                self.assertNotIn("whisper_model", ml_providers["transcription"])

    def test_build_processing_metadata_with_summarization(self):
        """Test building processing metadata with summarization provider info."""
        cfg = create_test_config(
            summary_provider="transformers",
            summary_model="facebook/bart-large-cnn",
            summary_reduce_model="allenai/led-large-16384",
            summary_device="cpu",
        )

        result = metadata._build_processing_metadata(cfg, "/output")

        self.assertIn("ml_providers", result.config_snapshot)
        ml_providers = result.config_snapshot["ml_providers"]
        self.assertIn("summarization", ml_providers)
        self.assertEqual(ml_providers["summarization"]["provider"], "transformers")
        self.assertEqual(ml_providers["summarization"]["map_model"], "facebook/bart-large-cnn")
        self.assertEqual(ml_providers["summarization"]["reduce_model"], "allenai/led-large-16384")
        self.assertEqual(ml_providers["summarization"]["device"], "cpu")


class TestDetermineMetadataPath(unittest.TestCase):
    """Tests for _determine_metadata_path function."""

    def test_determine_metadata_path_json(self):
        """Test determining metadata path with JSON format."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg = create_test_config(metadata_format="json")

        result = metadata._determine_metadata_path(
            episode=episode, output_dir="/output", run_suffix=None, cfg=cfg
        )

        self.assertTrue(result.endswith(".json"))
        self.assertIn("Episode_Title", result)
        self.assertIn("/metadata/", result)

    def test_determine_metadata_path_yaml(self):
        """Test determining metadata path with YAML format."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg = create_test_config(metadata_format="yaml")

        result = metadata._determine_metadata_path(
            episode=episode, output_dir="/output", run_suffix=None, cfg=cfg
        )

        self.assertTrue(result.endswith(".yaml"))
        self.assertIn("Episode_Title", result)
        self.assertIn("/metadata/", result)

    def test_determine_metadata_path_with_subdirectory(self):
        """Test determining metadata path with subdirectory."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg = create_test_config(metadata_format="json", metadata_subdirectory="metadata")

        result = metadata._determine_metadata_path(
            episode=episode, output_dir="/output", run_suffix=None, cfg=cfg
        )

        self.assertIn("/metadata/", result)
        self.assertTrue(result.endswith(".json"))

    def test_determine_metadata_path_with_run_suffix(self):
        """Test determining metadata path with run suffix."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg = create_test_config(metadata_format="json")

        result = metadata._determine_metadata_path(
            episode=episode, output_dir="/output", run_suffix="run1", cfg=cfg
        )

        self.assertIn("run1", result)
        self.assertIn("/metadata/", result)


class TestSerializeMetadata(unittest.TestCase):
    """Tests for _serialize_metadata function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(metadata_format="json")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_serialize_metadata_json(self):
        """Test serializing metadata to JSON format."""
        feed = create_test_feed()
        episode = create_test_episode(idx=1, title="Test Episode")
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        episode_id = metadata.generate_episode_id(TEST_FEED_URL, "Test Episode")

        feed_metadata = metadata._build_feed_metadata(
            feed, TEST_FEED_URL, feed_id, self.cfg, None, None, None
        )
        episode_metadata = metadata._build_episode_metadata(
            episode, episode_id, None, None, None, None, None, None, None
        )
        content_metadata = metadata._build_content_metadata(
            episode, [], None, None, None, None, None, None
        )
        processing_metadata = metadata._build_processing_metadata(self.cfg, self.temp_dir)

        metadata_doc = metadata.EpisodeMetadataDocument(
            feed=feed_metadata,
            episode=episode_metadata,
            content=content_metadata,
            processing=processing_metadata,
        )

        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        metadata._serialize_metadata(metadata_doc, metadata_path, self.cfg)

        # Verify file was created
        self.assertTrue(os.path.exists(metadata_path))
        # Verify it's valid JSON
        import json

        with open(metadata_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["episode"]["title"], "Test Episode")

    def test_serialize_metadata_yaml(self):
        """Test serializing metadata to YAML format."""
        self.cfg = create_test_config(metadata_format="yaml")
        feed = create_test_feed()
        episode = create_test_episode(idx=1, title="Test Episode")
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        episode_id = metadata.generate_episode_id(TEST_FEED_URL, "Test Episode")

        feed_metadata = metadata._build_feed_metadata(
            feed, TEST_FEED_URL, feed_id, self.cfg, None, None, None
        )
        episode_metadata = metadata._build_episode_metadata(
            episode, episode_id, None, None, None, None, None, None, None
        )
        content_metadata = metadata._build_content_metadata(
            episode, [], None, None, None, None, None, None
        )
        processing_metadata = metadata._build_processing_metadata(self.cfg, self.temp_dir)

        metadata_doc = metadata.EpisodeMetadataDocument(
            feed=feed_metadata,
            episode=episode_metadata,
            content=content_metadata,
            processing=processing_metadata,
        )

        metadata_path = os.path.join(self.temp_dir, "test.metadata.yaml")
        metadata._serialize_metadata(metadata_doc, metadata_path, self.cfg)

        # Verify file was created
        self.assertTrue(os.path.exists(metadata_path))
        # Verify it's valid YAML
        import yaml

        with open(metadata_path, "r") as f:
            data = yaml.safe_load(f)
            self.assertEqual(data["episode"]["title"], "Test Episode")

    def test_serialize_metadata_creates_directory(self):
        """Test that serialization creates directory if it doesn't exist."""
        metadata_path = os.path.join(self.temp_dir, "subdir", "test.metadata.json")
        feed = create_test_feed()
        episode = create_test_episode(idx=1, title="Test Episode")
        feed_id = metadata.generate_feed_id(TEST_FEED_URL)
        episode_id = metadata.generate_episode_id(TEST_FEED_URL, "Test Episode")

        feed_metadata = metadata._build_feed_metadata(
            feed, TEST_FEED_URL, feed_id, self.cfg, None, None, None
        )
        episode_metadata = metadata._build_episode_metadata(
            episode, episode_id, None, None, None, None, None, None, None
        )
        content_metadata = metadata._build_content_metadata(
            episode, [], None, None, None, None, None, None
        )
        processing_metadata = metadata._build_processing_metadata(self.cfg, self.temp_dir)

        metadata_doc = metadata.EpisodeMetadataDocument(
            feed=feed_metadata,
            episode=episode_metadata,
            content=content_metadata,
            processing=processing_metadata,
        )

        metadata._serialize_metadata(metadata_doc, metadata_path, self.cfg)

        # Verify directory was created
        self.assertTrue(os.path.exists(os.path.dirname(metadata_path)))
        self.assertTrue(os.path.exists(metadata_path))


class TestGenerateEpisodeSummary(unittest.TestCase):
    """Tests for _generate_episode_summary function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(generate_summaries=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_episode_summary_disabled(self):
        """Test that summary generation is skipped when disabled."""
        self.cfg = create_test_config(generate_summaries=False)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    def test_generate_episode_summary_dry_run(self):
        """Test that summary generation is skipped in dry-run mode."""
        self.cfg = create_test_config(generate_summaries=True, dry_run=True)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    def test_generate_episode_summary_file_not_found(self):
        """Test handling when transcript file doesn't exist."""
        result = metadata._generate_episode_summary(
            transcript_file_path="nonexistent.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    def test_generate_episode_summary_too_short(self):
        """Test that summary generation is skipped for very short transcripts."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("Short")  # Less than 50 chars

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.metadata.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_with_provider_success(self, mock_clean, mock_time):
        """Test successful summary generation with provider."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        # Create actual file for reading (tempfile operations are allowed)
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # Use function-based side effect to avoid StopIteration if called extra times
        time_values = iter([0.0, 1.0])
        mock_time.side_effect = lambda: next(time_values, 1.0)
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {
            "summary": "This is a summary",
            "metadata": {"model_used": "test-model"},
        }
        # Disable cleaned transcript saving to avoid filesystem I/O
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
            whisper_model="base.en",  # Test new parameter
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.short_summary, "This is a summary")
        self.assertEqual(result.model_used, "test-model")
        self.assertEqual(result.whisper_model_used, "base.en")
        mock_provider.summarize.assert_called_once()

    @patch("podcast_scraper.metadata.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_provider_returns_empty(self, mock_clean, mock_time):
        """Test handling when provider returns empty summary."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # Use a function-based side effect to avoid StopIteration
        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            return 0.0 if call_count[0] == 1 else 1.0

        mock_time.side_effect = time_side_effect
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {"summary": "", "metadata": {}}
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.metadata.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_provider_exception(self, mock_clean, mock_time):
        """Test handling when provider raises exception."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_time.side_effect = [0.0, 1.0]
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.side_effect = Exception("Provider error")
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
        )

        self.assertIsNone(result)

    def test_generate_episode_summary_no_provider(self):
        """Test that summary generation returns None when no provider."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=None,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.metadata.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_with_reduce_model(self, mock_clean, mock_time):
        """Test summary generation with reduce model in metadata."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        time_values = iter([0.0, 1.0])
        mock_time.side_effect = lambda: next(time_values, 1.0)
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {
            "summary": "This is a summary",
            "metadata": {
                "model_used": "facebook/bart-large-cnn",
                "reduce_model_used": "allenai/led-large-16384",
            },
        }
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        result = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
            whisper_model="base.en",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.short_summary, "This is a summary")
        self.assertEqual(result.model_used, "facebook/bart-large-cnn")
        self.assertEqual(result.reduce_model_used, "allenai/led-large-16384")
        self.assertEqual(result.whisper_model_used, "base.en")


class TestGenerateEpisodeMetadataEdgeCases(unittest.TestCase):
    """Tests for generate_episode_metadata edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(generate_metadata=True)
        self.feed = create_test_feed()
        self.episode = create_test_episode(idx=1, title="Test Episode")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_episode_metadata_disabled(self):
        """Test that metadata generation is skipped when disabled."""
        self.cfg = create_test_config(generate_metadata=False)

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.metadata._serialize_metadata")
    @patch("podcast_scraper.metadata._determine_metadata_path")
    def test_generate_episode_metadata_dry_run(self, mock_determine, mock_serialize):
        """Test metadata generation in dry-run mode."""
        self.cfg = create_test_config(generate_metadata=True, dry_run=True)
        mock_determine.return_value = os.path.join(self.temp_dir, "test.metadata.json")

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
        )

        self.assertEqual(result, os.path.join(self.temp_dir, "test.metadata.json"))
        # Should not serialize in dry-run
        mock_serialize.assert_not_called()

    @patch("podcast_scraper.metadata._serialize_metadata")
    @patch("podcast_scraper.metadata._determine_metadata_path")
    @patch("os.path.exists")
    def test_generate_episode_metadata_skip_existing(
        self, mock_exists, mock_determine, mock_serialize
    ):
        """Test that metadata generation is skipped when file exists."""
        self.cfg = create_test_config(generate_metadata=True, skip_existing=True)
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        mock_exists.return_value = True

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
        )

        self.assertIsNone(result)
        mock_serialize.assert_not_called()

    @patch("podcast_scraper.metadata._serialize_metadata")
    @patch("podcast_scraper.metadata._determine_metadata_path")
    @patch("os.path.exists")
    def test_generate_episode_metadata_overwrites_for_whisper(
        self, mock_exists, mock_determine, mock_serialize
    ):
        """Test that metadata is overwritten for whisper_transcription source."""
        self.cfg = create_test_config(generate_metadata=True, skip_existing=True)
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        mock_exists.return_value = True

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
            transcript_source="whisper_transcription",
        )

        # Should overwrite even if exists
        self.assertEqual(result, metadata_path)
        mock_serialize.assert_called_once()

    @patch("podcast_scraper.metadata._serialize_metadata")
    @patch("podcast_scraper.metadata._determine_metadata_path")
    @patch("podcast_scraper.metadata._generate_episode_summary")
    def test_generate_episode_metadata_with_summary(
        self, mock_generate_summary, mock_determine, mock_serialize
    ):
        """Test metadata generation with summary."""
        self.cfg = create_test_config(generate_metadata=True, generate_summaries=True)
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        from datetime import datetime

        mock_summary = metadata.SummaryMetadata(
            short_summary="Test summary",
            generated_at=datetime.now(),
            model_used="test-model",
            reduce_model_used=None,  # Optional field
            whisper_model_used=None,  # Optional field
            provider="local",
            word_count=100,
        )
        mock_generate_summary.return_value = mock_summary

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
            transcript_file_path="transcript.txt",
            summary_provider=Mock(),
        )

        self.assertEqual(result, metadata_path)
        mock_generate_summary.assert_called_once()
        mock_serialize.assert_called_once()

    @patch("podcast_scraper.metadata._serialize_metadata")
    @patch("podcast_scraper.metadata._determine_metadata_path")
    def test_generate_episode_metadata_serialization_error(self, mock_determine, mock_serialize):
        """Test handling serialization errors."""
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        mock_serialize.side_effect = OSError("Disk full")

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.metadata._serialize_metadata")
    @patch("podcast_scraper.metadata._determine_metadata_path")
    def test_generate_episode_metadata_records_metrics(self, mock_determine, mock_serialize):
        """Test that metadata generation records metrics."""
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        from podcast_scraper import metrics

        mock_metrics = metrics.Metrics()

        result = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
            pipeline_metrics=mock_metrics,
        )

        self.assertEqual(result, metadata_path)
        self.assertEqual(mock_metrics.metadata_files_generated, 1)
