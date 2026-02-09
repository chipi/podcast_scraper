#!/usr/bin/env python3
"""Tests for metadata generation functionality."""

# Note: Entity normalization tests added for Issue #387

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
    from podcast_scraper import config
    from podcast_scraper.rss import parser as rss_parser
    from podcast_scraper.workflow import metadata_generation as metadata

# Import from parent conftest explicitly to avoid conflicts

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

# Import directly from tests.conftest (works with pytest-xdist)
from tests.conftest import (  # noqa: E402
    create_test_config,
    create_test_episode,
    create_test_feed,
    TEST_FEED_URL,
    TEST_MEDIA_URL,
    TEST_TRANSCRIPT_TYPE_VTT,
    TEST_TRANSCRIPT_URL,
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

    @patch("podcast_scraper.workflow.metadata_generation.transformers", create=True)
    @patch("podcast_scraper.workflow.metadata_generation.torch", create=True)
    def test_generate_metadata_dry_run(self, mock_torch, mock_transformers):
        """Test metadata generation in dry-run mode."""
        # Mock transformers and torch to prevent actual imports
        mock_transformers.__version__ = "4.0.0"
        mock_torch.__version__ = "2.0.0"
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

        speakers = metadata._build_speakers_from_detected_names(
            detected_hosts=["Host"], detected_guests=["Guest"]
        )
        result = metadata._build_content_metadata(
            episode=episode,
            transcript_infos=transcript_infos,
            media_id=media_id,
            transcript_file_path="transcript.txt",
            transcript_source="direct_download",
            whisper_model=None,
            speakers=speakers,
        )

        self.assertEqual(result.transcript_urls, transcript_infos)
        self.assertEqual(result.media_url, TEST_MEDIA_URL)
        self.assertEqual(result.media_id, media_id)
        self.assertEqual(result.transcript_file_path, "transcript.txt")
        self.assertEqual(result.transcript_source, "direct_download")
        # Check speakers list instead of detected_hosts/detected_guests
        self.assertEqual(len(result.speakers), 2)
        host_names = [s.name for s in result.speakers if s.role == "host"]
        guest_names = [s.name for s in result.speakers if s.role == "guest"]
        self.assertEqual(host_names, ["Host"])
        self.assertEqual(guest_names, ["Guest"])

    def test_build_content_metadata_with_none_values(self):
        """Test building content metadata with None values."""
        episode = create_test_episode(idx=1, title="Test Episode")
        transcript_infos = []

        speakers = metadata._build_speakers_from_detected_names(
            detected_hosts=None, detected_guests=None
        )
        result = metadata._build_content_metadata(
            episode=episode,
            transcript_infos=transcript_infos,
            media_id=None,
            transcript_file_path=None,
            transcript_source=None,
            whisper_model=None,
            speakers=speakers,
        )

        self.assertEqual(result.transcript_urls, [])
        self.assertIsNone(result.media_id)
        self.assertIsNone(result.transcript_file_path)
        self.assertIsNone(result.transcript_source)
        # Check speakers list instead of detected_hosts/detected_guests
        self.assertEqual(result.speakers, [])


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
        # whisper_model is now in ml_providers.transcription.whisper_model
        self.assertEqual(
            result.config_snapshot["ml_providers"]["transcription"]["whisper_model"],
            config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""),
        )
        self.assertEqual(result.config_snapshot["auto_speakers"], True)
        self.assertEqual(result.config_snapshot["screenplay"], True)
        self.assertEqual(result.config_snapshot["max_episodes"], 10)
        self.assertIsNotNone(result.processing_timestamp)

    def test_build_processing_metadata_transcribe_disabled(self):
        """Test building processing metadata when transcription is disabled."""
        cfg = create_test_config(transcribe_missing=False)

        result = metadata._build_processing_metadata(cfg, "/output")

        # When transcription is disabled, whisper_model should not be in transcription config
        if "transcription" in result.config_snapshot.get("ml_providers", {}):
            self.assertNotIn(
                "whisper_model", result.config_snapshot["ml_providers"]["transcription"]
            )


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
        speakers = metadata._build_speakers_from_detected_names(None, None)
        content_metadata = metadata._build_content_metadata(
            episode, [], None, None, None, None, speakers
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
        speakers = metadata._build_speakers_from_detected_names(None, None)
        content_metadata = metadata._build_content_metadata(
            episode, [], None, None, None, None, speakers
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
        speakers = metadata._build_speakers_from_detected_names(None, None)
        content_metadata = metadata._build_content_metadata(
            episode, [], None, None, None, None, speakers
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

        result, _ = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    def test_generate_episode_summary_dry_run(self):
        """Test that summary generation is skipped in dry-run mode."""
        self.cfg = create_test_config(generate_summaries=True, dry_run=True)

        result, _ = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    def test_generate_episode_summary_file_not_found(self):
        """Test handling when transcript file doesn't exist."""
        result, _ = metadata._generate_episode_summary(
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

        result, _ = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.metadata_generation.time.time")
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

        result, call_metrics = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.short_summary, "This is a summary")
        # model_used is no longer stored in SummaryMetadata - it's in processing.config_snapshot
        mock_provider.summarize.assert_called_once()

    @patch("podcast_scraper.workflow.metadata_generation.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_provider_returns_empty(self, mock_clean, mock_time):
        """Test that empty summary raises RuntimeError when generate_summaries=True."""
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

        # Should raise RuntimeError when summary is empty and generate_summaries=True
        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.temp_dir,
                cfg=self.cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("empty result", str(context.exception))
        self.assertIn("generate_summaries=True", str(context.exception))

    @patch("podcast_scraper.workflow.metadata_generation.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_provider_exception(self, mock_clean, mock_time):
        """Test provider exception raises RuntimeError when generate_summaries=True (fail-fast)."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # Provide enough time.time() values for all logging calls
        # (debug at start, error logging, etc.)
        mock_time.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0]
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.side_effect = Exception("Provider error")
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        # Should raise RuntimeError when provider raises exception and generate_summaries=True
        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.temp_dir,
                cfg=self.cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("Failed to generate summary", str(context.exception))
        self.assertIn("generate_summaries=True", str(context.exception))

    def test_generate_episode_summary_no_provider(self):
        """Test summary generation raises ValueError when no provider and generate_summaries=True."""  # noqa: E501
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # When generate_summaries=True but no provider, should raise ValueError (fail-fast)
        with self.assertRaises(ValueError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.temp_dir,
                cfg=self.cfg,
                episode_idx=1,
                summary_provider=None,
            )

        self.assertIn("summary_provider is required", str(context.exception))

    @patch("podcast_scraper.workflow.metadata_generation.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_mock_object_without_return_value(self, mock_clean, mock_time):
        """Test handling of Mock object without return_value (should skip summary)."""
        import os

        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # Use a callable that returns values indefinitely (logging also calls time.time())
        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            return float(call_count[0] - 1)  # First call returns 0.0, second returns 1.0, etc.

        mock_time.side_effect = time_side_effect
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        # Create a Mock object that has _mock_name but no return_value
        mock_summary = Mock()
        mock_summary._mock_name = "mock_summary"
        mock_provider.summarize.return_value = mock_summary
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        # Should return None (skip summary) when Mock object has no proper string value
        result, _ = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.metadata_generation.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_mock_object_string_conversion_fails(
        self, mock_clean, mock_time
    ):
        """Test handling of Mock object that fails string conversion (should skip summary)."""
        import os

        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # Use a callable that returns values indefinitely (logging also calls time.time())
        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            return float(call_count[0] - 1)  # First call returns 0.0, second returns 1.0, etc.

        mock_time.side_effect = time_side_effect
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        # Create a Mock object that raises exception when converted to string
        mock_summary = Mock()
        mock_summary.__str__ = Mock(side_effect=Exception("Cannot convert to string"))
        mock_provider.summarize.return_value = mock_summary
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        # Should return None (skip summary) when string conversion fails
        result, _ = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.metadata_generation.time.time")
    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_generate_episode_summary_mock_object_with_return_value_string(
        self, mock_clean, mock_time
    ):
        """Test handling of Mock object with return_value that is a string (should use it)."""
        import os

        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        # Use a callable that returns values indefinitely (logging also calls time.time())
        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            return float(call_count[0] - 1)  # First call returns 0.0, second returns 1.0, etc.

        mock_time.side_effect = time_side_effect
        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        # Return a dict with summary string (not a Mock)
        mock_provider.summarize.return_value = {"summary": "This is a valid summary string"}
        self.cfg = create_test_config(generate_summaries=True, save_cleaned_transcript=False)

        # Should use the summary string from the dict
        result, _ = metadata._generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=self.temp_dir,
            cfg=self.cfg,
            episode_idx=1,
            summary_provider=mock_provider,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.short_summary, "This is a valid summary string")


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

    @patch("podcast_scraper.workflow.metadata_generation._serialize_metadata")
    @patch("podcast_scraper.workflow.metadata_generation._determine_metadata_path")
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

    @patch("podcast_scraper.workflow.metadata_generation._serialize_metadata")
    @patch("podcast_scraper.workflow.metadata_generation._determine_metadata_path")
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

    @patch("podcast_scraper.workflow.metadata_generation._serialize_metadata")
    @patch("podcast_scraper.workflow.metadata_generation._determine_metadata_path")
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

    @patch("podcast_scraper.workflow.metadata_generation._serialize_metadata")
    @patch("podcast_scraper.workflow.metadata_generation._determine_metadata_path")
    @patch("podcast_scraper.workflow.metadata_generation._generate_episode_summary")
    def test_generate_episode_metadata_with_summary(
        self, mock_generate_summary, mock_determine, mock_serialize
    ):
        """Test metadata generation with summary."""
        self.cfg = create_test_config(generate_metadata=True, generate_summaries=True)
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        from datetime import datetime

        mock_summary = metadata.SummaryMetadata(
            bullets=["Test bullet 1", "Test bullet 2"],  # Required field
            generated_at=datetime.now(),
            word_count=100,
            schema_status="valid",
        )
        mock_generate_summary.return_value = (mock_summary, None)

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

    @patch("podcast_scraper.workflow.metadata_generation._serialize_metadata")
    @patch("podcast_scraper.workflow.metadata_generation._determine_metadata_path")
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

    @patch("podcast_scraper.workflow.metadata_generation._serialize_metadata")
    @patch("podcast_scraper.workflow.metadata_generation._determine_metadata_path")
    def test_generate_episode_metadata_records_metrics(self, mock_determine, mock_serialize):
        """Test that metadata generation records metrics."""
        metadata_path = os.path.join(self.temp_dir, "test.metadata.json")
        mock_determine.return_value = metadata_path
        from podcast_scraper.workflow import metrics

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


class TestEntityNormalization(unittest.TestCase):
    """Tests for entity normalization functionality (Issue #387)."""

    def test_normalize_entity_full_name(self):
        """Test normalization of full names."""
        result = metadata._normalize_entity("John Doe")
        self.assertEqual(result.canonical, "john doe")
        self.assertEqual(result.original, "John Doe")
        self.assertIn("john doe", result.aliases)
        self.assertIn("doe", result.aliases)
        self.assertIn("john", result.aliases)
        # Default provenance is "transcript"
        self.assertEqual(result.provenance, "transcript")

    def test_normalize_entity_single_name(self):
        """Test normalization of single names."""
        result = metadata._normalize_entity("Madonna")
        self.assertEqual(result.canonical, "madonna")
        self.assertEqual(result.original, "Madonna")
        self.assertIn("madonna", result.aliases)

    def test_normalize_entity_with_punctuation(self):
        """Test normalization removes punctuation."""
        result = metadata._normalize_entity("John O'Brien")
        self.assertEqual(result.canonical, "john obrien")
        self.assertEqual(result.original, "John O'Brien")
        # Punctuation should be removed
        self.assertNotIn("o'brien", result.aliases)
        self.assertIn("obrien", result.aliases)

    def test_normalize_entity_whitespace(self):
        """Test normalization handles whitespace."""
        result = metadata._normalize_entity("  John   Doe  ")
        self.assertEqual(result.canonical, "john doe")
        # Original is stripped (strip() is called first)
        self.assertEqual(result.original, "John   Doe")
        self.assertIn("john doe", result.aliases)

    def test_normalize_entity_empty(self):
        """Test normalization of empty string."""
        result = metadata._normalize_entity("")
        self.assertEqual(result.canonical, "")
        self.assertEqual(result.original, "")
        self.assertEqual(result.aliases, [])

    def test_normalize_entity_three_part_name(self):
        """Test normalization of three-part names."""
        result = metadata._normalize_entity("Mary Jane Watson")
        self.assertEqual(result.canonical, "mary jane watson")
        self.assertEqual(result.original, "Mary Jane Watson")
        self.assertIn("mary jane watson", result.aliases)
        self.assertIn("watson", result.aliases)  # Last name
        self.assertIn("mary", result.aliases)  # First name

    def test_normalize_entity_in_content_metadata(self):
        """Test that normalized entities are stored in ContentMetadata."""
        speakers = []
        detected_hosts = ["John Doe"]
        detected_guests = ["Jane Smith"]

        content = metadata._build_content_metadata(
            episode=create_test_episode(),
            transcript_infos=[],
            media_id=None,
            transcript_file_path=None,
            transcript_source=None,
            whisper_model=None,
            speakers=speakers,
            detected_hosts=detected_hosts,
            detected_guests=detected_guests,
        )

        self.assertEqual(len(content.normalized_entities), 2)
        # Check that entities are normalized
        entity_names = [e.original for e in content.normalized_entities]
        self.assertIn("John Doe", entity_names)
        self.assertIn("Jane Smith", entity_names)
        # Check aliases are populated
        for entity in content.normalized_entities:
            self.assertGreater(len(entity.aliases), 0)
            self.assertEqual(entity.canonical, entity.aliases[0])


class TestEntityExtractionFromSummary(unittest.TestCase):
    """Tests for entity extraction from summary (Issue #387)."""

    def test_extract_entities_from_summary(self):
        """Test that entities are extracted from summary text."""
        speakers = []
        detected_hosts = ["John Doe"]
        detected_guests = ["Jane Smith"]
        summary_text = "This episode features Alice Johnson and Bob Williams."

        # Mock NLP model
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent1.text = "Alice Johnson"
        mock_ent1.label_ = "PERSON"
        mock_ent2 = MagicMock()
        mock_ent2.text = "Bob Williams"
        mock_ent2.label_ = "PERSON"
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_nlp.return_value = mock_doc

        # Mock extract_all_entities
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            mock_extract.return_value = [
                {"text": "Alice Johnson", "start": 0, "end": 13, "label": "PERSON"},
                {"text": "Bob Williams", "start": 30, "end": 42, "label": "PERSON"},
            ]

            content = metadata._build_content_metadata(
                episode=create_test_episode(),
                transcript_infos=[],
                media_id=None,
                transcript_file_path=None,
                transcript_source=None,
                whisper_model=None,
                speakers=speakers,
                detected_hosts=detected_hosts,
                detected_guests=detected_guests,
                summary_text=summary_text,
                nlp=mock_nlp,
            )

            # Should have entities from transcript (John Doe, Jane Smith)
            # and from summary (Alice Johnson, Bob Williams)
            entity_names = {e.original for e in content.normalized_entities}
            self.assertIn("John Doe", entity_names)
            self.assertIn("Jane Smith", entity_names)
            self.assertIn("Alice Johnson", entity_names)
            self.assertIn("Bob Williams", entity_names)

    def test_entity_provenance_tracking(self):
        """Test that entity provenance is tracked correctly."""
        speakers = []
        detected_hosts = ["John Doe"]
        detected_guests = ["Jane Smith"]
        summary_text = "This episode features Jane Smith and Alice Johnson."

        # Mock NLP model
        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            # Jane Smith appears in both transcript and summary
            # Alice Johnson only in summary
            mock_extract.return_value = [
                {"text": "Jane Smith", "start": 0, "end": 10, "label": "PERSON"},
                {"text": "Alice Johnson", "start": 30, "end": 43, "label": "PERSON"},
            ]

            content = metadata._build_content_metadata(
                episode=create_test_episode(),
                transcript_infos=[],
                media_id=None,
                transcript_file_path=None,
                transcript_source=None,
                whisper_model=None,
                speakers=speakers,
                detected_hosts=detected_hosts,
                detected_guests=detected_guests,
                summary_text=summary_text,
                nlp=mock_nlp,
            )

            # Find entities by original name
            entity_map = {e.original: e for e in content.normalized_entities}

            # John Doe: only in transcript
            self.assertEqual(entity_map["John Doe"].provenance, "transcript")

            # Jane Smith: in both transcript and summary
            self.assertEqual(entity_map["Jane Smith"].provenance, "both")

            # Alice Johnson: only in summary
            self.assertEqual(entity_map["Alice Johnson"].provenance, "summary")

    def test_entity_extraction_no_summary(self):
        """Test that entity extraction works without summary."""
        speakers = []
        detected_hosts = ["John Doe"]
        detected_guests = ["Jane Smith"]

        content = metadata._build_content_metadata(
            episode=create_test_episode(),
            transcript_infos=[],
            media_id=None,
            transcript_file_path=None,
            transcript_source=None,
            whisper_model=None,
            speakers=speakers,
            detected_hosts=detected_hosts,
            detected_guests=detected_guests,
            summary_text=None,  # No summary
            nlp=None,
        )

        # Should only have transcript entities
        entity_names = {e.original for e in content.normalized_entities}
        self.assertIn("John Doe", entity_names)
        self.assertIn("Jane Smith", entity_names)
        self.assertEqual(len(entity_names), 2)

        # All should have transcript provenance
        for entity in content.normalized_entities:
            self.assertEqual(entity.provenance, "transcript")


class TestQABackfillAndMismatchSeverity(unittest.TestCase):
    """Tests for QA backfill and mismatch severity logic (Issue #387)."""

    def test_entity_consistency_with_fuzzy_matching(self):
        """Test entity consistency uses fuzzy matching, doesn't flag normalization."""
        # "Kevin Warsh" in extracted, "Kevin Walsh" in summary
        # Should NOT flag as mismatch (fuzzy match with constraints)
        extracted_entities = ["Kevin Warsh"]
        summary_text = "Kevin Walsh discusses monetary policy."

        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            mock_extract.return_value = [
                {"text": "Kevin Walsh", "start": 0, "end": 12, "label": "PERSON"}
            ]

            mismatch, has_entities = metadata._check_entity_consistency(
                extracted_entities, summary_text, nlp=mock_nlp
            )

            # Should NOT flag as mismatch (fuzzy match found)
            self.assertFalse(mismatch, "Should not flag fuzzy matches as mismatches")
            self.assertTrue(has_entities, "Summary has named entities")

    def test_entity_consistency_flags_zero_evidence(self):
        """Test that entity consistency flags zero-evidence entities."""
        # "John Doe" in extracted, "Alice Johnson" in summary (completely different)
        # Should flag as mismatch (zero evidence)
        extracted_entities = ["John Doe"]
        summary_text = "Alice Johnson discusses the topic."

        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            mock_extract.return_value = [
                {"text": "Alice Johnson", "start": 0, "end": 13, "label": "PERSON"}
            ]

            mismatch, has_entities = metadata._check_entity_consistency(
                extracted_entities, summary_text, nlp=mock_nlp
            )

            # Should flag as mismatch (zero evidence)
            self.assertTrue(mismatch, "Should flag zero-evidence entities as mismatches")
            self.assertTrue(has_entities, "Summary has named entities")

    def test_entity_consistency_rejects_common_words(self):
        """Test that entity consistency rejects common words from matching."""
        # "John Doe" in extracted, "The" in summary (common word)
        # Common words are rejected from matching, so "The" won't match "John Doe"
        # But if NER extracts it, it will be flagged as zero-evidence
        extracted_entities = ["John Doe"]
        summary_text = "The host discusses the topic."

        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            # Even if NER incorrectly extracts "The" as a person
            mock_extract.return_value = [{"text": "The", "start": 0, "end": 3, "label": "PERSON"}]

            mismatch, has_entities = metadata._check_entity_consistency(
                extracted_entities, summary_text, nlp=mock_nlp
            )

            # Common words are rejected from matching (constraints prevent match)
            # But if NER extracts them, they're still flagged as zero-evidence
            # This is correct - "The" doesn't match "John Doe" and is zero evidence
            self.assertTrue(mismatch, "Common words that don't match are zero-evidence")
            self.assertTrue(has_entities, "Summary has named entities (even if incorrect)")

    def test_qa_backfill_detects_missing_entities(self):
        """Test that QA backfill detects entities missing from summary."""
        # "John Doe" in transcript, but not in summary
        # Should be included with "transcript" provenance (backfilled)
        speakers = []
        detected_hosts = ["John Doe"]
        detected_guests = ["Jane Smith"]
        summary_text = "This episode features Alice Johnson."  # Missing John Doe and Jane Smith

        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            mock_extract.return_value = [
                {"text": "Alice Johnson", "start": 0, "end": 13, "label": "PERSON"}
            ]

            content = metadata._build_content_metadata(
                episode=create_test_episode(),
                transcript_infos=[],
                media_id=None,
                transcript_file_path=None,
                transcript_source=None,
                whisper_model=None,
                speakers=speakers,
                detected_hosts=detected_hosts,
                detected_guests=detected_guests,
                summary_text=summary_text,
                nlp=mock_nlp,
            )

            # Should have all entities: John Doe, Jane Smith (transcript), Alice Johnson (summary)
            entity_map = {e.original: e for e in content.normalized_entities}
            self.assertIn("John Doe", entity_map)
            self.assertIn("Jane Smith", entity_map)
            self.assertIn("Alice Johnson", entity_map)

            # John Doe and Jane Smith should have "transcript" provenance (backfilled)
            self.assertEqual(entity_map["John Doe"].provenance, "transcript")
            self.assertEqual(entity_map["Jane Smith"].provenance, "transcript")
            # Alice Johnson should have "summary" provenance
            self.assertEqual(entity_map["Alice Johnson"].provenance, "summary")


class TestSummaryFaithfulnessCheck(unittest.TestCase):
    """Tests for summary faithfulness check (Issue #387)."""

    def test_faithfulness_check_detects_hallucinations(self):
        """Test that faithfulness check detects entities not in source."""
        # "John Doe" in transcript, "Alice Johnson" in summary (hallucination)
        transcript_text = "John Doe discusses the topic with the host."
        episode_description = "Episode featuring John Doe"
        summary_text = "Alice Johnson discusses the topic."

        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            # Mock transcript entities
            mock_extract.side_effect = [
                [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}],  # transcript
                [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}],  # description
                [{"text": "Alice Johnson", "start": 0, "end": 13, "label": "PERSON"}],  # summary
            ]

            has_out_of_source, out_of_source_entities = metadata._check_summary_faithfulness(
                transcript_text=transcript_text,
                episode_description=episode_description,
                summary_text=summary_text,
                nlp=mock_nlp,
            )

            # Should detect "Alice Johnson" as out-of-source
            self.assertTrue(has_out_of_source, "Should detect out-of-source entities")
            self.assertIn("Alice Johnson", out_of_source_entities)

    def test_faithfulness_check_allows_valid_entities(self):
        """Test that faithfulness check allows entities that are in source."""
        # "John Doe" in both transcript and summary
        transcript_text = "John Doe discusses the topic with the host."
        episode_description = "Episode featuring John Doe"
        summary_text = "John Doe discusses the topic."

        mock_nlp = MagicMock()
        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            # Mock entities - all have "John Doe"
            mock_extract.side_effect = [
                [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}],  # transcript
                [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}],  # description
                [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}],  # summary
            ]

            has_out_of_source, out_of_source_entities = metadata._check_summary_faithfulness(
                transcript_text=transcript_text,
                episode_description=episode_description,
                summary_text=summary_text,
                nlp=mock_nlp,
            )

            # Should NOT detect out-of-source entities
            self.assertFalse(has_out_of_source, "Should not flag valid entities")
            self.assertEqual(len(out_of_source_entities), 0)

    def test_faithfulness_check_uses_top_n_entities(self):
        """Test that faithfulness check uses top N entities from transcript."""
        # Create transcript with multiple entities
        transcript_text = "John Doe and Jane Smith discuss. John Doe mentions Bob Williams."
        # "John Doe" appears twice, "Jane Smith" once, "Bob Williams" once
        summary_text = "Alice Johnson discusses the topic."  # Hallucination

        mock_nlp = MagicMock()

        def extract_side_effect(text, nlp, labels=None):
            # Return different results based on text content
            if "John Doe" in text and "Jane Smith" in text:
                # Transcript
                return [
                    {"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"},
                    {"text": "Jane Smith", "start": 13, "end": 23, "label": "PERSON"},
                    {"text": "John Doe", "start": 30, "end": 38, "label": "PERSON"},
                    {"text": "Bob Williams", "start": 50, "end": 62, "label": "PERSON"},
                ]
            elif "Alice Johnson" in text:
                # Summary
                return [{"text": "Alice Johnson", "start": 0, "end": 13, "label": "PERSON"}]
            else:
                # Description or other
                return []

        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            mock_extract.side_effect = extract_side_effect

            has_out_of_source, out_of_source_entities = metadata._check_summary_faithfulness(
                transcript_text=transcript_text,
                episode_description=None,
                summary_text=summary_text,
                nlp=mock_nlp,
                top_n_entities=20,
            )

            # Should detect "Alice Johnson" as out-of-source
            self.assertTrue(has_out_of_source)
            self.assertIn("Alice Johnson", out_of_source_entities)

    def test_faithfulness_check_in_qa_flags(self):
        """Test that faithfulness check is integrated into QA flags."""
        speakers = []
        detected_hosts = ["John Doe"]
        detected_guests = []
        summary_text = "Alice Johnson discusses the topic."  # Hallucination
        transcript_text = "John Doe discusses the topic."
        episode_description = "Episode with John Doe"

        mock_nlp = MagicMock()

        def extract_side_effect(text, nlp, labels=None):
            # Return different results based on text content
            if "John Doe" in text and "discusses" in text:
                # Transcript
                return [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}]
            elif "Episode" in text:
                # Description
                return [{"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"}]
            elif "Alice Johnson" in text:
                # Summary
                return [{"text": "Alice Johnson", "start": 0, "end": 13, "label": "PERSON"}]
            else:
                return []

        with patch(
            "podcast_scraper.providers.ml.ner_extraction.extract_all_entities"
        ) as mock_extract:
            mock_extract.side_effect = extract_side_effect

            qa_flags = metadata._build_qa_flags(
                speakers=speakers,
                detected_hosts=detected_hosts,
                detected_guests=detected_guests,
                summary_text=summary_text,
                nlp=mock_nlp,
                transcript_text=transcript_text,
                episode_description=episode_description,
            )

            # Should flag out-of-source entities
            self.assertTrue(
                qa_flags.summary_entity_out_of_source,
                "Should flag out-of-source entities in QA flags",
            )
            self.assertIn("Alice Johnson", qa_flags.summary_out_of_source_entities)


class TestFuzzyMatchingConstraints(unittest.TestCase):
    """Tests for fuzzy matching with constraints (Issue #387)."""

    def test_is_rare_last_name(self):
        """Test rare last name detection."""
        # Rare names should return True
        self.assertTrue(metadata._is_rare_last_name("warsh"))
        self.assertTrue(metadata._is_rare_last_name("walsh"))
        self.assertTrue(metadata._is_rare_last_name("smith"))
        self.assertTrue(metadata._is_rare_last_name("johnson"))

        # Common words should return False
        self.assertFalse(metadata._is_rare_last_name("the"))
        self.assertFalse(metadata._is_rare_last_name("and"))
        self.assertFalse(metadata._is_rare_last_name("for"))
        self.assertFalse(metadata._is_rare_last_name("but"))

        # Short names should return False
        self.assertFalse(metadata._is_rare_last_name(""))
        self.assertFalse(metadata._is_rare_last_name("a"))
        self.assertFalse(metadata._is_rare_last_name("ab"))

    def test_has_paired_first_name(self):
        """Test paired first name detection."""
        # Test with full name in extracted entities
        extracted = ["Kevin Warsh"]
        aliases = {}
        for entity in extracted:
            alias = metadata._normalize_entity(entity)
            aliases[alias.canonical] = alias

        # "Warsh" should be paired with "Kevin"
        self.assertTrue(
            metadata._has_paired_first_name("warsh", extracted, aliases),
            "Warsh should be paired with Kevin",
        )

        # "Walsh" should not be paired (not in extracted)
        self.assertFalse(
            metadata._has_paired_first_name("walsh", extracted, aliases),
            "Walsh should not be paired",
        )

        # Test with multiple entities
        extracted2 = ["Kevin Warsh", "John Smith"]
        aliases2 = {}
        for entity in extracted2:
            alias = metadata._normalize_entity(entity)
            aliases2[alias.canonical] = alias

        self.assertTrue(metadata._has_paired_first_name("warsh", extracted2, aliases2))
        self.assertTrue(metadata._has_paired_first_name("smith", extracted2, aliases2))

    def test_fuzzy_matching_warsh_vs_walsh_constraints(self):
        """Test fuzzy matching constraints for Warsh vs Walsh scenario (Issue #387)."""
        # Test the constraint functions directly
        # "Kevin Warsh" in extracted, "Kevin Walsh" in summary
        # Should match because:
        # 1. Edit distance is 1 (Warsh vs Walsh)
        # 2. Both are rare last names
        # 3. Last name is paired with first name "Kevin"

        extracted_entities = ["Kevin Warsh"]
        aliases = {}
        for entity in extracted_entities:
            alias = metadata._normalize_entity(entity)
            aliases[alias.canonical] = alias

        # Test that "warsh" is rare
        self.assertTrue(metadata._is_rare_last_name("warsh"))
        # Test that "walsh" is rare
        self.assertTrue(metadata._is_rare_last_name("walsh"))
        # Test that "warsh" is paired with first name
        self.assertTrue(metadata._has_paired_first_name("warsh", extracted_entities, aliases))

        # Test edit distance
        distance = metadata._calculate_levenshtein_distance("warsh", "walsh")
        self.assertEqual(distance, 1, "Warsh and Walsh should have edit distance 1")

    def test_fuzzy_matching_rejects_common_words(self):
        """Test that fuzzy matching rejects common words."""
        # Common words should be rejected
        self.assertFalse(metadata._is_rare_last_name("the"))
        self.assertFalse(metadata._is_rare_last_name("and"))
        self.assertFalse(metadata._is_rare_last_name("for"))
        self.assertFalse(metadata._is_rare_last_name("but"))

    def test_fuzzy_matching_requires_paired_first_name(self):
        """Test that fuzzy matching requires paired first name for last-name-only matches."""
        # Test with full name
        extracted_entities = ["Kevin Warsh"]
        aliases = {}
        for entity in extracted_entities:
            alias = metadata._normalize_entity(entity)
            aliases[alias.canonical] = alias

        # "Warsh" should be paired with "Kevin"
        self.assertTrue(metadata._has_paired_first_name("warsh", extracted_entities, aliases))

        # Test with just last name (no pairing)
        extracted_entities2 = ["Warsh"]  # Just last name, no first name
        aliases2 = {}
        for entity in extracted_entities2:
            alias = metadata._normalize_entity(entity)
            aliases2[alias.canonical] = alias

        # "Warsh" should not be paired (no first name)
        self.assertFalse(metadata._has_paired_first_name("warsh", extracted_entities2, aliases2))
