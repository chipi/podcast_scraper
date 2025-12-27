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
    from podcast_scraper import config, metadata, rss_parser, speaker_detection

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


class TestSpeakerDetection(unittest.TestCase):
    """Tests for speaker detection using NER."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url=TEST_FEED_URL,
            auto_speakers=True,
            ner_model="en_core_web_sm",
        )
        # Clear spaCy model cache before each test to ensure clean state
        speaker_detection.clear_spacy_model_cache()

    @unittest.skip(
        "TODO: Fix spacy mocking setup - spacy.load() MagicMock interferes with test mocks"
    )
    @patch("podcast_scraper.speaker_detection._load_spacy_model")
    def test_get_ner_model_enabled(self, mock_load):
        """Test getting NER model when auto_speakers is enabled."""
        # Clear cache to ensure fresh load
        speaker_detection.clear_spacy_model_cache()

        mock_nlp = unittest.mock.MagicMock()
        mock_load.return_value = mock_nlp

        nlp = speaker_detection.get_ner_model(self.cfg)
        self.assertEqual(nlp, mock_nlp)
        mock_load.assert_called_once_with("en_core_web_sm")

        # Second call should use cache (no additional load call)
        mock_load.reset_mock()
        nlp2 = speaker_detection.get_ner_model(self.cfg)
        self.assertEqual(nlp2, mock_nlp)
        mock_load.assert_not_called()

    def test_get_ner_model_disabled(self):
        """Test getting NER model when auto_speakers is disabled."""
        cfg = config.Config(
            rss_url=TEST_FEED_URL,
            auto_speakers=False,
        )
        nlp = speaker_detection.get_ner_model(cfg)
        self.assertIsNone(nlp)

    @unittest.skip(
        "TODO: Fix spacy mocking setup - spacy.load() MagicMock interferes with test mocks"
    )
    @patch("podcast_scraper.speaker_detection._load_spacy_model")
    def test_extract_person_entities(self, mock_load):
        """Test extracting person entities from text."""
        # Clear cache to ensure fresh load
        speaker_detection.clear_spacy_model_cache()

        # Mock spaCy model
        mock_nlp = unittest.mock.MagicMock()
        mock_doc = unittest.mock.MagicMock()
        mock_ent = unittest.mock.MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "John Doe"
        mock_ent.score = 0.95
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp

        nlp = speaker_detection.get_ner_model(self.cfg)
        result = speaker_detection.extract_person_entities("Interview with John Doe", nlp)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "John Doe")
        self.assertEqual(result[0][1], 0.95)

    @unittest.skip(
        "TODO: Fix spacy mocking setup - spacy.load() MagicMock interferes with test mocks"
    )
    @patch("podcast_scraper.speaker_detection._load_spacy_model")
    def test_extract_person_entities_sanitizes_names(self, mock_load):
        """Test that person entity extraction sanitizes names."""
        # Clear cache to ensure fresh load
        speaker_detection.clear_spacy_model_cache()

        mock_nlp = unittest.mock.MagicMock()
        mock_doc = unittest.mock.MagicMock()
        mock_ent = unittest.mock.MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "John (Smith),"
        mock_ent.score = 0.9
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp

        nlp = speaker_detection.get_ner_model(self.cfg)
        result = speaker_detection.extract_person_entities("John (Smith),", nlp)

        # Should sanitize to "John"
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "John")

    @patch("spacy.load")
    @patch("podcast_scraper.speaker_detection._load_spacy_model")
    def test_extract_person_entities_filters_short_names(self, mock_load, mock_spacy_load):
        """Test that very short names are filtered out."""
        # Clear cache to ensure fresh load
        speaker_detection.clear_spacy_model_cache()

        mock_nlp = unittest.mock.MagicMock()
        mock_doc = unittest.mock.MagicMock()
        mock_ent = unittest.mock.MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "A"  # Too short
        mock_ent.score = 0.8
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp

        nlp = speaker_detection.get_ner_model(self.cfg)
        result = speaker_detection.extract_person_entities("A", nlp)

        # Should filter out short names
        self.assertEqual(len(result), 0)

    def test_detect_hosts_from_feed_authors(self):
        """Test detecting hosts from RSS feed authors."""
        hosts = speaker_detection.detect_hosts_from_feed(
            feed_title="Test Podcast",
            feed_description="A test podcast",
            feed_authors=["John Doe", "Jane Smith"],
        )
        self.assertEqual(len(hosts), 2)
        self.assertIn("John Doe", hosts)
        self.assertIn("Jane Smith", hosts)

    def test_detect_hosts_from_feed_author_email(self):
        """Test detecting hosts from RSS feed authors with email format."""
        hosts = speaker_detection.detect_hosts_from_feed(
            feed_title="Test Podcast",
            feed_description="A test podcast",
            feed_authors=["John Doe <john@example.com>"],
        )
        self.assertEqual(len(hosts), 1)
        self.assertIn("John Doe", hosts)

    @unittest.skip(
        "TODO: Fix spacy mocking setup - spacy.load() MagicMock interferes with test mocks"
    )
    @patch("podcast_scraper.speaker_detection.get_ner_model")
    @patch("podcast_scraper.speaker_detection.extract_person_entities")
    def test_detect_hosts_from_feed_ner(self, mock_extract, mock_get_model):
        """Test detecting hosts from feed title using NER."""
        mock_nlp = unittest.mock.MagicMock()
        mock_get_model.return_value = mock_nlp
        mock_extract.side_effect = [
            [("John Doe", 0.9)],  # Title entities
            [("Jane Smith", 0.85)],  # Description entities
        ]

        hosts = speaker_detection.detect_hosts_from_feed(
            feed_title="John Doe Podcast",
            feed_description="Hosted by Jane Smith",
            feed_authors=None,
            nlp=mock_nlp,
        )

        self.assertEqual(len(hosts), 2)
        self.assertIn("John Doe", hosts)
        self.assertIn("Jane Smith", hosts)

    @unittest.skip(
        "TODO: Fix spacy mocking setup - spacy.load() MagicMock interferes with test mocks"
    )
    @patch("podcast_scraper.speaker_detection.get_ner_model")
    @patch("podcast_scraper.speaker_detection.extract_person_entities")
    def test_detect_speaker_names_guests(self, mock_extract, mock_get_model):
        """Test detecting guest names from episode title."""
        mock_nlp = unittest.mock.MagicMock()
        mock_get_model.return_value = mock_nlp
        mock_extract.return_value = [("Guest Name", 0.9)]

        cfg = config.Config(
            rss_url=TEST_FEED_URL,
            auto_speakers=True,
        )

        speakers, hosts, succeeded = speaker_detection.detect_speaker_names(
            episode_title="Interview with Guest Name",
            episode_description=None,
            cfg=cfg,
            known_hosts={"Host Name"},
        )

        self.assertTrue(succeeded)
        self.assertIn("Guest Name", speakers)
        self.assertIn("Host Name", hosts)

    @patch("podcast_scraper.speaker_detection.get_ner_model")
    def test_detect_speaker_names_disabled(self, mock_get_model):
        """Test that speaker detection returns defaults when disabled."""
        cfg = config.Config(
            rss_url=TEST_FEED_URL,
            auto_speakers=False,
        )

        speakers, hosts, succeeded = speaker_detection.detect_speaker_names(
            episode_title="Test Episode",
            episode_description=None,
            cfg=cfg,
        )

        self.assertFalse(succeeded)
        self.assertEqual(speakers, ["Host", "Guest"])
        self.assertEqual(len(hosts), 0)

    def test_sanitize_person_name(self):
        """Test sanitizing person names."""
        # Test removing parentheses
        result = speaker_detection._sanitize_person_name("John (Smith)")
        self.assertEqual(result, "John")

        # Test removing trailing punctuation
        result = speaker_detection._sanitize_person_name("John Doe,")
        self.assertEqual(result, "John Doe")

        # Test removing leading punctuation
        result = speaker_detection._sanitize_person_name(",John Doe")
        self.assertEqual(result, "John Doe")

        # Test hyphenated names
        result = speaker_detection._sanitize_person_name("Mary-Jane Watson")
        self.assertEqual(result, "Mary-Jane Watson")

        # Test apostrophes
        result = speaker_detection._sanitize_person_name("O'Brien")
        self.assertEqual(result, "O'Brien")

        # Test too short
        result = speaker_detection._sanitize_person_name("A")
        self.assertIsNone(result)

        # Test only numbers
        result = speaker_detection._sanitize_person_name("123")
        self.assertIsNone(result)


class TestSpeakerDetectionHelpers(unittest.TestCase):
    """Tests for extracted helper functions in speaker_detection module."""

    def test_validate_person_entity(self):
        """Test validating person entity names."""
        # Valid names
        self.assertTrue(speaker_detection._validate_person_entity("John Doe"))
        self.assertTrue(speaker_detection._validate_person_entity("Mary-Jane"))

        # Invalid: too short
        self.assertFalse(speaker_detection._validate_person_entity("A"))
        self.assertFalse(speaker_detection._validate_person_entity(""))

        # Invalid: pure numbers
        self.assertFalse(speaker_detection._validate_person_entity("123"))

        # Invalid: HTML-like patterns
        self.assertFalse(speaker_detection._validate_person_entity("<script>"))

    def test_extract_confidence_score(self):
        """Test extracting confidence score from spaCy entity."""
        # Mock entity with score attribute
        mock_ent = unittest.mock.MagicMock()
        mock_ent.score = 0.85
        score = speaker_detection._extract_confidence_score(mock_ent)
        self.assertEqual(score, 0.85)

        # Mock entity with _ score attribute
        mock_ent2 = unittest.mock.MagicMock()
        del mock_ent2.score
        mock_ent2._.score = 0.9
        score = speaker_detection._extract_confidence_score(mock_ent2)
        self.assertEqual(score, 0.9)

        # Mock entity without score (defaults to 1.0)
        mock_ent3 = unittest.mock.MagicMock()
        # Remove score attribute if it exists
        if hasattr(mock_ent3, "score"):
            delattr(mock_ent3, "score")
        # Ensure _ attribute doesn't exist or doesn't have score
        if hasattr(mock_ent3, "_"):
            if hasattr(mock_ent3._, "score"):
                delattr(mock_ent3._, "score")
        score = speaker_detection._extract_confidence_score(mock_ent3)
        self.assertEqual(score, 1.0)

    def test_split_text_on_separators(self):
        """Test splitting text on common separators."""
        # Test pipe separator
        segments, last = speaker_detection._split_text_on_separators("Title | Guest Name")
        self.assertEqual(len(segments), 2)
        self.assertEqual(last, "Guest Name")

        # Test em dash
        segments, last = speaker_detection._split_text_on_separators("Title — Guest Name")
        self.assertEqual(len(segments), 2)
        self.assertEqual(last, "Guest Name")

        # Test no separator
        segments, last = speaker_detection._split_text_on_separators("Title with Guest Name")
        self.assertEqual(len(segments), 1)
        self.assertIsNone(last)

    def test_calculate_heuristic_score(self):
        """Test calculating heuristic score for guest names."""
        heuristics = {
            "title_position_preference": "end",
            "common_prefixes": ["with"],
            "common_suffixes": [],
        }

        # Name at end of title (should get position bonus)
        score = speaker_detection._calculate_heuristic_score(
            "Guest Name", "Interview with Guest Name", heuristics
        )
        self.assertGreater(score, 0.0)

        # Name not in title (should return 0)
        score = speaker_detection._calculate_heuristic_score(
            "Guest Name", "Different Title", heuristics
        )
        self.assertEqual(score, 0.0)

        # No heuristics (should return 0)
        score = speaker_detection._calculate_heuristic_score("Guest Name", "Title", None)
        self.assertEqual(score, 0.0)

    def test_build_guest_candidates(self):
        """Test building guest candidates dictionary."""
        title_guests = [("Guest1", 0.9), ("Guest2", 0.8)]
        desc_guests = [("Guest1", 0.85)]  # Overlap with Guest1
        heuristics = {"title_position_preference": "end"}

        candidates = speaker_detection._build_guest_candidates(
            title_guests, desc_guests, "Title with Guest1", heuristics
        )

        self.assertIn("Guest1", candidates)
        self.assertIn("Guest2", candidates)
        # Guest1 should have overlap=True
        self.assertTrue(candidates["Guest1"][1])  # appears_in_both
        # Guest2 should have overlap=False
        self.assertFalse(candidates["Guest2"][1])

    def test_select_best_guest(self):
        """Test selecting best guest from candidates."""
        candidates = {
            "Guest1": (0.7, False, 0.2),  # Lower combined score
            "Guest2": (0.8, True, 0.3),  # Higher combined score (overlap + heuristic)
        }

        guest, conf, overlap, heuristic = speaker_detection._select_best_guest(candidates)
        self.assertEqual(guest, "Guest2")
        self.assertEqual(conf, 0.8)
        self.assertTrue(overlap)
        self.assertEqual(heuristic, 0.3)

    def test_analyze_title_position(self):
        """Test analyzing guest name position in title."""
        # Name at start
        pos = speaker_detection._analyze_title_position("John", "John Doe Interview")
        self.assertEqual(pos, "start")

        # Name at end
        pos = speaker_detection._analyze_title_position("John", "Interview with John")
        self.assertEqual(pos, "end")

        # Name in middle
        pos = speaker_detection._analyze_title_position("John", "Interview with John Doe")
        self.assertEqual(pos, "middle")

        # Name not found
        pos = speaker_detection._analyze_title_position("John", "Different Title")
        self.assertIsNone(pos)

    def test_extract_prefix_suffix(self):
        """Test extracting prefix and suffix around guest name."""
        prefix, suffix = speaker_detection._extract_prefix_suffix("John", "Interview with John Doe")
        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)

        # Name not found
        prefix, suffix = speaker_detection._extract_prefix_suffix("John", "Different Title")
        self.assertIsNone(prefix)
        self.assertIsNone(suffix)

    def test_find_common_patterns(self):
        """Test finding common patterns."""
        patterns = ["with", "with", "featuring", "with", "featuring"]
        common = speaker_detection._find_common_patterns(patterns, min_count=2)
        self.assertIn("with", common)
        self.assertIn("featuring", common)

        # No patterns
        common = speaker_detection._find_common_patterns([])
        self.assertEqual(len(common), 0)

    def test_determine_title_position_preference(self):
        """Test determining title position preference."""
        positions = ["end", "end", "end", "start"]
        pref = speaker_detection._determine_title_position_preference(positions)
        self.assertEqual(pref, "end")

        # Not consistent enough
        positions = ["end", "start", "middle"]
        pref = speaker_detection._determine_title_position_preference(positions)
        self.assertIsNone(pref)

    def test_build_speaker_names_list(self):
        """Test building speaker names list."""
        # Hosts and guests
        names, succeeded = speaker_detection._build_speaker_names_list(
            {"Host1", "Host2"}, ["Guest1"], 5
        )
        self.assertTrue(succeeded)
        self.assertIn("Host1", names)
        self.assertIn("Guest1", names)

        # Only hosts, no guests - should return actual host names
        names, succeeded = speaker_detection._build_speaker_names_list({"Host1"}, [], 5)
        self.assertTrue(succeeded)
        self.assertEqual(names[0], "Host1")
        self.assertEqual(len(names), 1)

        # Multiple hosts, no guests - should return sorted host names
        names, succeeded = speaker_detection._build_speaker_names_list(
            {"Jane Smith", "John Doe"}, [], 5
        )
        self.assertTrue(succeeded)
        self.assertEqual(len(names), 2)
        # Should be sorted deterministically
        self.assertEqual(names, ["Jane Smith", "John Doe"])

        # No hosts or guests
        names, succeeded = speaker_detection._build_speaker_names_list(set(), [], 5)
        self.assertFalse(succeeded)
        self.assertEqual(names, ["Host", "Guest"])


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
