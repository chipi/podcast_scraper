#!/usr/bin/env python3
"""Tests for utility functions."""

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
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Mock ML dependencies before importing modules that require them
# Unit tests run without ML dependencies installed
with patch.dict("sys.modules", {"spacy": MagicMock()}):
    from podcast_scraper.providers.ml import speaker_detection
    from podcast_scraper.providers.ml.ml_provider import MLProvider

# Import from parent conftest explicitly to avoid conflicts
import importlib.util

parent_tests_dir = Path(__file__).parent.parent
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

from podcast_scraper import config


class TestFormatScreenplay(unittest.TestCase):
    """Tests for format_screenplay_from_segments function (now in MLProvider)."""

    def test_format_screenplay(self):
        """Test formatting screenplay from segments using MLProvider."""
        cfg = create_test_config()
        provider = MLProvider(cfg)

        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello, this is speaker one."},
            {"start": 6.0, "end": 10.0, "text": "And this is speaker two."},
            {"start": 15.0, "end": 20.0, "text": "Speaker one again."},
        ]
        result = provider.format_screenplay_from_segments(
            segments, 2, ["Speaker1", "Speaker2"], 2.0
        )
        self.assertIn("Speaker1", result)
        self.assertIn("Speaker2", result)
        self.assertIn("Hello", result)


# TestIntegrationMain moved to tests/test_integration.py


class TestModelLoading(unittest.TestCase):
    """Test that Whisper and spaCy models can be loaded (via MLProvider)."""

    # Note: Whisper model loading tests moved to test_ml_provider.py
    # These tests now focus on spaCy model loading

    @patch("podcast_scraper.providers.ml.speaker_detection._load_spacy_model")
    def test_spacy_model_loading(self, mock_load):
        """Test that spaCy model can be loaded."""
        cfg = create_test_config(
            auto_speakers=True,
            language="en",
            ner_model=None,  # Should default to en_core_web_sm
        )

        # Mock spaCy model
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        # Test that get_ner_model works
        nlp = speaker_detection.get_ner_model(cfg)
        self.assertIsNotNone(nlp)
        self.assertEqual(nlp, mock_nlp)
        mock_load.assert_called_once_with(config.DEFAULT_NER_MODEL)

    def test_spacy_model_validation(self):
        """Test that spaCy model name validation works."""
        # Test valid model names
        self.assertTrue(speaker_detection._validate_model_name("en_core_web_sm"))
        self.assertTrue(speaker_detection._validate_model_name("en_core_web_md"))
        self.assertTrue(speaker_detection._validate_model_name("fr_core_news_sm"))
        # Test invalid model names (should prevent command injection)
        self.assertFalse(speaker_detection._validate_model_name("en_core_web_sm; rm -rf /"))
        self.assertFalse(speaker_detection._validate_model_name("en_core_web_sm && ls"))
        self.assertFalse(speaker_detection._validate_model_name(""))
        self.assertFalse(speaker_detection._validate_model_name("a" * 101))  # Too long

    # Whisper model selection tests moved to test_ml_provider.py
    # These tests are now covered by MLProvider tests


# Metadata test classes moved to tests/test_metadata.py
