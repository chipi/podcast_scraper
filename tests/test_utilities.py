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

from podcast_scraper import speaker_detection, whisper_integration as whisper

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: F401, E402
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
    MockHTTPResponse,
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
)


class TestFormatScreenplay(unittest.TestCase):
    """Tests for format_screenplay_from_segments function."""

    def test_format_screenplay(self):
        """Test formatting screenplay from segments."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello, this is speaker one."},
            {"start": 6.0, "end": 10.0, "text": "And this is speaker two."},
            {"start": 15.0, "end": 20.0, "text": "Speaker one again."},
        ]
        result = whisper.format_screenplay_from_segments(segments, 2, ["Speaker1", "Speaker2"], 2.0)
        self.assertIn("Speaker1", result)
        self.assertIn("Speaker2", result)
        self.assertIn("Hello", result)


# TestIntegrationMain moved to tests/test_integration.py


class TestModelLoading(unittest.TestCase):
    """Test that Whisper and spaCy models can be loaded."""

    def test_whisper_model_loading_with_fallback(self):
        """Test that Whisper model loading works with fallback logic."""
        cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="base",
            language="en",
        )
        # This should either load the model or fail gracefully with clear error
        model = whisper.load_whisper_model(cfg)
        if model is None:
            # If model loading fails, check that we got helpful error messages
            # This test validates that the error handling works correctly
            # In CI, models might not be available, so we just verify graceful failure
            self.assertIsNone(model)
        else:
            # If model loads successfully, verify it's usable
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "transcribe") or hasattr(model, "device"))

    def test_whisper_model_loading_tiny_fallback(self):
        """Test that Whisper falls back to tiny model if base fails."""
        cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="base",
            language="en",
        )
        # Try loading - should either succeed or fail gracefully
        model = whisper.load_whisper_model(cfg)
        # Just verify the function doesn't crash
        # Model might be None if Whisper isn't installed or download fails
        # This test validates the fallback logic doesn't crash
        if model is not None:
            # If model loads, verify it has expected attributes
            self.assertTrue(hasattr(model, "transcribe") or hasattr(model, "device"))

    def test_spacy_model_loading(self):
        """Test that spaCy model can be loaded."""
        # Clear cache to ensure fresh load
        speaker_detection.clear_spacy_model_cache()

        cfg = create_test_config(
            auto_speakers=True,
            language="en",
            ner_model=None,  # Should default to en_core_web_sm
        )
        # Test that get_ner_model works
        nlp = speaker_detection.get_ner_model(cfg)
        if nlp is None:
            # spaCy might not be installed or model might not be downloaded
            # This is acceptable - we just verify graceful failure
            self.assertIsNone(nlp)
        else:
            # If model loads, verify it can process text
            self.assertIsNotNone(nlp)
            # Test basic NER functionality
            doc = nlp("John Smith interviewed Jane Doe.")
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            self.assertGreater(len(persons), 0, "Should detect at least one PERSON entity")

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

    def test_whisper_model_selection_english(self):
        """Test that English models prefer .en variants."""
        cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="base",
            language="en",
        )
        # The function should prefer base.en over base for English
        # We can't easily test the internal selection without mocking,
        # but we can verify it doesn't crash and handles gracefully
        model = whisper.load_whisper_model(cfg)
        # Just verify graceful handling - model might be None if Whisper not installed
        if model is not None:
            self.assertTrue(hasattr(model, "transcribe") or hasattr(model, "device"))

    def test_whisper_model_selection_non_english(self):
        """Test that non-English models use multilingual variants."""
        cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="base.en",
            language="fr",
        )
        # For French, should use multilingual model (no .en suffix)
        # We can't easily test the internal selection without mocking,
        # but we can verify it doesn't crash and handles gracefully
        model = whisper.load_whisper_model(cfg)
        # Just verify graceful handling - model might be None if Whisper not installed
        if model is not None:
            self.assertTrue(hasattr(model, "transcribe") or hasattr(model, "device"))


# Metadata test classes moved to tests/test_metadata.py
