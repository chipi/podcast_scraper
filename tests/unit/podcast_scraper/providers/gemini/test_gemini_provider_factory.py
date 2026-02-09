#!/usr/bin/env python3
"""Unit tests for Gemini providers via factory (Issue #194).

These tests verify Gemini provider implementations with mocked API calls,
using factory functions to create providers (tests factory integration).

For standalone provider tests, see test_gemini_provider.py.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock google.genai before importing modules that require it
# Unit tests run without google-genai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_google = MagicMock()
mock_genai_module = MagicMock()
mock_genai_module.configure = Mock()
mock_genai_module.GenerativeModel = Mock()
mock_api_core = MagicMock()
mock_api_core.exceptions = MagicMock()
_patch_google = patch.dict(
    "sys.modules",
    {
        "google": mock_google,
        "google.genai": mock_genai_module,
        "google.api_core": mock_api_core,
    },
)
_patch_google.start()
from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_gemini_providers]


@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiTranscriptionProviderFactory(unittest.TestCase):
    """Test Gemini transcription provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            gemini_api_key="test-api-key-123",
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_initialization(self, mock_genai):
        """Test that Gemini transcription provider initializes correctly via factory."""
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Verify genai.Client was called
        mock_genai.Client.assert_called_once_with(api_key="test-api-key-123")
        self.assertTrue(provider._transcription_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open, mock_genai):
        """Test successful transcription via Gemini API via factory."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        # Mock Gemini Client API response
        mock_response = Mock()
        mock_response.text = "Transcribed text"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3", language="en")
        self.assertEqual(result, "Transcribed text")
        mock_client.models.generate_content.assert_called_once()

    def test_transcribe_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml", transcription_provider="gemini"
                )
            self.assertIn("Gemini API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_factory_creates_gemini_provider(self, mock_genai):
        """Test that factory creates Gemini transcription provider."""
        # Ensure genai is not None (module-level import check)
        if mock_genai is None:
            mock_genai = Mock()
        provider = create_transcription_provider(self.cfg)
        # Factory now returns unified GeminiProvider, not separate provider classes
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")


@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiSpeakerDetectorFactory(unittest.TestCase):
    """Test Gemini speaker detection provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="gemini",
            gemini_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_genai):
        """Test successful speaker detection via Gemini API via factory."""
        # Mock Gemini client and response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "speakers": ["John Doe", "Jane Smith"],
                "hosts": ["John Doe"],
                "guests": ["Jane Smith"],
            }
        )

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        # Mock prompt rendering
        mock_render_prompt.side_effect = lambda name, **kwargs: "test prompt"

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with John Doe and Jane Smith",
            episode_description="A test episode",
            known_hosts={"John Doe"},
        )

        self.assertEqual(len(speakers), 2)
        self.assertIn("John Doe", speakers)
        self.assertIn("Jane Smith", speakers)
        self.assertEqual(detected_hosts, {"John Doe"})
        self.assertTrue(success)

    def test_detect_speakers_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="gemini",
                    auto_speakers=True,
                )
            self.assertIn("Gemini API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_factory_creates_gemini_detector(self, mock_genai):
        """Test that factory creates Gemini speaker detector."""
        # Ensure genai is not None (module-level import check)
        if mock_genai is None:
            mock_genai = Mock()
        detector = create_speaker_detector(self.cfg)
        # Factory now returns unified GeminiProvider, not separate provider classes
        self.assertEqual(detector.__class__.__name__, "GeminiProvider")


@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiSummarizationProviderFactory(unittest.TestCase):
    """Test Gemini summarization provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_metadata=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_genai):
        """Test successful summarization via Gemini API via factory."""
        # Mock Gemini Client API response
        mock_response = Mock()
        mock_response.text = "This is a summary of the transcript."

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        # Mock prompt rendering
        mock_render_prompt.side_effect = lambda name, **kwargs: "test prompt"

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text that needs summarization.")

        self.assertEqual(result["summary"], "This is a summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "gemini")

    def test_summarize_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="gemini",
                    generate_summaries=True,
                    generate_metadata=True,
                )
            self.assertIn("Gemini API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_factory_creates_gemini_provider(self, mock_genai):
        """Test that factory creates Gemini summarization provider."""
        # Ensure genai is not None and has required methods
        if mock_genai is None:
            mock_genai = Mock()
        mock_genai.configure = Mock()
        provider = create_summarization_provider(self.cfg)
        # Factory now returns unified GeminiProvider, not separate provider classes
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")
