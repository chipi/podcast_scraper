#!/usr/bin/env python3
"""Unit tests for OpenAI providers (Stage 6).

These tests verify OpenAI provider implementations with mocked API calls.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_openai_providers]


@pytest.mark.llm
@pytest.mark.openai
class TestOpenAITranscriptionProvider(unittest.TestCase):
    """Test OpenAI transcription provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )

    @patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
    def test_provider_initialization(self, mock_openai_class):
        """Test that OpenAI transcription provider initializes correctly via factory."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # base_url is only included if openai_api_base is set
        expected_kwargs = {"api_key": "sk-test123"}
        if self.cfg.openai_api_base:
            expected_kwargs["base_url"] = self.cfg.openai_api_base
        mock_openai_class.assert_called_once_with(**expected_kwargs)
        self.assertTrue(provider._transcription_initialized)

    @patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
    def test_transcribe_success(self, mock_openai_class):
        """Test successful transcription via OpenAI API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = "Transcribed text"

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Create a temporary audio file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            result = provider.transcribe(audio_path, language="en")
            self.assertEqual(result, "Transcribed text")
            mock_client.audio.transcriptions.create.assert_called_once()
        finally:
            import os

            os.unlink(audio_path)

    def test_transcribe_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml", transcription_provider="openai"
                )
            self.assertIn("OpenAI API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_factory_creates_openai_provider(self):
        """Test that factory creates OpenAI transcription provider."""
        provider = create_transcription_provider(self.cfg)
        # Factory now returns unified OpenAIProvider, not separate provider classes
        self.assertEqual(provider.__class__.__name__, "OpenAIProvider")


@pytest.mark.llm
@pytest.mark.openai
class TestOpenAISpeakerDetector(unittest.TestCase):
    """Test OpenAI speaker detection provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_openai_class):
        """Test successful speaker detection via OpenAI API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock prompt rendering
        mock_render_prompt.side_effect = [
            "System prompt",  # system prompt
            "User prompt",  # user prompt
        ]

        # Mock API response with JSON
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "speakers": ["John Doe", "Jane Smith"],
                            "hosts": ["John Doe"],
                            "guests": ["Jane Smith"],
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

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
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="openai",
                    auto_speakers=True,
                )
            self.assertIn("OpenAI API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_factory_creates_openai_detector(self):
        """Test that factory creates OpenAI speaker detector."""
        detector = create_speaker_detector(self.cfg)
        # Factory now returns unified OpenAIProvider, not separate provider classes
        self.assertEqual(detector.__class__.__name__, "OpenAIProvider")


@pytest.mark.llm
@pytest.mark.openai
class TestOpenAISummarizationProvider(unittest.TestCase):
    """Test OpenAI summarization provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_metadata=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_openai_class):
        """Test successful summarization via OpenAI API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock prompt rendering
        mock_render_prompt.side_effect = [
            "System prompt",  # system prompt
            "User prompt with transcript",  # user prompt
        ]

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="This is a test summary of the transcript."))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize(
            text="This is a long transcript that needs to be summarized.",
            episode_title="Test Episode",
        )

        self.assertIn("summary", result)
        self.assertEqual(result["summary"], "This is a test summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "openai")

    def test_summarize_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="openai",
                    generate_metadata=True,
                    generate_summaries=True,
                )
            self.assertIn("OpenAI API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_factory_creates_openai_provider(self):
        """Test that factory creates OpenAI summarization provider."""
        provider = create_summarization_provider(self.cfg)
        # Factory now returns unified OpenAIProvider, not separate provider classes
        self.assertEqual(provider.__class__.__name__, "OpenAIProvider")


@pytest.mark.llm
@pytest.mark.openai
class TestOpenAIProviderErrorHandling(unittest.TestCase):
    """Test error handling for OpenAI providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )

    @patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
    def test_transcribe_api_error(self, mock_openai_class):
        """Test that API errors are handled gracefully via factory."""
        # Mock OpenAI client to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("API Error")

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            from podcast_scraper.exceptions import ProviderRuntimeError

            with self.assertRaises(ProviderRuntimeError) as cm:
                provider.transcribe(audio_path)
            self.assertIn("OpenAI transcription failed", str(cm.exception))
        finally:
            import os

            os.unlink(audio_path)
