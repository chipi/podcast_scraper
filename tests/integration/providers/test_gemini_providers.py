"""Integration tests for Gemini providers.

These tests verify Gemini provider implementations with mocked API calls.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_gemini_providers]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiTranscriptionProvider(unittest.TestCase):
    """Test Gemini transcription provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_initialization(self, mock_genai):
        """Test that Gemini transcription provider initializes correctly via factory."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Verify genai was configured with API key
        mock_genai.configure.assert_called_once_with(api_key="test-api-key-123")
        self.assertTrue(provider._transcription_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open, mock_genai):
        """Test successful transcription via Gemini API via factory."""
        # Mock Gemini SDK response
        mock_response = Mock()
        mock_response.text = "Transcribed text from Gemini"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Create a temporary audio file for testing
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            result = provider.transcribe(audio_path, language="en")
            self.assertEqual(result, "Transcribed text from Gemini")
            mock_model.generate_content.assert_called_once()
        finally:
            os.unlink(audio_path)

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

    def test_factory_creates_gemini_provider(self):
        """Test that factory creates unified Gemini provider."""
        provider = create_transcription_provider(self.cfg)
        # Verify it's the unified Gemini provider
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiSpeakerDetector(unittest.TestCase):
    """Test Gemini speaker detection provider."""

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
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider.GeminiProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider.GeminiProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(
        self, mock_parse, mock_build_prompt, mock_render_prompt, mock_genai
    ):
        """Test successful speaker detection via Gemini API via factory."""
        # Mock Gemini SDK response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "speakers": ["John Doe", "Jane Smith"],
                "hosts": ["John Doe"],
                "guests": ["Jane Smith"],
            }
        )
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_build_prompt.return_value = "User Prompt"
        mock_render_prompt.return_value = "System Prompt"
        # _parse_speakers_from_response returns (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["John Doe", "Jane Smith"], {"John Doe"}, True)

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

    def test_factory_creates_gemini_detector(self):
        """Test that factory creates unified Gemini provider."""
        detector = create_speaker_detector(self.cfg)
        # Verify it's the unified Gemini provider
        self.assertEqual(detector.__class__.__name__, "GeminiProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiSummarizationProvider(unittest.TestCase):
    """Test Gemini summarization provider."""

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
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider.GeminiProvider._build_summarization_prompts"
    )
    def test_summarize_success(
        self, mock_build_prompts, mock_get_metadata, mock_render_prompt, mock_genai
    ):
        """Test successful summarization via Gemini API via factory."""
        # Mock Gemini SDK response
        mock_response = Mock()
        mock_response.text = "This is a test summary of the transcript."
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        # _build_summarization_prompts returns (system_prompt, user_prompt, system_prompt_name, user_prompt_name, paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "summarization/system_v1",
            "summarization/user_v1",
            1,
            5,
        )
        mock_render_prompt.return_value = "Rendered Prompt"
        mock_get_metadata.return_value = {"version": "v1"}

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize(
            text="This is a long transcript that needs to be summarized.",
            episode_title="Test Episode",
        )

        self.assertIn("summary", result)
        self.assertEqual(result["summary"], "This is a test summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

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
                    generate_metadata=True,
                    generate_summaries=True,
                )
            self.assertIn("Gemini API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key

    def test_factory_creates_gemini_provider(self):
        """Test that factory creates unified Gemini provider."""
        provider = create_summarization_provider(self.cfg)
        # Verify it's the unified Gemini provider
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiProviderErrorHandling(unittest.TestCase):
    """Test error handling for Gemini providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open, mock_genai):
        """Test that API errors are handled gracefully via factory."""
        # Mock Gemini SDK to raise exception
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            from podcast_scraper.exceptions import ProviderRuntimeError

            with self.assertRaises(ProviderRuntimeError) as cm:
                provider.transcribe(audio_path)
            self.assertIn("Gemini transcription failed", str(cm.exception))
        finally:
            os.unlink(audio_path)
