"""Integration tests for Mistral providers.

These tests verify Mistral provider implementations with mocked API calls.
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

pytestmark = [pytest.mark.integration, pytest.mark.module_mistral_providers]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.mistral
class TestMistralTranscriptionProvider(unittest.TestCase):
    """Test Mistral transcription provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="mistral",
            mistral_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_initialization(self, mock_mistral_class):
        """Test that Mistral transcription provider initializes correctly via factory."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Verify client was created with API key
        mock_mistral_class.assert_called_once()
        self.assertTrue(provider._transcription_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open, mock_mistral_class):
        """Test successful transcription via Mistral Voxtral API via factory."""
        # Mock Mistral client and response
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        # Mock Voxtral transcription response
        mock_response = Mock()
        mock_response.text = "Transcribed text from Mistral Voxtral"
        # Mistral uses 'complete' method, not 'create'
        mock_client.audio.transcriptions.complete.return_value = mock_response
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"  # Return actual bytes
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        # Create a temporary audio file for testing
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            result = provider.transcribe(audio_path, language="en")
            self.assertEqual(result, "Transcribed text from Mistral Voxtral")
            mock_client.audio.transcriptions.complete.assert_called_once()
        finally:
            os.unlink(audio_path)

    def test_transcribe_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml", transcription_provider="mistral"
                )
            self.assertIn("Mistral API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["MISTRAL_API_KEY"] = original_key

    def test_factory_creates_mistral_provider(self):
        """Test that factory creates unified Mistral provider."""
        provider = create_transcription_provider(self.cfg)
        # Verify it's the unified Mistral provider
        self.assertEqual(provider.__class__.__name__, "MistralProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.mistral
class TestMistralSpeakerDetector(unittest.TestCase):
    """Test Mistral speaker detection provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="mistral",
            mistral_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider.MistralProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider.MistralProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(
        self, mock_parse, mock_build_prompt, mock_render_prompt, mock_mistral_class
    ):
        """Test successful speaker detection via Mistral API via factory."""
        # Mock Mistral client and response
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_build_prompt.return_value = "User Prompt"
        mock_render_prompt.return_value = "System Prompt"

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "speakers": ["John Doe", "Jane Smith"],
                "hosts": ["John Doe"],
                "guests": ["Jane Smith"],
            }
        )
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        # Mistral uses 'complete' method, not 'create'
        mock_client.chat.complete.return_value = mock_response
        # _parse_speakers_from_response returns (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["John Doe", "Jane Smith"], {"John Doe"}, True)

        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
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
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="mistral",
                    auto_speakers=True,
                )
            self.assertIn("Mistral API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["MISTRAL_API_KEY"] = original_key

    def test_factory_creates_mistral_detector(self):
        """Test that factory creates unified Mistral provider."""
        detector = create_speaker_detector(self.cfg)
        # Verify it's the unified Mistral provider
        self.assertEqual(detector.__class__.__name__, "MistralProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.mistral
class TestMistralSummarizationProvider(unittest.TestCase):
    """Test Mistral summarization provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="mistral",
            mistral_api_key="test-api-key-123",
            generate_metadata=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider.MistralProvider._build_summarization_prompts"
    )
    def test_summarize_success(
        self, mock_build_prompts, mock_get_metadata, mock_render_prompt, mock_mistral_class
    ):
        """Test successful summarization via Mistral API via factory."""
        # Mock Mistral client and response
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
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

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="This is a test summary of the transcript."))
        ]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        # Mistral uses 'complete' method, not 'create'
        mock_client.chat.complete.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.client = mock_client
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
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="mistral",
                    generate_metadata=True,
                    generate_summaries=True,
                )
            self.assertIn("Mistral API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["MISTRAL_API_KEY"] = original_key

    def test_factory_creates_mistral_provider(self):
        """Test that factory creates unified Mistral provider."""
        provider = create_summarization_provider(self.cfg)
        # Verify it's the unified Mistral provider
        self.assertEqual(provider.__class__.__name__, "MistralProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.mistral
class TestMistralProviderErrorHandling(unittest.TestCase):
    """Test error handling for Mistral providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="mistral",
            mistral_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open, mock_mistral_class):
        """Test that API errors are handled gracefully via factory."""
        # Mock Mistral client to raise exception
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        # Mistral uses 'complete' method, not 'create'
        mock_client.audio.transcriptions.complete.side_effect = Exception("API Error")
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"  # Return actual bytes
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            from podcast_scraper.exceptions import ProviderRuntimeError

            with self.assertRaises(ProviderRuntimeError) as cm:
                provider.transcribe(audio_path)
            self.assertIn("Mistral transcription failed", str(cm.exception))
        finally:
            os.unlink(audio_path)
