#!/usr/bin/env python3
"""Integration tests for Gemini E2E server mock endpoints.

These tests verify that Gemini providers correctly use the E2E server's
Gemini mock endpoints. Note: The Gemini SDK (google-genai) may not
support custom base URLs directly, so these tests mock the SDK calls.

These tests verify component interactions with infrastructure, not complete
user workflows.
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.gemini
class TestGeminiE2EServerIntegration:
    """Test that Gemini providers correctly use E2E server endpoints.

    Note: The Gemini SDK may not support custom base URLs directly.
    These tests mock the SDK calls to simulate E2E server usage.
    """

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_transcription_provider_uses_e2e_server(self, mock_genai, e2e_server):
        """Test that Gemini transcription provider can be used with E2E server."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            gemini_api_key="test-api-key-123",
            gemini_api_base=e2e_server.urls.gemini_api_base(),  # Use E2E server
            transcription_provider="gemini",
            transcribe_missing=True,
        )

        # Mock Gemini SDK response
        mock_response = Mock()
        mock_response.text = "This is a test transcription from Gemini E2E server."
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"FAKE AUDIO DATA")
            audio_path = tmp_file.name

        try:
            # Transcribe should work with mocked SDK
            transcript = provider.transcribe(audio_path)

            # Verify transcript is returned
            assert isinstance(transcript, str)
            assert len(transcript) > 0
            assert "test transcription" in transcript.lower() or "gemini" in transcript.lower()
        finally:
            # Clean up provider
            provider.cleanup()
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_summarization_provider_uses_e2e_server(self, mock_genai, e2e_server):
        """Test that Gemini summarization provider can be used with E2E server."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            gemini_api_key="test-api-key-123",
            gemini_api_base=e2e_server.urls.gemini_api_base(),  # Use E2E server
            summary_provider="gemini",
            generate_summaries=True,
            generate_metadata=True,
        )

        # Mock Gemini SDK response
        mock_response = Mock()
        mock_response.text = "This is a test summary from Gemini E2E server."

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Summarize should work with mocked SDK
        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )

        # Verify summary is returned
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "test summary" in result["summary"].lower() or "gemini" in result["summary"].lower()

        # Clean up provider
        provider.cleanup()

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_speaker_detector_uses_e2e_server(self, mock_genai, e2e_server):
        """Test that Gemini speaker detector can be used with E2E server."""
        import json

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            gemini_api_key="test-api-key-123",
            gemini_api_base=e2e_server.urls.gemini_api_base(),  # Use E2E server
            speaker_detector_provider="gemini",
            auto_speakers=True,
        )

        # Mock Gemini SDK response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "speakers": ["Alice", "Bob"],
                "hosts": ["Alice"],
                "guests": ["Bob"],
            }
        )

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Detect speakers should work with mocked SDK
        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with Alice and Bob",
            episode_description="Alice interviews Bob about their work",
            known_hosts={"Alice"},
        )

        # Verify speakers are returned
        assert isinstance(speakers, list)
        assert len(speakers) > 0
        assert isinstance(detected_hosts, set)
        assert isinstance(success, bool)

        # Clean up detector
        detector.cleanup()
