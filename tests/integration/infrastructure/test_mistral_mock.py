#!/usr/bin/env python3
"""Integration tests for Mistral E2E server mock endpoints.

These tests verify that Mistral providers correctly use the E2E server's
Mistral mock endpoints via the fake SDK client. Mistral uses OpenAI-compatible
API format, so it uses the same endpoints as OpenAI:
- /v1/chat/completions: For summarization and speaker detection
- /v1/audio/transcriptions: For transcription

These tests verify component interactions with infrastructure, not complete
user workflows.
"""

import os
import sys
import tempfile

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
@pytest.mark.mistral
class TestMistralE2EServerIntegration:
    """Test that Mistral providers correctly use E2E server endpoints via fake SDK."""

    def test_mistral_transcription_provider_uses_e2e_server(self, e2e_server, monkeypatch):
        """Test that Mistral transcription provider uses E2E server endpoints."""
        # Configure fake Mistral SDK to use E2E server (same as e2e conftest)
        from tests.fixtures.mock_server.mistral_mock_client import create_fake_mistral_client

        mistral_api_base = e2e_server.urls.mistral_api_base()
        FakeMistral = create_fake_mistral_client(mistral_api_base)

        # Replace Mistral SDK with fake client (same as e2e conftest does)
        monkeypatch.setattr("mistralai.Mistral", FakeMistral)

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            mistral_api_key="test-api-key-123",
            mistral_api_base=mistral_api_base,  # Use E2E server
            transcription_provider="mistral",
            transcribe_missing=True,
        )

        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"FAKE AUDIO DATA")
            audio_path = tmp_file.name

        try:
            # Transcribe should use E2E server endpoints (real HTTP request via fake SDK)
            transcript = provider.transcribe(audio_path)

            # Verify transcript is returned (from E2E server mock)
            assert isinstance(transcript, str)
            assert len(transcript) > 0
            assert "test transcription" in transcript.lower()
        finally:
            # Clean up provider
            provider.cleanup()
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_mistral_summarization_provider_uses_e2e_server(self, e2e_server, monkeypatch):
        """Test that Mistral summarization provider uses E2E server endpoints."""
        # Configure fake Mistral SDK to use E2E server (same as e2e conftest)
        from tests.fixtures.mock_server.mistral_mock_client import create_fake_mistral_client

        mistral_api_base = e2e_server.urls.mistral_api_base()
        FakeMistral = create_fake_mistral_client(mistral_api_base)

        # Replace Mistral SDK with fake client (same as e2e conftest does)
        monkeypatch.setattr("mistralai.Mistral", FakeMistral)

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            mistral_api_key="test-api-key-123",
            mistral_api_base=mistral_api_base,  # Use E2E server
            summary_provider="mistral",
            generate_summaries=True,
            generate_metadata=True,
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Summarize should use E2E server endpoints (real HTTP request via fake SDK)
        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )

        # Verify summary is returned (from E2E server mock)
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "test summary" in result["summary"].lower()

        # Clean up provider
        provider.cleanup()

    def test_mistral_speaker_detector_uses_e2e_server(self, e2e_server, monkeypatch):
        """Test that Mistral speaker detector uses E2E server endpoints."""
        # Configure fake Mistral SDK to use E2E server (same as e2e conftest)
        from tests.fixtures.mock_server.mistral_mock_client import create_fake_mistral_client

        mistral_api_base = e2e_server.urls.mistral_api_base()
        FakeMistral = create_fake_mistral_client(mistral_api_base)

        # Replace Mistral SDK with fake client (same as e2e conftest does)
        monkeypatch.setattr("mistralai.Mistral", FakeMistral)

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            mistral_api_key="test-api-key-123",
            mistral_api_base=mistral_api_base,  # Use E2E server
            speaker_detector_provider="mistral",
            auto_speakers=True,
        )

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Detect speakers should use E2E server endpoints (real HTTP request via fake SDK)
        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with Alice and Bob",
            episode_description="Alice interviews Bob about their work",
            known_hosts={"Alice"},
        )

        # Verify speakers are returned (from E2E server mock)
        assert isinstance(speakers, list)
        assert len(speakers) > 0
        assert isinstance(detected_hosts, set)
        assert isinstance(success, bool)

        # Clean up detector
        detector.cleanup()
