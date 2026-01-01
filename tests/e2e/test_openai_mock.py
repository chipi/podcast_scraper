#!/usr/bin/env python3
"""OpenAI E2E server integration tests.

These tests verify that OpenAI providers correctly use the E2E server's
OpenAI mock endpoints via HTTP requests, testing the full HTTP flow.
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


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.openai
class TestOpenAIE2EServerIntegration:
    """Test that OpenAI providers correctly use E2E server endpoints."""

    def test_openai_transcription_provider_uses_e2e_server(self, e2e_server):
        """Test that OpenAI transcription provider uses E2E server endpoints."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            transcription_provider="openai",
        )

        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Verify provider is configured to use E2E server
        assert str(provider.client.base_url).rstrip(
            "/"
        ) == e2e_server.urls.openai_api_base().rstrip(
            "/"
        ), "Provider should use E2E server base URL"

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"FAKE AUDIO DATA")
            audio_path = tmp_file.name

        try:
            # Transcribe should use E2E server endpoints (real HTTP request)
            transcript = provider.transcribe(audio_path)

            # Verify transcript is returned (from E2E server mock)
            assert isinstance(transcript, str)
            assert len(transcript) > 0
            assert "test transcription" in transcript.lower()
        finally:
            # Clean up
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_openai_summarization_provider_uses_e2e_server(self, e2e_server):
        """Test that OpenAI summarization provider uses E2E server endpoints."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            summary_provider="openai",
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Verify provider is configured to use E2E server
        assert str(provider.client.base_url).rstrip(
            "/"
        ) == e2e_server.urls.openai_api_base().rstrip(
            "/"
        ), "Provider should use E2E server base URL"

        # Summarize should use E2E server endpoints (real HTTP request)
        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )

        # Verify summary is returned (from E2E server mock)
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "test summary" in result["summary"].lower()

    def test_openai_speaker_detector_uses_e2e_server(self, e2e_server):
        """Test that OpenAI speaker detector uses E2E server endpoints."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
            speaker_detector_provider="openai",
        )

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Verify detector is configured to use E2E server
        assert str(detector.client.base_url).rstrip(
            "/"
        ) == e2e_server.urls.openai_api_base().rstrip(
            "/"
        ), "Detector should use E2E server base URL"

        # Detect speakers should use E2E server endpoints (real HTTP request)
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
