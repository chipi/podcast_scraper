#!/usr/bin/env python3
"""OpenAI mock verification tests for E2E tests.

These tests verify that OpenAI providers use mocked clients in E2E tests
and return realistic mock responses.
"""

import os
import sys
import tempfile
import unittest

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
@pytest.mark.skip(
    reason="OpenAI E2E tests skipped for now - infrastructure ready but tests disabled"
)
class TestOpenAIMock(unittest.TestCase):
    """Test that OpenAI providers use mocked clients in E2E tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
        )

    def test_openai_transcription_provider_uses_mock(self):
        """Test that OpenAI transcription provider uses mocked client."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"FAKE AUDIO DATA")
            audio_path = tmp_file.name

        try:
            # Transcribe should use mocked client (no real API call)
            transcript = provider.transcribe(audio_path)

            # Verify transcript is returned (from mock)
            self.assertIsInstance(transcript, str)
            self.assertGreater(len(transcript), 0)
            self.assertIn("test transcription", transcript.lower())
        finally:
            # Clean up
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_openai_summarization_provider_uses_mock(self):
        """Test that OpenAI summarization provider uses mocked client."""
        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Summarize should use mocked client (no real API call)
        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )

        # Verify summary is returned (from mock)
        self.assertIn("summary", result)
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 0)
        self.assertIn("test summary", result["summary"].lower())

    def test_openai_speaker_detector_uses_mock(self):
        """Test that OpenAI speaker detector uses mocked client."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Detect speakers should use mocked client (no real API call)
        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with Alice and Bob",
            episode_description="Alice interviews Bob about their work",
            known_hosts={"Alice"},
        )

        # Verify speakers are returned (from mock)
        self.assertIsInstance(speakers, list)
        self.assertGreater(len(speakers), 0)
        self.assertIsInstance(detected_hosts, set)
        self.assertIsInstance(success, bool)
