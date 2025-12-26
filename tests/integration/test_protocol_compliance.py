#!/usr/bin/env python3
"""Protocol compliance tests for providers (Stage 5).

These tests verify that all providers correctly implement their protocols.
"""

import inspect
import unittest
from typing import get_type_hints

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.base import SpeakerDetector
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.base import SummarizationProvider
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.base import TranscriptionProvider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
class TestTranscriptionProviderProtocol(unittest.TestCase):
    """Test TranscriptionProvider protocol compliance."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        self.provider = create_transcription_provider(self.cfg)

    def test_provider_implements_transcribe_method(self):
        """Test that provider implements transcribe() method."""
        self.assertTrue(hasattr(self.provider, "transcribe"))
        self.assertTrue(callable(getattr(self.provider, "transcribe")))

    def test_transcribe_method_signature(self):
        """Test that transcribe() has correct signature."""
        sig = inspect.signature(self.provider.transcribe)
        params = list(sig.parameters.keys())

        # Should have audio_path and optional language
        self.assertIn("audio_path", params)
        self.assertIn("language", params)

    def test_provider_type_checking(self):
        """Test that provider can be used as TranscriptionProvider type."""
        # This test verifies that mypy would accept the provider as TranscriptionProvider
        # We can't actually run mypy here, but we can verify the protocol structure
        protocol_methods = ["transcribe"]
        for method_name in protocol_methods:
            self.assertTrue(
                hasattr(self.provider, method_name),
                f"Provider missing protocol method: {method_name}",
            )


@pytest.mark.integration
class TestSpeakerDetectorProtocol(unittest.TestCase):
    """Test SpeakerDetector protocol compliance."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="ner"
        )
        self.detector = create_speaker_detector(self.cfg)

    def test_detector_implements_detect_speakers_method(self):
        """Test that detector implements detect_speakers() method."""
        self.assertTrue(hasattr(self.detector, "detect_speakers"))
        self.assertTrue(callable(getattr(self.detector, "detect_speakers")))

    def test_detector_implements_analyze_patterns_method(self):
        """Test that detector implements analyze_patterns() method."""
        self.assertTrue(hasattr(self.detector, "analyze_patterns"))
        self.assertTrue(callable(getattr(self.detector, "analyze_patterns")))

    def test_detect_speakers_method_signature(self):
        """Test that detect_speakers() has correct signature."""
        sig = inspect.signature(self.detector.detect_speakers)
        params = list(sig.parameters.keys())

        # Should have episode_title, episode_description, known_hosts
        self.assertIn("episode_title", params)
        self.assertIn("episode_description", params)
        self.assertIn("known_hosts", params)

    def test_analyze_patterns_method_signature(self):
        """Test that analyze_patterns() has correct signature."""
        sig = inspect.signature(self.detector.analyze_patterns)
        params = list(sig.parameters.keys())

        # Should have episodes and known_hosts
        self.assertIn("episodes", params)
        self.assertIn("known_hosts", params)

    def test_detector_type_checking(self):
        """Test that detector can be used as SpeakerDetector type."""
        protocol_methods = ["detect_speakers", "analyze_patterns"]
        for method_name in protocol_methods:
            self.assertTrue(
                hasattr(self.detector, method_name),
                f"Detector missing protocol method: {method_name}",
            )


@pytest.mark.integration
class TestSummarizationProviderProtocol(unittest.TestCase):
    """Test SummarizationProvider protocol compliance."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        self.provider = create_summarization_provider(self.cfg)

    def test_provider_implements_summarize_method(self):
        """Test that provider implements summarize() method."""
        self.assertTrue(hasattr(self.provider, "summarize"))
        self.assertTrue(callable(getattr(self.provider, "summarize")))

    def test_summarize_method_signature(self):
        """Test that summarize() has correct signature."""
        sig = inspect.signature(self.provider.summarize)
        params = list(sig.parameters.keys())

        # Should have text, episode_title, episode_description, params
        self.assertIn("text", params)
        self.assertIn("episode_title", params)
        self.assertIn("episode_description", params)
        self.assertIn("params", params)

    def test_summarize_return_type(self):
        """Test that summarize() returns correct type."""
        # We can't actually call it without initializing, but we can check the signature
        sig = inspect.signature(self.provider.summarize)
        return_annotation = sig.return_annotation

        # Should return Dict[str, Any]
        self.assertIn("Dict", str(return_annotation))

    def test_provider_type_checking(self):
        """Test that provider can be used as SummarizationProvider type."""
        protocol_methods = ["summarize"]
        for method_name in protocol_methods:
            self.assertTrue(
                hasattr(self.provider, method_name),
                f"Provider missing protocol method: {method_name}",
            )


@pytest.mark.integration
class TestProtocolTypeHints(unittest.TestCase):
    """Test that protocols have proper type hints."""

    def test_transcription_provider_type_hints(self):
        """Test that TranscriptionProvider protocol has type hints."""
        hints = get_type_hints(TranscriptionProvider.transcribe)
        self.assertIn("return", hints)

    def test_speaker_detector_provider_type_hints(self):
        """Test that SpeakerDetector protocol has type hints."""
        hints = get_type_hints(SpeakerDetector.detect_speakers)
        self.assertIn("return", hints)

        hints = get_type_hints(SpeakerDetector.analyze_patterns)
        self.assertIn("return", hints)

    def test_summarization_provider_type_hints(self):
        """Test that SummarizationProvider protocol has type hints."""
        hints = get_type_hints(SummarizationProvider.summarize)
        self.assertIn("return", hints)
