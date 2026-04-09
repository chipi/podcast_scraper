#!/usr/bin/env python3
"""Integration tests for mixed provider configurations.

These tests verify that different capabilities can use different providers
and that they work correctly together.
"""

import unittest

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider

# Mock and patch not used in this file
# from unittest.mock import Mock, patch  # noqa: F401


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.openai
class TestMixedProviderConfigurations(unittest.TestCase):
    """Test mixed provider configurations."""

    def test_whisper_transcription_with_openai_speaker_detection(self):
        """Test using Whisper for transcription and OpenAI for speaker detection."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=False,
            auto_speakers=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)

        # Transcription should be MLProvider
        self.assertEqual(transcription_provider.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(transcription_provider, "transcribe"))

        # Speaker detection should be OpenAIProvider
        self.assertEqual(speaker_detector.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))

    def test_openai_transcription_with_ner_speaker_detection(self):
        """Test using OpenAI for transcription and NER for speaker detection."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            speaker_detector_provider="spacy",
            openai_api_key="sk-test123",
            transcribe_missing=False,
            auto_speakers=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)

        # Transcription should be OpenAIProvider
        self.assertEqual(transcription_provider.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(transcription_provider, "transcribe"))

        # Speaker detection should be MLProvider
        self.assertEqual(speaker_detector.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))

    def test_transformers_summarization_with_openai_transcription(self):
        """Test using Transformers summarization with OpenAI transcription."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            summary_provider="transformers",
            openai_api_key="sk-test123",
            transcribe_missing=False,
            generate_summaries=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # Transcription should be OpenAIProvider
        self.assertEqual(transcription_provider.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(transcription_provider, "transcribe"))

        # Summarization should be MLProvider
        self.assertEqual(summarization_provider.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(summarization_provider, "summarize"))

    def test_all_different_providers(self):
        """Test using different providers for all three capabilities."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="openai",
            summary_provider="transformers",
            openai_api_key="sk-test123",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # All should be different provider types
        self.assertEqual(transcription_provider.__class__.__name__, "MLProvider")
        self.assertEqual(speaker_detector.__class__.__name__, "OpenAIProvider")
        self.assertEqual(summarization_provider.__class__.__name__, "MLProvider")

        # All should implement their protocols
        self.assertTrue(hasattr(transcription_provider, "transcribe"))
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))
        self.assertTrue(hasattr(summarization_provider, "summarize"))

    def test_mixed_providers_have_correct_requires_separate_instances(self):
        """Test that mixed providers have correct _requires_separate_instances values."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="openai",
            summary_provider="transformers",
            openai_api_key="sk-test123",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # MLProvider should require separate instances
        self.assertTrue(getattr(transcription_provider, "_requires_separate_instances", False))
        self.assertTrue(getattr(summarization_provider, "_requires_separate_instances", False))

        # OpenAIProvider should not require separate instances
        self.assertFalse(getattr(speaker_detector, "_requires_separate_instances", False))
