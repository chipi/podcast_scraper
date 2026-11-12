#!/usr/bin/env python3
"""Integration tests for mixed provider configurations.

These tests verify that different capabilities can use different providers
and that they work correctly together.
"""

import unittest

import pytest

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider

# Mock and patch not used in this file
# from unittest.mock import Mock, patch  # noqa: F401


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.openai
class TestMixedProviderConfigurations(unittest.TestCase):
    """Test mixed provider configurations."""

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
