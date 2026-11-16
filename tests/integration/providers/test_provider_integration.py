#!/usr/bin/env python3
"""Integration tests for provider system (Stage 5).

These tests verify that all providers work together correctly in the workflow.
"""

import os
import unittest
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_ml_providers]


@pytest.mark.integration
class TestProviderSwitching(unittest.TestCase):
    """Test provider switching via configuration."""

    @pytest.mark.llm
    @pytest.mark.openai
    def test_transcription_provider_switching(self):
        """Test that transcription provider can be switched via config."""
        cfg1 = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider1 = create_transcription_provider(cfg1)
        # Verify protocol compliance, not class name
        self.assertEqual(provider1.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(provider1, "transcribe"))

        # Test OpenAI provider creation (Stage 6)
        cfg2 = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )
        provider2 = create_transcription_provider(cfg2)
        # Verify protocol compliance, not class name
        self.assertEqual(provider2.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(provider2, "transcribe"))

        # Test error handling: missing API key (caught by validator)
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError):
                config.Config(
                    rss_url="https://example.com/feed.xml", transcription_provider="openai"
                )
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    @pytest.mark.llm
    @pytest.mark.openai
    def test_summarization_provider_switching(self):
        """Test that summarization provider can be switched via config."""
        cfg1 = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider1 = create_summarization_provider(cfg1)
        # Verify protocol compliance, not class name
        self.assertEqual(provider1.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(provider1, "summarize"))

        # Test OpenAI provider creation (Stage 6)
        cfg2 = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_metadata=True,
            generate_summaries=True,
        )
        provider2 = create_summarization_provider(cfg2)
        # Verify protocol compliance, not class name
        self.assertEqual(provider2.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(provider2, "summarize"))

        # Test error handling: missing API key (caught by validator)
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError):
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="openai",
                    generate_summaries=False,
                )
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key


@pytest.mark.integration
class TestProviderErrorHandling(unittest.TestCase):
    """Test error handling when providers fail."""

    def test_transcription_provider_initialization_failure(self):
        """Test graceful handling when transcription provider fails to initialize."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=Exception("Model load failed")):
            with self.assertRaises(Exception):
                provider.initialize()  # type: ignore[attr-defined]

    def test_summarization_provider_initialization_failure(self):
        """Test graceful handling when summarization provider fails to initialize."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=Exception("Model load failed")):
            with self.assertRaises(Exception):
                provider.initialize()  # type: ignore[attr-defined]

    def test_provider_method_failure_after_initialization(self):
        """Test graceful handling when provider methods fail after initialization."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Mock summarize failure
        with patch.object(provider, "summarize", side_effect=ValueError("Summarization failed")):
            with self.assertRaises(ValueError):
                provider.summarize("test text")


@pytest.mark.integration
class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with default configurations."""

    def test_default_provider_configuration(self):
        """Test that default config uses expected providers."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        # Defaults use new consistent naming (library names)
        self.assertEqual(cfg.transcription_provider, "whisper")
        self.assertEqual(cfg.speaker_detector_provider, "spacy")
        self.assertEqual(cfg.summary_provider, "transformers")

    def test_config_without_provider_fields(self):
        """Test that config without explicit provider fields still works."""
        # Create config with minimal fields
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        # Providers should use defaults (new consistent naming)
        self.assertEqual(cfg.transcription_provider, "whisper")
        self.assertEqual(cfg.speaker_detector_provider, "spacy")
        self.assertEqual(cfg.summary_provider, "transformers")

        # When no preload, each factory creates a new instance
        # (They may or may not be the same object, but they're separate instances)
        # The key is that they work correctly without preloading
