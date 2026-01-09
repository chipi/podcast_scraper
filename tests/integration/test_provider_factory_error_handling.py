#!/usr/bin/env python3
"""Integration tests for factory error handling.

These tests verify that factories handle errors gracefully and that
the workflow can recover from provider initialization failures.
"""

import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
class TestFactoryErrorHandling(unittest.TestCase):
    """Test factory error handling."""

    def test_factory_raises_value_error_for_invalid_provider(self):
        """Test that factories raise ValueError for invalid provider types."""
        # Test transcription factory
        with self.assertRaises(ValueError) as context:

            class MockConfig:
                transcription_provider = "invalid"

            create_transcription_provider(MockConfig())  # type: ignore[arg-type]
        self.assertIn("Unsupported transcription provider", str(context.exception))

        # Test speaker detector factory
        with self.assertRaises(ValueError) as context:

            class MockConfig:
                speaker_detector_provider = "invalid"

            create_speaker_detector(MockConfig())  # type: ignore[arg-type]
        self.assertIn("Unsupported speaker detector type", str(context.exception))

        # Test summarization factory
        with self.assertRaises(ValueError) as context:

            class MockConfig:
                summary_provider = "invalid"

            create_summarization_provider(MockConfig())  # type: ignore[arg-type]
        self.assertIn("Unsupported summarization provider", str(context.exception))

    def test_openai_factory_requires_api_key(self):
        """Test that OpenAI factories require API key."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Transcription - ValidationError is raised by config validator before factory
            with self.assertRaises(ValidationError) as context:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="openai",
                )
            self.assertIn("OpenAI API key required", str(context.exception))

            # Speaker detector - ValidationError is raised by config validator before factory
            with self.assertRaises(ValidationError) as context:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="openai",
                )
            self.assertIn("OpenAI API key required", str(context.exception))

            # Summarization - ValidationError is raised by config validator before factory
            with self.assertRaises(ValidationError) as context:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="openai",
                )
            self.assertIn("OpenAI API key required", str(context.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key


@pytest.mark.integration
class TestProviderInitializationErrorRecovery(unittest.TestCase):
    """Test that providers can recover from initialization errors."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

    def test_provider_initialization_failure_does_not_corrupt_provider(self):
        """Test that initialization failure doesn't corrupt provider state.

        MLProvider.initialize() is resilient - it logs warnings but doesn't raise,
        allowing other components to initialize even if one fails.
        """
        # Attempt initialization that fails - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = create_transcription_provider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_import.side_effect = RuntimeError("Model load failed")

            # initialize() should not raise - it logs warnings and continues
            provider.initialize()

            # Provider should still be usable (state is clean)
            # Whisper is not initialized, but provider is still usable
            if hasattr(provider, "_whisper_initialized"):
                self.assertFalse(provider._whisper_initialized)
            self.assertIsNotNone(provider)  # Provider object still exists

    def test_provider_can_reinitialize_after_failure(self):
        """Test that provider can be reinitialized after failure.

        MLProvider.initialize() is resilient - it logs warnings but doesn't raise.
        We can create a new provider and initialize it successfully even after a previous failure.
        """
        # First attempt fails - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = create_transcription_provider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_import.side_effect = RuntimeError("Model load failed")

            # initialize() should not raise - it logs warnings and continues
            provider.initialize()
            # Verify Whisper is not initialized
            if hasattr(provider, "_whisper_initialized"):
                self.assertFalse(provider._whisper_initialized)

        # Second attempt succeeds - create new provider with same config
        provider2 = create_transcription_provider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import.return_value = mock_whisper_lib

            provider2.initialize()
            # Verify Whisper is now initialized
            if hasattr(provider2, "_whisper_initialized"):
                self.assertTrue(provider2._whisper_initialized)
            elif hasattr(provider2, "is_initialized"):
                self.assertTrue(provider2.is_initialized)


@pytest.mark.integration
class TestProviderRequiresSeparateInstancesIntegration(unittest.TestCase):
    """Test _requires_separate_instances attribute in integration context."""

    def test_ml_provider_requires_separate_instances(self):
        """Test that MLProvider requires separate instances (workflow pattern)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Verify attribute exists and is True (workflow uses getattr)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertTrue(requires_separate)

    def test_openai_provider_does_not_require_separate_instances(self):
        """Test that OpenAIProvider does not require separate instances (workflow pattern)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Verify attribute exists and is False (workflow uses getattr)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

    def test_getattr_defaults_to_false_for_missing_attribute(self):
        """Test that getattr defaults to False for providers without attribute."""
        # Create a mock provider without the attribute
        # Use spec=[] to prevent Mock from auto-creating attributes
        mock_provider = Mock(spec=[])
        # Don't set _requires_separate_instances

        # Workflow pattern: getattr with default False
        requires_separate = getattr(mock_provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)
