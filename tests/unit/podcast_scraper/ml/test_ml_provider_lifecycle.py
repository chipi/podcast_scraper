#!/usr/bin/env python3
"""Lifecycle and edge case tests for MLProvider.

These tests focus on provider lifecycle management, error recovery,
and edge cases that strengthen the refactor.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.ml.ml_provider import MLProvider


@pytest.mark.unit
class TestMLProviderLifecycle(unittest.TestCase):
    """Test provider lifecycle edge cases."""

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

    def test_multiple_initialize_calls_idempotent(self):
        """Test that multiple initialize() calls are idempotent."""
        # First initialization - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = MLProvider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import.return_value = mock_whisper_lib

            provider.initialize()
            self.assertTrue(provider._whisper_initialized)

            # Second initialization should not reload
            provider.initialize()
            self.assertTrue(provider._whisper_initialized)
            # Should only load once
            self.assertEqual(mock_whisper_lib.load_model.call_count, 1)

    def test_cleanup_when_not_initialized(self):
        """Test that cleanup() works even when not initialized."""
        provider = MLProvider(self.cfg)

        # Cleanup should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    def test_partial_initialization_cleanup(self):
        """Test cleanup when only some capabilities are initialized."""
        # Initialize only Whisper - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = MLProvider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import.return_value = mock_whisper_lib

            provider.initialize()
            self.assertTrue(provider._whisper_initialized)
            self.assertFalse(provider._spacy_initialized)
            self.assertFalse(provider._transformers_initialized)

        # Cleanup should work
        provider.cleanup()
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider.is_initialized)

    def test_initialization_failure_does_not_corrupt_state(self):
        """Test that initialization failure doesn't leave provider in corrupted state."""
        # Attempt initialization that fails - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = MLProvider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_import.side_effect = RuntimeError("Model load failed")

            with self.assertRaises(RuntimeError):
                provider.initialize()

            # State should be clean
            self.assertFalse(provider._whisper_initialized)
            self.assertIsNone(provider._whisper_model)
            self.assertFalse(provider.is_initialized)

    def test_cleanup_after_initialization_failure(self):
        """Test that cleanup works after initialization failure."""
        # Attempt initialization that fails - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = MLProvider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_import.side_effect = RuntimeError("Model load failed")

            try:
                provider.initialize()
            except RuntimeError:
                pass

        # Cleanup should still work
        provider.cleanup()
        self.assertFalse(provider.is_initialized)


@pytest.mark.unit
class TestMLProviderRequiresSeparateInstances(unittest.TestCase):
    """Test _requires_separate_instances attribute behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
        )

    def test_ml_provider_requires_separate_instances(self):
        """Test that MLProvider requires separate instances for thread safety."""
        provider = MLProvider(self.cfg)

        # MLProvider should require separate instances (models are not thread-safe)
        self.assertTrue(provider._requires_separate_instances)

        # Verify attribute exists and is accessible via getattr (workflow pattern)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertTrue(requires_separate)

    def test_requires_separate_instances_accessible_via_getattr(self):
        """Test that _requires_separate_instances is accessible via getattr (workflow pattern)."""
        provider = MLProvider(self.cfg)

        # Workflow uses getattr() to check this attribute
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertTrue(requires_separate)

        # Should default to False if attribute doesn't exist (defensive)
        # Use spec=[] to prevent Mock from auto-creating attributes
        fake_provider = Mock(spec=[])
        default_value = getattr(fake_provider, "_requires_separate_instances", False)
        self.assertFalse(default_value)


@pytest.mark.unit
class TestMLProviderStateConsistency(unittest.TestCase):
    """Test provider state consistency after operations."""

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

    def test_is_initialized_reflects_any_capability(self):
        """Test that is_initialized reflects if any capability is initialized."""
        provider = MLProvider(self.cfg)
        self.assertFalse(provider.is_initialized)

        # Initialize only Whisper - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = MLProvider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import.return_value = mock_whisper_lib

            provider.initialize()
            self.assertTrue(provider.is_initialized)
            self.assertTrue(provider._whisper_initialized)

    def test_state_after_cleanup(self):
        """Test that state is clean after cleanup."""
        # Initialize - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = MLProvider(cfg)
        with patch("podcast_scraper.ml.ml_provider._import_third_party_whisper") as mock_import:
            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import.return_value = mock_whisper_lib

            provider.initialize()
            self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()

        # State should be clean
        self.assertFalse(provider.is_initialized)
        self.assertFalse(provider._whisper_initialized)
        self.assertIsNone(provider._whisper_model)
        self.assertFalse(provider._spacy_initialized)
        self.assertIsNone(provider._spacy_nlp)
        self.assertFalse(provider._transformers_initialized)
        self.assertIsNone(provider._map_model)
        self.assertIsNone(provider._reduce_model)
