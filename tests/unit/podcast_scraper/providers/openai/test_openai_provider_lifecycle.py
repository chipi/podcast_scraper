#!/usr/bin/env python3
"""Lifecycle and edge case tests for OpenAIProvider.

These tests focus on provider lifecycle management, error recovery,
and edge cases that strengthen the refactor.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock openai before importing modules that require it
# Unit tests run without openai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper import config
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider


@pytest.mark.unit
class TestOpenAIProviderLifecycle(unittest.TestCase):
    """Test provider lifecycle edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

    def test_multiple_initialize_calls_idempotent(self):
        """Test that multiple initialize() calls are idempotent."""
        # Create config with at least one capability enabled
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            transcribe_missing=True,  # Enable transcription
        )
        provider = OpenAIProvider(cfg)

        # First initialization
        provider.initialize()
        self.assertTrue(
            provider._transcription_initialized
            or provider._speaker_detection_initialized
            or provider._summarization_initialized
        )

        # Second initialization should not cause issues
        provider.initialize()
        # State should remain consistent
        self.assertTrue(provider.is_initialized)

    def test_cleanup_when_not_initialized(self):
        """Test that cleanup() works even when not initialized."""
        provider = OpenAIProvider(self.cfg)

        # Cleanup should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    def test_partial_initialization_cleanup(self):
        """Test cleanup when only some capabilities are initialized."""
        # Initialize only transcription - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = OpenAIProvider(cfg)
        provider.initialize()
        self.assertTrue(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

        # Cleanup should work
        provider.cleanup()
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider.is_initialized)

    def test_requires_separate_instances_false(self):
        """Test that OpenAIProvider does not require separate instances."""
        provider = OpenAIProvider(self.cfg)

        # OpenAIProvider should not require separate instances (API clients are thread-safe)
        self.assertFalse(provider._requires_separate_instances)

        # Verify attribute exists and is accessible via getattr (workflow pattern)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

    def test_requires_separate_instances_accessible_via_getattr(self):
        """Test that _requires_separate_instances is accessible via getattr (workflow pattern)."""
        provider = OpenAIProvider(self.cfg)

        # Workflow uses getattr() to check this attribute
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

        # Should default to False if attribute doesn't exist (defensive)
        # Use spec=[] to prevent Mock from auto-creating attributes
        fake_provider = Mock(spec=[])
        default_value = getattr(fake_provider, "_requires_separate_instances", False)
        self.assertFalse(default_value)

    def test_state_after_cleanup(self):
        """Test that state is clean after cleanup."""
        # Initialize all capabilities - create new config and provider
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )
        provider = OpenAIProvider(cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()

        # State should be clean
        self.assertFalse(provider.is_initialized)
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    def test_client_persists_after_cleanup(self):
        """Test that OpenAI client persists after cleanup (can be reused)."""
        provider = OpenAIProvider(self.cfg)
        original_client = provider.client

        # Initialize and cleanup
        provider.initialize()
        provider.cleanup()

        # Client should still exist (cleanup doesn't destroy it)
        self.assertIsNotNone(provider.client)
        self.assertIs(provider.client, original_client)
