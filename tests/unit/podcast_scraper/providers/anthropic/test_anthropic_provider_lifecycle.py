#!/usr/bin/env python3
"""Lifecycle and edge case tests for AnthropicProvider.

These tests focus on provider lifecycle management, error recovery,
and edge cases that strengthen the refactor.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider


@pytest.mark.unit
class TestAnthropicProviderLifecycle(unittest.TestCase):
    """Test provider lifecycle edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            auto_speakers=False,
            generate_summaries=False,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_multiple_initialize_calls_idempotent(self, mock_anthropic):
        """Test that multiple initialize() calls are idempotent."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Create config with at least one capability enabled
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            speaker_detector_provider="anthropic",
            auto_speakers=True,  # Enable speaker detection
        )
        provider = AnthropicProvider(cfg)

        # First initialization
        provider.initialize()
        self.assertTrue(
            provider._speaker_detection_initialized or provider._summarization_initialized
        )

        # Second initialization should not cause issues
        provider.initialize()
        # State should remain consistent
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_cleanup_when_not_initialized(self, mock_anthropic):
        """Test that cleanup() works even when not initialized."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

        # Cleanup should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_partial_initialization_cleanup(self, mock_anthropic):
        """Test cleanup when only some capabilities are initialized."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Initialize only speaker detection
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            speaker_detector_provider="anthropic",
            auto_speakers=True,
            generate_summaries=False,
        )
        provider = AnthropicProvider(cfg)
        provider.initialize()
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

        # Cleanup should work
        provider.cleanup()
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_requires_separate_instances_false(self, mock_anthropic):
        """Test that AnthropicProvider does not require separate instances."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

        # AnthropicProvider should not require separate instances (API clients are thread-safe)
        self.assertFalse(provider._requires_separate_instances)

        # Verify attribute exists and is accessible via getattr (workflow pattern)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_requires_separate_instances_accessible_via_getattr(self, mock_anthropic):
        """Test that _requires_separate_instances is accessible via getattr (workflow pattern)."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

        # Workflow uses getattr() to check this attribute
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

        # Should default to False if attribute doesn't exist (defensive)
        fake_provider = Mock(spec=[])
        default_value = getattr(fake_provider, "_requires_separate_instances", False)
        self.assertFalse(default_value)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_state_after_cleanup(self, mock_anthropic):
        """Test that state is clean after cleanup."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Initialize all capabilities
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )
        provider = AnthropicProvider(cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()

        # State should be clean
        self.assertFalse(provider.is_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_client_persists_after_cleanup(self, mock_anthropic):
        """Test that Anthropic client persists after cleanup (can be reused)."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        original_client = provider.client

        # Initialize and cleanup
        provider.initialize()
        provider.cleanup()

        # Client should still exist (cleanup doesn't destroy it)
        self.assertIsNotNone(provider.client)
        self.assertIs(provider.client, original_client)
