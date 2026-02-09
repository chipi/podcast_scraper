#!/usr/bin/env python3
"""Lifecycle and edge case tests for OllamaProvider.

These tests focus on provider lifecycle management, error recovery,
and edge cases that strengthen the refactor.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock openai and httpx before importing modules that require them
# Unit tests run without openai/httpx packages installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
mock_httpx = MagicMock()
_patch_ollama = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
        "httpx": mock_httpx,
    },
)
_patch_ollama.start()

from podcast_scraper import config
from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider


@pytest.mark.unit
class TestOllamaProviderLifecycle(unittest.TestCase):
    """Test provider lifecycle edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            summary_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=False,
            generate_summaries=False,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_multiple_initialize_calls_idempotent(self, mock_openai, mock_httpx):
        """Test that multiple initialize() calls are idempotent."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock health check and model validation
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.3:latest"},
            ]
        }
        mock_httpx.get.return_value = mock_response

        # Create config with at least one capability enabled
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,  # Enable speaker detection
        )
        provider = OllamaProvider(cfg)

        # First initialization
        provider.initialize()
        self.assertTrue(
            provider._speaker_detection_initialized or provider._summarization_initialized
        )

        # Second initialization should not cause issues
        provider.initialize()
        # State should remain consistent
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_cleanup_when_not_initialized(self, mock_openai, mock_httpx):
        """Test that cleanup() works even when not initialized."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock httpx to prevent real network calls
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)

        # Cleanup should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_partial_initialization_cleanup(self, mock_openai, mock_httpx):
        """Test cleanup when only some capabilities are initialized."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock health check and model validation
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.3:latest"},
            ]
        }
        mock_httpx.get.return_value = mock_response

        # Initialize only speaker detection
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,
            generate_summaries=False,
        )
        provider = OllamaProvider(cfg)
        provider.initialize()
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

        # Cleanup should work
        provider.cleanup()
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_requires_separate_instances_false(self, mock_openai, mock_httpx):
        """Test that OllamaProvider does not require separate instances."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock httpx to prevent real network calls
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)

        # OllamaProvider should not require separate instances (API clients are thread-safe)
        self.assertFalse(provider._requires_separate_instances)

        # Verify attribute exists and is accessible via getattr (workflow pattern)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_requires_separate_instances_accessible_via_getattr(self, mock_openai, mock_httpx):
        """Test that _requires_separate_instances is accessible via getattr (workflow pattern)."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock httpx to prevent real network calls
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)

        # Workflow uses getattr() to check this attribute
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

        # Should default to False if attribute doesn't exist (defensive)
        fake_provider = Mock(spec=[])
        default_value = getattr(fake_provider, "_requires_separate_instances", False)
        self.assertFalse(default_value)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_state_after_cleanup(self, mock_openai, mock_httpx):
        """Test that state is clean after cleanup."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock health check and model validation
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.2:latest"},
                {"name": "llama3.3:latest"},
            ]
        }
        mock_httpx.get.return_value = mock_response

        # Initialize all capabilities
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            summary_provider="ollama",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )
        provider = OllamaProvider(cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()

        # State should be clean
        self.assertFalse(provider.is_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_client_persists_after_cleanup(self, mock_openai, mock_httpx):
        """Test that Ollama (OpenAI) client persists after cleanup (can be reused)."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock httpx to prevent real network calls
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        original_client = provider.client

        # Initialize and cleanup
        provider.initialize()
        provider.cleanup()

        # Client should still exist (cleanup doesn't destroy it)
        self.assertIsNotNone(provider.client)
        self.assertIs(provider.client, original_client)
