#!/usr/bin/env python3
"""Tests for transcription provider (Stage 2).

These tests verify that the transcription provider pattern works correctly.
"""

import unittest
from unittest.mock import Mock, patch

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider


class TestTranscriptionProviderFactory(unittest.TestCase):
    """Test transcription provider factory."""

    def test_create_whisper_provider(self):
        """Test that factory creates WhisperTranscriptionProvider for 'whisper'."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "WhisperTranscriptionProvider")

    def test_create_invalid_provider(self):
        """Test that factory raises ValueError for invalid provider."""
        # Create a config with invalid provider type
        # Note: Config validation should prevent this, but test factory error handling
        with self.assertRaises(ValueError) as context:
            # We can't actually create a config with invalid provider due to validation
            # So we'll test the factory directly with a mock config
            from podcast_scraper.transcription.factory import create_transcription_provider

            class MockConfig:
                transcription_provider = "invalid"

            create_transcription_provider(MockConfig())  # type: ignore[arg-type]

        self.assertIn("Unsupported transcription provider", str(context.exception))

    def test_factory_returns_provider_instance(self):
        """Test that factory returns a provider instance."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)
        # Verify it has the expected methods
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "cleanup"))


class TestWhisperTranscriptionProvider(unittest.TestCase):
    """Test WhisperTranscriptionProvider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
            whisper_model="tiny",
        )

    def test_provider_initialization(self):
        """Test that provider can be initialized."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        provider = WhisperTranscriptionProvider(self.cfg)
        self.assertFalse(provider.is_initialized)
        self.assertIsNone(provider.model)

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    def test_provider_initialize_loads_model(self, mock_load_model):
        """Test that initialize() loads the Whisper model."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        mock_model = Mock()
        mock_load_model.return_value = mock_model

        provider = WhisperTranscriptionProvider(self.cfg)
        provider.initialize()

        self.assertTrue(provider.is_initialized)
        self.assertEqual(provider.model, mock_model)
        mock_load_model.assert_called_once_with(self.cfg)

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    def test_provider_initialize_fails_on_no_model(self, mock_load_model):
        """Test that initialize() raises RuntimeError if model loading fails."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        mock_load_model.return_value = None

        provider = WhisperTranscriptionProvider(self.cfg)
        with self.assertRaises(RuntimeError) as context:
            provider.initialize()

        self.assertIn("Failed to load Whisper model", str(context.exception))

    @patch(
        "podcast_scraper.transcription.whisper_provider.whisper_integration.transcribe_with_whisper"
    )
    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    def test_provider_transcribe(self, mock_load_model, mock_transcribe):
        """Test that transcribe() calls transcribe_with_whisper."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_transcribe.return_value = ({"text": "Test transcription"}, 1.5)

        provider = WhisperTranscriptionProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Test transcription")
        mock_transcribe.assert_called_once_with(mock_model, "/path/to/audio.mp3", self.cfg)

    def test_provider_transcribe_not_initialized(self):
        """Test that transcribe() raises RuntimeError if not initialized."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        provider = WhisperTranscriptionProvider(self.cfg)
        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch(
        "podcast_scraper.transcription.whisper_provider.whisper_integration.transcribe_with_whisper"
    )
    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    def test_provider_transcribe_empty_text(self, mock_load_model, mock_transcribe):
        """Test that transcribe() raises ValueError if transcription returns empty text."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_transcribe.return_value = ({"text": ""}, 1.5)

        provider = WhisperTranscriptionProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("empty text", str(context.exception))

    def test_provider_cleanup(self):
        """Test that cleanup() marks provider as uninitialized."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        provider = WhisperTranscriptionProvider(self.cfg)
        # Cleanup on uninitialized provider should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    def test_provider_cleanup_after_initialization(self, mock_load_model):
        """Test that cleanup() works after initialization."""
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        mock_model = Mock()
        mock_load_model.return_value = mock_model

        provider = WhisperTranscriptionProvider(self.cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        provider.cleanup()
        self.assertFalse(provider.is_initialized)
        self.assertIsNone(provider.model)


class TestTranscriptionProviderProtocol(unittest.TestCase):
    """Test that WhisperTranscriptionProvider implements TranscriptionProvider protocol."""

    def test_provider_implements_protocol(self):
        """Test that WhisperTranscriptionProvider implements TranscriptionProvider protocol."""
        from podcast_scraper.transcription.base import TranscriptionProvider
        from podcast_scraper.transcription.whisper_provider import WhisperTranscriptionProvider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = WhisperTranscriptionProvider(cfg)

        # Check that provider has required protocol methods
        self.assertTrue(hasattr(provider, "transcribe"))
        # Protocol requires transcribe(audio_path, language) -> str
        import inspect

        sig = inspect.signature(provider.transcribe)
        params = list(sig.parameters.keys())
        self.assertIn("audio_path", params)
        self.assertIn("language", params)
