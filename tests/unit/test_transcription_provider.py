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
        """Test that factory creates MLProvider for 'whisper'."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

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
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = create_transcription_provider(cfg)
        # Verify it has the expected methods
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "cleanup"))


class TestMLProviderTranscriptionViaFactory(unittest.TestCase):
    """Test MLProvider transcription capability via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=False,  # Disable to avoid loading models
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model="tiny",
        )

    def test_provider_creation_via_factory(self):
        """Test that provider can be created via factory."""
        provider = create_transcription_provider(self.cfg)
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_provider_initialization_state(self):
        """Test that provider tracks initialization state."""
        provider = create_transcription_provider(self.cfg)
        # Initially not initialized (transcribe_missing is False)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_provider_initialize_loads_model(self, mock_import_whisper):
        """Test that initialize() loads the Whisper model via factory."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=self.cfg.whisper_model,
        )

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider = create_transcription_provider(cfg)
        provider.initialize()

        self.assertTrue(provider.is_initialized)
        self.assertEqual(provider.model, mock_model)
        mock_whisper_lib.load_model.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_provider_initialize_fails_on_no_model(self, mock_import_whisper):
        """Test that initialize() raises RuntimeError if model loading fails."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=self.cfg.whisper_model,
        )

        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.side_effect = FileNotFoundError("Model not found")
        mock_import_whisper.return_value = mock_whisper_lib

        provider = create_transcription_provider(cfg)
        with self.assertRaises(RuntimeError) as context:
            provider.initialize()

        self.assertIn("Failed to load", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    def test_provider_transcribe(self, mock_progress, mock_import_whisper):
        """Test that transcribe() works via factory."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=self.cfg.whisper_model,
        )

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "Test transcription", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_progress.return_value.__enter__.return_value = None

        provider = create_transcription_provider(cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Test transcription")
        mock_model.transcribe.assert_called_once()

    def test_provider_transcribe_not_initialized(self):
        """Test that transcribe() raises RuntimeError if not initialized."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=self.cfg.whisper_model,
        )
        provider = create_transcription_provider(cfg)
        # Don't call initialize()
        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    def test_provider_transcribe_empty_text(self, mock_progress, mock_import_whisper):
        """Test that transcribe() raises ValueError if transcription returns empty text."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=self.cfg.whisper_model,
        )

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_progress.return_value.__enter__.return_value = None

        provider = create_transcription_provider(cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("empty text", str(context.exception))

    def test_provider_cleanup(self):
        """Test that cleanup() marks provider as uninitialized."""
        provider = create_transcription_provider(self.cfg)
        # Cleanup on uninitialized provider should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_provider_cleanup_after_initialization(self, mock_import_whisper):
        """Test that cleanup() works after initialization."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=self.cfg.whisper_model,
        )

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider = create_transcription_provider(cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        provider.cleanup()
        self.assertFalse(provider.is_initialized)
        self.assertIsNone(provider.model)


class TestTranscriptionProviderProtocol(unittest.TestCase):
    """Test that MLProvider implements TranscriptionProvider protocol (via factory)."""

    def test_provider_implements_protocol(self):
        """Test that MLProvider implements TranscriptionProvider protocol."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = create_transcription_provider(cfg)

        # Check that provider has required protocol methods
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # Protocol requires transcribe(audio_path, language) -> str
        import inspect

        sig = inspect.signature(provider.transcribe)
        params = list(sig.parameters.keys())
        self.assertIn("audio_path", params)
        self.assertIn("language", params)
