#!/usr/bin/env python3
"""Tests for transcription provider (Stage 2).

These tests verify that the transcription provider pattern works correctly.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.exceptions import ProviderNotInitializedError, ProviderRuntimeError
from podcast_scraper.transcription.factory import create_transcription_provider

pytestmark = [pytest.mark.unit]


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
            from unittest.mock import MagicMock

            from podcast_scraper import config
            from podcast_scraper.transcription.factory import create_transcription_provider

            mock_cfg = MagicMock(spec=config.Config)
            mock_cfg.transcription_provider = "invalid"
            # Make isinstance check pass
            mock_cfg.__class__ = config.Config

            create_transcription_provider(mock_cfg)

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


class TestMLProviderTranscription(unittest.TestCase):
    """Test MLProvider transcription capability (via factory)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=False,  # Disable to avoid loading models
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_provider_initialize_fails_on_no_model(self, mock_import_whisper):
        """Test that initialize() logs warning and continues if model loading fails.

        MLProvider.initialize() is resilient - it logs warnings but doesn't raise,
        allowing other components (e.g., Transformers) to initialize even if Whisper fails.
        """
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
        # initialize() should not raise - it logs warnings and continues
        provider.initialize()

        # Verify Whisper is not initialized (but provider is still usable)
        if hasattr(provider, "_whisper_initialized"):
            self.assertFalse(provider._whisper_initialized)

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.providers.ml.ml_provider.progress.progress_context")
    def test_provider_transcribe(self, mock_progress, mock_import_whisper):
        """Test that transcribe() calls internal transcription method via factory."""
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
        with self.assertRaises(ProviderNotInitializedError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))
        self.assertEqual(context.exception.provider, "MLProvider/Whisper")

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.providers.ml.ml_provider.progress.progress_context")
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

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("empty text", str(context.exception))

    def test_provider_cleanup(self):
        """Test that cleanup() marks provider as uninitialized."""
        provider = create_transcription_provider(self.cfg)
        # Cleanup on uninitialized provider should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
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


class TestTranscriptionFactoryExperimentMode(unittest.TestCase):
    """Test experiment-mode paths in create_transcription_provider."""

    def test_experiment_mode_invalid_provider_raises_value_error(self):
        """Passing an unrecognised provider string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_transcription_provider("invalid_provider")
        self.assertIn("Invalid provider type", str(ctx.exception))

    def test_experiment_mode_invalid_params_type_raises_type_error(self):
        """Non-dict / non-TranscriptionParams params raise TypeError."""
        with self.assertRaises(TypeError) as ctx:
            create_transcription_provider("whisper", params=42)
        self.assertIn("params must be TranscriptionParams or dict", str(ctx.exception))

    @patch("podcast_scraper.providers.ml.ml_provider.MLProvider.__init__", return_value=None)
    def test_experiment_mode_dict_params_creates_provider(self, mock_init):
        """Dict params are converted to TranscriptionParams and provider is created."""
        provider = create_transcription_provider("whisper", params={"model_name": "base.en"})
        self.assertIsNotNone(provider)
        mock_init.assert_called_once()

    @patch("podcast_scraper.providers.ml.ml_provider.MLProvider.__init__", return_value=None)
    def test_experiment_mode_none_params_uses_defaults(self, mock_init):
        """Omitting params uses default TranscriptionParams."""
        provider = create_transcription_provider("whisper")
        self.assertIsNotNone(provider)
        mock_init.assert_called_once()


class TestTranscriptionFactoryConfigBranches(unittest.TestCase):
    """Test config-mode branches for gemini, mistral, and anthropic."""

    @patch(
        "podcast_scraper.providers.gemini.gemini_provider.GeminiProvider.__init__",
        return_value=None,
    )
    def test_config_gemini_creates_gemini_provider(self, mock_init):
        """Config with transcription_provider='gemini' creates GeminiProvider."""
        cfg = config.Config(
            rss="",
            transcription_provider="gemini",
            gemini_api_key="test-key",
            auto_speakers=False,
        )
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        mock_init.assert_called_once()

    @patch(
        "podcast_scraper.providers.mistral.mistral_provider.MistralProvider.__init__",
        return_value=None,
    )
    def test_config_mistral_creates_mistral_provider(self, mock_init):
        """Config with transcription_provider='mistral' creates MistralProvider."""
        cfg = config.Config(
            rss="",
            transcription_provider="mistral",
            mistral_api_key="test-key",
            auto_speakers=False,
        )
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        mock_init.assert_called_once()

    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider.AnthropicProvider.__init__",
        return_value=None,
    )
    def test_config_anthropic_creates_anthropic_provider(self, mock_init):
        """Config with transcription_provider='anthropic' creates AnthropicProvider.

        Config validation doesn't allow 'anthropic' directly, so we use a
        mock config (same pattern as test_create_invalid_provider).
        """
        from unittest.mock import MagicMock

        mock_cfg = MagicMock(spec=config.Config)
        mock_cfg.transcription_provider = "anthropic"
        mock_cfg.__class__ = config.Config

        provider = create_transcription_provider(mock_cfg)
        self.assertIsNotNone(provider)
        mock_init.assert_called_once()

    def test_config_with_params_raises_type_error(self):
        """Passing params alongside a Config object raises TypeError."""
        from podcast_scraper.providers.params import TranscriptionParams

        cfg = config.Config(
            rss="",
            transcription_provider="whisper",
            auto_speakers=False,
        )
        with self.assertRaises(TypeError) as ctx:
            create_transcription_provider(cfg, params=TranscriptionParams(model_name="base.en"))
        self.assertIn("Cannot provide params", str(ctx.exception))


class TestTranscriptionFactoryAnthropicExperiment(unittest.TestCase):
    """Test that anthropic is NOT available in experiment mode."""

    def test_anthropic_not_in_experiment_mode(self):
        """'anthropic' is config-only; experiment mode rejects it."""
        with self.assertRaises(ValueError) as ctx:
            create_transcription_provider("anthropic")
        self.assertIn("Invalid provider type", str(ctx.exception))
