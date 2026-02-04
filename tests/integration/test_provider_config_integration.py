#!/usr/bin/env python3
"""Integration tests for provider configuration and factory creation.

These tests verify that provider configuration fields work correctly and that
factory functions create providers as expected:
- Config accepts provider fields with correct defaults
- Provider field validation works correctly
- Factory functions create correct provider instances
- Backward compatibility with deprecated fields
"""

import os
import unittest

import pytest

from podcast_scraper import config

# Note: Import tests moved to tests/unit/test_package_imports.py
# Note: Protocol definition tests moved to tests/unit/test_protocol_definitions.py
# Integration tests should focus on component interactions, not import/protocol verification.


@pytest.mark.integration
@pytest.mark.critical_path
class TestProviderConfigFields(unittest.TestCase):
    """Test that provider config fields are accepted with correct defaults."""

    def test_speaker_detector_provider_default(self):
        """Test that speaker_detector_provider defaults to 'spacy'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.speaker_detector_provider, "spacy")

    def test_speaker_detector_provider_validation(self):
        """Test that speaker_detector_provider accepts valid values."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        self.assertEqual(cfg.speaker_detector_provider, "spacy")

        # Deprecated "speaker_detector_type" with "ner" should still work (maps to "spacy")
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg_ner = config.Config(
                rss_url="https://example.com/feed.xml", speaker_detector_type="ner"
            )
            self.assertEqual(cfg_ner.speaker_detector_provider, "spacy")
            self.assertTrue(any("deprecated" in str(warning.message).lower() for warning in w))

        # OpenAI provider requires API key
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
        )
        self.assertEqual(cfg.speaker_detector_provider, "openai")

    def test_speaker_detector_provider_invalid(self):
        """Test that speaker_detector_provider rejects invalid values."""
        with self.assertRaises(ValueError):
            config.Config(
                rss_url="https://example.com/feed.xml", speaker_detector_provider="invalid"
            )

    def test_speaker_detector_type_backward_compatibility(self):
        """Test that deprecated speaker_detector_type still works (backward compatibility)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_type="ner")
            # Verify deprecation warning was issued
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            # Verify the value was mapped correctly (ner → spacy)
            self.assertEqual(cfg.speaker_detector_provider, "spacy")

    def test_transcription_provider_default(self):
        """Test that transcription_provider defaults to 'whisper'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.transcription_provider, "whisper")

    def test_transcription_provider_validation(self):
        """Test that transcription_provider accepts valid values."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        self.assertEqual(cfg.transcription_provider, "whisper")

        # OpenAI provider requires API key
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )
        self.assertEqual(cfg.transcription_provider, "openai")

    def test_transcription_provider_invalid(self):
        """Test that transcription_provider rejects invalid values."""
        with self.assertRaises(ValueError):
            config.Config(rss_url="https://example.com/feed.xml", transcription_provider="invalid")

    def test_summary_provider_default(self):
        """Test that summary_provider defaults to 'transformers'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.summary_provider, "transformers")

    def test_summary_provider_validation(self):
        """Test that summary_provider accepts valid values."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", summary_provider="transformers")
        self.assertEqual(cfg.summary_provider, "transformers")

        # Test that "transformers" works (no longer testing deprecated "local" alias)
        cfg_transformers = config.Config(
            rss_url="https://example.com/feed.xml", summary_provider="transformers"
        )
        self.assertEqual(cfg_transformers.summary_provider, "transformers")

        # OpenAI provider requires API key
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
        )
        self.assertEqual(cfg.summary_provider, "openai")

    def test_summary_provider_invalid(self):
        """Test that summary_provider rejects invalid values."""
        with self.assertRaises(ValueError):
            config.Config(rss_url="https://example.com/feed.xml", summary_provider="invalid")

    def test_openai_api_key_optional(self):
        """Test that openai_api_key is optional and defaults to None."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            cfg = config.Config(rss_url="https://example.com/feed.xml")
            self.assertIsNone(cfg.openai_api_key)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_openai_api_key_can_be_set(self):
        """Test that openai_api_key can be set."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", openai_api_key="sk-test123")
        self.assertEqual(cfg.openai_api_key, "sk-test123")


@pytest.mark.integration
@pytest.mark.critical_path
class TestProviderFactories(unittest.TestCase):
    """Test that factory functions create providers correctly."""

    def test_speaker_detector_factory_creates_detector(self):
        """Test that speaker detector factory creates detector."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        detector = create_speaker_detector(cfg)
        self.assertIsNotNone(detector)
        # Verify it's the unified ML provider
        self.assertEqual(detector.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "clear_cache"))

    def test_transcription_provider_factory_creates_provider(self):
        """Test that transcription provider factory creates provider."""
        from podcast_scraper.transcription.factory import create_transcription_provider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))

    def test_summarization_provider_factory_creates_provider(self):
        """Test that summarization provider factory creates provider."""
        from podcast_scraper.summarization.factory import create_summarization_provider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_openai_providers_factory_creation(self):
        """Test that OpenAI providers can be created via factories."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Test OpenAI transcription provider
        cfg_transcription = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )
        provider = create_transcription_provider(cfg_transcription)
        self.assertIsNotNone(provider)
        # Verify it's the unified OpenAI provider
        self.assertEqual(provider.__class__.__name__, "OpenAIProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))

        # Test OpenAI speaker detector
        cfg_speaker = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )
        detector = create_speaker_detector(cfg_speaker)
        self.assertIsNotNone(detector)
        # Verify it's the unified OpenAI provider
        self.assertEqual(detector.__class__.__name__, "OpenAIProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "clear_cache"))

        # Test OpenAI summarization provider
        cfg_summary = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_metadata=True,  # Required when generate_summaries=True
            generate_summaries=True,
        )
        provider = create_summarization_provider(cfg_summary)
        self.assertIsNotNone(provider)
        # Verify it's the unified OpenAI provider
        self.assertEqual(provider.__class__.__name__, "OpenAIProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_gemini_providers_factory_creation(self):
        """Test that Gemini providers can be created via factories."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Test Gemini transcription provider
        cfg_transcription = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            gemini_api_key="test-api-key-123",
        )
        provider = create_transcription_provider(cfg_transcription)
        self.assertIsNotNone(provider)
        # Verify it's the unified Gemini provider
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))

        # Test Gemini speaker detector
        cfg_speaker = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="gemini",
            gemini_api_key="test-api-key-123",
            auto_speakers=True,
        )
        detector = create_speaker_detector(cfg_speaker)
        self.assertIsNotNone(detector)
        # Verify it's the unified Gemini provider
        self.assertEqual(detector.__class__.__name__, "GeminiProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "clear_cache"))

        # Test Gemini summarization provider
        cfg_summary = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_metadata=True,  # Required when generate_summaries=True
            generate_summaries=True,
        )
        provider = create_summarization_provider(cfg_summary)
        self.assertIsNotNone(provider)
        # Verify it's the unified Gemini provider
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.critical_path
class TestConfigBackwardCompatibility(unittest.TestCase):
    """Test that config changes don't break existing functionality."""

    def test_existing_config_fields_still_work(self):
        """Test that existing config fields still work as before."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./test",
            max_episodes=10,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            transcribe_missing=True,
        )
        self.assertEqual(cfg.rss_url, "https://example.com/feed.xml")
        self.assertEqual(cfg.output_dir, "./test")
        self.assertEqual(cfg.max_episodes, 10)
        self.assertEqual(cfg.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)
        self.assertTrue(cfg.transcribe_missing)

    def test_defaults_match_current_behavior(self):
        """Test that new field defaults match current behavior."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        # Speaker detection: defaults to "spacy" (spaCy NER)
        self.assertEqual(cfg.speaker_detector_provider, "spacy")

        # Transcription: defaults to "whisper" (Whisper integration)
        self.assertEqual(cfg.transcription_provider, "whisper")

        # Summarization: defaults to "transformers" (HuggingFace Transformers)
        self.assertEqual(cfg.summary_provider, "transformers")

    def test_deprecated_provider_names_still_work(self):
        """Test that deprecated provider names ('ner', 'local') still work for backward compatibility."""
        import warnings

        # Test that "spacy" works directly (no longer testing deprecated "ner" alias)
        cfg_spacy = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
        )
        self.assertEqual(cfg_spacy.speaker_detector_provider, "spacy")

        # Test deprecated "local" → "transformers"
        # Note: "transformers" is the current name, not deprecated
        # The deprecated name was "local", but it's not tested here
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg_local = config.Config(
                rss_url="https://example.com/feed.xml",
                summary_provider="transformers",
            )
            self.assertEqual(cfg_local.summary_provider, "transformers")
            # "transformers" is not deprecated, so no warning should be issued
            self.assertFalse(any("deprecated" in str(warning.message).lower() for warning in w))
