#!/usr/bin/env python3
"""Tests for Stage 0: Foundation & Preparation.

These tests verify that Stage 0 infrastructure is correctly set up:
- New packages can be imported
- Protocols are defined and type-checkable
- Config accepts new fields with defaults
- All existing functionality remains unchanged
"""

import unittest

import pytest

from podcast_scraper import config

# Note: Import tests moved to tests/unit/test_package_imports.py
# Note: Protocol definition tests moved to tests/unit/test_protocol_definitions.py
# Integration tests should focus on component interactions, not import/protocol verification.


@pytest.mark.integration
class TestStage0ConfigFields(unittest.TestCase):
    """Test that new config fields are accepted with correct defaults."""

    def test_speaker_detector_provider_default(self):
        """Test that speaker_detector_provider defaults to 'ner'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.speaker_detector_provider, "ner")

    def test_speaker_detector_provider_validation(self):
        """Test that speaker_detector_provider accepts valid values."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_provider="ner")
        self.assertEqual(cfg.speaker_detector_provider, "ner")

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
            # Verify the value was mapped correctly
            self.assertEqual(cfg.speaker_detector_provider, "ner")

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
        """Test that summary_provider defaults to 'local'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.summary_provider, "local")

    def test_summary_provider_validation(self):
        """Test that summary_provider accepts valid values."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", summary_provider="local")
        self.assertEqual(cfg.summary_provider, "local")

        # OpenAI provider requires API key
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
        )
        self.assertEqual(cfg.summary_provider, "openai")

        # Anthropic provider also requires API key (future implementation)
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="anthropic",
            openai_api_key="sk-test123",  # Using OpenAI key for now, will need separate key later
        )
        self.assertEqual(cfg.summary_provider, "anthropic")

    def test_summary_provider_invalid(self):
        """Test that summary_provider rejects invalid values."""
        with self.assertRaises(ValueError):
            config.Config(rss_url="https://example.com/feed.xml", summary_provider="invalid")

    def test_openai_api_key_optional(self):
        """Test that openai_api_key is optional and defaults to None."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertIsNone(cfg.openai_api_key)

    def test_openai_api_key_can_be_set(self):
        """Test that openai_api_key can be set."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", openai_api_key="sk-test123")
        self.assertEqual(cfg.openai_api_key, "sk-test123")


@pytest.mark.integration
class TestStage0Factories(unittest.TestCase):
    """Test that factory functions exist but raise NotImplementedError (Stage 0)."""

    def test_speaker_detector_factory_creates_detector(self):
        """Test that speaker detector factory creates detector (Stage 3)."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_provider="ner")
        # Stage 3: Factory now creates NERSpeakerDetector
        detector = create_speaker_detector(cfg)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "NERSpeakerDetector")

    def test_transcription_provider_factory_creates_provider(self):
        """Test that transcription provider factory creates provider (Stage 2)."""
        from podcast_scraper.transcription.factory import create_transcription_provider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        # Stage 2: Factory now creates WhisperTranscriptionProvider
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "WhisperTranscriptionProvider")

    def test_summarization_provider_factory_creates_provider(self):
        """Test that summarization provider factory creates provider (Stage 4)."""
        from podcast_scraper.summarization.factory import create_summarization_provider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        # Stage 4: Factory now creates TransformersSummarizationProvider
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "TransformersSummarizationProvider")

    def test_openai_providers_factory_creation(self):
        """Test that OpenAI providers can be created via factories (Stage 6)."""
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
        self.assertEqual(provider.__class__.__name__, "OpenAITranscriptionProvider")

        # Test OpenAI speaker detector
        cfg_speaker = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )
        detector = create_speaker_detector(cfg_speaker)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "OpenAISpeakerDetector")

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
        self.assertEqual(provider.__class__.__name__, "OpenAISummarizationProvider")


@pytest.mark.integration
class TestStage0BackwardCompatibility(unittest.TestCase):
    """Test that Stage 0 changes don't break existing functionality."""

    def test_existing_config_fields_still_work(self):
        """Test that existing config fields still work as before."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./test",
            max_episodes=10,
            whisper_model="base",
            transcribe_missing=True,
        )
        self.assertEqual(cfg.rss_url, "https://example.com/feed.xml")
        self.assertEqual(cfg.output_dir, "./test")
        self.assertEqual(cfg.max_episodes, 10)
        self.assertEqual(cfg.whisper_model, "base")
        self.assertTrue(cfg.transcribe_missing)

    def test_defaults_match_current_behavior(self):
        """Test that new field defaults match current behavior."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        # Speaker detection: defaults to "ner" (current spaCy NER)
        self.assertEqual(cfg.speaker_detector_provider, "ner")

        # Transcription: defaults to "whisper" (current Whisper integration)
        self.assertEqual(cfg.transcription_provider, "whisper")

        # Summarization: defaults to "local" (current transformers)
        self.assertEqual(cfg.summary_provider, "local")
