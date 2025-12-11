#!/usr/bin/env python3
"""Tests for Stage 0: Foundation & Preparation.

These tests verify that Stage 0 infrastructure is correctly set up:
- New packages can be imported
- Protocols are defined and type-checkable
- Config accepts new fields with defaults
- All existing functionality remains unchanged
"""

import unittest
from typing import get_type_hints

from podcast_scraper import config


class TestStage0PackageStructure(unittest.TestCase):
    """Test that new package structure exists and can be imported."""

    def test_preprocessing_module_importable(self):
        """Test that preprocessing.py can be imported."""
        import podcast_scraper.preprocessing

        self.assertIsNotNone(podcast_scraper.preprocessing)

    def test_speaker_detectors_package_importable(self):
        """Test that speaker_detectors package can be imported."""
        from podcast_scraper.speaker_detectors import base, factory

        self.assertIsNotNone(base)
        self.assertIsNotNone(factory)

    def test_transcription_package_importable(self):
        """Test that transcription package can be imported."""
        from podcast_scraper.transcription import base, factory

        self.assertIsNotNone(base)
        self.assertIsNotNone(factory)

    def test_summarization_package_importable(self):
        """Test that summarization package can be imported."""
        from podcast_scraper.summarization import base, factory

        self.assertIsNotNone(base)
        self.assertIsNotNone(factory)


class TestStage0Protocols(unittest.TestCase):
    """Test that Protocol definitions are valid and type-checkable."""

    def test_speaker_detector_protocol_exists(self):
        """Test that SpeakerDetector protocol is defined."""
        from podcast_scraper.speaker_detectors.base import SpeakerDetector

        # Verify protocol has required methods
        hints = get_type_hints(SpeakerDetector.detect_speakers)
        self.assertIn("episode_title", hints)
        self.assertIn("episode_description", hints)
        self.assertIn("known_hosts", hints)

    def test_transcription_provider_protocol_exists(self):
        """Test that TranscriptionProvider protocol is defined."""
        from podcast_scraper.transcription.base import TranscriptionProvider

        # Verify protocol has required methods
        hints = get_type_hints(TranscriptionProvider.transcribe)
        self.assertIn("audio_path", hints)
        self.assertIn("language", hints)

    def test_summarization_provider_protocol_exists(self):
        """Test that SummarizationProvider protocol is defined."""
        from podcast_scraper.summarization.base import SummarizationProvider

        # Verify protocol has required methods
        hints = get_type_hints(SummarizationProvider.summarize)
        self.assertIn("text", hints)
        self.assertIn("episode_title", hints)
        self.assertIn("episode_description", hints)
        self.assertIn("params", hints)


class TestStage0ConfigFields(unittest.TestCase):
    """Test that new config fields are accepted with correct defaults."""

    def test_speaker_detector_type_default(self):
        """Test that speaker_detector_type defaults to 'ner'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.speaker_detector_type, "ner")

    def test_speaker_detector_type_validation(self):
        """Test that speaker_detector_type accepts valid values."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_type="ner")
        self.assertEqual(cfg.speaker_detector_type, "ner")

        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_type="openai")
        self.assertEqual(cfg.speaker_detector_type, "openai")

    def test_speaker_detector_type_invalid(self):
        """Test that speaker_detector_type rejects invalid values."""
        with self.assertRaises(ValueError):
            config.Config(rss_url="https://example.com/feed.xml", speaker_detector_type="invalid")

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

        cfg = config.Config(rss_url="https://example.com/feed.xml", transcription_provider="openai")
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

        cfg = config.Config(rss_url="https://example.com/feed.xml", summary_provider="openai")
        self.assertEqual(cfg.summary_provider, "openai")

        cfg = config.Config(rss_url="https://example.com/feed.xml", summary_provider="anthropic")
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


class TestStage0Factories(unittest.TestCase):
    """Test that factory functions exist but raise NotImplementedError (Stage 0)."""

    def test_speaker_detector_factory_not_implemented(self):
        """Test that speaker detector factory raises NotImplementedError."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        cfg = config.Config(rss_url="https://example.com/feed.xml")
        with self.assertRaises(NotImplementedError):
            create_speaker_detector(cfg)

    def test_transcription_provider_factory_not_implemented(self):
        """Test that transcription provider factory raises NotImplementedError."""
        from podcast_scraper.transcription.factory import create_transcription_provider

        cfg = config.Config(rss_url="https://example.com/feed.xml")
        with self.assertRaises(NotImplementedError):
            create_transcription_provider(cfg)

    def test_summarization_provider_factory_not_implemented(self):
        """Test that summarization provider factory raises NotImplementedError."""
        from podcast_scraper.summarization.factory import create_summarization_provider

        cfg = config.Config(rss_url="https://example.com/feed.xml")
        with self.assertRaises(NotImplementedError):
            create_summarization_provider(cfg)


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
        self.assertEqual(cfg.speaker_detector_type, "ner")

        # Transcription: defaults to "whisper" (current Whisper integration)
        self.assertEqual(cfg.transcription_provider, "whisper")

        # Summarization: defaults to "local" (current transformers)
        self.assertEqual(cfg.summary_provider, "local")
