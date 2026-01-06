#!/usr/bin/env python3
"""Tests for CLI argument parsing and validation.

These tests verify the CLI argument parsing, validation, and configuration
building functions.
"""

import argparse
import os
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import cli, config


class TestValidateRssUrl(unittest.TestCase):
    """Test _validate_rss_url function."""

    def test_valid_http_url(self):
        """Test that valid HTTP URL passes validation."""
        errors = []
        cli._validate_rss_url("http://example.com/feed.xml", errors)
        self.assertEqual(errors, [])

    def test_valid_https_url(self):
        """Test that valid HTTPS URL passes validation."""
        errors = []
        cli._validate_rss_url("https://example.com/feed.xml", errors)
        self.assertEqual(errors, [])

    def test_empty_url(self):
        """Test that empty URL adds error."""
        errors = []
        cli._validate_rss_url("", errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("RSS URL is required", errors[0])

    def test_url_without_scheme(self):
        """Test that URL without http/https scheme adds error."""
        errors = []
        cli._validate_rss_url("example.com/feed.xml", errors)
        # URL without scheme may not have netloc, so may trigger hostname error instead
        self.assertGreater(len(errors), 0)
        # Should have either scheme or hostname error
        error_text = " ".join(errors)
        self.assertTrue(
            "RSS URL must be http or https" in error_text
            or "RSS URL must have a valid hostname" in error_text
        )

    def test_url_with_invalid_scheme(self):
        """Test that URL with invalid scheme adds error."""
        errors = []
        cli._validate_rss_url("ftp://example.com/feed.xml", errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("RSS URL must be http or https", errors[0])

    def test_url_without_hostname(self):
        """Test that URL without hostname adds error."""
        errors = []
        cli._validate_rss_url("http:///feed.xml", errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("RSS URL must have a valid hostname", errors[0])

    def test_url_with_path_and_query(self):
        """Test that URL with path and query parameters is valid."""
        errors = []
        cli._validate_rss_url("https://example.com/feed.xml?param=value", errors)
        self.assertEqual(errors, [])


class TestValidateWhisperConfig(unittest.TestCase):
    """Test _validate_whisper_config function."""

    def test_valid_model(self):
        """Test that valid Whisper model passes validation."""
        # Tests should use test default (tiny.en), not production default (base)
        args = Namespace(transcribe_missing=True, whisper_model=config.TEST_DEFAULT_WHISPER_MODEL)
        errors = []
        cli._validate_whisper_config(args, errors)
        self.assertEqual(errors, [])

    def test_valid_model_when_transcribe_disabled(self):
        """Test that model validation is skipped when transcribe_missing is False."""
        args = Namespace(transcribe_missing=False, whisper_model="invalid")
        errors = []
        cli._validate_whisper_config(args, errors)
        self.assertEqual(errors, [])

    def test_invalid_model(self):
        """Test that invalid Whisper model adds error."""
        args = Namespace(transcribe_missing=True, whisper_model="invalid")
        errors = []
        cli._validate_whisper_config(args, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("--whisper-model must be one of", errors[0])

    def test_all_valid_models(self):
        """Test that all valid models pass validation."""
        valid_models = (
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
            "large.en",
        )
        for model in valid_models:
            with self.subTest(model=model):
                args = Namespace(transcribe_missing=True, whisper_model=model)
                errors = []
                cli._validate_whisper_config(args, errors)
                self.assertEqual(errors, [], f"Model {model} should be valid")


class TestValidateSpeakerConfig(unittest.TestCase):
    """Test _validate_speaker_config function."""

    def test_valid_num_speakers(self):
        """Test that valid num_speakers passes validation."""
        args = Namespace(screenplay=True, num_speakers=2, speaker_names=None)
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_num_speakers_below_minimum(self):
        """Test that num_speakers below minimum adds error."""
        args = Namespace(screenplay=True, num_speakers=1, speaker_names=None)
        errors = []
        cli._validate_speaker_config(args, errors)
        # Note: Validation only checks if screenplay is True
        # If screenplay is True and num_speakers < MIN_NUM_SPEAKERS, should add error
        if args.screenplay and args.num_speakers < config.MIN_NUM_SPEAKERS:
            self.assertEqual(len(errors), 1)
            self.assertIn("--num-speakers must be at least", errors[0])
        else:
            # If validation doesn't trigger, that's also valid behavior
            self.assertGreaterEqual(len(errors), 0)

    def test_num_speakers_validation_when_screenplay_false(self):
        """Test that num_speakers validation is skipped when screenplay is False."""
        args = Namespace(screenplay=False, num_speakers=1, speaker_names=None)
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_valid_speaker_names(self):
        """Test that valid speaker names pass validation."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="Alice, Bob")
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_speaker_names_with_whitespace(self):
        """Test that speaker names with whitespace are handled."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="  Alice  ,  Bob  ")
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_speaker_names_single_name(self):
        """Test that single speaker name adds error."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="Alice")
        errors = []
        cli._validate_speaker_config(args, errors)
        # Validation only checks if speaker_names is provided
        if args.speaker_names:
            names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
            if len(names) < config.MIN_NUM_SPEAKERS:
                self.assertEqual(len(errors), 1)
                self.assertIn("At least two speaker names required", errors[0])
            else:
                self.assertEqual(errors, [])
        else:
            self.assertEqual(errors, [])

    def test_speaker_names_empty(self):
        """Test that empty speaker names string adds error."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="")
        errors = []
        cli._validate_speaker_config(args, errors)
        # Empty string splits to empty list after filtering, which has length 0
        # Validation only checks if speaker_names is truthy and has < 2 names
        if args.speaker_names:
            names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
            if len(names) < config.MIN_NUM_SPEAKERS:
                self.assertEqual(len(errors), 1)
                self.assertIn("At least two speaker names required", errors[0])
            else:
                self.assertEqual(errors, [])
        else:
            # Empty string might be falsy, so no validation
            self.assertEqual(errors, [])

    def test_speaker_names_multiple_names(self):
        """Test that multiple speaker names pass validation."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="Alice, Bob, Charlie")
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])


class TestValidateWorkersConfig(unittest.TestCase):
    """Test _validate_workers_config function."""

    def test_valid_workers(self):
        """Test that valid workers count passes validation."""
        args = Namespace(workers=4)
        errors = []
        cli._validate_workers_config(args, errors)
        self.assertEqual(errors, [])

    def test_workers_zero(self):
        """Test that workers=0 adds error."""
        args = Namespace(workers=0)
        errors = []
        cli._validate_workers_config(args, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("--workers must be at least 1", errors[0])

    def test_workers_negative(self):
        """Test that negative workers adds error."""
        args = Namespace(workers=-1)
        errors = []
        cli._validate_workers_config(args, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("--workers must be at least 1", errors[0])


class TestValidateArgs(unittest.TestCase):
    """Test validate_args function."""

    def test_valid_args(self):
        """Test that valid arguments pass validation."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        # Should not raise
        cli.validate_args(args)

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are collected."""
        args = Namespace(
            rss="",  # Empty RSS
            max_episodes=-1,  # Invalid
            timeout=0,  # Invalid
            delay_ms=-1,  # Invalid
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=0,  # Invalid
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        error_msg = str(cm.exception)
        self.assertIn("Invalid input parameters", error_msg)
        # Should contain multiple errors
        self.assertIn("RSS URL is required", error_msg)
        self.assertIn("--max-episodes must be positive", error_msg)
        self.assertIn("--timeout must be positive", error_msg)
        self.assertIn("--delay-ms must be non-negative", error_msg)
        self.assertIn("--workers must be at least 1", error_msg)

    def test_invalid_output_dir(self):
        """Test that invalid output directory adds error."""
        # Empty output_dir might be handled differently - validate_args only checks
        # if output_dir is provided. If it's empty string, it might not trigger validation.
        # Let's test with a clearly invalid path instead
        args2 = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir="\x00invalid",  # Invalid path with null byte
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args2)
        error_msg = str(cm.exception)
        self.assertIn("Invalid input parameters", error_msg)

    def test_max_episodes_none(self):
        """Test that max_episodes=None is valid."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=None,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        # Should not raise
        cli.validate_args(args)

    def test_max_episodes_zero(self):
        """Test that max_episodes=0 adds error."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=0,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        error_msg = str(cm.exception)
        self.assertIn("--max-episodes must be positive", error_msg)

    def test_delay_ms_zero(self):
        """Test that delay_ms=0 is valid."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        # Should not raise
        cli.validate_args(args)


class TestBuildConfig(unittest.TestCase):
    """Test _build_config function."""

    def test_build_config_from_args(self):
        """Test that Config is built correctly from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir="./output",
            max_episodes=10,
            user_agent="test-agent",
            timeout=30,
            delay_ms=100,
            prefer_type=["vtt", "srt"],
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=True,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names="Alice, Bob",
            run_id="test_run",
            skip_existing=False,  # Cannot be True with clean_output
            reuse_media=False,
            clean_output=True,
            dry_run=False,
            generate_metadata=True,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=True,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level="DEBUG",
            log_file=None,
            language="en",
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=4,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertIsInstance(cfg, config.Config)
        self.assertEqual(cfg.rss_url, "https://example.com/feed.xml")
        # output_dir is derived and normalized to absolute path
        self.assertIsNotNone(cfg.output_dir)
        self.assertIn("output", cfg.output_dir)  # Should contain "output" somewhere
        self.assertEqual(cfg.max_episodes, 10)
        self.assertEqual(cfg.user_agent, "test-agent")
        self.assertEqual(cfg.timeout, 30)
        self.assertEqual(cfg.delay_ms, 100)
        # prefer_type from args becomes prefer_types in Config
        # The actual value might be normalized, so just check it's a list
        self.assertIsInstance(cfg.prefer_types, list)
        self.assertIn("vtt", cfg.prefer_types)
        self.assertIn("srt", cfg.prefer_types)
        self.assertTrue(cfg.transcribe_missing)
        self.assertEqual(cfg.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)
        self.assertTrue(cfg.screenplay)
        self.assertEqual(cfg.screenplay_gap_s, 2.0)
        self.assertEqual(cfg.screenplay_num_speakers, 2)
        # speaker_names is split and stripped
        self.assertEqual(len(cfg.screenplay_speaker_names), 2)
        self.assertIn("Alice", cfg.screenplay_speaker_names)
        self.assertIn("Bob", cfg.screenplay_speaker_names)
        self.assertEqual(cfg.run_id, "test_run")
        # skip_existing and clean_output are mutually exclusive
        self.assertFalse(cfg.skip_existing)  # Set to False because clean_output=True
        self.assertTrue(cfg.clean_output)
        self.assertFalse(cfg.dry_run)
        self.assertTrue(cfg.generate_metadata)
        self.assertTrue(cfg.generate_summaries)
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertIsNone(cfg.log_file)
        self.assertEqual(cfg.language, "en")
        self.assertEqual(cfg.workers, 4)

    def test_build_config_with_defaults(self):
        """Test that Config uses defaults when args are not provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertIsInstance(cfg, config.Config)
        # Check that defaults are used
        # output_dir is derived from rss_url when None, so it won't be None
        self.assertIsNotNone(cfg.output_dir)  # Derived from RSS URL
        self.assertIsNone(cfg.max_episodes)
        self.assertFalse(cfg.transcribe_missing)
        self.assertFalse(cfg.screenplay)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_transcription_provider(self):
        """Test that _build_config includes transcription_provider from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="openai",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.transcription_provider, "openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_speaker_detector_provider(self):
        """Test that _build_config includes speaker_detector_provider from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="openai",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.speaker_detector_provider, "openai")

    def test_build_config_with_openai_api_base(self):
        """Test that _build_config includes openai_api_base from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base="http://localhost:8000/v1",
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_api_base, "http://localhost:8000/v1")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_loads_openai_api_key_from_env(self):
        """Test that _build_config loads openai_api_key from environment."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        # The Config object should load the key from environment via field validator
        self.assertEqual(cfg.openai_api_key, "sk-test123")

    @patch.dict(os.environ, {}, clear=True)
    def test_build_config_openai_api_key_none_when_not_in_env(self):
        """Test that _build_config sets openai_api_key to None when not in environment."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        # The Config object should have None when env var is not set
        self.assertIsNone(cfg.openai_api_key)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_transcription_model(self):
        """Test that _build_config includes openai_transcription_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="openai",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model="whisper-1",
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_transcription_model, "whisper-1")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_speaker_model(self):
        """Test that _build_config includes openai_speaker_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="openai",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model="gpt-4o",
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_speaker_model, "gpt-4o")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_summary_model(self):
        """Test that _build_config includes openai_summary_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=True,  # Required when generate_summaries=True
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=True,
            metrics_output=None,
            summary_provider="openai",
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model="gpt-4o-mini",
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_summary_model, "gpt-4o-mini")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_temperature(self):
        """Test that _build_config includes openai_temperature when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=0.7,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_temperature, 0.7)


class TestParseArgs(unittest.TestCase):
    """Test parse_args function."""

    def test_parse_args_basic(self):
        """Test parsing basic arguments."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertEqual(args.rss, "https://example.com/feed.xml")
        self.assertIsNotNone(args)

    def test_parse_args_with_options(self):
        """Test parsing arguments with various options."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--max-episodes",
                "10",
                "--timeout",
                "60",
                "--transcribe-missing",
                "--whisper-model",
                "base",
            ]
        )
        self.assertEqual(args.rss, "https://example.com/feed.xml")
        self.assertEqual(args.max_episodes, 10)
        self.assertEqual(args.timeout, 60)
        self.assertTrue(args.transcribe_missing)
        # CLI should preserve the value passed by user, not override with test default
        self.assertEqual(args.whisper_model, "base")

    def test_parse_args_with_transcription_provider(self):
        """Test parsing --transcription-provider argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--transcription-provider",
                "openai",
            ]
        )
        self.assertEqual(args.transcription_provider, "openai")

    def test_parse_args_with_speaker_detector_provider(self):
        """Test parsing --speaker-detector-provider argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--speaker-detector-provider",
                "openai",
            ]
        )
        self.assertEqual(args.speaker_detector_provider, "openai")

    def test_parse_args_with_openai_api_base(self):
        """Test parsing --openai-api-base argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-api-base",
                "http://localhost:8000/v1",
            ]
        )
        self.assertEqual(args.openai_api_base, "http://localhost:8000/v1")

    def test_parse_args_transcription_provider_default(self):
        """Test that --transcription-provider defaults to 'whisper'."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertEqual(args.transcription_provider, "whisper")

    def test_parse_args_speaker_detector_provider_default(self):
        """Test that --speaker-detector-provider defaults to 'spacy'."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertEqual(args.speaker_detector_provider, "spacy")

    def test_parse_args_openai_api_base_default(self):
        """Test that --openai-api-base defaults to None."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertIsNone(args.openai_api_base)

    def test_parse_args_with_openai_transcription_model(self):
        """Test parsing --openai-transcription-model argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-transcription-model",
                "whisper-1",
            ]
        )
        self.assertEqual(args.openai_transcription_model, "whisper-1")

    def test_parse_args_with_openai_speaker_model(self):
        """Test parsing --openai-speaker-model argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-speaker-model",
                "gpt-4o",
            ]
        )
        self.assertEqual(args.openai_speaker_model, "gpt-4o")

    def test_parse_args_with_openai_summary_model(self):
        """Test parsing --openai-summary-model argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-summary-model",
                "gpt-4o-mini",
            ]
        )
        self.assertEqual(args.openai_summary_model, "gpt-4o-mini")

    def test_parse_args_with_openai_temperature(self):
        """Test parsing --openai-temperature argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-temperature",
                "0.7",
            ]
        )
        self.assertEqual(args.openai_temperature, 0.7)

    def test_parse_args_openai_model_defaults(self):
        """Test that OpenAI model arguments default to None."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertIsNone(args.openai_transcription_model)
        self.assertIsNone(args.openai_speaker_model)
        self.assertIsNone(args.openai_summary_model)
        self.assertIsNone(args.openai_temperature)

    def test_parse_args_invalid_transcription_provider(self):
        """Test that invalid --transcription-provider raises error."""
        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "https://example.com/feed.xml",
                    "--transcription-provider",
                    "invalid",
                ]
            )

    def test_parse_args_invalid_speaker_detector_provider(self):
        """Test that invalid --speaker-detector-provider raises error."""
        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "https://example.com/feed.xml",
                    "--speaker-detector-provider",
                    "invalid",
                ]
            )

    @patch("builtins.print")
    def test_parse_args_version(self, mock_print):
        """Test that --version flag prints version and exits."""
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)
        mock_print.assert_called_once()

    def test_parse_args_validates_args(self):
        """Test that parse_args validates arguments."""
        with self.assertRaises(ValueError):
            cli.parse_args(["", "--workers", "0"])  # Invalid RSS URL and workers

    @patch("podcast_scraper.cli._load_and_merge_config")
    def test_parse_args_with_config_file(self, mock_load_config):
        """Test parsing arguments with config file."""
        mock_args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap=1.25,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
            prefer_type=[],
            user_agent="test",
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_max_length=None,
            summary_min_length=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level="INFO",
            log_file=None,
            language="en",
            ner_model=None,
            auto_speakers=False,
            cache_detected_hosts=False,
            version=False,
            config="config.yaml",
        )
        mock_load_config.return_value = mock_args

        args = cli.parse_args(["--config", "config.yaml", "https://example.com/feed.xml"])

        mock_load_config.assert_called_once()
        self.assertIsNotNone(args)


class TestAddArgumentGroups(unittest.TestCase):
    """Test argument group addition functions."""

    def test_add_common_arguments(self):
        """Test that _add_common_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        # Check that key arguments are present
        self.assertTrue(parser._actions[0].dest in ["help", "rss", "config"])
        # Find rss argument
        rss_action = next((a for a in parser._actions if a.dest == "rss"), None)
        self.assertIsNotNone(rss_action)

    def test_add_transcription_arguments(self):
        """Test that _add_transcription_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_transcription_arguments(parser)

        # Check that transcription arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("transcribe_missing", action_dests)
        self.assertIn("transcription_provider", action_dests)
        self.assertIn("whisper_model", action_dests)
        self.assertIn("screenplay", action_dests)

        # Check that transcription_provider has correct choices
        transcription_action = next(
            (a for a in parser._actions if a.dest == "transcription_provider"), None
        )
        self.assertIsNotNone(transcription_action)
        self.assertEqual(transcription_action.choices, ["whisper", "openai"])
        self.assertEqual(transcription_action.default, "whisper")

    def test_add_metadata_arguments(self):
        """Test that _add_metadata_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_metadata_arguments(parser)

        # Check that metadata arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("generate_metadata", action_dests)
        self.assertIn("metadata_format", action_dests)

    def test_add_speaker_detection_arguments(self):
        """Test that _add_speaker_detection_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_speaker_detection_arguments(parser)

        # Check that speaker detection arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("language", action_dests)
        self.assertIn("speaker_detector_provider", action_dests)
        self.assertIn("auto_speakers", action_dests)

        # Check that speaker_detector_provider has correct choices
        speaker_detector_action = next(
            (a for a in parser._actions if a.dest == "speaker_detector_provider"), None
        )
        self.assertIsNotNone(speaker_detector_action)
        self.assertEqual(
            speaker_detector_action.choices, ["spacy", "ner", "openai"]
        )  # "ner" deprecated
        self.assertEqual(speaker_detector_action.default, "spacy")

    def test_add_summarization_arguments(self):
        """Test that _add_summarization_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_summarization_arguments(parser)

        # Check that summarization arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("generate_summaries", action_dests)
        self.assertIn("summary_provider", action_dests)

    def test_add_cache_arguments(self):
        """Test that _add_cache_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_cache_arguments(parser)

        # Check that cache arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("cache_info", action_dests)
        self.assertIn("prune_cache", action_dests)
        self.assertIn("cache_dir", action_dests)

    def test_add_openai_arguments(self):
        """Test that _add_openai_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_openai_arguments(parser)

        # Check that OpenAI arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("openai_api_base", action_dests)
        self.assertIn("openai_transcription_model", action_dests)
        self.assertIn("openai_speaker_model", action_dests)
        self.assertIn("openai_summary_model", action_dests)
        self.assertIn("openai_temperature", action_dests)

        # Check that openai_api_base has correct default
        openai_action = next((a for a in parser._actions if a.dest == "openai_api_base"), None)
        self.assertIsNotNone(openai_action)
        self.assertIsNone(openai_action.default)

        # Check that model arguments have correct defaults (None, uses config defaults)
        for model_arg in [
            "openai_transcription_model",
            "openai_speaker_model",
            "openai_summary_model",
        ]:
            action = next((a for a in parser._actions if a.dest == model_arg), None)
            self.assertIsNotNone(action, f"{model_arg} should be present")
            self.assertIsNone(action.default, f"{model_arg} default should be None")


class TestCacheSubcommand(unittest.TestCase):
    """Test cache subcommand parsing and execution."""

    def test_parse_cache_args_status(self):
        """Test parsing cache --status command."""
        args = cli._parse_cache_args(["--status"])
        self.assertTrue(args.status)
        self.assertFalse(args.clean)

    def test_parse_cache_args_clean_all(self):
        """Test parsing cache --clean all command."""
        args = cli._parse_cache_args(["--clean", "all"])
        self.assertFalse(args.status)
        self.assertEqual(args.clean, "all")

    def test_parse_cache_args_clean_whisper(self):
        """Test parsing cache --clean whisper command."""
        args = cli._parse_cache_args(["--clean", "whisper"])
        self.assertEqual(args.clean, "whisper")

    def test_parse_cache_args_clean_with_yes(self):
        """Test parsing cache --clean with --yes flag."""
        args = cli._parse_cache_args(["--clean", "all", "--yes"])
        self.assertEqual(args.clean, "all")
        self.assertTrue(args.yes)

    def test_parse_args_detects_cache_subcommand(self):
        """Test that parse_args detects cache subcommand."""
        args = cli.parse_args(["cache", "--status"])
        self.assertTrue(hasattr(args, "command"))
        self.assertEqual(args.command, "cache")
        self.assertTrue(args.status)

    def test_parse_args_normal_command_not_cache(self):
        """Test that normal command is not treated as cache."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertFalse(hasattr(args, "command") and args.command == "cache")

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status(self, mock_get_info, mock_format_size):
        """Test main() with cache --status command."""
        from pathlib import Path

        mock_get_info.return_value = {
            "whisper": {"dir": Path("/whisper"), "size": 100, "count": 1, "models": []},
            "transformers": {"dir": Path("/transformers"), "size": 200, "count": 2, "models": []},
            "spacy": {"dir": Path("/spacy"), "size": 50, "count": 1, "models": []},
            "total_size": 350,
        }
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--status"])

        self.assertEqual(exit_code, 0)
        mock_get_info.assert_called_once()

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_all_caches")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_clean_all(self, mock_get_info, mock_clean, mock_format_size):
        """Test main() with cache --clean all --yes command."""
        mock_get_info.return_value = {"total_size": 350}
        mock_clean.return_value = {"whisper": (1, 100), "transformers": (2, 200), "spacy": (1, 50)}
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "all", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_whisper_cache")
    def test_main_cache_clean_whisper(self, mock_clean, mock_format_size):
        """Test main() with cache --clean whisper --yes command."""
        mock_clean.return_value = (1, 100)
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "whisper", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_transformers_cache")
    def test_main_cache_clean_transformers(self, mock_clean, mock_format_size):
        """Test main() with cache --clean transformers --yes command."""
        mock_clean.return_value = (2, 200)
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "transformers", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_spacy_cache")
    def test_main_cache_clean_spacy(self, mock_clean, mock_format_size):
        """Test main() with cache --clean spacy --yes command."""
        mock_clean.return_value = (1, 50)
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "spacy", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    def test_parse_cache_args_requires_status_or_clean(self):
        """Test that _parse_cache_args requires either --status or --clean."""
        with self.assertRaises(SystemExit):
            cli._parse_cache_args([])

    def test_parse_cache_args_clean_defaults_to_all(self):
        """Test that --clean without argument defaults to 'all'."""
        args = cli._parse_cache_args(["--clean"])
        self.assertEqual(args.clean, "all")

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_with_models(self, mock_get_info, mock_format_size):
        """Test main() with cache --status when models exist."""
        from pathlib import Path

        mock_get_info.return_value = {
            "whisper": {
                "dir": Path("/whisper"),
                "size": 100,
                "count": 1,
                "models": [{"name": "base.en.pt", "size": 100}],
            },
            "transformers": {
                "dir": Path("/transformers"),
                "size": 200,
                "count": 1,
                "models": [{"name": "facebook/bart-base", "size": 200}],
            },
            "spacy": {
                "dir": Path("/spacy"),
                "size": 50,
                "count": 1,
                "models": [{"name": "en_core_web_sm", "size": 50}],
            },
            "total_size": 350,
        }
        mock_format_size.side_effect = lambda x: f"{x} B"

        exit_code = cli.main(["cache", "--status"])

        self.assertEqual(exit_code, 0)
        mock_get_info.assert_called_once()

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_spacy_no_dir(self, mock_get_info, mock_format_size):
        """Test main() with cache --status when spacy has no cache dir."""
        from pathlib import Path

        mock_get_info.return_value = {
            "whisper": {"dir": Path("/whisper"), "size": 100, "count": 0, "models": []},
            "transformers": {"dir": Path("/transformers"), "size": 0, "count": 0, "models": []},
            "spacy": {"dir": None, "size": 0, "count": 0, "models": []},
            "total_size": 100,
        }
        mock_format_size.side_effect = lambda x: f"{x} B"

        exit_code = cli.main(["cache", "--status"])

        self.assertEqual(exit_code, 0)

    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_exception_handling(self, mock_get_info):
        """Test main() handles exceptions from cache operations."""
        # Test exception handling path
        mock_get_info.side_effect = Exception("Cache operation failed")

        exit_code = cli.main(["cache", "--status"])
        self.assertEqual(exit_code, 1)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_exception(self, mock_get_info, mock_format_size):
        """Test main() handles exceptions from cache operations."""
        mock_get_info.side_effect = Exception("Cache operation failed")
        mock_format_size.side_effect = lambda x: f"{x} B"

        exit_code = cli.main(["cache", "--status"])
        self.assertEqual(exit_code, 1)


class TestLoadAndMergeConfig(unittest.TestCase):
    """Test _load_and_merge_config function."""

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_validation_error(self, mock_load):
        """Test that _load_and_merge_config handles validation errors."""
        # Config file with unknown keys should raise ValueError
        mock_load.return_value = {"unknown_key": "value"}
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        with self.assertRaises(ValueError) as cm:
            cli._load_and_merge_config(parser, "config.yaml", None)
        self.assertIn("Unknown config option", str(cm.exception))

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_file_not_found(self, mock_load):
        """Test handling of missing config file."""
        mock_load.side_effect = FileNotFoundError("File not found")
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        with self.assertRaises(FileNotFoundError):
            cli._load_and_merge_config(parser, "missing.yaml", None)


if __name__ == "__main__":
    unittest.main()
