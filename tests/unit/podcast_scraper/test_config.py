#!/usr/bin/env python3
"""Tests for Config cross-field validation."""

import os
import sys
import unittest
import warnings

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pydantic import ValidationError

from podcast_scraper import Config, config


class TestSummaryValidation(unittest.TestCase):
    """Test summary-related cross-field validation."""

    def test_word_overlap_less_than_chunk_size(self):
        """Test that summary_word_overlap must be less than summary_word_chunk_size."""
        # Suppress expected warnings about values outside recommended ranges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaises(ValidationError) as context:
                Config(
                    rss_url="https://example.com/feed.xml",
                    summary_word_chunk_size=500,
                    summary_word_overlap=600,
                )
            self.assertIn("must be less than", str(context.exception))

    def test_word_overlap_equal_to_chunk_size_fails(self):
        """Test that overlap equal to chunk size fails."""
        # Suppress expected warnings about values outside recommended ranges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaises(ValidationError) as context:
                Config(
                    rss_url="https://example.com/feed.xml",
                    summary_word_chunk_size=500,
                    summary_word_overlap=500,
                )
            self.assertIn("must be less than", str(context.exception))

    def test_word_overlap_less_than_chunk_size_succeeds(self):
        """Test that valid overlap < chunk_size configuration succeeds."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            summary_word_chunk_size=900,
            summary_word_overlap=150,
        )
        self.assertEqual(cfg.summary_word_chunk_size, 900)
        self.assertEqual(cfg.summary_word_overlap, 150)

    def test_summaries_require_metadata(self):
        """Test that generate_summaries requires generate_metadata."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                generate_summaries=True,
                generate_metadata=False,
            )
        self.assertIn("requires generate_metadata", str(context.exception))

    def test_summaries_with_metadata_succeeds(self):
        """Test that summaries work when metadata is enabled."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
        )
        self.assertTrue(cfg.generate_summaries)
        self.assertTrue(cfg.generate_metadata)

    def test_word_chunk_size_outside_range_warns(self):
        """Test that word_chunk_size outside recommended range warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Config(
                rss_url="https://example.com/feed.xml",
                summary_word_chunk_size=500,  # Below 800
            )
            self.assertEqual(len(w), 1)
            self.assertIn("outside recommended range", str(w[0].message))
            self.assertIn("800-1200", str(w[0].message))

    def test_word_overlap_outside_range_warns(self):
        """Test that word_overlap outside recommended range warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Config(
                rss_url="https://example.com/feed.xml",
                summary_word_overlap=50,  # Below 100
            )
            self.assertEqual(len(w), 1)
            self.assertIn("outside recommended range", str(w[0].message))
            self.assertIn("100-200", str(w[0].message))


class TestOutputControlValidation(unittest.TestCase):
    """Test output control flag validation."""

    def test_clean_output_and_skip_existing_conflict(self):
        """Test that clean_output and skip_existing are mutually exclusive."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                clean_output=True,
                skip_existing=True,
            )
        self.assertIn("mutually exclusive", str(context.exception))
        self.assertIn("clean_output", str(context.exception))
        self.assertIn("skip_existing", str(context.exception))

    def test_clean_output_and_reuse_media_conflict(self):
        """Test that clean_output and reuse_media are mutually exclusive."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                clean_output=True,
                reuse_media=True,
            )
        self.assertIn("mutually exclusive", str(context.exception))
        self.assertIn("clean_output", str(context.exception))
        self.assertIn("reuse_media", str(context.exception))

    def test_skip_existing_and_reuse_media_compatible(self):
        """Test that skip_existing and reuse_media can be used together."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            skip_existing=True,
            reuse_media=True,
        )
        self.assertTrue(cfg.skip_existing)
        self.assertTrue(cfg.reuse_media)

    def test_clean_output_alone_succeeds(self):
        """Test that clean_output alone works fine."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            clean_output=True,
        )
        self.assertTrue(cfg.clean_output)
        self.assertFalse(cfg.skip_existing)
        self.assertFalse(cfg.reuse_media)


class TestTranscriptionValidation(unittest.TestCase):
    """Test transcription-related validation."""

    def test_transcribe_missing_requires_whisper_model(self):
        """Test that transcribe_missing requires a valid whisper_model."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                transcribe_missing=True,
                whisper_model="",
            )
        self.assertIn("requires a valid whisper_model", str(context.exception))

    def test_transcribe_missing_with_valid_model_succeeds(self):
        """Test that transcribe_missing works with valid whisper_model."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
        )
        self.assertTrue(cfg.transcribe_missing)
        self.assertEqual(cfg.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)

    def test_transcribe_missing_false_allows_empty_model(self):
        """Test that empty whisper_model is OK when transcribe_missing is False."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
            whisper_model="",
        )
        self.assertFalse(cfg.transcribe_missing)


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases in validation logic."""

    def test_multiple_validation_errors(self):
        """Test configuration with multiple validation errors."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                clean_output=True,
                skip_existing=True,  # Error: contradictory flags
            )
        # Should report validation errors
        error_str = str(context.exception)
        # Should report mutually exclusive error
        self.assertIn("mutually exclusive", error_str)

    def test_valid_complex_configuration(self):
        """Test that a complex valid configuration succeeds."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./transcripts",
            max_episodes=10,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=True,
            screenplay_num_speakers=2,
            auto_speakers=True,
            generate_metadata=True,
            metadata_format="yaml",
            generate_summaries=True,
            summary_word_chunk_size=900,
            summary_word_overlap=150,
            skip_existing=True,
            reuse_media=True,
        )
        # All settings should be applied correctly
        self.assertTrue(cfg.transcribe_missing)
        self.assertTrue(cfg.generate_summaries)
        self.assertTrue(cfg.skip_existing)
        self.assertTrue(cfg.reuse_media)
        self.assertFalse(cfg.clean_output)

    def test_preload_models_default(self):
        """Test that preload_models defaults to True."""
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertTrue(cfg.preload_models)

    def test_preload_models_can_be_disabled(self):
        """Test that preload_models can be set to False."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            preload_models=False,
        )
        self.assertFalse(cfg.preload_models)

    def test_preload_models_can_be_enabled(self):
        """Test that preload_models can be explicitly set to True."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
        )
        self.assertTrue(cfg.preload_models)


class TestConfigFieldValidators(unittest.TestCase):
    """Tests for Config field validators."""

    def test_rss_url_validator_handles_none(self):
        """Test that rss_url validator handles None and whitespace."""
        # Validator strips whitespace
        cfg = Config(rss_url="  https://example.com/feed.xml  ")
        self.assertEqual(cfg.rss_url, "https://example.com/feed.xml")

        # Empty string becomes None after validator
        cfg = Config(rss_url="")
        self.assertIsNone(cfg.rss_url)

        # None is allowed (validation happens in _validate_openai_provider_requirements)
        cfg = Config(rss_url=None)
        self.assertIsNone(cfg.rss_url)

    def test_output_dir_validator_handles_none(self):
        """Test that output_dir validator handles None."""
        # output_dir can be None in Config - it's derived later in _build_config
        cfg = Config(rss_url="https://example.com/feed.xml", output_dir=None)
        # Validator allows None (derivation happens in cli._build_config)
        self.assertIsNone(cfg.output_dir)

        # Test that validator handles environment variable
        import os

        original_env = os.environ.get("OUTPUT_DIR")
        try:
            os.environ["OUTPUT_DIR"] = "/tmp/test_output"
            cfg = Config(rss_url="https://example.com/feed.xml", output_dir=None)
            # Should load from environment if not provided
            self.assertEqual(cfg.output_dir, "/tmp/test_output")
        finally:
            if original_env:
                os.environ["OUTPUT_DIR"] = original_env
            elif "OUTPUT_DIR" in os.environ:
                del os.environ["OUTPUT_DIR"]

    def test_whisper_model_validator_handles_empty(self):
        """Test that whisper_model validator handles empty string."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
            whisper_model="",
        )
        self.assertEqual(cfg.whisper_model, "")

    def test_user_agent_validator_handles_none(self):
        """Test that user_agent validator handles None."""
        cfg = Config(rss_url="https://example.com/feed.xml", user_agent=None)
        # Should use default
        self.assertIsNotNone(cfg.user_agent)

    def test_log_level_validator_invalid(self):
        """Test that log_level validator rejects invalid values."""
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", log_level="INVALID")

    def test_log_level_validator_case_insensitive(self):
        """Test that log_level validator handles case-insensitive input."""
        cfg = Config(rss_url="https://example.com/feed.xml", log_level="debug")
        self.assertEqual(cfg.log_level, "DEBUG")

    def test_run_id_validator_too_long(self):
        """Test that run_id validator rejects too long values."""
        long_run_id = "a" * 101  # MAX_RUN_ID_LENGTH is 100
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", run_id=long_run_id)

    def test_max_episodes_validator_handles_none(self):
        """Test that max_episodes validator handles None."""
        cfg = Config(rss_url="https://example.com/feed.xml", max_episodes=None)
        self.assertIsNone(cfg.max_episodes)

    def test_timeout_validator_too_small(self):
        """Test that timeout validator enforces minimum."""
        # MIN_TIMEOUT_SECONDS is 1, validator enforces minimum
        # Value below minimum is clamped to minimum
        cfg = Config(rss_url="https://example.com/feed.xml", timeout=0)
        self.assertEqual(cfg.timeout, 1)  # Clamped to minimum

        # Value at minimum should pass
        cfg = Config(rss_url="https://example.com/feed.xml", timeout=1)
        self.assertEqual(cfg.timeout, 1)

        # Value above minimum should pass
        cfg = Config(rss_url="https://example.com/feed.xml", timeout=30)
        self.assertEqual(cfg.timeout, 30)

    def test_delay_ms_validator_negative(self):
        """Test that delay_ms validator rejects negative values."""
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", delay_ms=-1)

    def test_prefer_types_validator_handles_list(self):
        """Test that prefer_types validator handles list input."""
        cfg = Config(
            rss_url="https://example.com/feed.xml", prefer_types=["audio/mpeg", "audio/mp4"]
        )
        self.assertEqual(cfg.prefer_types, ["audio/mpeg", "audio/mp4"])

    def test_screenplay_speaker_names_validator_handles_string(self):
        """Test that screenplay_speaker_names validator handles string input."""
        cfg = Config(rss_url="https://example.com/feed.xml", screenplay_speaker_names="Alice, Bob")
        self.assertEqual(cfg.screenplay_speaker_names, ["Alice", "Bob"])

    def test_screenplay_gap_s_validator_handles_float(self):
        """Test that screenplay_gap_s validator handles float input."""
        cfg = Config(rss_url="https://example.com/feed.xml", screenplay_gap_s=1.5)
        self.assertEqual(cfg.screenplay_gap_s, 1.5)

    def test_screenplay_num_speakers_validator_handles_string(self):
        """Test that screenplay_num_speakers validator handles string input."""
        cfg = Config(rss_url="https://example.com/feed.xml", screenplay_num_speakers="3")
        self.assertEqual(cfg.screenplay_num_speakers, 3)

    def test_workers_validator_handles_string(self):
        """Test that workers validator handles string input."""
        cfg = Config(rss_url="https://example.com/feed.xml", workers="4")
        self.assertEqual(cfg.workers, 4)

    def test_language_validator_normalizes(self):
        """Test that language validator normalizes language codes."""
        cfg = Config(rss_url="https://example.com/feed.xml", language="EN")
        self.assertEqual(cfg.language, "en")

    def test_ner_model_validator_handles_none(self):
        """Test that ner_model validator handles None."""
        cfg = Config(rss_url="https://example.com/feed.xml", ner_model=None)
        self.assertIsNone(cfg.ner_model)

    def test_metadata_format_validator_invalid(self):
        """Test that metadata_format validator rejects invalid values."""
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", metadata_format="invalid")

    def test_metadata_subdirectory_validator_too_long(self):
        """Test that metadata_subdirectory validator rejects too long values."""
        long_subdir = "a" * 256  # MAX_METADATA_SUBDIRECTORY_LENGTH is 255
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", metadata_subdirectory=long_subdir)

    def test_openai_api_key_validator_handles_none(self):
        """Test that openai_api_key validator handles None when OpenAI providers are not used."""
        # Temporarily unset OPENAI_API_KEY to ensure it's not loaded from environment
        import os

        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Ensure no OpenAI providers are used to avoid cross-field validation error
            cfg = Config(
                rss_url="https://example.com/feed.xml",
                openai_api_key=None,
                transcription_provider="whisper",  # Not OpenAI
                speaker_detector_provider="spacy",  # Not OpenAI
                summary_provider="transformers",  # Not OpenAI
            )
            self.assertIsNone(cfg.openai_api_key)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_openai_api_base_validator_handles_none(self):
        """Test that openai_api_base validator handles None."""
        # Unset environment variable to ensure it's not loaded
        original_base = os.environ.pop("OPENAI_API_BASE", None)
        try:
            cfg = Config(rss_url="https://example.com/feed.xml", openai_api_base=None)
            self.assertIsNone(cfg.openai_api_base)
        finally:
            # Restore original environment variable if it existed
            if original_base is not None:
                os.environ["OPENAI_API_BASE"] = original_base

    def test_speaker_detector_provider_validator_invalid(self):
        """Test that speaker_detector_provider validator rejects invalid values."""
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", speaker_detector_provider="invalid")


if __name__ == "__main__":
    unittest.main()
