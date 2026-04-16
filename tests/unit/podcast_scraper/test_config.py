#!/usr/bin/env python3
"""Tests for Config cross-field validation."""

import os
import sys
import unittest
import warnings
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from pydantic import ValidationError

from podcast_scraper import Config, config

pytestmark = [pytest.mark.unit, pytest.mark.module_config]


@pytest.mark.unit
class TestLoadEnvVariableHelpers(unittest.TestCase):
    """Tests for environment variable loading helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = {}

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear environment variables
        env_vars = [
            "OUTPUT_DIR",
            "WORKERS",
            "OPENAI_TEMPERATURE",
            "MPS_EXCLUSIVE",
            "SUMMARY_DEVICE",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_load_string_env_var_success(self):
        """Test successful loading of string environment variable."""
        os.environ["OUTPUT_DIR"] = "/tmp/test_output"

        config.Config._load_string_env_var(self.data, "output_dir", "OUTPUT_DIR")

        self.assertEqual(self.data["output_dir"], "/tmp/test_output")

    def test_load_string_env_var_not_set(self):
        """Test that nothing is loaded when env var is not set."""
        config.Config._load_string_env_var(self.data, "output_dir", "OUTPUT_DIR")

        self.assertNotIn("output_dir", self.data)

    def test_load_string_env_var_already_in_data(self):
        """Test that existing value in data is not overwritten."""
        self.data["output_dir"] = "/existing/path"
        os.environ["OUTPUT_DIR"] = "/tmp/test_output"

        config.Config._load_string_env_var(self.data, "output_dir", "OUTPUT_DIR")

        self.assertEqual(self.data["output_dir"], "/existing/path")

    def test_load_string_env_var_empty_string(self):
        """Test that empty string env var is not loaded."""
        os.environ["OUTPUT_DIR"] = "   "

        config.Config._load_string_env_var(self.data, "output_dir", "OUTPUT_DIR")

        self.assertNotIn("output_dir", self.data)

    def test_load_int_env_var_success(self):
        """Test successful loading of integer environment variable."""
        os.environ["WORKERS"] = "4"

        config.Config._load_int_env_var(self.data, "workers", "WORKERS")

        self.assertEqual(self.data["workers"], 4)

    def test_load_int_env_var_below_minimum(self):
        """Test that value below minimum is not loaded."""
        os.environ["WORKERS"] = "0"

        config.Config._load_int_env_var(self.data, "workers", "WORKERS", min_value=1)

        self.assertNotIn("workers", self.data)

    def test_load_int_env_var_invalid(self):
        """Test that invalid integer value is not loaded."""
        os.environ["WORKERS"] = "not_a_number"

        config.Config._load_int_env_var(self.data, "workers", "WORKERS")

        self.assertNotIn("workers", self.data)

    def test_load_int_env_var_already_in_data(self):
        """Test that existing value in data is not overwritten."""
        self.data["workers"] = 2
        os.environ["WORKERS"] = "4"

        config.Config._load_int_env_var(self.data, "workers", "WORKERS")

        self.assertEqual(self.data["workers"], 2)

    def test_load_float_env_var_success(self):
        """Test successful loading of float environment variable."""
        os.environ["OPENAI_TEMPERATURE"] = "0.7"

        config.Config._load_float_env_var(
            self.data, "openai_temperature", "OPENAI_TEMPERATURE", 0.0, 2.0
        )

        self.assertEqual(self.data["openai_temperature"], 0.7)

    def test_load_float_env_var_below_minimum(self):
        """Test that value below minimum is not loaded."""
        os.environ["OPENAI_TEMPERATURE"] = "-0.1"

        config.Config._load_float_env_var(
            self.data, "openai_temperature", "OPENAI_TEMPERATURE", 0.0, 2.0
        )

        self.assertNotIn("openai_temperature", self.data)

    def test_load_float_env_var_above_maximum(self):
        """Test that value above maximum is not loaded."""
        os.environ["OPENAI_TEMPERATURE"] = "2.1"

        config.Config._load_float_env_var(
            self.data, "openai_temperature", "OPENAI_TEMPERATURE", 0.0, 2.0
        )

        self.assertNotIn("openai_temperature", self.data)

    def test_load_float_env_var_invalid(self):
        """Test that invalid float value is not loaded."""
        os.environ["OPENAI_TEMPERATURE"] = "not_a_float"

        config.Config._load_float_env_var(
            self.data, "openai_temperature", "OPENAI_TEMPERATURE", 0.0, 2.0
        )

        self.assertNotIn("openai_temperature", self.data)

    def test_load_bool_env_var_true(self):
        """Test loading boolean environment variable with true values."""
        for true_value in ["1", "true", "yes", "on", "TRUE", "YES", "ON"]:
            self.data = {}
            os.environ["MPS_EXCLUSIVE"] = true_value

            config.Config._load_bool_env_var(self.data, "mps_exclusive", "MPS_EXCLUSIVE")

            self.assertTrue(self.data["mps_exclusive"], f"Should be True for '{true_value}'")

    def test_load_bool_env_var_false(self):
        """Test loading boolean environment variable with false values."""
        for false_value in ["0", "false", "no", "off", "FALSE", "NO", "OFF"]:
            self.data = {}
            os.environ["MPS_EXCLUSIVE"] = false_value

            config.Config._load_bool_env_var(self.data, "mps_exclusive", "MPS_EXCLUSIVE")

            self.assertFalse(self.data["mps_exclusive"], f"Should be False for '{false_value}'")

    def test_load_bool_env_var_invalid(self):
        """Test that invalid boolean value is not loaded."""
        os.environ["MPS_EXCLUSIVE"] = "maybe"

        config.Config._load_bool_env_var(self.data, "mps_exclusive", "MPS_EXCLUSIVE")

        self.assertNotIn("mps_exclusive", self.data)

    def test_load_bool_env_var_already_in_data(self):
        """Test that existing value in data is not overwritten."""
        self.data["mps_exclusive"] = False
        os.environ["MPS_EXCLUSIVE"] = "true"

        config.Config._load_bool_env_var(self.data, "mps_exclusive", "MPS_EXCLUSIVE")

        self.assertFalse(self.data["mps_exclusive"])

    def test_load_device_env_var_valid_devices(self):
        """Test loading valid device environment variables."""
        for device in ["cpu", "cuda", "mps", "CPU", "CUDA", "MPS"]:
            self.data = {}
            os.environ["SUMMARY_DEVICE"] = device

            config.Config._load_device_env_var(self.data, "summary_device", "SUMMARY_DEVICE")

            self.assertEqual(
                self.data["summary_device"], device.lower(), f"Should load '{device}' as lowercase"
            )

    def test_load_device_env_var_invalid(self):
        """Test that invalid device value is not loaded."""
        os.environ["SUMMARY_DEVICE"] = "invalid_device"

        config.Config._load_device_env_var(self.data, "summary_device", "SUMMARY_DEVICE")

        self.assertNotIn("summary_device", self.data)

    def test_load_device_env_var_already_in_data(self):
        """Test that existing value in data is not overwritten."""
        self.data["summary_device"] = "cpu"
        os.environ["SUMMARY_DEVICE"] = "cuda"

        config.Config._load_device_env_var(self.data, "summary_device", "SUMMARY_DEVICE")

        self.assertEqual(self.data["summary_device"], "cpu")


class TestSummaryModeProfileDefaults(unittest.TestCase):
    """Tests for dev/prod profile selection of summary mode defaults."""

    def tearDown(self):
        """Clean up environment variables set by tests."""
        os.environ.pop("PODCAST_SCRAPER_PROFILE", None)

    def test_default_summary_mode_id_is_prod_when_profile_unset(self):
        """When profile is unset (and not testing), default mode should be production."""
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            os.environ.pop("PODCAST_SCRAPER_PROFILE", None)
            mode_id = config._get_default_summary_mode_id()
            self.assertEqual(mode_id, config.config_constants.PROD_DEFAULT_SUMMARY_MODE_ID)

    def test_default_summary_mode_id_is_dev_when_profile_dev(self):
        """When profile is dev (and not testing), default mode should be dev baseline."""
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            os.environ["PODCAST_SCRAPER_PROFILE"] = "dev"
            mode_id = config._get_default_summary_mode_id()
            self.assertEqual(mode_id, config.config_constants.DEV_DEFAULT_SUMMARY_MODE_ID)

    def test_default_summary_tokenize_uses_selected_mode_id(self):
        """Tokenize defaults should be sourced from the selected promoted mode."""
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            os.environ["PODCAST_SCRAPER_PROFILE"] = "dev"
            expected = {
                "map_max_input_tokens": 111,
                "reduce_max_input_tokens": 222,
                "truncation": True,
            }
            with patch(
                "podcast_scraper.providers.ml.model_registry.ModelRegistry.get_mode_configuration",
                return_value=SimpleNamespace(tokenize=expected),
            ) as mock_get_mode:
                got = config._get_default_summary_tokenize()
                self.assertEqual(got, expected)
                mock_get_mode.assert_called_once_with(
                    config.config_constants.DEV_DEFAULT_SUMMARY_MODE_ID
                )


class TestOpenAICleaningModelDefaults(unittest.TestCase):
    """OpenAI transcript-cleaning defaults align with config_constants (RFC-044 adjacency)."""

    def test_get_default_openai_cleaning_model_test_environment(self):
        """Test/CI uses TEST_DEFAULT_OPENAI_CLEANING_MODEL."""
        with patch("podcast_scraper.config._is_test_environment", return_value=True):
            self.assertEqual(
                config._get_default_openai_cleaning_model(),
                config.TEST_DEFAULT_OPENAI_CLEANING_MODEL,
            )
            self.assertEqual(
                config._get_default_openai_cleaning_model(),
                config.config_constants.TEST_DEFAULT_OPENAI_CLEANING_MODEL,
            )

    def test_get_default_openai_cleaning_model_production_environment(self):
        """Non-test uses PROD_DEFAULT_OPENAI_CLEANING_MODEL."""
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            self.assertEqual(
                config._get_default_openai_cleaning_model(),
                config.PROD_DEFAULT_OPENAI_CLEANING_MODEL,
            )

    def test_config_openai_cleaning_model_matches_test_default(self):
        """Config field resolves via factory (pytest runs in test environment)."""
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.openai_cleaning_model, config.TEST_DEFAULT_OPENAI_CLEANING_MODEL)


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

    def test_generate_gi_requires_metadata(self):
        """Test that generate_gi=True requires generate_metadata=True."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                generate_gi=True,
                generate_metadata=False,
            )
        self.assertIn("requires generate_metadata", str(context.exception))

    def test_generate_gi_with_metadata_succeeds(self):
        """Test that generate_gi works when metadata is enabled."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_gi=True,
            generate_metadata=True,
        )
        self.assertTrue(cfg.generate_gi)
        self.assertTrue(cfg.generate_metadata)

    def test_gi_qa_window_overlap_must_be_less_than_window(self):
        """gi_qa_window_overlap_chars must be < gi_qa_window_chars when windowing is on."""
        with self.assertRaises(ValidationError) as ctx:
            Config(
                rss_url="https://example.com/feed.xml",
                gi_qa_window_chars=100,
                gi_qa_window_overlap_chars=100,
            )
        self.assertIn("gi_qa_window_overlap", str(ctx.exception))

    def test_gi_qa_window_zero_allows_large_overlap(self):
        """When windowing is disabled (0), overlap is unconstrained."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            gi_qa_window_chars=0,
            gi_qa_window_overlap_chars=50_000,
        )
        self.assertEqual(cfg.gi_qa_window_chars, 0)

    def test_generate_kg_requires_metadata(self):
        """Test that generate_kg=True requires generate_metadata=True."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                generate_kg=True,
                generate_metadata=False,
            )
        self.assertIn("requires generate_metadata", str(context.exception))

    def test_generate_kg_with_metadata_succeeds(self):
        """Test that generate_kg works when metadata is enabled."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_kg=True,
            generate_metadata=True,
        )
        self.assertTrue(cfg.generate_kg)
        self.assertTrue(cfg.generate_metadata)

    def test_summary_provider_hybrid_ml_succeeds(self):
        """hybrid_ml summary provider keeps default hybrid model fields."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="hybrid_ml",
        )
        self.assertEqual(cfg.summary_provider, "hybrid_ml")
        self.assertIsNotNone(cfg.hybrid_map_model)
        self.assertIsNotNone(cfg.hybrid_reduce_model)

    def test_hybrid_reduce_instruction_style_paragraph(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            hybrid_reduce_instruction_style="paragraph",
        )
        self.assertEqual(cfg.hybrid_reduce_instruction_style, "paragraph")

    def test_gi_models_override_when_generate_gi(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_gi=True,
            generate_metadata=True,
            gi_embedding_model="org/custom-embed",
            extractive_qa_model="org/custom-qa",
            nli_model="org/custom-nli",
        )
        self.assertEqual(cfg.gi_embedding_model, "org/custom-embed")
        self.assertEqual(cfg.extractive_qa_model, "org/custom-qa")
        self.assertEqual(cfg.nli_model, "org/custom-nli")

    def test_gil_evidence_aligns_with_openai_summary_when_match_enabled(self):
        """Default gil_evidence_match_summary_provider upgrades transformers → openai."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_gi=True,
            generate_metadata=True,
            summary_provider="openai",
            openai_api_key="sk-test",
            gil_evidence_match_summary_provider=True,
        )
        self.assertEqual(cfg.quote_extraction_provider, "openai")
        self.assertEqual(cfg.entailment_provider, "openai")

    def test_gil_evidence_stays_local_when_match_disabled(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_gi=True,
            generate_metadata=True,
            summary_provider="openai",
            openai_api_key="sk-test",
            gil_evidence_match_summary_provider=False,
        )
        self.assertEqual(cfg.quote_extraction_provider, "transformers")
        self.assertEqual(cfg.entailment_provider, "transformers")

    def test_gil_evidence_not_changed_for_transformers_summary(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_gi=True,
            generate_metadata=True,
            summary_provider="transformers",
        )
        self.assertEqual(cfg.quote_extraction_provider, "transformers")
        self.assertEqual(cfg.entailment_provider, "transformers")

    def test_gil_evidence_aligns_with_hybrid_ml_summary_when_match_enabled(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_gi=True,
            generate_metadata=True,
            summary_provider="hybrid_ml",
            gil_evidence_match_summary_provider=True,
        )
        self.assertEqual(cfg.quote_extraction_provider, "hybrid_ml")
        self.assertEqual(cfg.entailment_provider, "hybrid_ml")

    def test_default_summary_prompt_params_include_bullet_defaults(self):
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertEqual(cfg.summary_prompt_params.get("bullet_min"), 3)
        self.assertEqual(cfg.summary_prompt_params.get("max_words_per_bullet"), 45)
        self.assertIsNone(cfg.summary_prompt_params.get("bullet_max"))

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

    def test_clean_output_and_append_conflict(self):
        """Test that clean_output and append are mutually exclusive (GitHub #444)."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                clean_output=True,
                append=True,
            )
        self.assertIn("mutually exclusive", str(context.exception))
        self.assertIn("clean_output", str(context.exception))
        self.assertIn("append", str(context.exception))

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


class TestEpisodeSelectionConfig(unittest.TestCase):
    """Episode order / offset / date range (GitHub #521)."""

    def test_episode_since_after_until_raises(self):
        """episode_since must not be after episode_until."""
        with self.assertRaises(ValidationError) as ctx:
            Config(
                rss_url="https://example.com/feed.xml",
                episode_since=date(2024, 6, 1),
                episode_until=date(2024, 1, 1),
            )
        self.assertIn("episode_since", str(ctx.exception))
        self.assertIn("episode_until", str(ctx.exception))

    def test_episode_since_equal_until_ok(self):
        """Same calendar day for since and until is valid."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            episode_since=date(2024, 3, 15),
            episode_until=date(2024, 3, 15),
        )
        self.assertEqual(cfg.episode_since, date(2024, 3, 15))
        self.assertEqual(cfg.episode_until, date(2024, 3, 15))

    def test_episode_offset_negative_raises(self):
        """episode_offset must be non-negative."""
        with self.assertRaises(ValidationError) as ctx:
            Config(
                rss_url="https://example.com/feed.xml",
                episode_offset=-1,
            )
        self.assertIn("episode_offset", str(ctx.exception))

    def test_episode_order_invalid_raises(self):
        """episode_order must be newest or oldest."""
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                episode_order="first",  # type: ignore[arg-type]
            )


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

    def test_vector_search_defaults_false(self):
        """vector_search is off by default until semantic corpus search is enabled (PRD-021)."""
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertFalse(cfg.vector_search)

    def test_vector_chunk_overlap_must_be_less_than_chunk_size(self):
        """vector_chunk_overlap_tokens must be strictly less than vector_chunk_size_tokens."""
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                vector_chunk_size_tokens=50,
                vector_chunk_overlap_tokens=50,
            )

    def test_vector_index_types_empty_list_normalized_to_none(self):
        """Validator maps [] to None (same as unset) for vector_index_types."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            vector_index_types=[],
        )
        self.assertIsNone(cfg.vector_index_types)

    def test_vector_index_types_explicit_list(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            vector_index_types=["insight", "transcript"],
        )
        self.assertEqual(cfg.vector_index_types, ["insight", "transcript"])

    def test_vector_backend_qdrant_literal_stored(self):
        """qdrant is allowed for forward compatibility (RFC-061 Phase 2)."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            vector_backend="qdrant",
        )
        self.assertEqual(cfg.vector_backend, "qdrant")

    def test_vector_search_true_with_embedding_model(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            vector_search=True,
            vector_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_faiss_index_mode="flat",
        )
        self.assertTrue(cfg.vector_search)
        self.assertEqual(cfg.vector_faiss_index_mode, "flat")
        self.assertIn("MiniLM", cfg.vector_embedding_model)

    def test_skip_auto_vector_index_defaults_false(self):
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertFalse(cfg.skip_auto_vector_index)

    def test_skip_auto_vector_index_can_enable(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            skip_auto_vector_index=True,
        )
        self.assertTrue(cfg.skip_auto_vector_index)

    def test_vector_index_path_optional_relative(self):
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            vector_index_path="indexes/semantic",
        )
        self.assertEqual(cfg.vector_index_path, "indexes/semantic")

    def test_append_with_skip_existing_allowed(self):
        """Append/resume stack may combine with skip_existing (GitHub #444)."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            append=True,
            skip_existing=True,
            clean_output=False,
        )
        self.assertTrue(cfg.append)
        self.assertTrue(cfg.skip_existing)

    def test_evidence_stack_fields_defaults(self):
        """Test that GIL evidence stack config fields exist with defaults (Issue #435)."""
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertIsNotNone(cfg.embedding_model)
        self.assertIn("/", cfg.embedding_model)
        self.assertIsNone(cfg.embedding_device)
        self.assertIsNotNone(cfg.extractive_qa_model)
        self.assertIn("/", cfg.extractive_qa_model)
        self.assertIsNone(cfg.extractive_qa_device)
        self.assertIsNotNone(cfg.nli_model)
        self.assertIn("/", cfg.nli_model)
        self.assertIsNone(cfg.nli_device)

    def test_mps_exclusive_default(self):
        """Test that mps_exclusive defaults to True."""
        cfg = Config(rss_url="https://example.com/feed.xml")
        self.assertTrue(cfg.mps_exclusive)

    def test_mps_exclusive_can_be_disabled(self):
        """Test that mps_exclusive can be set to False."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            mps_exclusive=False,
        )
        self.assertFalse(cfg.mps_exclusive)

    def test_mps_exclusive_can_be_enabled(self):
        """Test that mps_exclusive can be set to True."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            mps_exclusive=True,
        )
        self.assertTrue(cfg.mps_exclusive)


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

    def test_rss_url_rejects_non_http_scheme(self):
        """Non-http(s) RSS URLs must fail (aligns with CLI _validate_rss_url)."""
        with self.assertRaises(ValidationError) as ctx:
            Config(rss_url="file:///tmp/feed.xml")
        err = str(ctx.exception).lower()
        self.assertIn("http", err)

    def test_rss_url_rejects_missing_hostname(self):
        with self.assertRaises(ValidationError):
            Config(rss_url="https://")

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

    def test_hybrid_map_device_invalid_raises(self):
        """hybrid_*_device must be cuda, mps, cpu, auto, or empty."""
        with self.assertRaises(ValidationError) as ctx:
            Config(
                rss_url="https://example.com/feed.xml",
                hybrid_map_device="invalid-device",
            )
        self.assertIn("hybrid_*_device", str(ctx.exception))

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


@pytest.mark.unit
class TestPathTraversalAndTokenValidators(unittest.TestCase):
    """Hardening validators: no .. in paths; temperatures; max_tokens."""

    def test_output_dir_rejects_parent_components(self):
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", output_dir="/tmp/../etc")

    def test_log_file_rejects_traversal(self):
        with self.assertRaises(ValidationError):
            Config(rss_url="https://example.com/feed.xml", log_file="logs/../../secret.log")

    def test_summary_cache_dir_rejects_traversal(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                summary_cache_dir="cache/../outside",
            )

    def test_anthropic_temperature_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                anthropic_temperature=1.1,
            )

    def test_mistral_temperature_above_two_rejected(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                mistral_temperature=2.1,
            )

    def test_anthropic_cleaning_temperature_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                anthropic_cleaning_temperature=1.5,
            )

    def test_openai_max_tokens_zero_rejected(self):
        with self.assertRaises(ValidationError):
            Config(
                rss_url="https://example.com/feed.xml",
                openai_max_tokens=0,
            )


class TestMultiFeedConfig440(unittest.TestCase):
    """GitHub #440: rss_urls / feeds and corpus parent validation."""

    def tearDown(self):
        if "OUTPUT_DIR" in os.environ:
            del os.environ["OUTPUT_DIR"]

    def test_rss_urls_two_without_output_dir_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            Config(rss_urls=["https://a.example/feed.xml", "https://b.example/feed.xml"])
        self.assertIn("output_dir", str(ctx.exception).lower())

    def test_feeds_alias_two_with_output_dir_ok(self):
        cfg = Config.model_validate(
            {
                "feeds": ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                "output_dir": "/tmp/corpus_parent",
            }
        )
        self.assertEqual(len(cfg.rss_urls or []), 2)

    @patch.dict(os.environ, {"OUTPUT_DIR": "/tmp/corpus_from_env"}, clear=False)
    def test_rss_urls_two_ok_when_output_dir_from_env(self):
        cfg = Config(
            rss_urls=["https://a.example/feed.xml", "https://b.example/feed.xml"],
        )
        self.assertTrue(cfg.output_dir)

    def test_rss_field_as_list_requires_output_dir(self):
        """YAML-style rss: [url, url] promotes to rss_urls (GitHub #440)."""
        with self.assertRaises(ValidationError) as ctx:
            Config.model_validate(
                {
                    "rss": ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                }
            )
        self.assertIn("output_dir", str(ctx.exception).lower())

    def test_rss_field_as_list_with_output_dir_ok(self):
        cfg = Config.model_validate(
            {
                "rss": ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                "output_dir": "/tmp/corpus_multi",
            }
        )
        self.assertEqual(len(cfg.rss_urls or []), 2)

    def test_deprecated_multi_feed_soft_fail_exit_zero_maps_to_multi_feed_strict(self):
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            cfg = Config.model_validate(
                {
                    "rss": ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                    "output_dir": "/tmp/corpus_dep",
                    "multi_feed_soft_fail_exit_zero": True,
                }
            )
        self.assertFalse(cfg.multi_feed_strict)
        self.assertTrue(any("multi_feed_soft_fail_exit_zero" in str(w.message) for w in wrec))

    def test_deprecated_multi_feed_soft_fail_exit_zero_false_maps_strict_true(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = Config.model_validate(
                {
                    "rss": ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                    "output_dir": "/tmp/corpus_dep2",
                    "multi_feed_soft_fail_exit_zero": False,
                }
            )
        self.assertTrue(cfg.multi_feed_strict)

    def test_multi_feed_strict_and_deprecated_legacy_key_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            Config.model_validate(
                {
                    "rss": ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                    "output_dir": "/tmp/corpus_both",
                    "multi_feed_strict": False,
                    "multi_feed_soft_fail_exit_zero": True,
                }
            )
        self.assertIn("multi_feed_strict", str(ctx.exception))


@pytest.mark.unit
class TestScreenplayApiTranscriptionCoerce562(unittest.TestCase):
    """GitHub #562: screenplay only for whisper; coerce for API transcription."""

    def tearDown(self) -> None:
        config.reset_screenplay_issue_562_gates()

    def test_openai_transcription_coerces_screenplay_false(self) -> None:
        cfg = Config(
            rss="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test",
            screenplay=True,
        )
        self.assertFalse(cfg.screenplay)

    def test_whisper_keeps_screenplay_true(self) -> None:
        cfg = Config(
            rss="https://example.com/feed.xml",
            transcription_provider="whisper",
            screenplay=True,
        )
        self.assertTrue(cfg.screenplay)

    def test_coerce_info_emitted_once_per_process(self) -> None:
        with self.assertLogs("podcast_scraper.config", level="INFO") as cm:
            Config(
                rss="https://example.com/f1.xml",
                transcription_provider="openai",
                openai_api_key="sk-test",
                screenplay=True,
            )
            Config(
                rss="https://example.com/f2.xml",
                transcription_provider="gemini",
                gemini_api_key="g-test",
                screenplay=True,
            )
        matched = [x for x in cm.output if "562" in x and "screenplay" in x.lower()]
        self.assertEqual(len(matched), 1, msg=str(cm.output))

    def test_screenplay_integer_one_coerced_for_openai(self) -> None:
        cfg = Config.model_validate(
            {
                "rss": "https://example.com/feed.xml",
                "transcription_provider": "openai",
                "openai_api_key": "sk-test",
                "screenplay": 1,
            }
        )
        self.assertFalse(cfg.screenplay)

    @patch.dict(os.environ, {"PODCAST_SCRAPER_SCREENPLAY_STRICT": "1"}, clear=False)
    def test_screenplay_strict_env_rejects_api_transcription(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            Config(
                rss="https://example.com/feed.xml",
                transcription_provider="openai",
                openai_api_key="sk-test",
                screenplay=True,
            )
        self.assertIn("PODCAST_SCRAPER_SCREENPLAY_STRICT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
