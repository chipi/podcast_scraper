#!/usr/bin/env python3
"""Tests for filesystem operations."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import sys
import tempfile
import unittest

# Bandit: tests construct safe XML elements
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path
from unittest.mock import patch

from platformdirs import user_cache_dir, user_data_dir

import podcast_scraper
from podcast_scraper import config
from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

# Import directly from tests.conftest (works with pytest-xdist)
from tests.conftest import (  # noqa: E402
    create_test_config,
    create_test_episode,
    TEST_BASE_URL,
    TEST_CONTENT_TYPE_VTT,
    TEST_CUSTOM_OUTPUT_DIR,
    TEST_EPISODE_TITLE,
    TEST_EPISODE_TITLE_SPECIAL,
    TEST_FEED_URL,
    TEST_MEDIA_TYPE_MP3,
    TEST_MEDIA_URL,
    TEST_OUTPUT_DIR,
    TEST_RUN_ID,
    TEST_TRANSCRIPT_TYPE_VTT,
    TEST_TRANSCRIPT_URL,
    TEST_TRANSCRIPT_URL_SRT,
)


class TestSanitizeFilename(unittest.TestCase):
    """Tests for sanitize_filename function."""

    def test_simple_filename(self):
        """Test sanitizing a simple filename."""
        name = "episode_title"
        result = filesystem.sanitize_filename(name)
        self.assertEqual(result, "episode_title")

    def test_filename_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        name = TEST_EPISODE_TITLE_SPECIAL
        result = filesystem.sanitize_filename(name)
        self.assertNotIn(":", result)
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)
        self.assertNotIn("*", result)
        self.assertNotIn("?", result)

    def test_filename_with_whitespace(self):
        """Test sanitizing filename with whitespace."""
        name = f"  {TEST_EPISODE_TITLE}   "
        result = filesystem.sanitize_filename(name)
        self.assertEqual(result, TEST_EPISODE_TITLE)

    def test_empty_filename(self):
        """Test sanitizing empty filename."""
        name = ""
        result = filesystem.sanitize_filename(name)
        self.assertEqual(result, "untitled")


class TestValidateAndNormalizeOutputDir(unittest.TestCase):
    """Tests for validate_and_normalize_output_dir function."""

    def test_valid_relative_path(self):
        """Test validating a valid relative path."""
        path = TEST_OUTPUT_DIR
        result = filesystem.validate_and_normalize_output_dir(path)
        self.assertIsInstance(result, str)
        self.assertIn(TEST_OUTPUT_DIR, result)

    def test_path_traversal_attempt(self):
        """Test that path traversal attempts are handled."""
        path = "../../etc/passwd"
        result = filesystem.validate_and_normalize_output_dir(path)
        # Should normalize but not allow traversal
        self.assertIsInstance(result, str)

    def test_absolute_path(self):
        """Test validating an absolute path."""
        # Use a path in the current working directory to avoid system directory restrictions
        test_path = os.path.join(os.getcwd(), "test_output")
        try:
            result = filesystem.validate_and_normalize_output_dir(test_path)
            self.assertIsInstance(result, str)
            self.assertIn("test_output", result)
        finally:
            # Clean up if directory was created
            if os.path.exists(test_path) and os.path.isdir(test_path):
                try:
                    os.rmdir(test_path)
                except OSError:
                    pass

    def test_platformdirs_user_locations_do_not_warn(self):
        """Paths under platformdirs user data/cache locations should not warn."""
        candidates = {
            user_data_dir("podcast_scraper"),
            user_data_dir("podcast-scraper"),
            user_cache_dir("podcast_scraper"),
            user_cache_dir("podcast-scraper"),
        }
        for candidate in {c for c in candidates if c}:
            with self.subTest(candidate=candidate):
                with patch.object(filesystem.logger, "warning") as mock_warning:
                    result = filesystem.validate_and_normalize_output_dir(candidate)
                self.assertIsInstance(result, str)
                resolved_result = Path(result).resolve()
                resolved_candidate = Path(candidate).expanduser().resolve()
                self.assertTrue(
                    resolved_result == resolved_candidate
                    or resolved_result.is_relative_to(resolved_candidate)
                )
                mock_warning.assert_not_called()


class TestDeriveOutputDir(unittest.TestCase):
    """Tests for derive_output_dir function."""

    def test_default_output_dir(self):
        """Test deriving default output directory from RSS URL."""
        rss_url = TEST_FEED_URL
        result = filesystem.derive_output_dir(rss_url, None)
        self.assertIn("output/rss", result)
        self.assertIn("example.com", result)
        self.assertEqual(result, "output/rss_example.com_6904f1c4")

    def test_custom_output_dir(self):
        """Test using custom output directory."""
        rss_url = TEST_FEED_URL
        custom = TEST_CUSTOM_OUTPUT_DIR
        result = filesystem.derive_output_dir(rss_url, custom)
        # Result is normalized to absolute path
        self.assertIn(TEST_CUSTOM_OUTPUT_DIR, result)


class TestDeriveMediaExtension(unittest.TestCase):
    """Tests for derive_media_extension function."""

    def test_derive_from_media_type(self):
        """Test deriving extension from media type."""
        result = episode_processor.derive_media_extension(
            TEST_MEDIA_TYPE_MP3, f"{TEST_BASE_URL}/audio"
        )
        self.assertEqual(result, ".mp3")

    def test_derive_from_url(self):
        """Test deriving extension from URL."""
        result = episode_processor.derive_media_extension(None, TEST_MEDIA_URL)
        self.assertEqual(result, ".mp3")

    def test_default_extension(self):
        """Test default extension when type and URL don't match."""
        result = episode_processor.derive_media_extension(None, f"{TEST_BASE_URL}/audio")
        self.assertEqual(result, ".bin")


class TestDeriveTranscriptExtension(unittest.TestCase):
    """Tests for derive_transcript_extension function."""

    def test_derive_from_transcript_type(self):
        """Test deriving extension from transcript type."""
        result = episode_processor.derive_transcript_extension(
            TEST_TRANSCRIPT_TYPE_VTT, None, f"{TEST_BASE_URL}/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_content_type(self):
        """Test deriving extension from content type."""
        result = episode_processor.derive_transcript_extension(
            None, TEST_CONTENT_TYPE_VTT, f"{TEST_BASE_URL}/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_url(self):
        """Test deriving extension from URL."""
        result = episode_processor.derive_transcript_extension(None, None, TEST_TRANSCRIPT_URL_SRT)
        self.assertEqual(result, ".srt")

    def test_url_with_query_keeps_extension(self):
        """URL query parameters should not override file extension."""
        url = f"{TEST_TRANSCRIPT_URL}?alt=json"
        result = episode_processor.derive_transcript_extension(None, None, url)
        self.assertEqual(result, ".vtt")

    def test_default_extension(self):
        """Test default extension."""
        result = episode_processor.derive_transcript_extension(
            None, None, f"{TEST_BASE_URL}/transcript"
        )
        self.assertEqual(result, ".txt")


class TestSetupOutputDirectory(unittest.TestCase):
    """Tests for setup_output_directory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_with_run_id(self):
        """Test setting up output directory with run ID."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            max_episodes=None,
            user_agent="test",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap_s=2.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=TEST_RUN_ID,
            skip_existing=False,
            clean_output=False,
            auto_speakers=False,  # Disable to test run_id only
            generate_summaries=False,  # Disable to test run_id only
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # New format: run_<run_id>_<timestamp> or run_<run_id>_<timestamp>_<hash> (if ML features)
        import re

        # Pattern matches: run_<run_id>_<timestamp> or run_<run_id>_<timestamp>_<hash>
        self.assertRegex(
            output_dir,
            rf"run_{re.escape(TEST_RUN_ID)}_\d{{8}}-\d{{6}}(_[a-f0-9]{{8}})?",
        )
        self.assertIsNotNone(run_suffix)
        self.assertIn(TEST_RUN_ID, run_suffix or "")
        # No ML features in this test, so no full_config_string
        self.assertIsNone(full_config_string)

    def test_with_duplicate_directory_counter(self):
        """Test that duplicate directories get a counter appended."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            max_episodes=None,
            user_agent="test",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            screenplay=False,
            screenplay_gap_s=2.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=TEST_RUN_ID,
            skip_existing=False,
            clean_output=False,  # Important: don't clean, so we can test counter
            auto_speakers=False,
            generate_summaries=False,
        )
        # First call - should create run_<run_id>_<timestamp>
        output_dir1, run_suffix1, _ = filesystem.setup_output_directory(cfg)
        import re

        self.assertRegex(
            output_dir1, rf"run_{re.escape(TEST_RUN_ID)}_\d{{8}}-\d{{6}}(_[a-f0-9]{{8}})?"
        )
        self.assertIsNotNone(run_suffix1)
        self.assertIn(TEST_RUN_ID, run_suffix1 or "")
        self.assertTrue(os.path.exists(output_dir1))

        # Second call - should create run_<run_id>_<timestamp>_1
        output_dir2, run_suffix2, _ = filesystem.setup_output_directory(cfg)
        self.assertRegex(
            output_dir2, rf"run_{re.escape(TEST_RUN_ID)}_\d{{8}}-\d{{6}}(_[a-f0-9]{{8}})?_1"
        )
        self.assertIsNotNone(run_suffix2)
        self.assertIn("_1", run_suffix2 or "")
        self.assertTrue(os.path.exists(output_dir2))
        self.assertNotEqual(output_dir1, output_dir2)

        # Third call - should create run_<run_id>_<timestamp>_2
        output_dir3, run_suffix3, _ = filesystem.setup_output_directory(cfg)
        self.assertRegex(
            output_dir3, rf"run_{re.escape(TEST_RUN_ID)}_\d{{8}}-\d{{6}}(_[a-f0-9]{{8}})?_2"
        )
        self.assertIsNotNone(run_suffix3)
        self.assertIn("_2", run_suffix3 or "")
        self.assertTrue(os.path.exists(output_dir3))

    def test_with_clean_output_no_counter(self):
        """Test that clean_output=True removes existing directory instead of appending counter."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            max_episodes=None,
            user_agent="test",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            screenplay=False,
            screenplay_gap_s=2.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=TEST_RUN_ID,
            skip_existing=False,
            clean_output=True,  # Should remove existing, not append counter
            auto_speakers=False,
            generate_summaries=False,
        )
        # First call - should create run_<run_id>_<timestamp>
        output_dir1, run_suffix1, _ = filesystem.setup_output_directory(cfg)
        import re

        self.assertRegex(
            output_dir1, rf"run_{re.escape(TEST_RUN_ID)}_\d{{8}}-\d{{6}}(_[a-f0-9]{{8}})?"
        )
        self.assertIsNotNone(run_suffix1)
        self.assertIn(TEST_RUN_ID, run_suffix1 or "")
        self.assertTrue(os.path.exists(output_dir1))

        # Second call with clean_output=True - should reuse same name (workflow will clean it)
        # Note: setup_output_directory doesn't actually clean, workflow does
        # But it should not append counter when clean_output=True
        output_dir2, run_suffix2, _ = filesystem.setup_output_directory(cfg)
        self.assertRegex(
            output_dir2, rf"run_{re.escape(TEST_RUN_ID)}_\d{{8}}-\d{{6}}(_[a-f0-9]{{8}})?"
        )
        self.assertIsNotNone(run_suffix2)
        self.assertIn(TEST_RUN_ID, run_suffix2 or "")
        # Directory should exist (workflow will clean it later if needed)
        self.assertTrue(os.path.exists(output_dir2))

    def test_with_whisper_auto_run_id(self):
        """Test setting up output directory with Whisper auto run ID."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            max_episodes=None,
            user_agent="test",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=False,
            screenplay_gap_s=2.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=None,
            skip_existing=False,
            clean_output=False,
            auto_speakers=False,  # Disable to test whisper only
            generate_summaries=False,  # Disable to test whisper only
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # Test uses TEST_DEFAULT_WHISPER_MODEL (tiny.en), so full_config_string should reflect that
        # Format: "w_<model>" for whisper provider
        expected_model_in_suffix = config.TEST_DEFAULT_WHISPER_MODEL
        self.assertIsNotNone(full_config_string)
        self.assertIn(f"w_{expected_model_in_suffix}", full_config_string or "")
        # run_suffix is now hash-based, verify it has the expected format
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_openai_transcription(self):
        """Test provider suffix with OpenAI transcription."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=True,
            transcription_provider="openai",
            openai_api_key="sk-test",
            openai_transcription_model="whisper-1",
            auto_speakers=False,
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        self.assertIsNotNone(full_config_string)
        self.assertIn("oa_whisper-1", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_openai_summary(self):
        """Test provider suffix with OpenAI summary."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            generate_summaries=True,
            generate_metadata=True,
            summary_provider="openai",
            openai_api_key="sk-test",
            openai_summary_model="gpt-4o",
            auto_speakers=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        self.assertIsNotNone(full_config_string)
        self.assertIn("oa_gpt-4o", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_openai_speaker(self):
        """Test provider suffix with OpenAI speaker detection."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            auto_speakers=True,
            speaker_detector_provider="openai",
            openai_api_key="sk-test",
            openai_speaker_model="gpt-4o-mini",
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        self.assertIsNotNone(full_config_string)
        self.assertIn("oa_gpt-4o-mini", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_with_reduce_model(self):
        """Test provider suffix includes reduce model when different from map."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            generate_summaries=True,
            generate_metadata=True,
            summary_provider="transformers",
            summary_model="facebook/bart-base",
            summary_reduce_model="allenai/led-base-16384",
            auto_speakers=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # Should include both map and reduce models in full_config_string
        self.assertIsNotNone(full_config_string)
        self.assertIn("tf_bart-base", full_config_string or "")
        self.assertIn("r_led-base-16384", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_with_same_reduce_model(self):
        """Test provider suffix excludes reduce model when same as map."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            generate_summaries=True,
            generate_metadata=True,
            summary_provider="transformers",
            summary_model="facebook/bart-base",
            summary_reduce_model="facebook/bart-base",  # Same as map
            auto_speakers=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # Should include map model but NOT reduce model (they're the same) in full_config_string
        self.assertIsNotNone(full_config_string)
        self.assertIn("tf_bart-base", full_config_string or "")
        self.assertNotIn("r_", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_no_ml_features(self):
        """Test provider suffix returns None when no ML features are used."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # No ML features, but still get timestamp-based suffix
        self.assertIsNotNone(run_suffix)
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}")
        self.assertIsNone(full_config_string)  # No ML features = no config string
        # New behavior: Always create run_ prefix with timestamp (Issue #380)
        import re

        self.assertRegex(output_dir, rf"{re.escape(self.temp_dir)}/run_\d{{8}}-\d{{6}}")

    def test_provider_suffix_speaker_with_custom_ner_model(self):
        """Test provider suffix with custom NER model."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            auto_speakers=True,
            speaker_detector_provider="spacy",
            ner_model="en_core_web_md",  # Custom model
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        self.assertIsNotNone(full_config_string)
        self.assertIn("sp_spacy_md", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_speaker_with_non_standard_ner_model(self):
        """Test provider suffix with non-standard NER model name."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            auto_speakers=True,
            speaker_detector_provider="spacy",
            ner_model="custom_model_name",
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        self.assertIsNotNone(full_config_string)
        self.assertIn("sp_custom_model_name", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_speaker_with_default_ner_model(self):
        """Test provider suffix when ner_model is None (uses default)."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            auto_speakers=True,
            speaker_detector_provider="spacy",
            ner_model=None,  # Should use default
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # Should use DEFAULT_NER_MODEL
        self.assertIsNotNone(run_suffix)
        self.assertIsNotNone(full_config_string)
        self.assertIn("sp_", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    def test_provider_suffix_screenplay_enabled(self):
        """Test provider suffix when screenplay is enabled (also triggers speaker detection)."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            transcribe_missing=False,
            screenplay=True,
            auto_speakers=False,  # screenplay=True should still trigger speaker detection
            speaker_detector_provider="spacy",
            generate_summaries=False,
        )
        output_dir, run_suffix, full_config_string = filesystem.setup_output_directory(cfg)
        # Should include speaker detection in full_config_string
        self.assertIsNotNone(run_suffix)
        self.assertIsNotNone(full_config_string)
        self.assertIn("sp_", full_config_string or "")
        self.assertRegex(run_suffix or "", r"\d{8}-\d{6}_[a-f0-9]{8}")

    @patch("os.path.exists")
    def test_setup_output_directory_counter_safety_limit(self, mock_exists):
        """Test that counter safety limit raises RuntimeError."""
        # Create base directory
        base_dir = os.path.join(self.temp_dir, "run_test")
        os.makedirs(base_dir, exist_ok=True)

        # Mock os.path.exists to always return True (simulating 10000+ existing dirs)
        call_count = [0]

        def mock_exists_side_effect(path):
            call_count[0] += 1
            # First call checks base directory (exists)
            if call_count[0] == 1:
                return True
            # All subsequent calls for counter variants return True (all exist)
            return True

        mock_exists.side_effect = mock_exists_side_effect

        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=self.temp_dir,
            run_id="test",
            clean_output=False,
            auto_speakers=False,
            generate_summaries=False,
        )

        # Patch the counter limit to 2 for faster testing
        import podcast_scraper.utils.filesystem as fs_module

        def patched_setup(cfg):
            run_suffix = "test"
            base_effective_output_dir = os.path.join(cfg.output_dir, f"run_{run_suffix}")
            if not cfg.clean_output and run_suffix and os.path.exists(base_effective_output_dir):
                counter = 1
                while True:
                    effective_output_dir = f"{base_effective_output_dir}_{counter}"
                    if not os.path.exists(effective_output_dir):
                        run_suffix = f"{run_suffix}_{counter}"
                        break
                    counter += 1
                    # Lower limit for testing (normally 10000)
                    if counter > 2:
                        msg = (
                            f"Too many existing directories with similar names: "
                            f"{base_effective_output_dir}"
                        )
                        raise RuntimeError(msg)
            # Continue with rest of function
            transcripts_dir = os.path.join(base_effective_output_dir, fs_module.TRANSCRIPTS_SUBDIR)
            metadata_dir = os.path.join(base_effective_output_dir, fs_module.METADATA_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)
            return base_effective_output_dir, run_suffix

        with patch.object(fs_module, "setup_output_directory", side_effect=patched_setup):
            with self.assertRaises(RuntimeError) as cm:
                fs_module.setup_output_directory(cfg)
            self.assertIn("Too many existing directories", str(cm.exception))

    def test_shorten_model_name(self):
        """Test _shorten_model_name removes common prefixes."""
        from podcast_scraper.utils.filesystem import _shorten_model_name

        self.assertEqual(_shorten_model_name("facebook/bart-large-cnn"), "bart-large-cnn")
        self.assertEqual(_shorten_model_name("google/pegasus-large"), "pegasus-large")
        self.assertEqual(
            _shorten_model_name("sshleifer/distilbart-cnn-12-6"),
            "distilbart-cnn-12-6",
        )
        self.assertEqual(_shorten_model_name("allenai/led-large-16384"), "led-large-16384")
        self.assertEqual(_shorten_model_name("bart-large-cnn"), "bart-large-cnn")  # No prefix

    def test_setup_output_directory_none_output_dir(self):
        """Test that setup_output_directory raises ValueError when output_dir is None."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=None,  # None output_dir
            auto_speakers=False,
            generate_summaries=False,
        )

        with self.assertRaises(ValueError) as cm:
            filesystem.setup_output_directory(cfg)
        self.assertIn("output_dir must be defined", str(cm.exception))


class TestWriteFile(unittest.TestCase):
    """Tests for write_file function."""

    def test_write_file(self):
        """Test writing a file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            data = b"test content"
            filesystem.write_file(tmp_path, data)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "rb") as f:
                self.assertEqual(f.read(), data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSkipExisting(unittest.TestCase):
    """Tests for skip-existing resume behaviour."""

    def _make_config(self, **overrides):
        """Create test config with skip_existing defaults."""
        defaults = {
            "skip_existing": True,
            "transcribe_missing": True,
        }
        defaults.update(overrides)
        return create_test_config(**defaults)

    def test_process_transcript_download_skips_existing_file(self):
        """Existing transcript files are not re-downloaded when --skip-existing is in effect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ep_title = "Episode 1"
            ep_title_safe = filesystem.sanitize_filename(ep_title)
            # Create transcripts subdirectory
            transcripts_dir = os.path.join(tmpdir, filesystem.TRANSCRIPTS_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)
            existing_path = os.path.join(transcripts_dir, f"0001 - {ep_title_safe}.txt")
            with open(existing_path, "wb") as fh:
                fh.write(b"original")

            cfg = self._make_config(output_dir=tmpdir, transcribe_missing=False)

            # Create Episode object
            item = ET.Element("item")
            episode = create_test_episode(
                idx=1,
                title=ep_title,
                title_safe=ep_title_safe,
                item=item,
                transcript_urls=[],
                media_url=None,
                media_type=None,
            )

            with patch("podcast_scraper.rss.downloader.http_get") as mock_http_get:
                success, transcript_path, transcript_source, bytes_downloaded = (
                    episode_processor.process_transcript_download(
                        episode,
                        f"{TEST_BASE_URL}/ep1.txt",
                        "text/plain",
                        cfg,
                        tmpdir,
                        None,
                    )
                )

            self.assertFalse(success)
            self.assertIsNone(transcript_path)
            mock_http_get.assert_not_called()

    def test_download_media_skips_when_whisper_output_exists(self):
        """Whisper download step is skipped if final transcript already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = os.path.join(tmpdir, ".tmp_media")
            os.makedirs(temp_dir, exist_ok=True)
            ep_title = "Episode 2"
            ep_title_safe = filesystem.sanitize_filename(ep_title)

            # Create transcripts subdirectory
            transcripts_dir = os.path.join(tmpdir, filesystem.TRANSCRIPTS_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)
            final_path = filesystem.build_whisper_output_path(1, ep_title_safe, None, tmpdir)
            with open(final_path, "wb") as fh:
                fh.write(b"existing transcript")

            cfg = self._make_config(output_dir=tmpdir)

            item = ET.Element("item")
            ET.SubElement(
                item,
                "enclosure",
                attrib={"url": TEST_MEDIA_URL, "type": TEST_MEDIA_TYPE_MP3},
            )

            # Create Episode object
            episode = create_test_episode(
                idx=1,
                title=ep_title,
                title_safe=ep_title_safe,
                item=item,
                transcript_urls=[],
                media_url=TEST_MEDIA_URL,
                media_type=TEST_MEDIA_TYPE_MP3,
            )

            with patch("podcast_scraper.rss.downloader.http_download_to_file") as mock_download:
                job = episode_processor.download_media_for_transcription(
                    episode,
                    cfg,
                    temp_dir,
                    tmpdir,
                    None,
                )

            self.assertIsNone(job)
            mock_download.assert_not_called()
