#!/usr/bin/env python3
"""Tests for filesystem operations."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util

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
from podcast_scraper import config, episode_processor, filesystem

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

# Import helper functions from parent conftest
build_rss_xml_with_media = parent_conftest.build_rss_xml_with_media
build_rss_xml_with_speakers = parent_conftest.build_rss_xml_with_speakers
build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_media_response = parent_conftest.create_media_response
create_mock_spacy_model = parent_conftest.create_mock_spacy_model
create_rss_response = parent_conftest.create_rss_response
create_test_args = parent_conftest.create_test_args
create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
create_transcript_response = parent_conftest.create_transcript_response
MockHTTPResponse = parent_conftest.MockHTTPResponse
TEST_BASE_URL = parent_conftest.TEST_BASE_URL
TEST_CONTENT_TYPE_SRT = parent_conftest.TEST_CONTENT_TYPE_SRT
TEST_CONTENT_TYPE_VTT = parent_conftest.TEST_CONTENT_TYPE_VTT
TEST_CUSTOM_OUTPUT_DIR = parent_conftest.TEST_CUSTOM_OUTPUT_DIR
TEST_EPISODE_TITLE = parent_conftest.TEST_EPISODE_TITLE
TEST_EPISODE_TITLE_SPECIAL = parent_conftest.TEST_EPISODE_TITLE_SPECIAL
TEST_FEED_TITLE = parent_conftest.TEST_FEED_TITLE
TEST_FEED_URL = parent_conftest.TEST_FEED_URL
TEST_FULL_URL = parent_conftest.TEST_FULL_URL
TEST_MEDIA_TYPE_M4A = parent_conftest.TEST_MEDIA_TYPE_M4A
TEST_MEDIA_TYPE_MP3 = parent_conftest.TEST_MEDIA_TYPE_MP3
TEST_MEDIA_URL = parent_conftest.TEST_MEDIA_URL
TEST_OUTPUT_DIR = parent_conftest.TEST_OUTPUT_DIR
TEST_PATH = parent_conftest.TEST_PATH
TEST_RELATIVE_MEDIA = parent_conftest.TEST_RELATIVE_MEDIA
TEST_RELATIVE_TRANSCRIPT = parent_conftest.TEST_RELATIVE_TRANSCRIPT
TEST_RUN_ID = parent_conftest.TEST_RUN_ID
TEST_TRANSCRIPT_TYPE_SRT = parent_conftest.TEST_TRANSCRIPT_TYPE_SRT
TEST_TRANSCRIPT_TYPE_VTT = parent_conftest.TEST_TRANSCRIPT_TYPE_VTT
TEST_TRANSCRIPT_URL = parent_conftest.TEST_TRANSCRIPT_URL
TEST_TRANSCRIPT_URL_SRT = parent_conftest.TEST_TRANSCRIPT_URL_SRT


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
        self.assertIn("output_rss", result)
        self.assertIn("example.com", result)
        self.assertEqual(result, "output_rss_example.com_6904f1c4")

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

    def test_with_run_id(self):
        """Test setting up output directory with run ID."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=TEST_OUTPUT_DIR,
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
        )
        output_dir, run_suffix = filesystem.setup_output_directory(cfg)
        self.assertIn(f"run_{TEST_RUN_ID}", output_dir)
        self.assertEqual(run_suffix, TEST_RUN_ID)

    def test_with_whisper_auto_run_id(self):
        """Test setting up output directory with Whisper auto run ID."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=TEST_OUTPUT_DIR,
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
        )
        output_dir, run_suffix = filesystem.setup_output_directory(cfg)
        # Test uses TEST_DEFAULT_WHISPER_MODEL (tiny.en), so run_suffix should reflect that
        expected_model_in_suffix = config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", "")
        self.assertIn(f"whisper_{expected_model_in_suffix}", run_suffix or "")


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

            with patch("podcast_scraper.downloader.http_get") as mock_http_get:
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

            with patch("podcast_scraper.downloader.http_download_to_file") as mock_download:
                job = episode_processor.download_media_for_transcription(
                    episode,
                    cfg,
                    temp_dir,
                    tmpdir,
                    None,
                )

            self.assertIsNone(job)
            mock_download.assert_not_called()
