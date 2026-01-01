#!/usr/bin/env python3
"""Tests for episode processor utility functions.

These tests verify pure utility functions in episode_processor.py that
can be tested without I/O operations.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import create_test_config, create_test_episode  # noqa: E402

from podcast_scraper import episode_processor


class TestDeriveMediaExtension(unittest.TestCase):
    """Test derive_media_extension function."""

    def test_derive_from_media_type_mp3(self):
        """Test deriving .mp3 from audio/mpeg."""
        result = episode_processor.derive_media_extension("audio/mpeg", "http://example.com/audio")
        self.assertEqual(result, ".mp3")

    def test_derive_from_media_type_m4a(self):
        """Test deriving .m4a from audio/mp4."""
        result = episode_processor.derive_media_extension("audio/mp4", "http://example.com/audio")
        self.assertEqual(result, ".m4a")

    def test_derive_from_media_type_aac(self):
        """Test deriving .m4a from audio/aac."""
        result = episode_processor.derive_media_extension("audio/aac", "http://example.com/audio")
        self.assertEqual(result, ".m4a")

    def test_derive_from_media_type_ogg(self):
        """Test deriving .ogg from audio/ogg."""
        result = episode_processor.derive_media_extension("audio/ogg", "http://example.com/audio")
        self.assertEqual(result, ".ogg")

    def test_derive_from_media_type_oga(self):
        """Test deriving .ogg from audio/oga."""
        result = episode_processor.derive_media_extension("audio/oga", "http://example.com/audio")
        self.assertEqual(result, ".ogg")

    def test_derive_from_media_type_wav(self):
        """Test deriving .wav from audio/wav."""
        result = episode_processor.derive_media_extension("audio/wav", "http://example.com/audio")
        self.assertEqual(result, ".wav")

    def test_derive_from_media_type_webm(self):
        """Test deriving .webm from audio/webm."""
        result = episode_processor.derive_media_extension("audio/webm", "http://example.com/audio")
        self.assertEqual(result, ".webm")

    def test_derive_from_url_mp3(self):
        """Test deriving .mp3 from URL ending."""
        result = episode_processor.derive_media_extension(None, "http://example.com/audio.mp3")
        self.assertEqual(result, ".mp3")

    def test_derive_from_url_m4a(self):
        """Test deriving .m4a from URL ending."""
        result = episode_processor.derive_media_extension(None, "http://example.com/audio.m4a")
        self.assertEqual(result, ".m4a")

    def test_derive_from_url_uppercase(self):
        """Test that URL case is handled correctly."""
        result = episode_processor.derive_media_extension(None, "http://example.com/AUDIO.MP3")
        self.assertEqual(result, ".mp3")

    def test_derive_from_url_with_query(self):
        """Test that URL query parameters are handled."""
        # The function checks if URL ends with extension, query params prevent match
        result = episode_processor.derive_media_extension(
            None, "http://example.com/audio.mp3?param=value"
        )
        # Query params prevent ending match, so falls back to default
        self.assertEqual(result, ".bin")

    def test_default_extension_when_unknown(self):
        """Test that default .bin extension is returned for unknown types."""
        result = episode_processor.derive_media_extension(None, "http://example.com/audio")
        self.assertEqual(result, ".bin")

    def test_media_type_takes_precedence_over_url(self):
        """Test that media type takes precedence over URL."""
        result = episode_processor.derive_media_extension(
            "audio/mpeg", "http://example.com/audio.m4a"
        )
        self.assertEqual(result, ".mp3")  # Media type wins

    def test_unknown_media_type_falls_back_to_url(self):
        """Test that unknown media type falls back to URL."""
        result = episode_processor.derive_media_extension(
            "audio/unknown", "http://example.com/audio.mp3"
        )
        self.assertEqual(result, ".mp3")  # Falls back to URL

    def test_empty_media_type_and_url(self):
        """Test handling of empty media type and URL without extension."""
        result = episode_processor.derive_media_extension(None, "http://example.com/audio")
        self.assertEqual(result, ".bin")  # Default

    def test_media_type_without_slash(self):
        """Test handling of media type without slash."""
        result = episode_processor.derive_media_extension("invalid", "http://example.com/audio.mp3")
        self.assertEqual(result, ".mp3")  # Falls back to URL


class TestDeriveTranscriptExtension(unittest.TestCase):
    """Test derive_transcript_extension function."""

    def test_derive_from_transcript_type_vtt(self):
        """Test deriving .vtt from transcript_type."""
        result = episode_processor.derive_transcript_extension(
            "text/vtt", None, "http://example.com/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_transcript_type_srt(self):
        """Test deriving .srt from transcript_type."""
        result = episode_processor.derive_transcript_extension(
            "text/srt", None, "http://example.com/transcript"
        )
        self.assertEqual(result, ".srt")

    def test_derive_from_content_type_vtt(self):
        """Test deriving .vtt from content_type."""
        result = episode_processor.derive_transcript_extension(
            None, "text/vtt", "http://example.com/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_content_type_srt(self):
        """Test deriving .srt from content_type."""
        result = episode_processor.derive_transcript_extension(
            None, "text/srt", "http://example.com/transcript"
        )
        self.assertEqual(result, ".srt")

    def test_derive_from_url_vtt(self):
        """Test deriving .vtt from URL."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript.vtt"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_url_srt(self):
        """Test deriving .srt from URL."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript.srt"
        )
        self.assertEqual(result, ".srt")

    def test_derive_from_url_json(self):
        """Test deriving .json from URL."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript.json"
        )
        self.assertEqual(result, ".json")

    def test_derive_from_url_html(self):
        """Test deriving .html from URL."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript.html"
        )
        self.assertEqual(result, ".html")

    def test_transcript_type_takes_precedence(self):
        """Test that transcript_type takes precedence over content_type and URL."""
        result = episode_processor.derive_transcript_extension(
            "text/vtt", "text/srt", "http://example.com/transcript.json"
        )
        self.assertEqual(result, ".vtt")  # transcript_type wins

    def test_content_type_takes_precedence_over_url(self):
        """Test that content_type takes precedence over URL."""
        result = episode_processor.derive_transcript_extension(
            None, "text/srt", "http://example.com/transcript.vtt"
        )
        self.assertEqual(result, ".srt")  # content_type wins

    def test_url_with_query_parameters(self):
        """Test that URL query parameters don't interfere."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript.vtt?param=value"
        )
        self.assertEqual(result, ".vtt")

    def test_url_with_fragment(self):
        """Test that URL fragment doesn't interfere."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript.vtt#fragment"
        )
        self.assertEqual(result, ".vtt")

    def test_url_with_path_and_filename(self):
        """Test extracting extension from filename in URL path."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/path/to/transcript.srt"
        )
        self.assertEqual(result, ".srt")

    def test_content_type_with_subtype(self):
        """Test content_type with subtype like text/vtt+charset."""
        result = episode_processor.derive_transcript_extension(
            None, "text/vtt; charset=utf-8", "http://example.com/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_content_type_with_plus_subtype(self):
        """Test content_type with plus subtype like application/vtt+json."""
        # The function checks if subtype ends with +token, so vtt+json matches +json
        result = episode_processor.derive_transcript_extension(
            None, "application/vtt+json", "http://example.com/transcript"
        )
        # Matches json token because vtt+json ends with +json
        self.assertEqual(result, ".json")

    def test_default_extension_when_all_none(self):
        """Test that default .txt extension is returned when all inputs are None."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/transcript"
        )
        self.assertEqual(result, ".txt")

    def test_default_extension_when_unknown(self):
        """Test that default .txt extension is returned for unknown types."""
        result = episode_processor.derive_transcript_extension(
            "text/plain", "text/plain", "http://example.com/transcript"
        )
        self.assertEqual(result, ".txt")

    def test_url_case_insensitive(self):
        """Test that URL extension matching is case-insensitive."""
        result = episode_processor.derive_transcript_extension(
            None, None, "http://example.com/TRANSCRIPT.VTT"
        )
        self.assertEqual(result, ".vtt")

    def test_absolute_path_url(self):
        """Test handling of absolute path URLs."""
        result = episode_processor.derive_transcript_extension(
            None, None, "/path/to/transcript.srt"
        )
        self.assertEqual(result, ".srt")

    def test_relative_path_url(self):
        """Test handling of relative path URLs."""
        result = episode_processor.derive_transcript_extension(
            None, None, "../path/to/transcript.vtt"
        )
        self.assertEqual(result, ".vtt")


class TestDetermineOutputPath(unittest.TestCase):
    """Tests for _determine_output_path function."""

    def test_determine_output_path_basic(self):
        """Test basic output path determination."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        result = episode_processor._determine_output_path(
            episode=episode,
            transcript_url="http://example.com/transcript.vtt",
            transcript_type="text/vtt",
            effective_output_dir="/output",
            run_suffix=None,
            planned_ext=".vtt",
        )

        self.assertEqual(result, "/output/0001 - Episode_Title.vtt")

    def test_determine_output_path_with_run_suffix(self):
        """Test output path with run suffix."""
        episode = create_test_episode(idx=5, title_safe="Test_Episode")
        result = episode_processor._determine_output_path(
            episode=episode,
            transcript_url="http://example.com/transcript.srt",
            transcript_type="text/srt",
            effective_output_dir="/output",
            run_suffix="run1",
            planned_ext=".srt",
        )

        self.assertEqual(result, "/output/0005 - Test_Episode_run1.srt")

    def test_determine_output_path_single_digit_idx(self):
        """Test output path with single-digit episode index."""
        episode = create_test_episode(idx=3, title_safe="Short")
        result = episode_processor._determine_output_path(
            episode=episode,
            transcript_url="http://example.com/transcript.txt",
            transcript_type=None,
            effective_output_dir="./output",
            run_suffix=None,
            planned_ext=".txt",
        )

        # Should pad with zeros (EPISODE_NUMBER_FORMAT_WIDTH = 4)
        self.assertEqual(result, os.path.join(".", "output", "0003 - Short.txt"))

    def test_determine_output_path_double_digit_idx(self):
        """Test output path with double-digit episode index."""
        episode = create_test_episode(idx=42, title_safe="Episode_42")
        result = episode_processor._determine_output_path(
            episode=episode,
            transcript_url="http://example.com/transcript.vtt",
            transcript_type="text/vtt",
            effective_output_dir="/data/output",
            run_suffix="test",
            planned_ext=".vtt",
        )

        self.assertEqual(result, "/data/output/0042 - Episode_42_test.vtt")


class TestCheckExistingTranscript(unittest.TestCase):
    """Tests for _check_existing_transcript function."""

    def test_check_existing_transcript_skip_disabled(self):
        """Test that skip check returns False when skip_existing is False."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg_no_skip = create_test_config(skip_existing=False, output_dir="/output")
        result = episode_processor._check_existing_transcript(
            episode=episode,
            effective_output_dir="/output",
            run_suffix=None,
            cfg=cfg_no_skip,
        )

        self.assertFalse(result)

    @patch("podcast_scraper.episode_processor.Path")
    def test_check_existing_transcript_not_found(self, mock_path):
        """Test that skip check returns False when transcript doesn't exist."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg_skip = create_test_config(skip_existing=True, output_dir="/output")

        # Mock Path to return empty glob results
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = []
        mock_path.return_value = mock_path_instance

        result = episode_processor._check_existing_transcript(
            episode=episode,
            effective_output_dir="/output",
            run_suffix=None,
            cfg=cfg_skip,
        )

        self.assertFalse(result)

    @patch("podcast_scraper.episode_processor.Path")
    def test_check_existing_transcript_found(self, mock_path):
        """Test that skip check returns True when transcript exists."""
        episode = create_test_episode(idx=1, title_safe="Episode_Title")
        cfg_skip = create_test_config(skip_existing=True, output_dir="/output")

        # Mock Path to return a file match
        mock_file = Mock()
        mock_file.is_file.return_value = True
        mock_file.__str__ = lambda x: "/output/0001 - Episode_Title.vtt"
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = [mock_file]
        mock_path.return_value = mock_path_instance

        result = episode_processor._check_existing_transcript(
            episode=episode,
            effective_output_dir="/output",
            run_suffix=None,
            cfg=cfg_skip,
        )

        self.assertTrue(result)
        mock_file.is_file.assert_called_once()

    @patch("podcast_scraper.episode_processor.Path")
    def test_check_existing_transcript_with_run_suffix(self, mock_path):
        """Test that skip check works with run suffix."""
        episode = create_test_episode(idx=2, title_safe="Test_Episode")
        cfg_skip = create_test_config(skip_existing=True, output_dir="/output")

        # Mock Path to return a file match with run suffix
        mock_file = Mock()
        mock_file.is_file.return_value = True
        mock_file.__str__ = lambda x: "/output/0002 - Test_Episode_run1.srt"
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = [mock_file]
        mock_path.return_value = mock_path_instance

        result = episode_processor._check_existing_transcript(
            episode=episode,
            effective_output_dir="/output",
            run_suffix="run1",
            cfg=cfg_skip,
        )

        self.assertTrue(result)

    @patch("podcast_scraper.episode_processor.Path")
    def test_check_existing_transcript_ignores_directories(self, mock_path):
        """Test that skip check ignores directories, only checks files."""
        episode = create_test_episode(idx=3, title_safe="Episode_3")
        cfg_skip = create_test_config(skip_existing=True, output_dir="/output")

        # Mock Path to return a directory (not a file)
        mock_dir = Mock()
        mock_dir.is_file.return_value = False
        mock_dir.__str__ = lambda x: "/output/0003 - Episode_3.vtt"
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = [mock_dir]
        mock_path.return_value = mock_path_instance

        result = episode_processor._check_existing_transcript(
            episode=episode,
            effective_output_dir="/output",
            run_suffix=None,
            cfg=cfg_skip,
        )

        # Should return False because directory is not a file
        self.assertFalse(result)

    @patch("podcast_scraper.episode_processor.Path")
    def test_check_existing_transcript_dry_run(self, mock_path):
        """Test that skip check works with dry_run enabled."""
        cfg_dry_run = create_test_config(skip_existing=True, dry_run=True, output_dir="/output")
        episode = create_test_episode(idx=4, title_safe="Dry_Run_Episode")

        # Mock Path to return a file match
        mock_file = Mock()
        mock_file.is_file.return_value = True
        mock_file.__str__ = lambda x: "/output/0004 - Dry_Run_Episode.txt"
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = [mock_file]
        mock_path.return_value = mock_path_instance

        result = episode_processor._check_existing_transcript(
            episode=episode,
            effective_output_dir="/output",
            run_suffix=None,
            cfg=cfg_dry_run,
        )

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
