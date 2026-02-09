#!/usr/bin/env python3
"""Tests for episode processor utility functions.

These tests verify pure utility functions in episode_processor.py that
can be tested without I/O operations.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

# Mock openai before importing modules that require it
# Unit tests run without openai package installed
# (patch already imported above)

mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper.workflow import episode_processor

# Import directly from tests.conftest (works with pytest-xdist)
from tests.conftest import create_test_config, create_test_episode  # noqa: E402


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

        self.assertEqual(result, "/output/transcripts/0001 - Episode_Title.vtt")

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

        self.assertEqual(result, "/output/transcripts/0005 - Test_Episode_run1.srt")

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

        # Should pad with zeros (EPISODE_NUMBER_FORMAT_WIDTH = 4) and use transcripts/ subdirectory
        self.assertEqual(result, os.path.join(".", "output", "transcripts", "0003 - Short.txt"))

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

        self.assertEqual(result, "/data/output/transcripts/0042 - Episode_42_test.vtt")


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

    @patch("podcast_scraper.workflow.episode_processor.Path")
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

    @patch("podcast_scraper.workflow.episode_processor.Path")
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

    @patch("podcast_scraper.workflow.episode_processor.Path")
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

    @patch("podcast_scraper.workflow.episode_processor.Path")
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

    @patch("podcast_scraper.workflow.episode_processor.Path")
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


class TestProcessEpisodeDownload(unittest.TestCase):
    """Tests for process_episode_download function."""

    @patch("podcast_scraper.workflow.episode_processor.choose_transcript_url")
    @patch("podcast_scraper.workflow.episode_processor.process_transcript_download")
    def test_process_episode_download_with_transcript(self, mock_process_transcript, mock_choose):
        """Test process_episode_download when transcript URL is available."""
        episode = create_test_episode(
            idx=1, transcript_urls=[("https://example.com/transcript.vtt", "vtt")]
        )
        import queue

        cfg = create_test_config(transcribe_missing=False)
        transcription_jobs = queue.Queue()
        mock_choose.return_value = ("https://example.com/transcript.vtt", "vtt")
        mock_process_transcript.return_value = (True, "transcript.vtt", "direct_download", 1000)

        success, transcript_path, transcript_source, bytes_downloaded = (
            episode_processor.process_episode_download(
                episode=episode,
                cfg=cfg,
                temp_dir="/tmp",
                effective_output_dir="/output",
                run_suffix=None,
                transcription_jobs=transcription_jobs,
                transcription_jobs_lock=None,
            )
        )

        self.assertTrue(success)
        self.assertEqual(transcript_path, "transcript.vtt")
        self.assertEqual(transcript_source, "direct_download")
        self.assertEqual(bytes_downloaded, 1000)
        mock_choose.assert_called_once_with(episode.transcript_urls, cfg.prefer_types)
        mock_process_transcript.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.choose_transcript_url")
    @patch("podcast_scraper.workflow.episode_processor.process_transcript_download")
    def test_process_episode_download_with_delay(self, mock_process_transcript, mock_choose):
        """Test process_episode_download applies delay when configured."""
        episode = create_test_episode(
            idx=1, transcript_urls=[("https://example.com/transcript.vtt", "vtt")]
        )
        import queue

        cfg = create_test_config(transcribe_missing=False, delay_ms=100)
        transcription_jobs = queue.Queue()
        mock_choose.return_value = ("https://example.com/transcript.vtt", "vtt")
        mock_process_transcript.return_value = (True, "transcript.vtt", "direct_download", 1000)

        with patch("time.sleep") as mock_sleep:
            success, transcript_path, transcript_source, bytes_downloaded = (
                episode_processor.process_episode_download(
                    episode=episode,
                    cfg=cfg,
                    temp_dir="/tmp",
                    effective_output_dir="/output",
                    run_suffix=None,
                    transcription_jobs=transcription_jobs,
                    transcription_jobs_lock=None,
                )
            )

            self.assertTrue(success)
            mock_sleep.assert_called_once_with(0.1)  # 100ms / 1000

    @patch("podcast_scraper.workflow.episode_processor.choose_transcript_url")
    @patch("podcast_scraper.workflow.episode_processor.download_media_for_transcription")
    def test_process_episode_download_with_transcription(self, mock_download_media, mock_choose):
        """Test process_episode_download when transcription is needed."""
        import queue

        episode = create_test_episode(
            idx=1, transcript_urls=[], media_url="https://example.com/ep1.mp3"
        )
        cfg = create_test_config(transcribe_missing=True)
        transcription_jobs = queue.Queue()
        mock_choose.return_value = None  # No transcript URL
        mock_job = Mock()
        mock_job.idx = 1
        mock_download_media.return_value = mock_job

        success, transcript_path, transcript_source, bytes_downloaded = (
            episode_processor.process_episode_download(
                episode=episode,
                cfg=cfg,
                temp_dir="/tmp",
                effective_output_dir="/output",
                run_suffix=None,
                transcription_jobs=transcription_jobs,
                transcription_jobs_lock=None,
            )
        )

        self.assertFalse(success)
        self.assertIsNone(transcript_path)
        self.assertIsNone(transcript_source)
        self.assertEqual(bytes_downloaded, 0)
        self.assertEqual(transcription_jobs.qsize(), 1)
        self.assertEqual(transcription_jobs.get(), mock_job)
        mock_download_media.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.choose_transcript_url")
    @patch("podcast_scraper.workflow.episode_processor.download_media_for_transcription")
    def test_process_episode_download_with_transcription_lock(
        self, mock_download_media, mock_choose
    ):
        """Test process_episode_download uses lock when provided."""
        import queue

        episode = create_test_episode(
            idx=1, transcript_urls=[], media_url="https://example.com/ep1.mp3"
        )
        cfg = create_test_config(transcribe_missing=True)
        transcription_jobs = queue.Queue()
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(return_value=None)
        mock_lock.__exit__ = Mock(return_value=None)
        mock_choose.return_value = None
        mock_job = Mock()
        mock_download_media.return_value = mock_job

        success, transcript_path, transcript_source, bytes_downloaded = (
            episode_processor.process_episode_download(
                episode=episode,
                cfg=cfg,
                temp_dir="/tmp",
                effective_output_dir="/output",
                run_suffix=None,
                transcription_jobs=transcription_jobs,
                transcription_jobs_lock=mock_lock,
            )
        )

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
        self.assertEqual(transcription_jobs.qsize(), 1)

    @patch("podcast_scraper.workflow.episode_processor.choose_transcript_url")
    @patch("podcast_scraper.workflow.episode_processor.download_media_for_transcription")
    def test_process_episode_download_no_transcript_no_transcription(
        self, mock_download_media, mock_choose
    ):
        """Test process_episode_download when no transcript and transcription disabled."""
        episode = create_test_episode(
            idx=1, transcript_urls=[], media_url="https://example.com/ep1.mp3"
        )
        import queue

        cfg = create_test_config(transcribe_missing=False)
        transcription_jobs = queue.Queue()
        mock_choose.return_value = None

        success, transcript_path, transcript_source, bytes_downloaded = (
            episode_processor.process_episode_download(
                episode=episode,
                cfg=cfg,
                temp_dir="/tmp",
                effective_output_dir="/output",
                run_suffix=None,
                transcription_jobs=transcription_jobs,
                transcription_jobs_lock=None,
            )
        )

        self.assertFalse(success)
        self.assertIsNone(transcript_path)
        self.assertIsNone(transcript_source)
        self.assertEqual(bytes_downloaded, 0)
        self.assertEqual(transcription_jobs.qsize(), 0)
        mock_download_media.assert_not_called()

    @patch("podcast_scraper.workflow.episode_processor.choose_transcript_url")
    @patch("podcast_scraper.workflow.episode_processor.download_media_for_transcription")
    def test_process_episode_download_no_temp_dir(self, mock_download_media, mock_choose):
        """Test process_episode_download when temp_dir is None."""
        episode = create_test_episode(
            idx=1, transcript_urls=[], media_url="https://example.com/ep1.mp3"
        )
        import queue

        cfg = create_test_config(transcribe_missing=True)
        transcription_jobs = queue.Queue()
        mock_choose.return_value = None

        success, transcript_path, transcript_source, bytes_downloaded = (
            episode_processor.process_episode_download(
                episode=episode,
                cfg=cfg,
                temp_dir=None,
                effective_output_dir="/output",
                run_suffix=None,
                transcription_jobs=transcription_jobs,
                transcription_jobs_lock=None,
            )
        )

        self.assertFalse(success)
        mock_download_media.assert_not_called()


class TestTranscribeMediaToText(unittest.TestCase):
    """Tests for transcribe_media_to_text function."""

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    def test_transcribe_media_to_text_dry_run(self, mock_build_path):
        """Test transcribe_media_to_text in dry-run mode."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None
        cfg = create_test_config(dry_run=True)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=None,
            pipeline_metrics=None,
        )

        self.assertTrue(success)
        self.assertIsNotNone(transcript_path)
        self.assertEqual(bytes_downloaded, 0)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.relpath")
    def test_transcribe_media_to_text_reuse_existing(
        self, mock_relpath, mock_exists, mock_build_path
    ):
        """Test transcribe_media_to_text reuses existing transcript."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = ""  # Empty means reusing existing
        job.detected_speaker_names = None
        cfg = create_test_config(skip_existing=True)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = True
        mock_relpath.return_value = "0001 - Episode_1.txt"

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=None,
            pipeline_metrics=None,
        )

        self.assertTrue(success)
        self.assertEqual(transcript_path, "0001 - Episode_1.txt")
        self.assertEqual(bytes_downloaded, 0)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._format_transcript_if_needed")
    @patch("podcast_scraper.workflow.episode_processor._save_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_success(
        self,
        mock_cleanup,
        mock_save,
        mock_format,
        mock_getsize,
        mock_exists,
        mock_build_path,
    ):
        """Test successful transcription."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = ["Host", "Guest"]
        cfg = create_test_config(dry_run=False, preprocessing_enabled=False)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = True  # File exists for size check
        mock_getsize.return_value = 5000000  # 5MB

        mock_provider = Mock()
        mock_provider.transcribe_with_segments.return_value = (
            {"text": "Hello world", "segments": []},
            10.5,
        )
        mock_format.return_value = "Hello world"
        mock_save.return_value = "0001 - Episode_1.txt"

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=mock_provider,
            pipeline_metrics=None,
        )

        self.assertTrue(success)
        self.assertEqual(transcript_path, "0001 - Episode_1.txt")
        self.assertEqual(bytes_downloaded, 5000000)
        # Function now always passes pipeline_metrics, episode_duration_seconds, and call_metrics
        # Check that it was called with the audio path and language
        mock_provider.transcribe_with_segments.assert_called_once()
        call_args = mock_provider.transcribe_with_segments.call_args
        self.assertEqual(call_args[0][0], "/tmp/ep1.mp3")
        self.assertEqual(call_args[1]["language"], cfg.language)
        mock_cleanup.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_no_provider(
        self, mock_cleanup, mock_getsize, mock_exists, mock_build_path
    ):
        """Test transcribe_media_to_text when provider is None."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None
        cfg = create_test_config(dry_run=False)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = False

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=None,
            pipeline_metrics=None,
        )

        self.assertFalse(success)
        self.assertIsNone(transcript_path)
        self.assertEqual(bytes_downloaded, 0)
        mock_cleanup.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_transcription_error(
        self, mock_cleanup, mock_getsize, mock_exists, mock_build_path
    ):
        """Test transcribe_media_to_text handles transcription errors."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None
        cfg = create_test_config(dry_run=False, preprocessing_enabled=False)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = True  # File exists for size check
        mock_getsize.return_value = 5000000

        mock_provider = Mock()
        mock_provider.transcribe_with_segments.side_effect = RuntimeError("Transcription failed")

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=mock_provider,
            pipeline_metrics=None,
        )

        self.assertFalse(success)
        self.assertIsNone(transcript_path)
        # bytes_downloaded is set before transcription, so it's the file size even on error
        self.assertEqual(bytes_downloaded, 5000000)
        mock_cleanup.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_provider_without_pipeline_metrics(
        self, mock_cleanup, mock_getsize, mock_exists, mock_build_path
    ):
        """Test transcribe_media_to_text with provider that doesn't support pipeline_metrics."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None
        cfg = create_test_config(dry_run=False, preprocessing_enabled=False)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = True
        mock_getsize.return_value = 5000000

        # Create a mock provider that doesn't support pipeline_metrics parameter
        mock_provider = Mock()
        # Create a signature without pipeline_metrics
        from inspect import Parameter, Signature

        sig = Signature(
            [
                Parameter("audio_path", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("language", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            ]
        )
        mock_provider.transcribe_with_segments.__signature__ = sig
        mock_provider.transcribe_with_segments.return_value = (
            {"text": "Hello world", "segments": []},
            10.5,
        )

        mock_format = Mock(return_value="Hello world")
        mock_save = Mock(return_value="0001 - Episode_1.txt")
        with patch(
            "podcast_scraper.workflow.episode_processor._format_transcript_if_needed", mock_format
        ):
            with patch(
                "podcast_scraper.workflow.episode_processor._save_transcript_file", mock_save
            ):
                success, transcript_path, bytes_downloaded = (
                    episode_processor.transcribe_media_to_text(
                        job=job,
                        cfg=cfg,
                        whisper_model=None,
                        run_suffix=None,
                        effective_output_dir="/output",
                        transcription_provider=mock_provider,
                        pipeline_metrics=None,
                    )
                )

        self.assertTrue(success)
        self.assertEqual(transcript_path, "0001 - Episode_1.txt")
        # Function now always passes pipeline_metrics, episode_duration_seconds, and call_metrics
        # Check that it was called with the audio path and language
        mock_provider.transcribe_with_segments.assert_called_once()
        call_args = mock_provider.transcribe_with_segments.call_args
        self.assertEqual(call_args[0][0], "/tmp/ep1.mp3")
        self.assertEqual(call_args[1]["language"], cfg.language)
        mock_cleanup.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._format_transcript_if_needed")
    @patch("podcast_scraper.workflow.episode_processor._save_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_with_metrics(
        self,
        mock_cleanup,
        mock_save,
        mock_format,
        mock_getsize,
        mock_exists,
        mock_build_path,
    ):
        """Test transcribe_media_to_text records metrics."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None
        cfg = create_test_config(dry_run=False)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = False
        mock_getsize.return_value = 5000000

        mock_provider = Mock()
        mock_provider.transcribe_with_segments.return_value = (
            {"text": "Hello world", "segments": []},
            10.5,
        )
        mock_format.return_value = "Hello world"
        mock_save.return_value = "0001 - Episode_1.txt"

        mock_metrics = Mock()
        mock_metrics.record_transcribe_time = Mock()

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=mock_provider,
            pipeline_metrics=mock_metrics,
        )

        self.assertTrue(success)
        mock_metrics.record_transcribe_time.assert_called_once_with(10.5)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._format_transcript_if_needed")
    @patch("podcast_scraper.workflow.episode_processor._save_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_file_size_error(
        self,
        mock_cleanup,
        mock_save,
        mock_format,
        mock_getsize,
        mock_exists,
        mock_build_path,
    ):
        """Test transcribe_media_to_text handles file size check errors."""
        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None
        cfg = create_test_config(dry_run=False)
        mock_build_path.return_value = "/output/0001 - Episode_1.txt"
        mock_exists.return_value = True
        mock_getsize.side_effect = OSError("Permission denied")

        mock_provider = Mock()
        mock_provider.transcribe_with_segments.return_value = (
            {"text": "Hello world", "segments": []},
            10.5,
        )
        mock_format.return_value = "Hello world"
        mock_save.return_value = "0001 - Episode_1.txt"

        success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir="/output",
            transcription_provider=mock_provider,
            pipeline_metrics=None,
        )

        # Should succeed but with 0 bytes downloaded (file size check failed)
        self.assertTrue(success)
        self.assertEqual(bytes_downloaded, 0)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_transcribe_media_to_text_oversized_file_skipped(
        self, mock_cleanup, mock_getsize, mock_exists, mock_build_path
    ):
        """Test that oversized audio files are skipped gracefully with warning."""
        import tempfile

        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.detected_speaker_names = None

        # Create a temporary audio file larger than 25 MB
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            # Write 26 MB of data
            f.write(b"0" * (26 * 1024 * 1024))
            temp_media = f.name
            job.temp_media = temp_media

        try:
            cfg = create_test_config(
                transcribe_missing=True, transcription_provider="openai", openai_api_key="sk-test"
            )
            mock_build_path.return_value = "/output/0001 - Episode_1.txt"
            mock_exists.return_value = True
            mock_getsize.return_value = 26 * 1024 * 1024  # 26 MB

            # Create OpenAI provider that will validate file size
            from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

            provider = OpenAIProvider(cfg)
            provider.initialize()

            # transcribe_media_to_text should catch ValueError and return False (skipped)
            success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
                job=job,
                cfg=cfg,
                whisper_model=None,
                run_suffix=None,
                effective_output_dir="/output",
                transcription_provider=provider,
            )

            # Should return False (episode skipped, not failed)
            self.assertFalse(success)
            self.assertIsNone(transcript_path)
            # Should still report bytes downloaded (file was downloaded before validation)
            self.assertGreater(bytes_downloaded, 0)
            mock_cleanup.assert_called_once()
        finally:
            if os.path.exists(temp_media):
                os.unlink(temp_media)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._format_transcript_if_needed")
    @patch("podcast_scraper.workflow.episode_processor._save_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    @patch("podcast_scraper.cache.transcript_cache.save_transcript_to_cache")
    @patch("podcast_scraper.cache.transcript_cache.get_audio_hash")
    def test_transcribe_media_to_text_cache_with_whisper_model_object(
        self,
        mock_get_hash,
        mock_save_cache,
        mock_cleanup,
        mock_save,
        mock_format,
        mock_getsize,
        mock_exists,
        mock_build_path,
    ):
        """Test that non-string model objects are converted to config model names for cache."""
        import tempfile

        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None

        # Create temp media file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio" * 1000)
            temp_media = f.name
            job.temp_media = temp_media

        try:
            cfg = create_test_config(
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model="base.en",
                transcript_cache_enabled=True,
            )
            mock_build_path.return_value = "/output/0001 - Episode_1.txt"
            mock_exists.return_value = True
            mock_getsize.return_value = 10000
            mock_get_hash.return_value = "test_hash_12345"

            # Create mock provider with non-string model (simulating MLProvider.model)
            mock_provider = Mock()
            mock_provider.transcribe_with_segments.return_value = (
                {"text": "Hello world", "segments": []},
                10.5,
            )
            # Simulate MLProvider.model returning a Whisper object (not a string)
            mock_model_object = Mock()
            mock_model_object.__class__.__name__ = "Whisper"
            mock_provider.model = mock_model_object  # Non-string model
            mock_provider.name = "ml"  # Set name attribute directly
            mock_format.return_value = "Hello world"
            mock_save.return_value = "0001 - Episode_1.txt"

            success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
                job=job,
                cfg=cfg,
                whisper_model=None,
                run_suffix=None,
                effective_output_dir="/output",
                transcription_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertTrue(success)
            # Verify cache was called with string model name from config, not the object
            mock_save_cache.assert_called_once()
            call_args = mock_save_cache.call_args
            # Check that model is the string from config, not the object
            self.assertEqual(call_args[1]["model"], "base.en")
            self.assertEqual(call_args[1]["provider_name"], "ml")
        finally:
            if os.path.exists(temp_media):
                os.unlink(temp_media)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("podcast_scraper.workflow.episode_processor._format_transcript_if_needed")
    @patch("podcast_scraper.workflow.episode_processor._save_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    @patch("podcast_scraper.cache.transcript_cache.save_transcript_to_cache")
    @patch("podcast_scraper.cache.transcript_cache.get_audio_hash")
    def test_transcribe_media_to_text_cache_with_openai_provider(
        self,
        mock_get_hash,
        mock_save_cache,
        mock_cleanup,
        mock_save,
        mock_format,
        mock_getsize,
        mock_exists,
        mock_build_path,
    ):
        """Test that OpenAI provider uses transcription_model attribute for cache."""
        import tempfile

        job = Mock()
        job.idx = 1
        job.ep_title_safe = "Episode_1"
        job.temp_media = "/tmp/ep1.mp3"
        job.detected_speaker_names = None

        # Create temp media file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio" * 1000)
            temp_media = f.name
            job.temp_media = temp_media

        try:
            cfg = create_test_config(
                transcribe_missing=True,
                transcription_provider="openai",
                openai_api_key="sk-test",
                openai_transcription_model="whisper-1",
                transcript_cache_enabled=True,
            )
            mock_build_path.return_value = "/output/0001 - Episode_1.txt"
            mock_exists.return_value = True
            mock_getsize.return_value = 10000
            mock_get_hash.return_value = "test_hash_12345"

            # Create mock OpenAI provider with transcription_model attribute
            mock_provider = Mock()
            mock_provider.transcribe_with_segments.return_value = (
                {"text": "Hello world", "segments": []},
                10.5,
            )
            mock_provider.transcription_model = "whisper-1"  # String attribute
            mock_provider.name = "openai"  # Set name attribute directly
            mock_format.return_value = "Hello world"
            mock_save.return_value = "0001 - Episode_1.txt"

            success, transcript_path, bytes_downloaded = episode_processor.transcribe_media_to_text(
                job=job,
                cfg=cfg,
                whisper_model=None,
                run_suffix=None,
                effective_output_dir="/output",
                transcription_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertTrue(success)
            # Verify cache was called with transcription_model from provider
            mock_save_cache.assert_called_once()
            call_args = mock_save_cache.call_args
            self.assertEqual(call_args[1]["model"], "whisper-1")
            self.assertEqual(call_args[1]["provider_name"], "openai")
        finally:
            if os.path.exists(temp_media):
                os.unlink(temp_media)


class TestDownloadMediaForTranscription(unittest.TestCase):
    """Tests for download_media_for_transcription function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.episode = create_test_episode(
            idx=1, title="Test Episode", media_url="https://example.com/episode.mp3"
        )
        self.temp_dir = "/tmp"
        self.effective_output_dir = "/output"

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_download_media_skip_existing(self, mock_exists, mock_build_path):
        """Test that download is skipped when transcript exists and skip_existing is True."""
        self.cfg = create_test_config(skip_existing=True)
        mock_exists.return_value = True
        mock_build_path.return_value = "/output/transcript.txt"

        result = episode_processor.download_media_for_transcription(
            episode=self.episode,
            cfg=self.cfg,
            temp_dir=self.temp_dir,
            effective_output_dir=self.effective_output_dir,
            run_suffix=None,
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("podcast_scraper.workflow.episode_processor.downloader.http_download_to_file")
    def test_download_media_success(self, mock_download, mock_exists, mock_build_path):
        """Test successful media download."""
        self.cfg = create_test_config(skip_existing=False, dry_run=False)
        mock_exists.return_value = False
        mock_build_path.return_value = "/output/transcript.txt"
        mock_download.return_value = (True, 1000000)  # 1MB

        result = episode_processor.download_media_for_transcription(
            episode=self.episode,
            cfg=self.cfg,
            temp_dir=self.temp_dir,
            effective_output_dir=self.effective_output_dir,
            run_suffix=None,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.idx, 1)
        mock_download.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_download_media_dry_run(self, mock_exists, mock_build_path):
        """Test media download in dry-run mode."""
        self.cfg = create_test_config(skip_existing=False, dry_run=True)
        mock_exists.return_value = False
        mock_build_path.return_value = "/output/transcript.txt"

        result = episode_processor.download_media_for_transcription(
            episode=self.episode,
            cfg=self.cfg,
            temp_dir=self.temp_dir,
            effective_output_dir=self.effective_output_dir,
            run_suffix=None,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.idx, 1)
        self.assertEqual(result.temp_media, "")  # Empty in dry-run

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_download_media_no_media_url(self, mock_exists, mock_build_path):
        """Test that download returns None when episode has no media_url."""
        self.cfg = create_test_config(skip_existing=False)
        self.episode.media_url = None
        mock_exists.return_value = False
        mock_build_path.return_value = "/output/transcript.txt"

        result = episode_processor.download_media_for_transcription(
            episode=self.episode,
            cfg=self.cfg,
            temp_dir=self.temp_dir,
            effective_output_dir=self.effective_output_dir,
            run_suffix=None,
        )

        self.assertIsNone(result)


class TestFetchTranscriptContent(unittest.TestCase):
    """Tests for _fetch_transcript_content function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()

    @patch("podcast_scraper.workflow.episode_processor.downloader.http_get")
    def test_fetch_transcript_success(self, mock_get):
        """Test successfully fetching transcript content."""
        mock_get.return_value = (b"Transcript content", "text/vtt")

        result = episode_processor._fetch_transcript_content(
            transcript_url="https://example.com/transcript.vtt",
            cfg=self.cfg,
        )

        self.assertEqual(result, (b"Transcript content", "text/vtt"))
        mock_get.assert_called_once()

    @patch("podcast_scraper.workflow.episode_processor.downloader.http_get")
    def test_fetch_transcript_failure(self, mock_get):
        """Test handling transcript fetch failure."""
        mock_get.return_value = (None, None)

        result = episode_processor._fetch_transcript_content(
            transcript_url="https://example.com/transcript.vtt",
            cfg=self.cfg,
        )

        self.assertIsNone(result)


class TestWriteTranscriptFile(unittest.TestCase):
    """Tests for _write_transcript_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.episode = create_test_episode(idx=1, title="Test Episode")

    @patch("podcast_scraper.workflow.episode_processor.filesystem.write_file")
    @patch("os.path.exists")
    @patch("os.path.relpath")
    def test_write_transcript_success(self, mock_relpath, mock_exists, mock_write):
        """Test successfully writing transcript file."""
        mock_exists.return_value = False
        mock_relpath.return_value = "transcript.txt"

        result = episode_processor._write_transcript_file(
            data=b"Transcript content",
            out_path="/output/transcript.txt",
            cfg=self.cfg,
            episode=self.episode,
            effective_output_dir="/output",
        )

        self.assertEqual(result, "transcript.txt")
        mock_write.assert_called_once()

    @patch("os.path.exists")
    def test_write_transcript_skip_existing(self, mock_exists):
        """Test that writing is skipped when file exists and skip_existing is True."""
        self.cfg = create_test_config(skip_existing=True)
        mock_exists.return_value = True

        result = episode_processor._write_transcript_file(
            data=b"Transcript content",
            out_path="/output/transcript.txt",
            cfg=self.cfg,
            episode=self.episode,
            effective_output_dir="/output",
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.write_file")
    @patch("os.path.exists")
    def test_write_transcript_io_error(self, mock_exists, mock_write):
        """Test handling IOError during write."""
        mock_exists.return_value = False
        mock_write.side_effect = IOError("Disk full")

        result = episode_processor._write_transcript_file(
            data=b"Transcript content",
            out_path="/output/transcript.txt",
            cfg=self.cfg,
            episode=self.episode,
            effective_output_dir="/output",
        )

        self.assertIsNone(result)


class TestProcessTranscriptDownload(unittest.TestCase):
    """Tests for process_transcript_download function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(skip_existing=False)
        self.episode = create_test_episode(
            idx=1,
            title="Test Episode",
            transcript_urls=[("https://example.com/transcript.vtt", "text/vtt")],
        )

    @patch("podcast_scraper.workflow.episode_processor.os.path.relpath")
    @patch("podcast_scraper.workflow.episode_processor.os.path.exists")
    @patch("podcast_scraper.workflow.episode_processor.filesystem.write_file")
    @patch("podcast_scraper.workflow.episode_processor.derive_transcript_extension")
    @patch("podcast_scraper.workflow.episode_processor._write_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._fetch_transcript_content")
    @patch("podcast_scraper.workflow.episode_processor._determine_output_path")
    @patch("podcast_scraper.workflow.episode_processor._check_existing_transcript")
    def test_process_transcript_download_success(
        self,
        mock_check,
        mock_determine,
        mock_fetch,
        mock_write,
        mock_derive_ext,
        mock_write_file,
        mock_exists,
        mock_relpath,
    ):
        """Test successful transcript download."""
        mock_check.return_value = False  # Not found, proceed
        mock_determine.return_value = "/output/transcript.vtt"
        mock_fetch.return_value = (b"Transcript content", "text/vtt")
        mock_derive_ext.return_value = ".vtt"
        mock_exists.return_value = False
        mock_relpath.return_value = "transcript.vtt"
        mock_write.return_value = "transcript.vtt"

        result = episode_processor.process_transcript_download(
            episode=self.episode,
            transcript_url="https://example.com/transcript.vtt",
            transcript_type="text/vtt",
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
        )

        # Verify mocks were called
        self.assertTrue(mock_fetch.called, "_fetch_transcript_content should be called")
        self.assertTrue(mock_write.called, "_write_transcript_file should be called")

        self.assertTrue(result[0], f"Expected success but got: {result}")  # success
        self.assertEqual(result[1], "transcript.vtt")  # path
        self.assertEqual(result[2], "direct_download")  # source
        self.assertEqual(result[3], len(b"Transcript content"))  # bytes

    @patch("podcast_scraper.workflow.episode_processor._check_existing_transcript")
    def test_process_transcript_download_skip_existing(self, mock_check):
        """Test skipping transcript download when already exists."""
        mock_check.return_value = True  # Found, skip

        result = episode_processor.process_transcript_download(
            episode=self.episode,
            transcript_url="https://example.com/transcript.vtt",
            transcript_type="text/vtt",
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
        )

        self.assertFalse(result[0])  # skipped
        self.assertIsNone(result[1])  # no path

    @patch("podcast_scraper.workflow.episode_processor._fetch_transcript_content")
    @patch("podcast_scraper.workflow.episode_processor._determine_output_path")
    @patch("podcast_scraper.workflow.episode_processor._check_existing_transcript")
    def test_process_transcript_download_fetch_failure(
        self, mock_check, mock_determine, mock_fetch
    ):
        """Test handling transcript fetch failure."""
        mock_check.return_value = (False, None)  # Not found, proceed
        mock_determine.return_value = "/output/transcript.vtt"
        mock_fetch.return_value = None  # Fetch failed

        result = episode_processor.process_transcript_download(
            episode=self.episode,
            transcript_url="https://example.com/transcript.vtt",
            transcript_type="text/vtt",
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
        )

        self.assertFalse(result[0])  # failure
        self.assertIsNone(result[1])  # no path

    @patch("podcast_scraper.workflow.episode_processor.derive_transcript_extension")
    @patch("podcast_scraper.workflow.episode_processor._determine_output_path")
    @patch("podcast_scraper.workflow.episode_processor._check_existing_transcript")
    def test_process_transcript_download_dry_run(self, mock_check, mock_determine, mock_derive_ext):
        """Test transcript download in dry-run mode."""
        self.cfg = create_test_config(dry_run=True)
        mock_check.return_value = False  # Not found, proceed
        mock_determine.return_value = "/output/transcript.vtt"
        mock_derive_ext.return_value = ".vtt"

        result = episode_processor.process_transcript_download(
            episode=self.episode,
            transcript_url="https://example.com/transcript.vtt",
            transcript_type="text/vtt",
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
        )

        self.assertTrue(result[0])  # success (dry-run)
        self.assertEqual(result[1], "/output/transcript.vtt")  # path
        self.assertEqual(result[2], "direct_download")  # source
        self.assertEqual(result[3], 0)  # no bytes in dry-run


@pytest.mark.unit
class TestFormatTranscriptIfNeeded(unittest.TestCase):
    """Tests for _format_transcript_if_needed helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(screenplay=False)

    def test_format_transcript_if_needed_plain_text(self):
        """Test formatting with plain text (no screenplay)."""
        result = {"text": "Hello world", "segments": []}

        text = episode_processor._format_transcript_if_needed(result, self.cfg, None, None)

        self.assertEqual(text, "Hello world")

    def test_format_transcript_if_needed_screenplay_enabled(self):
        """Test formatting with screenplay enabled."""
        cfg = create_test_config(screenplay=True, screenplay_num_speakers=2)
        result = {"text": "Hello world", "segments": []}
        mock_provider = Mock()
        mock_provider.format_screenplay_from_segments = Mock(return_value="SCREENPLAY")

        text = episode_processor._format_transcript_if_needed(
            result, cfg, ["Speaker 1"], mock_provider
        )

        self.assertEqual(text, "SCREENPLAY")
        mock_provider.format_screenplay_from_segments.assert_called_once()

    def test_format_transcript_if_needed_screenplay_no_provider(self):
        """Test formatting with screenplay but no provider support."""
        cfg = create_test_config(screenplay=True)
        result = {"text": "Hello world", "segments": []}

        with patch("podcast_scraper.workflow.episode_processor.logger") as mock_logger:
            text = episode_processor._format_transcript_if_needed(result, cfg, None, None)

            self.assertEqual(text, "Hello world")
            mock_logger.warning.assert_called()

    def test_format_transcript_if_needed_empty_text(self):
        """Test formatting with empty text."""
        result = {"text": "", "segments": []}

        text = episode_processor._format_transcript_if_needed(result, self.cfg, None, None)

        self.assertEqual(text, "")


@pytest.mark.unit
class TestSaveTranscriptFile(unittest.TestCase):
    """Tests for _save_transcript_file helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.job = Mock()
        self.job.idx = 1
        self.job.ep_title_safe = "test_episode"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("podcast_scraper.workflow.episode_processor.filesystem.write_file")
    @patch("os.path.relpath")
    def test_save_transcript_file_success(self, mock_relpath, mock_write_file, mock_build_path):
        """Test successful transcript file save."""
        mock_build_path.return_value = os.path.join(self.temp_dir, "transcript.txt")
        mock_relpath.return_value = "transcript.txt"

        result = episode_processor._save_transcript_file(
            "Transcript text", self.job, None, self.temp_dir
        )

        self.assertEqual(result, "transcript.txt")
        mock_write_file.assert_called_once()

    def test_save_transcript_file_empty_text(self):
        """Test that empty text raises RuntimeError."""
        with self.assertRaises(RuntimeError) as context:
            episode_processor._save_transcript_file("", self.job, None, self.temp_dir)

        self.assertIn("empty transcription", str(context.exception))


@pytest.mark.unit
class TestCleanupTempMedia(unittest.TestCase):
    """Tests for _cleanup_temp_media helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.temp_media = os.path.join(self.temp_dir, "temp.mp3")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cleanup_temp_media_success(self):
        """Test successful cleanup of temp media file."""
        with open(self.temp_media, "w") as f:
            f.write("test")

        cfg = create_test_config(reuse_media=False)
        episode_processor._cleanup_temp_media(self.temp_media, cfg)

        self.assertFalse(os.path.exists(self.temp_media))

    def test_cleanup_temp_media_reuse_enabled(self):
        """Test that cleanup is skipped when reuse_media is enabled."""
        with open(self.temp_media, "w") as f:
            f.write("test")

        cfg = create_test_config(reuse_media=True)

        with patch("podcast_scraper.workflow.episode_processor.logger") as mock_logger:
            episode_processor._cleanup_temp_media(self.temp_media, cfg)

            self.assertTrue(os.path.exists(self.temp_media))
            mock_logger.debug.assert_called()

    def test_cleanup_temp_media_file_not_exists(self):
        """Test cleanup when file doesn't exist (no error)."""
        cfg = create_test_config(reuse_media=False)

        # Should not raise
        episode_processor._cleanup_temp_media(self.temp_media, cfg)


@pytest.mark.unit
class TestCheckAndReuseExistingTranscript(unittest.TestCase):
    """Tests for _check_and_reuse_existing_transcript helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.job = Mock()
        self.job.idx = 1
        self.job.ep_title_safe = "test_episode"
        self.job.temp_media = ""
        self.job.episode = create_test_episode(idx=1, title="Test Episode")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    @patch("os.path.relpath")
    def test_check_and_reuse_existing_transcript_found(
        self, mock_relpath, mock_exists, mock_build_path
    ):
        """Test reusing existing transcript when found."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        mock_build_path.return_value = transcript_path
        mock_exists.return_value = True
        mock_relpath.return_value = "transcript.txt"

        cfg = create_test_config(skip_existing=True)

        result = episode_processor._check_and_reuse_existing_transcript(
            self.job, cfg, None, self.temp_dir
        )

        self.assertIsNotNone(result)
        self.assertTrue(result[0])  # success
        self.assertEqual(result[1], "transcript.txt")

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_check_and_reuse_existing_transcript_not_found(self, mock_exists, mock_build_path):
        """Test when existing transcript is not found."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        mock_build_path.return_value = transcript_path
        mock_exists.return_value = False

        cfg = create_test_config(skip_existing=True)

        result = episode_processor._check_and_reuse_existing_transcript(
            self.job, cfg, None, self.temp_dir
        )

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.episode_processor.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_check_and_reuse_existing_transcript_skip_disabled(self, mock_exists, mock_build_path):
        """Test that reuse is skipped when skip_existing is disabled."""
        transcript_path = os.path.join(self.temp_dir, "transcript.txt")
        mock_build_path.return_value = transcript_path
        mock_exists.return_value = True

        cfg = create_test_config(skip_existing=False)

        result = episode_processor._check_and_reuse_existing_transcript(
            self.job, cfg, None, self.temp_dir
        )

        self.assertIsNone(result)


@pytest.mark.unit
class TestCheckTranscriptCache(unittest.TestCase):
    """Tests for _check_transcript_cache helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.temp_media = os.path.join(self.temp_dir, "temp.mp3")
        with open(self.temp_media, "w") as f:
            f.write("test audio")

        self.job = Mock()
        self.job.idx = 1
        self.job.ep_title_safe = "test_episode"
        self.job.episode = create_test_episode(idx=1, title="Test Episode")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.cache.transcript_cache.get_cached_transcript")
    @patch("podcast_scraper.cache.transcript_cache.get_audio_hash")
    @patch("podcast_scraper.workflow.episode_processor._save_transcript_file")
    @patch("podcast_scraper.workflow.episode_processor._cleanup_temp_media")
    def test_check_transcript_cache_hit(
        self, mock_cleanup, mock_save, mock_get_hash, mock_get_cached
    ):
        """Test transcript cache hit."""
        mock_get_hash.return_value = "test_hash"
        mock_get_cached.return_value = "Cached transcript text"
        mock_save.return_value = "transcript.txt"

        cfg = create_test_config(transcript_cache_enabled=True)

        result = episode_processor._check_transcript_cache(
            self.job, cfg, self.temp_media, None, self.temp_dir
        )

        self.assertIsNotNone(result)
        self.assertTrue(result[0])  # success
        self.assertEqual(result[1], "transcript.txt")
        mock_cleanup.assert_called_once()

    @patch("podcast_scraper.cache.transcript_cache.get_cached_transcript")
    def test_check_transcript_cache_miss(self, mock_get_cached):
        """Test transcript cache miss."""
        mock_get_cached.return_value = None

        cfg = create_test_config(transcript_cache_enabled=True)

        result = episode_processor._check_transcript_cache(
            self.job, cfg, self.temp_media, None, self.temp_dir
        )

        self.assertIsNone(result)

    def test_check_transcript_cache_disabled(self):
        """Test that cache check is skipped when disabled."""
        cfg = create_test_config(transcript_cache_enabled=False)

        result = episode_processor._check_transcript_cache(
            self.job, cfg, self.temp_media, None, self.temp_dir
        )

        self.assertIsNone(result)


@pytest.mark.unit
class TestGetProviderModelName(unittest.TestCase):
    """Tests for _get_provider_model_name helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()

    def test_get_provider_model_name_from_whisper_model(self):
        """Test extracting model name from cfg.whisper_model."""
        cfg = create_test_config(whisper_model="base")
        mock_provider = Mock()

        result = episode_processor._get_provider_model_name(mock_provider, cfg)

        self.assertEqual(result, "base")

    def test_get_provider_model_name_from_openai_model(self):
        """Test extracting model name from cfg.openai_transcription_model."""
        cfg = create_test_config(
            transcription_provider="openai",
            openai_transcription_model="whisper-1",
            openai_api_key="test-key",  # Required for OpenAI provider
        )
        mock_provider = Mock()
        # Mock provider has non-string model (like Whisper object)
        mock_provider.model = Mock()  # Non-string object

        result = episode_processor._get_provider_model_name(mock_provider, cfg)

        self.assertEqual(result, "whisper-1")

    def test_get_provider_model_name_from_provider_attribute(self):
        """Test extracting model name from provider attribute."""
        # Create a mock that doesn't have 'model' attribute
        # Use spec to limit attributes, or explicitly delete model
        mock_provider = Mock(spec=["transcription_model"])
        mock_provider.transcription_model = "custom-model"

        result = episode_processor._get_provider_model_name(mock_provider, self.cfg)

        # Should return transcription_model since model attribute doesn't exist
        self.assertEqual(result, "custom-model")

    def test_get_provider_model_name_returns_none(self):
        """Test that None is returned when model cannot be determined."""
        mock_provider = Mock()
        # Remove both model and transcription_model attributes
        if hasattr(mock_provider, "model"):
            delattr(mock_provider, "model")
        if hasattr(mock_provider, "transcription_model"):
            delattr(mock_provider, "transcription_model")

        result = episode_processor._get_provider_model_name(mock_provider, self.cfg)

        self.assertIsNone(result)


@pytest.mark.unit
class TestSaveTranscriptToCacheIfNeeded(unittest.TestCase):
    """Tests for _save_transcript_to_cache_if_needed helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.temp_media = os.path.join(self.temp_dir, "temp.mp3")
        with open(self.temp_media, "w") as f:
            f.write("test audio")

        self.job = Mock()
        self.job.idx = 1

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.cache.transcript_cache.save_transcript_to_cache")
    @patch("podcast_scraper.cache.transcript_cache.get_audio_hash")
    @patch("podcast_scraper.workflow.episode_processor._get_provider_model_name")
    @patch("podcast_scraper.workflow.episode_processor.logger")
    def test_save_transcript_to_cache_if_needed_success(
        self, mock_logger, mock_get_model, mock_get_hash, mock_save
    ):
        """Test successful transcript cache save."""
        mock_get_hash.return_value = "test_hash"
        mock_get_model.return_value = "base"

        cfg = create_test_config(transcript_cache_enabled=True)

        episode_processor._save_transcript_to_cache_if_needed(
            self.job, cfg, self.temp_media, "Transcript text", Mock()
        )

        mock_save.assert_called_once()

    def test_save_transcript_to_cache_if_needed_disabled(self):
        """Test that cache save is skipped when disabled."""
        cfg = create_test_config(transcript_cache_enabled=False)

        with patch("podcast_scraper.cache.transcript_cache.save_transcript_to_cache") as mock_save:
            episode_processor._save_transcript_to_cache_if_needed(
                self.job, cfg, self.temp_media, "Transcript text", Mock()
            )

            mock_save.assert_not_called()

    @patch("podcast_scraper.cache.transcript_cache.save_transcript_to_cache")
    @patch("podcast_scraper.cache.transcript_cache.get_audio_hash")
    @patch("podcast_scraper.workflow.episode_processor._get_provider_model_name")
    @patch("podcast_scraper.workflow.episode_processor.logger")
    def test_save_transcript_to_cache_if_needed_handles_exception(
        self, mock_logger, mock_get_model, mock_get_hash, mock_save
    ):
        """Test that exceptions during cache save are handled."""
        mock_get_hash.return_value = "test_hash"
        mock_get_model.return_value = "base"
        mock_save.side_effect = RuntimeError("Cache save failed")

        cfg = create_test_config(transcript_cache_enabled=True)

        # Should not raise
        episode_processor._save_transcript_to_cache_if_needed(
            self.job, cfg, self.temp_media, "Transcript text", Mock()
        )

        mock_logger.warning.assert_called()


@pytest.mark.unit
class TestRecordTranscriptionMetrics(unittest.TestCase):
    """Tests for _record_transcription_metrics helper function."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.cfg = create_test_config()
        self.pipeline_metrics = metrics.Metrics()
        self.job = Mock()
        self.job.idx = 1
        self.job.episode = create_test_episode(idx=1, title="Test Episode")
        self.call_metrics = Mock()
        self.call_metrics.retries = 0
        self.call_metrics.rate_limit_sleep_sec = 0.0
        self.call_metrics.prompt_tokens = None
        self.call_metrics.completion_tokens = None
        self.call_metrics.estimated_cost = None

    @patch("podcast_scraper.workflow.orchestration._log_episode_metrics")
    def test_record_transcription_metrics_success(self, mock_log_metrics):
        """Test successful recording of transcription metrics."""
        episode_processor._record_transcription_metrics(
            self.job, self.cfg, 10.5, self.call_metrics, self.pipeline_metrics
        )

        self.assertEqual(self.pipeline_metrics.transcribe_times, [10.5])
        mock_log_metrics.assert_called_once()

    def test_record_transcription_metrics_no_pipeline_metrics(self):
        """Test that function returns early when pipeline_metrics is None."""
        # Should not raise
        episode_processor._record_transcription_metrics(
            self.job, self.cfg, 10.5, self.call_metrics, None
        )


if __name__ == "__main__":
    unittest.main()
