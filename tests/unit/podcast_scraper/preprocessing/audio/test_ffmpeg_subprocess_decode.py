"""GitHub #558: ffmpeg/ffprobe subprocess text decoding (UTF-8 + replace)."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.preprocessing.audio.ffmpeg_processor import (
    extract_audio_metadata,
    FFmpegAudioPreprocessor,
)

pytestmark = [pytest.mark.unit, pytest.mark.module_utils]


class TestRunTextSubprocessKwargs(unittest.TestCase):
    """Assert subprocess.run receives safe decoding kwargs (GitHub #558)."""

    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor._check_ffprobe_available")
    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor.os.path.exists")
    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor.subprocess.run")
    def test_ffprobe_uses_utf8_replace(self, mock_run, mock_exists, mock_ffprobe):
        mock_ffprobe.return_value = True
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(stdout='{"streams":[]}', stderr="", returncode=0)

        out = extract_audio_metadata("/fake/audio.mp3")
        self.assertIsNone(out)
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs.get("encoding"), "utf-8")
        self.assertEqual(kwargs.get("errors"), "replace")
        self.assertTrue(kwargs.get("text"))

    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor._check_ffmpeg_available")
    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor.subprocess.run")
    def test_ffmpeg_preprocess_uses_utf8_replace(self, mock_run, mock_ffmpeg):
        mock_ffmpeg.return_value = True
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        pre = FFmpegAudioPreprocessor()
        ok, elapsed = pre.preprocess("/in.mp3", "/out.mp3")
        self.assertTrue(ok)
        self.assertGreaterEqual(elapsed, 0.0)
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs.get("encoding"), "utf-8")
        self.assertEqual(kwargs.get("errors"), "replace")
        self.assertTrue(kwargs.get("text"))
