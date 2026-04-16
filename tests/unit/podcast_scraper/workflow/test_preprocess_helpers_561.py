"""GitHub #561: preprocessing cache probe and API re-encode helpers."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.rss.downloader import OPENAI_MAX_FILE_SIZE_BYTES
from podcast_scraper.workflow import episode_processor
from tests.conftest import create_test_config

pytestmark = [pytest.mark.unit, pytest.mark.module_workflow]

_TARGET = episode_processor._PREPROCESSING_API_REENCODE_TARGET_BYTES


class TestPreprocessHelpers561(unittest.TestCase):
    """Unit tests for _preprocessing_probe_preprocessed_cache and re-encode ladder."""

    @patch("podcast_scraper.preprocessing.audio.cache.get_cached_audio_path")
    @patch("os.path.getsize")
    def test_probe_skips_oversized_openai_cache_hit(
        self, mock_getsize: MagicMock, mock_get_cached: MagicMock
    ) -> None:
        mock_get_cached.side_effect = ["/cache/huge.mp3", "/cache/ok.mp3"]
        mock_getsize.side_effect = [OPENAI_MAX_FILE_SIZE_BYTES + 1, 2048]
        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test",
            preprocessing_enabled=True,
        )
        path, cache_key, elapsed = episode_processor._preprocessing_probe_preprocessed_cache(
            cfg,
            "/tmp/source.mp3",
            "/cache/pre",
            [48, 40, 32, 24],
            "openai",
        )
        self.assertEqual(path, "/cache/ok.mp3")
        self.assertTrue(cache_key)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(mock_get_cached.call_count, 2)

    @patch("podcast_scraper.preprocessing.audio.cache.get_cached_audio_path")
    @patch("os.path.getsize")
    def test_whisper_does_not_skip_large_cached_file(
        self, mock_getsize: MagicMock, mock_get_cached: MagicMock
    ) -> None:
        mock_get_cached.return_value = "/cache/fat.mp3"
        cfg = create_test_config(rss_url="https://example.com/feed.xml")
        path, _, _ = episode_processor._preprocessing_probe_preprocessed_cache(
            cfg,
            "/tmp/source.mp3",
            "/cache/pre",
            [48],
            "whisper",
        )
        self.assertEqual(path, "/cache/fat.mp3")
        mock_getsize.assert_not_called()

    def test_reencode_openai_steps_down_then_stops(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pre = os.path.join(td, "pre.mp3")
            raw = os.path.join(td, "raw.mp3")
            Path(pre).write_bytes(b"x" * 20)
            Path(raw).write_bytes(b"y" * 10)

            preproc = MagicMock()
            preproc.mp3_bitrate_kbps = 48

            def reencode(inp: str, outp: str, kbps: int) -> tuple:
                Path(outp).write_bytes(b"s")
                return True, 0.05

            preproc.reencode_mp3_to_bitrate.side_effect = reencode

            with patch("podcast_scraper.workflow.episode_processor.os.path.getsize") as gs:
                gs.side_effect = [_TARGET + 1, 100]
                out_path, final_kbps, te = (
                    episode_processor._preprocessing_reencode_mp3_until_target(
                        7,
                        preproc,
                        raw,
                        pre,
                        "openai",
                        1.0,
                    )
                )
                self.assertEqual(out_path, f"{raw}.re_encode.40.mp3")
            self.assertEqual(final_kbps, 40)
            self.assertGreater(te, 0.9)

    def test_reencode_skipped_for_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pre = os.path.join(td, "pre.mp3")
            raw = os.path.join(td, "raw.mp3")
            Path(pre).write_bytes(b"a")
            preproc = MagicMock()
            preproc.mp3_bitrate_kbps = 48
            out, kb, te = episode_processor._preprocessing_reencode_mp3_until_target(
                1, preproc, raw, pre, "whisper", 2.5
            )
            self.assertEqual(out, pre)
            self.assertEqual(kb, 48)
            self.assertEqual(te, 2.5)
            preproc.reencode_mp3_to_bitrate.assert_not_called()
