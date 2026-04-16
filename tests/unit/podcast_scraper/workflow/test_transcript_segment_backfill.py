"""Transcript segment sidecar backfill for GI (GitHub #542)."""

from __future__ import annotations

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest

from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor
from tests.conftest import create_test_config, create_test_episode

pytestmark = [pytest.mark.unit]

TEST_MEDIA_URL = "https://example.com/audio.mp3"
TEST_MEDIA_TYPE_MP3 = "audio/mpeg"


@pytest.mark.unit
class TestTranscriptTxtMissingSegments(unittest.TestCase):
    """Unit tests for ``transcript_txt_missing_segments``."""

    def test_false_when_not_txt_extension(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as handle:
            path = handle.name
        try:
            self.assertFalse(episode_processor.transcript_txt_missing_segments(path))
        finally:
            os.unlink(path)

    def test_false_when_file_missing(self) -> None:
        self.assertFalse(
            episode_processor.transcript_txt_missing_segments("/nonexistent/path/file.txt")
        )

    def test_true_when_txt_exists_without_sidecar(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as handle:
            handle.write("hello")
            path = handle.name
        try:
            self.assertTrue(episode_processor.transcript_txt_missing_segments(path))
        finally:
            os.unlink(path)

    def test_false_when_sidecar_present(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as handle:
            handle.write("hello")
            txt_path = handle.name
        seg_path = os.path.splitext(txt_path)[0] + ".segments.json"
        try:
            with open(seg_path, "w", encoding="utf-8") as seg:
                seg.write("[]")
            self.assertFalse(episode_processor.transcript_txt_missing_segments(txt_path))
        finally:
            os.unlink(txt_path)
            if os.path.isfile(seg_path):
                os.unlink(seg_path)


@pytest.mark.unit
class TestDownloadMediaSegmentBackfill(unittest.TestCase):
    """``download_media_for_transcription`` with ``backfill_transcript_segments``."""

    def test_skips_when_txt_only_and_backfill_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = os.path.join(tmpdir, ".tmp_media")
            os.makedirs(temp_dir, exist_ok=True)
            ep_title = "Episode 2"
            ep_title_safe = filesystem.sanitize_filename(ep_title)
            transcripts_dir = os.path.join(tmpdir, filesystem.TRANSCRIPTS_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)
            final_path = filesystem.build_whisper_output_path(1, ep_title_safe, None, tmpdir)
            with open(final_path, "wb") as fh:
                fh.write(b"existing transcript")

            cfg = create_test_config(
                output_dir=tmpdir,
                skip_existing=True,
                transcribe_missing=True,
                generate_metadata=True,
                generate_gi=True,
                backfill_transcript_segments=False,
            )

            item = ET.Element("item")
            ET.SubElement(
                item,
                "enclosure",
                attrib={"url": TEST_MEDIA_URL, "type": TEST_MEDIA_TYPE_MP3},
            )
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

    def test_downloads_when_txt_only_backfill_and_gi_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = os.path.join(tmpdir, ".tmp_media")
            os.makedirs(temp_dir, exist_ok=True)
            ep_title = "Episode 2"
            ep_title_safe = filesystem.sanitize_filename(ep_title)
            transcripts_dir = os.path.join(tmpdir, filesystem.TRANSCRIPTS_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)
            final_path = filesystem.build_whisper_output_path(1, ep_title_safe, None, tmpdir)
            with open(final_path, "wb") as fh:
                fh.write(b"existing transcript")

            cfg = create_test_config(
                output_dir=tmpdir,
                skip_existing=True,
                transcribe_missing=True,
                generate_metadata=True,
                generate_gi=True,
                backfill_transcript_segments=True,
            )

            item = ET.Element("item")
            ET.SubElement(
                item,
                "enclosure",
                attrib={"url": TEST_MEDIA_URL, "type": TEST_MEDIA_TYPE_MP3},
            )
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
                mock_download.return_value = (True, 100)
                job = episode_processor.download_media_for_transcription(
                    episode,
                    cfg,
                    temp_dir,
                    tmpdir,
                    None,
                )

            self.assertIsNotNone(job)
            assert job is not None
            self.assertTrue(job.temp_media)
            mock_download.assert_called_once()

    def test_skips_when_segments_sidecar_exists_with_backfill_gi(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = os.path.join(tmpdir, ".tmp_media")
            os.makedirs(temp_dir, exist_ok=True)
            ep_title = "Episode 2"
            ep_title_safe = filesystem.sanitize_filename(ep_title)
            transcripts_dir = os.path.join(tmpdir, filesystem.TRANSCRIPTS_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)
            final_path = filesystem.build_whisper_output_path(1, ep_title_safe, None, tmpdir)
            with open(final_path, "wb") as fh:
                fh.write(b"existing transcript")
            seg_path = os.path.splitext(final_path)[0] + ".segments.json"
            with open(seg_path, "w", encoding="utf-8") as fh:
                fh.write("[]")

            cfg = create_test_config(
                output_dir=tmpdir,
                skip_existing=True,
                transcribe_missing=True,
                generate_metadata=True,
                generate_gi=True,
                backfill_transcript_segments=True,
            )

            item = ET.Element("item")
            ET.SubElement(
                item,
                "enclosure",
                attrib={"url": TEST_MEDIA_URL, "type": TEST_MEDIA_TYPE_MP3},
            )
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
