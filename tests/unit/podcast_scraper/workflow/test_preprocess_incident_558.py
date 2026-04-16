"""GitHub #558: preprocessing failure writes corpus_incidents.jsonl episode row."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from podcast_scraper.workflow import episode_processor
from tests.conftest import create_test_config, create_test_episode

pytestmark = [pytest.mark.unit, pytest.mark.module_workflow]


class TestPreprocessIncident558(unittest.TestCase):
    """Episode-scoped incident when ffmpeg preprocess fails and we fall back."""

    @patch("podcast_scraper.preprocessing.audio.cache.get_cached_audio_path")
    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor.extract_audio_metadata")
    @patch("podcast_scraper.preprocessing.audio.factory.create_audio_preprocessor")
    def test_preprocess_failure_appends_incident_jsonl(
        self, mock_factory, mock_extract_meta, mock_cached
    ) -> None:
        mock_cached.return_value = None
        mock_extract_meta.return_value = None
        pre = Mock()
        pre.get_cache_key.return_value = "cachekey12"
        pre.preprocess.return_value = (False, 0.05)
        mock_factory.return_value = pre

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"abc")
            media = f.name
        try:
            with tempfile.TemporaryDirectory() as td:
                inc = os.path.join(td, "corpus_incidents.jsonl")
                cfg = create_test_config(
                    preprocessing_enabled=True,
                    incident_log_path=inc,
                    rss_url="https://example.com/feed.xml",
                )
                job = Mock()
                job.idx = 1
                job.episode = create_test_episode(idx=1, title="Ep1")

                out = episode_processor._preprocess_audio_if_needed(job, cfg, media, None)
                self.assertEqual(out, media)

                self.assertTrue(os.path.isfile(inc))
                lines = Path(inc).read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 1)
                row = json.loads(lines[0])
                self.assertEqual(row["scope"], "episode")
                self.assertEqual(row["category"], "policy")
                self.assertEqual(row["stage"], "preprocessing")
                self.assertIn("ffmpeg", row["message"].lower())
        finally:
            if os.path.isfile(media):
                os.unlink(media)
