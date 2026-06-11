"""Cache-aware download behavior for the #947 raw-audio cache."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest

from podcast_scraper.utils import audio_cache
from podcast_scraper.workflow import episode_processor
from tests.conftest import create_test_config, create_test_episode


def _episode_with_guid(guid: str):
    item = ET.Element("item")
    g = ET.SubElement(item, "guid")
    g.text = guid
    return create_test_episode(item=item, media_url="https://example.com/ep.mp3")


@pytest.mark.unit
class TestCacheAwareDownload:
    def test_cache_hit_skips_http(self, tmp_path):
        cache = tmp_path / "cache"
        # seed the cache for this guid
        seed = tmp_path / "seed.mp3"
        seed.write_bytes(b"cached-audio" * 100)
        audio_cache.store(cache, "g-hit", str(seed))

        cfg = create_test_config(audio_cache_dir=str(cache))
        episode = _episode_with_guid("g-hit")
        temp_media = str(tmp_path / "run" / "ep.mp3")

        with patch.object(episode_processor.downloader, "http_download_to_file") as mock_dl:
            ok, total, elapsed = episode_processor._download_or_reuse_media(
                episode, cfg, temp_media, None, str(tmp_path / "corpus")
            )

        assert ok is True
        assert elapsed == 0.0
        mock_dl.assert_not_called()  # no feed fetch on a cache hit
        assert open(temp_media, "rb").read() == seed.read_bytes()

    def test_cache_miss_downloads_and_stores(self, tmp_path):
        cache = tmp_path / "cache"
        cfg = create_test_config(audio_cache_dir=str(cache))
        episode = _episode_with_guid("g-miss")
        temp_media = str(tmp_path / "run" / "ep.mp3")

        def fake_download(url, ua, timeout, out_path):
            import os

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(b"fresh-audio" * 100)
            return True, 1100

        with patch.object(
            episode_processor.downloader, "http_download_to_file", side_effect=fake_download
        ) as mock_dl:
            ok, total, elapsed = episode_processor._download_or_reuse_media(
                episode, cfg, temp_media, None, str(tmp_path / "corpus")
            )

        assert ok is True
        mock_dl.assert_called_once()
        # audio now lives in the cache for next time
        assert audio_cache.lookup_by_guid(cache, "g-miss") is not None

    def test_disabled_cache_no_store(self, tmp_path):
        cache = tmp_path / "cache"
        cfg = create_test_config(audio_cache_dir=str(cache), audio_cache_enabled=False)
        episode = _episode_with_guid("g-off")
        temp_media = str(tmp_path / "run" / "ep.mp3")

        def fake_download(url, ua, timeout, out_path):
            import os

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            open(out_path, "wb").write(b"x" * 100)
            return True, 100

        with patch.object(
            episode_processor.downloader, "http_download_to_file", side_effect=fake_download
        ):
            episode_processor._download_or_reuse_media(
                episode, cfg, temp_media, None, str(tmp_path / "corpus")
            )
        assert audio_cache.lookup_by_guid(cache, "g-off") is None  # nothing cached when disabled
