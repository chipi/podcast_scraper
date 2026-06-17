"""Cache-aware download behavior for the #947 raw-audio cache."""

from __future__ import annotations

import os
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


@pytest.mark.unit
class TestPersistMediaLinkMode:
    """G6: corpus media/ hardlinks to the retained #947 cache entry instead of copying."""

    def _persist(self, tmp_path, link_mode, *, seed_cache=True, guid="g-persist"):
        cache = tmp_path / "cache"
        corpus = tmp_path / "corpus"
        cfg = create_test_config(audio_cache_dir=str(cache), corpus_media_link_mode=link_mode)
        episode = _episode_with_guid(guid)
        cache_entry = None
        if seed_cache:
            seed = tmp_path / "seed.mp3"
            seed.write_bytes(b"cached-audio" * 100)
            cache_entry = audio_cache.store(cache, guid, str(seed))
        temp_media = tmp_path / "run" / "ep.mp3"
        os.makedirs(temp_media.parent, exist_ok=True)
        temp_media.write_bytes(b"cached-audio" * 100)
        episode_processor._maybe_persist_episode_media(
            cfg, str(temp_media), str(corpus), "transcripts/ep.txt", episode=episode
        )
        return corpus / "media" / "ep.mp3", cache_entry

    def test_hardlink_mode_shares_inode_with_cache_entry(self, tmp_path):
        dest, cache_entry = self._persist(tmp_path, "hardlink")
        assert dest.is_file()
        assert cache_entry is not None
        assert dest.stat().st_ino == os.stat(cache_entry).st_ino

    def test_copy_mode_default_does_not_link(self, tmp_path):
        dest, cache_entry = self._persist(tmp_path, "copy")
        assert dest.is_file() and not dest.is_symlink()
        assert dest.stat().st_ino != os.stat(cache_entry).st_ino

    def test_hardlink_without_cache_entry_falls_back_to_copy(self, tmp_path):
        dest, cache_entry = self._persist(tmp_path, "hardlink", seed_cache=False)
        assert cache_entry is None
        assert dest.is_file() and not dest.is_symlink()

    def test_persist_disabled_writes_nothing(self, tmp_path):
        cfg = create_test_config(persist_episode_media=False, corpus_media_link_mode="hardlink")
        temp_media = tmp_path / "run" / "ep.mp3"
        temp_media.parent.mkdir(parents=True)
        temp_media.write_bytes(b"ID3")
        episode_processor._maybe_persist_episode_media(
            cfg,
            str(temp_media),
            str(tmp_path / "corpus"),
            "transcripts/ep.txt",
            episode=_episode_with_guid("g1"),
        )
        assert not (tmp_path / "corpus" / "media" / "ep.mp3").exists()

    def test_missing_transcript_relpath_is_noop(self, tmp_path):
        cfg = create_test_config(corpus_media_link_mode="hardlink")
        temp_media = tmp_path / "run" / "ep.mp3"
        temp_media.parent.mkdir(parents=True)
        temp_media.write_bytes(b"ID3")
        episode_processor._maybe_persist_episode_media(
            cfg,
            str(temp_media),
            str(tmp_path / "corpus"),
            None,
            episode=_episode_with_guid("g1"),
        )
        assert not (tmp_path / "corpus" / "media").exists()

    def test_episode_without_guid_copies(self, tmp_path):
        cfg = create_test_config(
            audio_cache_dir=str(tmp_path / "cache"), corpus_media_link_mode="hardlink"
        )
        temp_media = tmp_path / "run" / "ep.mp3"
        temp_media.parent.mkdir(parents=True)
        temp_media.write_bytes(b"ID3")
        episode = create_test_episode(item=ET.Element("item"))  # no <guid> → no cache entry
        episode_processor._maybe_persist_episode_media(
            cfg, str(temp_media), str(tmp_path / "corpus"), "transcripts/ep.txt", episode=episode
        )
        dest = tmp_path / "corpus" / "media" / "ep.mp3"
        assert dest.is_file() and not dest.is_symlink()


@pytest.mark.unit
def test_resolve_audio_cache_entry_swallows_errors(tmp_path):
    """Best-effort: a broken episode never blocks persistence (returns None)."""
    cfg = create_test_config(audio_cache_dir=str(tmp_path / "cache"))

    class _BadEpisode:
        @property
        def item(self):
            raise RuntimeError("boom")

    assert episode_processor._resolve_audio_cache_entry(cfg, str(tmp_path), _BadEpisode()) is None
    assert episode_processor._resolve_audio_cache_entry(cfg, str(tmp_path), None) is None
