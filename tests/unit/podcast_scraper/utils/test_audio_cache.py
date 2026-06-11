"""Unit tests for the #947 durable raw-audio cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.utils import audio_cache


def _cfg(**kw):
    base = dict(audio_cache_enabled=True, audio_cache_in_corpus=False, audio_cache_dir=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.unit
class TestResolveCacheRoot:
    def test_disabled_returns_none(self):
        assert audio_cache.resolve_cache_root(_cfg(audio_cache_enabled=False), "/tmp/c") is None

    def test_explicit_dir(self, tmp_path):
        root = audio_cache.resolve_cache_root(_cfg(audio_cache_dir=str(tmp_path / "ac")), "/tmp/c")
        assert root == tmp_path / "ac"

    def test_default_dir(self):
        root = audio_cache.resolve_cache_root(_cfg(), "/tmp/c")
        assert root is not None and root.name == "audio"  # .cache/audio

    def test_in_corpus_under_corpus(self, tmp_path):
        root = audio_cache.resolve_cache_root(_cfg(audio_cache_in_corpus=True), str(tmp_path))
        assert root is not None
        assert ".podcast_scraper/audio-cache" in str(root)
        assert str(root).startswith(str(tmp_path.resolve()))


@pytest.mark.unit
class TestStoreLookup:
    def test_store_then_lookup_roundtrip(self, tmp_path):
        root = tmp_path / "cache"
        src = tmp_path / "ep.mp3"
        src.write_bytes(b"audio-bytes" * 50)
        stored = audio_cache.store(root, "guid-abc", str(src))
        assert stored is not None and stored.endswith(".mp3")
        # sharded by sha256(guid)
        digest = hashlib.sha256(b"guid-abc").hexdigest()
        assert f"sha256/{digest[:2]}/{digest[2:4]}/{digest}.mp3" in stored.replace("\\", "/")
        assert audio_cache.lookup_by_guid(root, "guid-abc") == stored

    def test_dedupe_by_existence(self, tmp_path):
        root = tmp_path / "cache"
        src = tmp_path / "ep.mp3"
        src.write_bytes(b"x" * 100)
        first = audio_cache.store(root, "g", str(src))
        before = Path(first).stat().st_mtime_ns
        # second store of same guid must not rewrite the file
        src.write_bytes(b"y" * 200)
        second = audio_cache.store(root, "g", str(src))
        assert second == first
        assert Path(first).stat().st_mtime_ns == before  # untouched
        assert Path(first).read_bytes() == b"x" * 100

    def test_missing_guid_skips(self, tmp_path):
        root = tmp_path / "cache"
        src = tmp_path / "ep.mp3"
        src.write_bytes(b"x" * 10)
        assert audio_cache.store(root, "", str(src)) is None
        assert audio_cache.store(root, None, str(src)) is None
        assert audio_cache.lookup_by_guid(root, "") is None

    def test_empty_source_skips(self, tmp_path):
        root = tmp_path / "cache"
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        assert audio_cache.store(root, "g", str(empty)) is None
        assert audio_cache.store(root, "g", str(tmp_path / "nope.mp3")) is None

    def test_lookup_miss(self, tmp_path):
        assert audio_cache.lookup_by_guid(tmp_path / "cache", "nope") is None
        assert audio_cache.lookup_by_guid(None, "g") is None

    def test_zero_byte_cache_entry_is_miss(self, tmp_path):
        root = tmp_path / "cache"
        src = tmp_path / "ep.m4a"
        src.write_bytes(b"a" * 20)
        stored = audio_cache.store(root, "g", str(src))
        Path(stored).write_bytes(b"")  # corrupt to zero bytes
        assert audio_cache.lookup_by_guid(root, "g") is None

    def test_copy_into(self, tmp_path):
        cached = tmp_path / "c.mp3"
        cached.write_bytes(b"hello" * 10)
        dest = tmp_path / "sub" / "temp.mp3"
        assert audio_cache.copy_into(str(cached), str(dest)) is True
        assert dest.read_bytes() == cached.read_bytes()
