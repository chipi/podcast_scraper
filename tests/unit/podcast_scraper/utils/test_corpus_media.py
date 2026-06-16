"""Unit tests for corpus media helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from podcast_scraper.utils.corpus_media import (
    audio_relpath_for_transcript,
    persist_episode_media,
    resolve_audio_relpath_for_metadata,
)

pytestmark = pytest.mark.unit


def test_audio_relpath_for_transcript() -> None:
    assert audio_relpath_for_transcript("transcripts/ep_01.txt") == "media/ep_01.mp3"


def test_persist_episode_media(tmp_path: Path) -> None:
    src = tmp_path / "dl.mp3"
    src.write_bytes(b"ID3")
    rel = persist_episode_media(str(src), str(tmp_path), "transcripts/show_ep.txt")
    assert rel == "media/show_ep.mp3"
    assert (tmp_path / "media" / "show_ep.mp3").is_file()


def test_resolve_audio_relpath_for_metadata(tmp_path: Path) -> None:
    media = tmp_path / "media" / "ep.mp3"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"ID3")
    resolved = resolve_audio_relpath_for_metadata(str(tmp_path), "transcripts/ep.txt")
    assert resolved == "media/ep.mp3"


def test_resolve_audio_relpath_finds_non_mp3_extension(tmp_path: Path) -> None:
    """Non-mp3 persisted audio (e.g. .m4a) must still resolve, not 404 (G1)."""
    media = tmp_path / "media" / "ep.m4a"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"ftyp")
    resolved = resolve_audio_relpath_for_metadata(str(tmp_path), "transcripts/ep.txt")
    assert resolved == "media/ep.m4a"


def test_persist_and_resolve_agree_on_non_mp3(tmp_path: Path) -> None:
    """What persist writes is exactly what the metadata resolver reports (G1)."""
    src = tmp_path / "dl.m4a"
    src.write_bytes(b"ftyp")
    rel = persist_episode_media(str(src), str(tmp_path), "transcripts/show_ep.txt")
    assert rel == "media/show_ep.m4a"
    assert resolve_audio_relpath_for_metadata(str(tmp_path), "transcripts/show_ep.txt") == rel


# --- G6: hardlink/symlink media modes ------------------------------------------------


def test_persist_default_mode_is_a_real_copy(tmp_path: Path) -> None:
    src = tmp_path / "dl.mp3"
    src.write_bytes(b"ID3")
    cache = tmp_path / "cache" / "audio.mp3"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"ID3")
    persist_episode_media(
        str(src), str(tmp_path), "transcripts/ep.txt", link_source=str(cache), link_mode="copy"
    )
    dest = tmp_path / "media" / "ep.mp3"
    assert dest.is_file() and not dest.is_symlink()
    # A copy has its own inode, distinct from the cache entry.
    assert dest.stat().st_ino != cache.stat().st_ino


def test_persist_hardlink_shares_inode_with_cache_entry(tmp_path: Path) -> None:
    src = tmp_path / "dl.mp3"
    src.write_bytes(b"ID3DATA")
    cache = tmp_path / "cache" / "audio.mp3"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"ID3DATA")  # same filesystem as tmp_path → hardlink-able
    rel = persist_episode_media(
        str(src), str(tmp_path), "transcripts/ep.txt", link_source=str(cache), link_mode="hardlink"
    )
    dest = tmp_path / "media" / "ep.mp3"
    assert rel == "media/ep.mp3"
    assert dest.stat().st_ino == cache.stat().st_ino  # one inode, no duplicated bytes


def test_persist_hardlink_falls_back_to_copy_without_link_source(tmp_path: Path) -> None:
    src = tmp_path / "dl.mp3"
    src.write_bytes(b"ID3")
    rel = persist_episode_media(
        str(src), str(tmp_path), "transcripts/ep.txt", link_source=None, link_mode="hardlink"
    )
    dest = tmp_path / "media" / "ep.mp3"
    assert rel == "media/ep.mp3"
    assert dest.is_file() and not dest.is_symlink()


def test_persist_symlink_to_in_corpus_target(tmp_path: Path) -> None:
    src = tmp_path / "dl.mp3"
    src.write_bytes(b"ID3")
    # Cache entry lives UNDER the corpus root → symlink is allowed.
    cache = tmp_path / ".podcast_scraper" / "audio-cache" / "audio.mp3"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"ID3")
    persist_episode_media(
        str(src), str(tmp_path), "transcripts/ep.txt", link_source=str(cache), link_mode="symlink"
    )
    dest = tmp_path / "media" / "ep.mp3"
    assert dest.is_symlink()
    assert Path(dest).resolve() == cache.resolve()
    # Relative link → the in-corpus snapshot survives a move/tar backup.
    assert not os.path.isabs(os.readlink(dest))


def test_persist_symlink_to_external_target_falls_back_to_copy(tmp_path: Path) -> None:
    """A symlink whose target is outside the corpus would 404 in the viewer → copy instead."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    src = tmp_path / "dl.mp3"
    src.write_bytes(b"ID3")
    external = tmp_path / "external_cache" / "audio.mp3"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"ID3")
    persist_episode_media(
        str(src), str(corpus), "transcripts/ep.txt", link_source=str(external), link_mode="symlink"
    )
    dest = corpus / "media" / "ep.mp3"
    assert dest.is_file() and not dest.is_symlink()
