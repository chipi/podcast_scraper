"""Unit tests for corpus media helpers."""

from __future__ import annotations

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
