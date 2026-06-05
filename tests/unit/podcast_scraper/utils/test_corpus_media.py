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
