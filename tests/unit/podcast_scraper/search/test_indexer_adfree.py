"""#974: search / enrich-edges resolve the ad-free processing base when present."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search.indexer import _transcript_path

pytestmark = pytest.mark.unit


def _doc(rel: str) -> dict:
    return {"content": {"transcript_file_path": rel}}


def test_prefers_adfree_sibling_when_present(tmp_path: Path) -> None:
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "ep.txt").write_text("raw with ads", encoding="utf-8")
    (tdir / "ep.adfree.txt").write_text("ad free", encoding="utf-8")

    resolved = _transcript_path(tmp_path, _doc("transcripts/ep.txt"))
    assert resolved is not None
    assert resolved.name == "ep.adfree.txt"


def test_falls_back_to_raw_when_no_adfree(tmp_path: Path) -> None:
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "ep.txt").write_text("raw only", encoding="utf-8")

    resolved = _transcript_path(tmp_path, _doc("transcripts/ep.txt"))
    assert resolved is not None
    assert resolved.name == "ep.txt"


def test_none_when_nothing_exists(tmp_path: Path) -> None:
    assert _transcript_path(tmp_path, _doc("transcripts/missing.txt")) is None
    assert _transcript_path(tmp_path, _doc("")) is None
