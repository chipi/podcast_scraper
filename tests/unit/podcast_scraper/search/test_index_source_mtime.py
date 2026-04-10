"""Unit tests for index fingerprint source mtimes (GitHub #507)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search import index_source_mtime as index_source_mtime_mod
from podcast_scraper.search.index_source_mtime import (
    invalidate_newest_index_source_mtime_cache,
    newest_index_source_mtime_epoch,
)


def _write_meta(root: Path, name: str, doc: dict) -> Path:
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True)
    p = meta_dir / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_newest_index_source_mtime_metadata_only(tmp_path: Path) -> None:
    doc = {"episode": {"episode_id": "e1"}, "feed": {}}
    meta = _write_meta(tmp_path, "ep1.metadata.json", doc)
    got = newest_index_source_mtime_epoch(tmp_path)
    assert got is not None
    assert abs(got - meta.stat().st_mtime) < 0.01


def test_newest_index_source_mtime_includes_gi_kg_transcript(tmp_path: Path) -> None:
    doc = {
        "episode": {"episode_id": "e1"},
        "feed": {},
        "content": {"transcript_file_path": "t.txt"},
    }
    _write_meta(tmp_path, "ep1.metadata.json", doc)
    gi = tmp_path / "metadata" / "ep1.gi.json"
    gi.write_text("{}", encoding="utf-8")
    kg = tmp_path / "metadata" / "ep1.kg.json"
    kg.write_text('{"nodes":[]}', encoding="utf-8")
    tr = tmp_path / "t.txt"
    tr.write_text("hello", encoding="utf-8")

    got = newest_index_source_mtime_epoch(tmp_path)
    assert got is not None
    assert got >= max(
        tr.stat().st_mtime,
        gi.stat().st_mtime,
        kg.stat().st_mtime,
    )


def test_newest_index_source_mtime_empty_corpus(tmp_path: Path) -> None:
    assert newest_index_source_mtime_epoch(tmp_path) is None


def test_newest_index_source_mtime_ttl_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_source_mtime_mod._cache.clear()
    calls = {"n": 0}

    def fake_compute(_root: Path) -> float:
        calls["n"] += 1
        return 42.0

    monkeypatch.setattr(
        index_source_mtime_mod,
        "_compute_newest_index_source_mtime_epoch",
        fake_compute,
    )
    monkeypatch.setattr(index_source_mtime_mod.time, "monotonic", lambda: 100.0)
    assert newest_index_source_mtime_epoch(tmp_path) == 42.0
    assert calls["n"] == 1
    assert newest_index_source_mtime_epoch(tmp_path) == 42.0
    assert calls["n"] == 1
    monkeypatch.setattr(index_source_mtime_mod.time, "monotonic", lambda: 200.0)
    assert newest_index_source_mtime_epoch(tmp_path) == 42.0
    assert calls["n"] == 2


def test_invalidate_newest_index_source_mtime_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_source_mtime_mod._cache.clear()
    calls = {"n": 0}

    def fake_compute(_root: Path) -> float:
        calls["n"] += 1
        return 1.0

    monkeypatch.setattr(
        index_source_mtime_mod,
        "_compute_newest_index_source_mtime_epoch",
        fake_compute,
    )
    monkeypatch.setattr(index_source_mtime_mod.time, "monotonic", lambda: 0.0)
    newest_index_source_mtime_epoch(tmp_path)
    invalidate_newest_index_source_mtime_cache(str(tmp_path))
    newest_index_source_mtime_epoch(tmp_path)
    assert calls["n"] == 2
