"""Latest ``run_*`` selection under ``feeds/<dir>/`` (GitHub #580 follow-up)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    feed_dir_and_run_segment_from_relpath,
    latest_feed_run_allowed_relpaths,
)


def test_feed_dir_and_run_segment_parses_nested_layout() -> None:
    rel = "feeds/pod_x/run_20260417-010000_abc/metadata/ep.metadata.json"
    assert feed_dir_and_run_segment_from_relpath(rel) == ("pod_x", "run_20260417-010000_abc")


def test_feed_dir_and_run_segment_rejects_non_run_layout() -> None:
    rel = "metadata/ep.metadata.json"
    assert feed_dir_and_run_segment_from_relpath(rel) == (None, None)


def test_latest_feed_run_allowed_relpaths_keeps_lexicographic_max() -> None:
    rels = [
        "feeds/pod/run_20260416-000000_a/metadata/1.metadata.json",
        "feeds/pod/run_20260417-000000_b/metadata/1.metadata.json",
        "search/index.json",
    ]
    allowed = latest_feed_run_allowed_relpaths(rels)
    assert "search/index.json" in allowed
    assert "feeds/pod/run_20260416-000000_a/metadata/1.metadata.json" not in allowed
    assert "feeds/pod/run_20260417-000000_b/metadata/1.metadata.json" in allowed


def test_discover_metadata_files_keeps_only_latest_run(tmp_path: Path) -> None:
    doc = {
        "feed": {"feed_id": "f1", "title": "S"},
        "episode": {"episode_id": "e1", "title": "T", "published_date": "2024-01-01"},
    }
    old = (
        tmp_path
        / "feeds"
        / "pod"
        / "run_20260416-000000_old"
        / "metadata"
        / "0001 - Old_20260416-000000_old.metadata.json"
    )
    new = (
        tmp_path
        / "feeds"
        / "pod"
        / "run_20260417-000000_new"
        / "metadata"
        / "0001 - New_20260417-000000_new.metadata.json"
    )
    old.parent.mkdir(parents=True, exist_ok=True)
    new.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps(doc), encoding="utf-8")
    new.write_text(json.dumps(doc), encoding="utf-8")

    paths = discover_metadata_files(tmp_path)
    assert len(paths) == 1
    assert paths[0].resolve() == new.resolve()
