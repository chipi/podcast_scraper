"""Latest ``run_*`` selection under ``feeds/<dir>/`` (GitHub #580 follow-up)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.corpus_scope import (
    discover_all_metadata_files,
    discover_metadata_files,
    feed_dir_and_run_segment_from_relpath,
    latest_feed_run_allowed_relpaths,
)


def _write_episode(tmp_path: Path, run_seg: str, episode_id: str) -> Path:
    """Write a metadata doc for *episode_id* under feeds/pod/<run_seg>/metadata/."""
    doc = {
        "feed": {"feed_id": "f1", "title": "S"},
        "episode": {"episode_id": episode_id, "title": episode_id, "published_date": "2024-01-01"},
    }
    p = (
        tmp_path
        / "feeds"
        / "pod"
        / run_seg
        / "metadata"
        / f"0001 - {episode_id}_{run_seg}.metadata.json"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


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


def test_discover_metadata_files_reprocessed_episode_keeps_newest_run(tmp_path: Path) -> None:
    """Same episode reprocessed into a newer run → newest run wins (supersession, no duplicate).

    (The central rule unions across runs; a collision on ``(feed_id, episode_id)`` resolves to the
    newest run — here the two docs share ``e1``/``f1`` so only the newer run survives.)
    """
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


def test_distinct_episodes_split_across_runs_all_included(tmp_path: Path) -> None:
    """#877 / 94-vs-106 FIX: two DIFFERENT episodes scattered across run dirs — BOTH included.

    prod (append=false) scatters a feed's episodes across timestamped run dirs (incremental add).
    The central discovery rule now unions across all runs, so ``discover_metadata_files`` returns
    every distinct episode — it no longer under-counts by dropping older runs.
    """
    _write_episode(tmp_path, "run_20260416-000000_old", "e1")
    _write_episode(tmp_path, "run_20260417-000000_new", "e2")

    latest = discover_metadata_files(tmp_path)
    all_runs = discover_all_metadata_files(tmp_path)

    assert len(latest) == 2  # union across runs — both episodes
    assert len(all_runs) == 2  # both episodes are physically present


def test_catalog_latest_and_cumulative_converge_on_distinct_total(tmp_path: Path) -> None:
    """#877 FIX: build_catalog_rows shares the central union+dedup rule, so it matches cumulative.

    Both readers report the true distinct-episode total; the old latest-run-only undercount is gone.
    """
    from podcast_scraper.server.corpus_catalog import (
        build_catalog_rows,
        build_catalog_rows_cumulative,
    )

    _write_episode(tmp_path, "run_20260416-000000_old", "e1")
    _write_episode(tmp_path, "run_20260417-000000_new", "e2")

    assert len(build_catalog_rows(tmp_path)) == 2  # converged on the true distinct total
    assert len(build_catalog_rows_cumulative(tmp_path)) == 2  # true distinct total
