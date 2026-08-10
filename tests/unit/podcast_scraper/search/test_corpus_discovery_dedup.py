"""Central corpus-discovery rule: union across ALL runs, one winner per ``(feed_id, episode_id)``.

Regression guard for the 94-vs-106 index bug (Planet Money): the old
``discover_metadata_files`` kept only the lexicographically-greatest ``run_*`` dir per feed, so an
incremental add (a NEW run dir holding only the new episode) made the indexer drop the feed's prior
run's episodes. The rule is now: discover every run's metadata, and when the SAME episode is
reprocessed into a newer run, the NEWEST run wins — newest = run-folder timestamp
(``run_YYYYMMDD-HHMMSS_*``), with file-mtime fallback only for timestamp-less ``run_append_*`` dirs.

Both production operations depend on this single rule: the incremental add (union — disjoint
episodes across runs all survive) and the full reindex (dedup — a reprocess supersedes its trophy).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from podcast_scraper.search.corpus_scope import discover_metadata_files


def _write_ep(
    corpus: Path,
    feed_dir: str,
    run_seg: str,
    filename: str,
    feed_id: str,
    episode_id: str,
    *,
    mtime: float | None = None,
) -> Path:
    meta_dir = corpus / "feeds" / feed_dir / run_seg / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    p = meta_dir / f"{filename}.metadata.json"
    p.write_text(
        json.dumps(
            {
                "feed": {"feed_id": feed_id},
                "episode": {"episode_id": episode_id, "title": filename},
            }
        ),
        encoding="utf-8",
    )
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _episode_ids(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        eid = (doc.get("episode") or {}).get("episode_id")
        if eid:
            out.add(eid)
    return out


def _run_seg_of(path: Path) -> str:
    for part in path.as_posix().split("/"):
        if part.startswith("run_"):
            return part
    return ""


def test_incremental_add_unions_across_runs(tmp_path: Path) -> None:
    """A new run dir holding only the new episode must NOT drop the feed's prior run."""
    corpus = tmp_path / "corpus"
    feed = "rss_example_abc123"
    _write_ep(corpus, feed, "run_20260101-000000_h1", "0001 - A", "F", "E1")
    _write_ep(corpus, feed, "run_20260101-000000_h1", "0002 - B", "F", "E2")
    _write_ep(corpus, feed, "run_20260101-000000_h1", "0003 - C", "F", "E3")
    # incremental add: brand-new run dir, single new episode
    _write_ep(corpus, feed, "run_20260102-000000_h1", "0001 - D", "F", "E4")

    found = discover_metadata_files(corpus)
    assert _episode_ids(found) == {"E1", "E2", "E3", "E4"}


def test_reprocess_supersedes_older_run_newest_wins(tmp_path: Path) -> None:
    """Same episode reprocessed into a newer run: newest run wins, trophy stays dead (no dup)."""
    corpus = tmp_path / "corpus"
    feed = "rss_example_abc123"
    _write_ep(corpus, feed, "run_20260101-000000_h1", "0001 - old", "F", "E1")
    _write_ep(corpus, feed, "run_20260105-000000_h1", "0001 - new", "F", "E1")

    found = discover_metadata_files(corpus)
    assert _episode_ids(found) == {"E1"}
    assert len(found) == 1
    assert _run_seg_of(found[0]) == "run_20260105-000000_h1"


def test_newest_by_run_timestamp_not_lexicographic_append_landmine(tmp_path: Path) -> None:
    """``run_append_*`` sorts lexicographically after every ``run_<ts>`` (``a`` > ``2``).

    Under the old lexicographic rule the append dir would ALWAYS win. Newest must be decided by
    real recency: an OLDER append copy must LOSE to a NEWER timestamped reprocess.
    """
    corpus = tmp_path / "corpus"
    feed = "rss_example_abc123"
    old_append_mtime = datetime(2026, 1, 1).timestamp()
    _write_ep(corpus, feed, "run_append_h1", "0001 - appended", "F", "E1", mtime=old_append_mtime)
    _write_ep(corpus, feed, "run_20260105-000000_h1", "0001 - reprocessed", "F", "E1")

    found = discover_metadata_files(corpus)
    assert len(found) == 1
    assert _run_seg_of(found[0]) == "run_20260105-000000_h1"


def test_newer_append_beats_older_timestamped(tmp_path: Path) -> None:
    """Converse: a NEWER append copy (by mtime) supersedes an OLDER timestamped run."""
    corpus = tmp_path / "corpus"
    feed = "rss_example_abc123"
    new_append_mtime = datetime(2026, 1, 10).timestamp()
    _write_ep(corpus, feed, "run_20260105-000000_h1", "0001 - old", "F", "E1")
    _write_ep(corpus, feed, "run_append_h1", "0001 - fresh", "F", "E1", mtime=new_append_mtime)

    found = discover_metadata_files(corpus)
    assert len(found) == 1
    assert _run_seg_of(found[0]) == "run_append_h1"


def test_two_feeds_disjoint_all_kept(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_ep(corpus, "rss_a_111", "run_20260101-000000_h1", "0001 - A", "FA", "E1")
    _write_ep(corpus, "rss_b_222", "run_20260101-000000_h1", "0001 - B", "FB", "E2")

    found = discover_metadata_files(corpus)
    assert _episode_ids(found) == {"E1", "E2"}


def test_planet_money_shape_12_plus_1(tmp_path: Path) -> None:
    """The exact prod shape: a 12-episode batch run + a 1-episode incremental add → 13, all kept."""
    corpus = tmp_path / "corpus"
    feed = "rss_feeds.npr.org_7ce5b183"
    for i in range(1, 13):
        _write_ep(corpus, feed, "run_20260805-175034_7a69fc41", f"{i:04d} - ep", "PM", f"E{i}")
    _write_ep(corpus, feed, "run_20260810-120000_8c97d853", "0001 - older workers", "PM", "E13")

    found = discover_metadata_files(corpus)
    assert _episode_ids(found) == {f"E{i}" for i in range(1, 14)}
