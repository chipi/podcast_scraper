"""Unit tests for the consumer catalog cache + O(1) slug index (perf remediation).

The cache must do two things and never a third: (1) serve repeated reads without reading the rows
again, (2) invalidate when the rows change (a stale catalog would silently hide new episodes), and
never (3) hand back a shared list a caller can mutate. resolve_slug must be O(1) over the cached
index and stay correct across an invalidation. The test corpus is written as files and imported
into the test database before the catalog reads it (``tests._corpus_rows``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import catalog_cache, slugs

pytestmark = [pytest.mark.unit]


def _write_episode(root: Path, *, stem: str, feed_id: str, episode_id: str) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": feed_id, "title": "Show"},
        "episode": {
            "episode_id": episode_id,
            "title": episode_id,
            "published_date": "2024-01-01T00:00:00",
        },
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def _spy_read(monkeypatch) -> list[int]:
    """Count real catalog reads of the rows so a cache HIT is provable, not assumed."""
    calls = [0]
    real = catalog_cache._catalog_rows

    def _counting(root: Path):
        calls[0] += 1
        return real(root)

    monkeypatch.setattr(catalog_cache, "_catalog_rows", _counting)
    return calls


def test_repeated_reads_read_the_rows_once(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    calls = _spy_read(monkeypatch)

    first = catalog_cache.cached_catalog(tmp_path)
    for _ in range(5):
        catalog_cache.cached_catalog(tmp_path)

    assert len(first) == 1
    assert calls[0] == 1, "the rows were read again on a cache hit"


def test_invalidates_when_the_rows_change(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    calls = _spy_read(monkeypatch)

    assert len(catalog_cache.cached_catalog(tmp_path)) == 1
    assert calls[0] == 1

    # A new episode's row lands and the row token moves: the next read gets the previous catalog
    # at once while the background builder rebuilds; reads after the rebuild reflect it.
    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")

    assert len(catalog_cache.cached_catalog(tmp_path)) == 1
    assert perf_cache.wait_for_rebuilds()
    after = catalog_cache.cached_catalog(tmp_path)
    assert len(after) == 2, "a stale catalog hid the new episode"
    assert calls[0] == 2, "invalidation did not trigger a rebuild"


def test_returned_list_is_a_copy_callers_cannot_corrupt_the_cache(tmp_path: Path) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")

    first = catalog_cache.cached_catalog(tmp_path)
    first.clear()  # a caller mutating its copy (e.g. /discover sorts in place)
    second = catalog_cache.cached_catalog(tmp_path)
    assert len(second) == 2, "mutating a returned list corrupted the shared cache entry"


def test_the_last_run_catalog_is_the_catalog_on_rows(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    calls = _spy_read(monkeypatch)

    assert catalog_cache.cached_catalog_last_run(tmp_path) == catalog_cache.cached_catalog(tmp_path)
    assert calls[0] == 1, "the last-run catalog read the rows a second time"


def test_resolve_slug_is_correct_and_survives_invalidation(tmp_path: Path) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")

    rows = catalog_cache.cached_catalog(tmp_path)
    slug = slugs.slug_for_row(rows[0])
    resolved = slugs.resolve_slug(tmp_path, slug)
    assert resolved is not None and resolved.episode_id == "e1"
    assert slugs.resolve_slug(tmp_path, "does-not-exist") is None

    # After an ingest, a newly-added episode's slug resolves through the refreshed index.
    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")
    catalog_cache.cached_catalog(tmp_path)
    slugs.resolve_slug(tmp_path, slug)
    assert perf_cache.wait_for_rebuilds()
    new_row = next(r for r in catalog_cache.cached_catalog(tmp_path) if r.episode_id == "e2")
    new_slug = slugs.slug_for_row(new_row)
    assert slugs.resolve_slug(tmp_path, new_slug) is not None


def test_the_token_is_read_once_in_its_hold_and_every_time_inside_fresh(monkeypatch) -> None:
    """A request that reads the catalog several times pays one token query (the Player's
    round-trip budgets); the operator's current reads (``fresh()``) never reuse a held token."""
    reads = [0]

    def read() -> float:
        reads[0] += 1
        return float(reads[0])

    clock = [100.0]
    monkeypatch.setattr(catalog_cache.time, "monotonic", lambda: clock[0])
    hold = catalog_cache._TokenHold(read, 2.0)

    assert [hold(), hold(), hold()] == [1.0, 1.0, 1.0]
    assert reads[0] == 1
    with perf_cache.fresh():
        assert hold() == 2.0
        assert hold() == 3.0
    clock[0] += 1.9
    assert hold() == 3.0
    clock[0] += 0.2
    assert hold() == 4.0


def test_a_settled_block_reads_the_token_once_and_serves_the_cached_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    """``corpus_rows_settled``: the API's own functions, so a measured request pays no token query
    the API would not (the Player's round-trip budgets)."""
    from tests._corpus_rows import corpus_rows_settled

    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    assert len(catalog_cache.cached_catalog(tmp_path)) == 1
    reads = [0]
    real = catalog_cache._read_token

    def _counting() -> float:
        reads[0] += 1
        return real()

    monkeypatch.setattr(catalog_cache, "_read_token", _counting)
    with corpus_rows_settled():
        for _ in range(3):
            assert len(catalog_cache.cached_catalog(tmp_path)) == 1
    assert reads[0] == 1, "a settled block read the token more than once"
