"""Unit tests for the consumer catalog cache + O(1) slug index (perf remediation).

The cache must do two things and never a third: (1) serve repeated reads without re-walking the
corpus, (2) invalidate when the corpus changes (a stale catalog would silently hide new episodes),
and never (3) hand back a shared list a caller can mutate. resolve_slug must be O(1) over the
cached index and stay correct across an invalidation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import app_catalog_cache, app_slugs

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


def _set_corpus_stamp(root: Path, mtime: float) -> None:
    """Write/age the run-summary that ``perf_cache.corpus_mtime`` keys on (the ingest signal)."""
    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    os.utime(stamp, (mtime, mtime))


@pytest.fixture(autouse=True)
def _fresh_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


def _spy_build(monkeypatch) -> list[int]:
    """Count real catalog walks so a cache HIT is provable, not assumed."""
    calls = [0]
    real = app_catalog_cache.build_catalog_rows_cumulative

    def _counting(root: Path):
        calls[0] += 1
        return real(root)

    monkeypatch.setattr(app_catalog_cache, "build_catalog_rows_cumulative", _counting)
    return calls


def test_repeated_reads_walk_the_corpus_once(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _set_corpus_stamp(tmp_path, 1_000_000.0)
    calls = _spy_build(monkeypatch)

    first = app_catalog_cache.cached_catalog(tmp_path)
    for _ in range(5):
        app_catalog_cache.cached_catalog(tmp_path)

    assert len(first) == 1
    assert calls[0] == 1, "the catalog was re-walked on a cache hit"


def test_invalidates_when_the_corpus_changes(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _set_corpus_stamp(tmp_path, 1_000_000.0)
    calls = _spy_build(monkeypatch)

    assert len(app_catalog_cache.cached_catalog(tmp_path)) == 1
    assert calls[0] == 1

    # A new episode lands and the ingest stamp advances → the next read MUST reflect it.
    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")
    _set_corpus_stamp(tmp_path, 2_000_000.0)

    after = app_catalog_cache.cached_catalog(tmp_path)
    assert len(after) == 2, "a stale catalog hid the new episode"
    assert calls[0] == 2, "invalidation did not trigger a rebuild"


def test_returned_list_is_a_copy_callers_cannot_corrupt_the_cache(tmp_path: Path) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")
    _set_corpus_stamp(tmp_path, 1_000_000.0)

    first = app_catalog_cache.cached_catalog(tmp_path)
    first.clear()  # a caller mutating its copy (e.g. /discover sorts in place)
    second = app_catalog_cache.cached_catalog(tmp_path)
    assert len(second) == 2, "mutating a returned list corrupted the shared cache entry"


def _spy_build_last_run(monkeypatch) -> list[int]:
    """Count real last-run catalog walks so a cache HIT on cached_catalog_last_run is provable."""
    calls = [0]
    real = app_catalog_cache.build_catalog_rows

    def _counting(root: Path):
        calls[0] += 1
        return real(root)

    monkeypatch.setattr(app_catalog_cache, "build_catalog_rows", _counting)
    return calls


def test_last_run_repeated_reads_walk_the_corpus_once(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _set_corpus_stamp(tmp_path, 1_000_000.0)
    calls = _spy_build_last_run(monkeypatch)

    first = app_catalog_cache.cached_catalog_last_run(tmp_path)
    for _ in range(5):
        app_catalog_cache.cached_catalog_last_run(tmp_path)

    assert len(first) == 1
    assert calls[0] == 1, "the last-run catalog was re-walked on a cache hit"


def test_last_run_invalidates_when_the_corpus_changes(tmp_path: Path, monkeypatch) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _set_corpus_stamp(tmp_path, 1_000_000.0)
    calls = _spy_build_last_run(monkeypatch)

    assert len(app_catalog_cache.cached_catalog_last_run(tmp_path)) == 1
    assert calls[0] == 1

    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")
    _set_corpus_stamp(tmp_path, 2_000_000.0)

    after = app_catalog_cache.cached_catalog_last_run(tmp_path)
    assert len(after) == 2, "a stale last-run catalog hid the new episode"
    assert calls[0] == 2, "invalidation did not trigger a rebuild"


def test_last_run_uses_a_distinct_namespace_from_cumulative(tmp_path: Path) -> None:
    """The two variants must not share a cache entry — swapping them would change the row set."""
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _set_corpus_stamp(tmp_path, 1_000_000.0)

    # Prime the cumulative cache first; the last-run read must still compute its own entry, not
    # return the cumulative one, even though both key on the same root + mtime.
    app_catalog_cache.cached_catalog(tmp_path)
    last_run = app_catalog_cache.cached_catalog_last_run(tmp_path)
    last_run.clear()  # mutating this copy must not affect the cumulative entry
    assert len(app_catalog_cache.cached_catalog(tmp_path)) == 1


def test_resolve_slug_is_correct_and_survives_invalidation(tmp_path: Path) -> None:
    _write_episode(tmp_path, stem="0001", feed_id="f1", episode_id="e1")
    _set_corpus_stamp(tmp_path, 1_000_000.0)

    rows = app_catalog_cache.cached_catalog(tmp_path)
    slug = app_slugs.slug_for_row(rows[0])
    resolved = app_slugs.resolve_slug(tmp_path, slug)
    assert resolved is not None and resolved.episode_id == "e1"
    assert app_slugs.resolve_slug(tmp_path, "does-not-exist") is None

    # After an ingest, a newly-added episode's slug resolves through the refreshed index.
    _write_episode(tmp_path, stem="0002", feed_id="f1", episode_id="e2")
    _set_corpus_stamp(tmp_path, 2_000_000.0)
    new_row = next(r for r in app_catalog_cache.cached_catalog(tmp_path) if r.episode_id == "e2")
    new_slug = app_slugs.slug_for_row(new_row)
    assert app_slugs.resolve_slug(tmp_path, new_slug) is not None
