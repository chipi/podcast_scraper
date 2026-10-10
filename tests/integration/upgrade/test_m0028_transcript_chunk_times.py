"""0028: the transcript rows already in the search index get the time they were said, in place.

A real LanceDB index shaped like prod's: transcript rows at 0 ms, the sidecar holding each row's
char span, the timed transcript on disk. No embedding model and no re-index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backend import SegmentDocument  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from podcast_scraper.search.indexer import _collect_docs_for_episode  # noqa: E402
from podcast_scraper.upgrade.migration import MigrationContext  # noqa: E402
from podcast_scraper.upgrade.migrations import m0028_transcript_chunk_times as m0028  # noqa: E402
from podcast_scraper.upgrade.registry import get_migrations  # noqa: E402

DIM = 4
SENTENCES = [f"Sentence number {i} says something about risk and time." for i in range(60)]


def _corpus(root: Path) -> list:
    run = root / "feeds" / "f" / "run_1"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    body = "\n\n".join(SENTENCES)
    (run / "transcripts" / "ep.txt").write_text(body, encoding="utf-8")
    segs = [{"start": 10.0 * i, "end": 10.0 * i + 8.0, "text": s} for i, s in enumerate(SENTENCES)]
    (run / "transcripts" / "ep.segments.json").write_text(json.dumps(segs), encoding="utf-8")
    doc = {
        "feed": {"feed_id": "f", "title": "F", "language": "en"},
        "episode": {"episode_id": "ep1", "title": "T", "published_date": "2026-06-15T00:00:00Z"},
        "content": {"transcript_file_path": "transcripts/ep.txt"},
    }
    meta_rel = "feeds/f/run_1/metadata/ep.metadata.json"
    (root / meta_rel).write_text(json.dumps(doc), encoding="utf-8")
    rows = [
        (doc_id, text, meta)
        for doc_id, text, meta in _collect_docs_for_episode(
            run,
            root / meta_rel,
            doc,
            target_tokens=60,
            overlap_tokens=0,
            metadata_relative_path=meta_rel,
        )
        if meta.get("doc_type") == "transcript"
    ]
    # What prod holds: every transcript row at 0 ms, in the index and in its sidecar.
    sidecar = {}
    for doc_id, _text, meta in rows:
        sidecar[doc_id] = {**meta, "timestamp_start_ms": None, "timestamp_end_ms": None}
    lance = root / "search" / "lance_index"
    b = LanceDBBackend(str(lance), embed_dim=DIM)
    b.upsert_segments(
        [
            SegmentDocument(
                id=doc_id,
                text=text,
                show_id="f",
                episode_id="ep1",
                start_time=0.0,
                end_time=0.0,
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            for doc_id, text, _meta in rows
        ]
    )
    b.create_indices()  # prod's index is built: the rewrite must keep keyword search working
    (lance.parent / "metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return rows


def _times(root: Path) -> dict:
    table = LanceDBBackend(str(root / "search" / "lance_index"), embed_dim=DIM)._open_if_exists(
        "segment"
    )
    n = table.count_rows()
    return {
        r["id"]: (r["start_time"], r["end_time"])
        for r in table.search().select(["id", "start_time", "end_time"]).limit(n).to_list()
    }


def _ctx(root: Path, dry_run: bool = False) -> MigrationContext:
    return MigrationContext(corpus_root=root, dry_run=dry_run)


def test_dry_run_counts_and_writes_nothing(tmp_path: Path) -> None:
    rows = _corpus(tmp_path)
    res = m0028.TranscriptChunkTimesMigration().apply(_ctx(tmp_path, dry_run=True))
    assert res.details["timed"] == len(rows)
    assert set(_times(tmp_path).values()) == {(0.0, 0.0)}


def test_every_row_gets_the_time_the_indexer_now_writes(tmp_path: Path) -> None:
    rows = _corpus(tmp_path)
    mig = m0028.TranscriptChunkTimesMigration()
    assert mig.verify(_ctx(tmp_path))[0] is False
    mig.apply(_ctx(tmp_path))
    got = _times(tmp_path)
    for doc_id, _text, meta in rows:
        # The indexer, fixed, computes the same chunk's time from the same segments.
        assert got[doc_id] == (
            meta["timestamp_start_ms"] / 1000.0,
            meta["timestamp_end_ms"] / 1000.0,
        ), doc_id
    assert max(s for s, _ in got.values()) > 0
    sidecar = json.loads((tmp_path / "search" / "metadata.json").read_text())
    assert all(sidecar[d]["timestamp_start_ms"] == m["timestamp_start_ms"] for d, _, m in rows)
    assert mig.verify(_ctx(tmp_path))[0] is True
    # Idempotent: nothing is left to time.
    assert mig.apply(_ctx(tmp_path)).details["timed"] == 0


def test_a_transcript_changed_since_indexing_is_left_alone(tmp_path: Path) -> None:
    rows = _corpus(tmp_path)
    body = tmp_path / "feeds" / "f" / "run_1" / "transcripts" / "ep.txt"
    body.write_text("Inserted at the top. " + body.read_text(), encoding="utf-8")
    res = m0028.TranscriptChunkTimesMigration().apply(_ctx(tmp_path))
    assert res.details["text_mismatch"] == len(rows)
    assert set(_times(tmp_path).values()) == {(0.0, 0.0)}


def test_undo_puts_every_row_and_the_sidecar_back(tmp_path: Path) -> None:
    _corpus(tmp_path)
    before = (tmp_path / "search" / "metadata.json").read_text()
    m0028.TranscriptChunkTimesMigration().apply(_ctx(tmp_path))
    restored, errors = m0028.undo(tmp_path)
    assert errors == [] and restored > 0
    assert set(_times(tmp_path).values()) == {(0.0, 0.0)}
    assert json.loads((tmp_path / "search" / "metadata.json").read_text()) == json.loads(before)


def test_registered_after_0027() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0028_transcript_chunk_times") == ids.index("0027_missing_covers_stored") + 1


def test_keyword_search_still_finds_the_rewritten_rows(tmp_path: Path) -> None:
    from podcast_scraper.search.backend import SearchQuery

    _corpus(tmp_path)
    m0028.TranscriptChunkTimesMigration().apply(_ctx(tmp_path))
    b = LanceDBBackend(str(tmp_path / "search" / "lance_index"), embed_dim=DIM)
    hits = b.search_bm25(
        SearchQuery(text="Sentence number 42", embedding=[1.0, 0.0, 0.0, 0.0], k=5, tier="segment")
    )
    assert hits and float(hits[0].payload["start_time"]) > 0
