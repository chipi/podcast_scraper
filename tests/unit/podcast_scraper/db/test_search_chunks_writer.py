"""The index build's transcript chunks as search_chunks rows (db.search_chunks_writer, r-search)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import engines, search_chunks_writer
from podcast_scraper.db.corpus_keys import source_natural_id
from podcast_scraper.db.corpus_writer import RETRANSCRIBED, sync_documents
from podcast_scraper.db.importers.corpus_documents import transcript_files
from podcast_scraper.db.schema import (
    content_items,
    corpus_metadata,
    corpus_revision,
    search_chunks,
    segments,
    sources,
)
from podcast_scraper.db.search_chunks_writer import sync_episode_chunks
from podcast_scraper.db.settings import ENV_PIPELINE_RW_URL, normalize_url
from podcast_scraper.search.indexer import _collect_docs_for_episode, chunked_bodies
from tests.unit.podcast_scraper.db.test_db_corpus import conn, url  # noqa: F401

BASE = "0001 - Episode"
BODY = "".join(f"Sentence number {n} of the episode, with a few more words.\n" for n in range(40))
FEED = "sha256:" + "cd" * 32


def _doc() -> dict[str, Any]:
    return {
        "feed": {"feed_id": FEED, "title": "A show", "language": "en"},
        "episode": {"episode_id": "item:sc", "title": "E", "language": "en"},
        "content": {"transcript_file_path": f"transcripts/{BASE}.txt"},
    }


def _episode(root: Path, body: str = BODY) -> tuple[Path, Path, dict[str, Any]]:
    run = root / "feeds" / "show" / "run_20261101-000000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(exist_ok=True)
    (run / "transcripts" / f"{BASE}.txt").write_text(body, encoding="utf-8")
    meta = run / "metadata" / f"{BASE}.metadata.json"
    doc = _doc()
    meta.write_text(json.dumps(doc), encoding="utf-8")
    return run, meta, doc


@pytest.fixture
def db(url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:  # noqa: F811
    monkeypatch.setenv(ENV_PIPELINE_RW_URL, url)
    engines.reset_engines()
    search_chunks_writer.OUTCOMES.clear()
    engine = create_engine(normalize_url(url))
    yield engine
    with engine.begin() as c:
        for table in reversed(corpus_metadata.sorted_tables):
            if table is not corpus_revision:
                c.execute(table.delete())
    engine.dispose()
    engines.reset_engines()


def _item_with_documents(engine: Any, run: Path, doc: dict[str, Any]) -> int:
    with engine.begin() as c:
        source_id = c.execute(
            insert(sources)
            .values(natural_id=source_natural_id(doc), content_type="podcast_episode", title="A")
            .returning(sources.c.source_id)
        ).scalar_one()
        item_id: int = c.execute(
            insert(content_items)
            .values(
                natural_id="item:sc",
                slug="sc",
                source_id=source_id,
                external_item_id="ext-sc",
                content_type="podcast_episode",
                title="E",
            )
            .returning(content_items.c.item_id)
        ).scalar_one()
        files = transcript_files(run / "transcripts", BASE, "en")
        sync_documents(c, item_id, files, change_kind=RETRANSCRIBED)
    return item_id


def _chunks(run: Path, meta: Path, doc: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    rows = _collect_docs_for_episode(
        run, meta, doc, target_tokens=40, overlap_tokens=8, metadata_relative_path="m"
    )
    transcript = [r for r in rows if r[2].get("doc_type") == "transcript"]
    return rows, transcript


def _range_text(c: Connection, row: Any) -> str:
    cuts: list[str] = [
        r[0]
        for r in c.execute(
            select(segments.c.text)
            .where(segments.c.document_id == row["document_id"])
            .where(segments.c.seq.between(row["start_seq"], row["end_seq"]))
            .order_by(segments.c.seq)
        )
    ]
    if len(cuts) == 1:
        return cuts[0][row["start_offset"] : row["end_offset"]]
    return cuts[0][row["start_offset"] :] + "".join(cuts[1:-1]) + cuts[-1][: row["end_offset"]]


def test_each_chunk_is_a_row_whose_range_is_its_text(
    db: Any, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    run, meta, doc = _episode(tmp_path)
    item_id = _item_with_documents(db, run, doc)
    rows, transcript = _chunks(run, meta, doc)
    assert len(transcript) > 3
    assert sync_episode_chunks(doc, chunked_bodies(run, meta, doc, rows), "en") == "written"
    stored = {
        r["natural_id"]: r
        for r in conn.execute(select(search_chunks).where(search_chunks.c.item_id == item_id))
        .mappings()
        .all()
    }
    assert set(stored) == {r[0] for r in transcript}
    for natural_id, chunk_text, _ in transcript:
        assert _range_text(conn, stored[natural_id]) == chunk_text, natural_id
        assert stored[natural_id]["speakers"] == []


def test_a_second_sync_of_the_same_chunks_changes_nothing(
    db: Any, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    run, meta, doc = _episode(tmp_path)
    _item_with_documents(db, run, doc)
    rows, _ = _chunks(run, meta, doc)
    bodies = chunked_bodies(run, meta, doc, rows)
    assert sync_episode_chunks(doc, bodies, "en") == "written"
    first = conn.execute(select(search_chunks.c.chunk_id)).scalars().all()
    conn.rollback()
    assert sync_episode_chunks(doc, bodies, "en") == "unchanged"
    assert conn.execute(select(search_chunks.c.chunk_id)).scalars().all() == first


def test_a_body_the_pipeline_has_not_synced_is_stale_and_not_cut(
    db: Any, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    run, meta, doc = _episode(tmp_path)
    _item_with_documents(db, run, doc)
    (run / "transcripts" / f"{BASE}.txt").write_text(BODY + "A newer line.\n", encoding="utf-8")
    rows, _ = _chunks(run, meta, doc)
    assert sync_episode_chunks(doc, chunked_bodies(run, meta, doc, rows), "en") == "stale"
    assert conn.execute(text("SELECT count(*) FROM corpus.search_chunks")).scalar_one() == 0


def _stored(c: Connection, item_id: int) -> list[tuple[Any, ...]]:
    return sorted(
        tuple(r)
        for r in c.execute(
            select(
                search_chunks.c.natural_id,
                search_chunks.c.document_id,
                search_chunks.c.start_seq,
                search_chunks.c.start_offset,
                search_chunks.c.end_seq,
                search_chunks.c.end_offset,
            ).where(search_chunks.c.item_id == item_id)
        )
    )


def test_the_import_writes_the_builds_rows_in_the_callers_transaction(
    db: Any, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    from podcast_scraper.search.two_tier_indexer import import_search_chunks

    run, meta, doc = _episode(tmp_path)
    item_id = _item_with_documents(db, run, doc)
    # A second run of a show the pipeline has not written: no item row.
    other = tmp_path / "feeds" / "other" / "run_20261101-000000"
    (other / "metadata").mkdir(parents=True)
    unsynced = {**_doc(), "episode": {"episode_id": "item:none", "title": "N", "language": "en"}}
    (other / "metadata" / f"{BASE}.metadata.json").write_text(json.dumps(unsynced), "utf-8")
    report = import_search_chunks(conn, tmp_path, target_tokens=40, overlap_tokens=8)
    assert report == {"written": 1, "no_item_row": 1}
    imported = _stored(conn, item_id)
    conn.rollback()
    # A dry run: the caller rolls it back, nothing is left.
    assert _stored(conn, item_id) == []
    rows, transcript = _chunks(run, meta, doc)
    assert sync_episode_chunks(doc, chunked_bodies(run, meta, doc, rows), "en") == "written"
    # The same rows, ids included, as the index build writes.
    assert imported == _stored(conn, item_id)
    assert {r[0] for r in imported} == {r[0] for r in transcript}
    assert import_search_chunks(conn, tmp_path, target_tokens=40, overlap_tokens=8) == {
        "unchanged": 1,
        "no_item_row": 1,
    }
    conn.rollback()


def test_the_import_chunks_as_the_pipelines_build_by_default(
    db: Any, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # indexer.index_corpus builds with the config's chunk sizes (300 / 50), not this module's
    # 256 / 32: a different size gives every chunk:<scope>:<n> another span of the body.
    from podcast_scraper import config
    from podcast_scraper.search.two_tier_indexer import import_search_chunks

    cfg = config.Config()
    body = "".join(f"Word{n} " * 7 + "end.\n" for n in range(150))
    run, meta, doc = _episode(tmp_path, body)
    item_id = _item_with_documents(db, run, doc)
    assert import_search_chunks(conn, tmp_path) == {"written": 1}
    imported = _stored(conn, item_id)
    conn.rollback()
    rows = _collect_docs_for_episode(
        run,
        meta,
        doc,
        target_tokens=cfg.vector_chunk_size_tokens,
        overlap_tokens=cfg.vector_chunk_overlap_tokens,
        metadata_relative_path="m",
    )
    assert sync_episode_chunks(doc, chunked_bodies(run, meta, doc, rows), "en") == "written"
    assert len(imported) > 2 and imported == _stored(conn, item_id)


def test_the_import_leaves_out_an_unusable_episode(
    db: Any, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    from podcast_scraper.languages import UNUSABLE_FIELD
    from podcast_scraper.search.two_tier_indexer import import_search_chunks

    run, meta, doc = _episode(tmp_path)
    item_id = _item_with_documents(db, run, doc)
    doc["episode"][UNUSABLE_FIELD] = {"reason": "silence"}
    meta.write_text(json.dumps(doc), encoding="utf-8")
    assert import_search_chunks(conn, tmp_path) == {}
    assert _stored(conn, item_id) == []
    conn.rollback()


def test_an_episode_the_pipeline_has_not_written_has_no_rows(db: Any, tmp_path: Path) -> None:
    run, meta, doc = _episode(tmp_path)
    rows, _ = _chunks(run, meta, doc)
    assert sync_episode_chunks(doc, chunked_bodies(run, meta, doc, rows), "en") == "no_item_row"


def test_without_a_pipeline_database_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_PIPELINE_RW_URL, raising=False)
    engines.reset_engines()
    run, meta, doc = _episode(tmp_path)
    rows, _ = _chunks(run, meta, doc)
    assert sync_episode_chunks(doc, chunked_bodies(run, meta, doc, rows), "en") == "not_configured"
