"""The pipeline's transcript documents and segments in the corpus database (db.corpus_writer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection

from podcast_scraper.db import corpus_writer, engines
from podcast_scraper.db.corpus_writer import (
    ENV_STRICT,
    OUTCOMES,
    RETRANSCRIBED,
    sync_documents,
    sync_episode,
    SyncFailed,
)
from podcast_scraper.db.importers.corpus_documents import joined_body, transcript_files
from podcast_scraper.db.settings import ENV_PIPELINE_RW_URL
from tests.unit.podcast_scraper.db.test_db_corpus import _item, _source, conn, url  # noqa: F401

BASE = "0001 - An episode_abc"


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_bytes(body.encode("utf-8"))
    return path


def _files(directory: Path) -> list:
    return transcript_files(directory, BASE, "en")


def _versions(db: Connection, item: int, variant: str = "raw") -> list[tuple]:
    rows = db.execute(
        text(
            "SELECT version, change_kind, supersedes_id IS NOT NULL FROM corpus.documents"
            " WHERE item_id = :i AND variant = :v ORDER BY version"
        ),
        {"i": item, "v": variant},
    ).all()
    return [tuple(row) for row in rows]


def test_a_first_sync_writes_version_one_and_its_segments(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    body = "Welcome back.\r\nToday, trails.\n"
    _write(tmp_path, BASE + ".txt", body)
    item = _item(conn, _source(conn))
    [synced] = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert (synced.variant, synced.version, synced.segments, synced.written) == ("raw", 1, 2, True)
    assert joined_body(conn, item, synced.document_id) == body
    assert _versions(conn, item) == [(1, "created", False)]
    conn.rollback()


def test_the_same_body_writes_nothing(conn: Connection, tmp_path: Path) -> None:  # noqa: F811
    _write(tmp_path, BASE + ".txt", "One line.\n")
    item = _item(conn, _source(conn))
    first = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    again = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert [d.written for d in again] == [False]
    assert again[0].document_id == first[0].document_id
    assert _versions(conn, item) == [(1, "created", False)]
    conn.rollback()


def test_a_new_body_supersedes_and_the_old_segments_stay(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, BASE + ".txt", "Old words.\n")
    item = _item(conn, _source(conn))
    [old] = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    _write(tmp_path, BASE + ".txt", "New words.\nAnd more.\n")
    [new] = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert (new.version, new.written) == (2, True)
    assert _versions(conn, item) == [(1, "created", False), (2, "retranscribed", True)]
    assert joined_body(conn, item, old.document_id) == "Old words.\n"
    assert joined_body(conn, item, new.document_id) == "New words.\nAnd more.\n"
    conn.rollback()


def test_an_unknown_cause_is_refused(conn: Connection, tmp_path: Path) -> None:  # noqa: F811
    _write(tmp_path, BASE + ".txt", "x\n")
    with pytest.raises(ValueError, match="change_kind"):
        sync_documents(conn, 1, _files(tmp_path), change_kind="created")


def _metadata(directory: Path, episode_id: str = "item:1") -> Path:
    path = directory / f"{BASE}.metadata.json"
    doc = {
        "feed": {"feed_id": "sha256:" + "ab" * 32, "title": "A show"},
        "episode": {"episode_id": episode_id, "title": "E", "language": "en"},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def pipeline_db(url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # noqa: F811
    monkeypatch.setenv(ENV_PIPELINE_RW_URL, url)
    engines.reset_engines()
    OUTCOMES.clear()
    yield
    engines.reset_engines()


def test_with_no_database_configured_nothing_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_PIPELINE_RW_URL, raising=False)
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    assert sync_episode(_metadata(tmp_path), transcript) == "not_configured"


def test_an_episode_with_no_item_row_is_counted(
    pipeline_db: None, tmp_path: Path  # noqa: F811
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    assert sync_episode(_metadata(tmp_path, "item:none"), transcript) == "no_item_row"
    assert OUTCOMES["no_item_row"] == 1


def test_an_episode_with_a_row_is_written_then_unchanged(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    item = _item(conn, _source(conn, "source:w"), "w1")
    conn.commit()
    try:
        transcript = _write(tmp_path, BASE + ".txt", "Hello.\n")
        meta = _metadata(tmp_path, "item:w1")
        assert sync_episode(meta, transcript) == "written"
        assert sync_episode(meta, transcript) == "unchanged"
        assert _versions(conn, item) == [(1, "created", False)]
    finally:
        conn.execute(text("DELETE FROM corpus.content_items WHERE item_id = :i"), {"i": item})
        conn.execute(text("DELETE FROM corpus.sources WHERE natural_id = 'source:w'"))
        conn.commit()


def test_a_refused_item_is_counted_and_fails_only_when_strict(
    pipeline_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    _write(tmp_path, BASE + ".cleaned.adfree.txt", "x\n")
    meta = _metadata(tmp_path)
    assert sync_episode(meta, transcript) == "refused"
    monkeypatch.setenv(ENV_STRICT, "1")
    with pytest.raises(SyncFailed, match="refused"):
        sync_episode(meta, transcript)
    assert OUTCOMES["refused"] == 2


def test_a_news_item_is_not_this_slices(pipeline_db: None, tmp_path: Path) -> None:
    meta = tmp_path / "n.metadata.json"
    meta.write_text(json.dumps({"content_type": "news_article", "item_id": "item:n"}))
    assert sync_episode(meta, _write(tmp_path, "n.txt", "x\n")) == "news_item"
    assert corpus_writer.OUTCOMES["news_item"] == 1
