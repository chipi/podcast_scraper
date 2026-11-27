"""The audio dedup gate once the corpus is served from rows (w-public): the gate reads and writes
``audio_fingerprints``, a fingerprint is held until its episode's ``sync_rows`` writes it with
the source row, and the file index is neither read nor written."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Connection

from podcast_scraper.db import audio_fingerprints as rows
from podcast_scraper.db.corpus_writer import sync_episode
from podcast_scraper.db.schema import audio_fingerprints, sources
from podcast_scraper.db.settings import ENV_SERVED_FROM_ROWS
from podcast_scraper.utils import audio_fingerprint as gate
from podcast_scraper.workflow.metadata_generation import generate_feed_id
from tests.unit.podcast_scraper.db.test_corpus_writer import (  # noqa: F401
    _write,
    BASE,
    pipeline_db,
)
from tests.unit.podcast_scraper.db.test_db_corpus import conn, url  # noqa: F401

FEED_URL = "https://feeds.example.com/show.xml"
A = "a" * 64


@pytest.fixture
def on_rows(pipeline_db: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # noqa: F811
    monkeypatch.setenv(ENV_SERVED_FROM_ROWS, "1")
    gate.reset_pending_for_tests()
    rows._HELD.clear()
    rows.OUTCOMES.clear()
    yield
    rows._HELD.clear()
    gate.reset_pending_for_tests()


def _metadata(directory: Path, guid: str) -> Path:
    doc = {
        "feed": {"feed_id": generate_feed_id(FEED_URL), "title": "A show", "language": "en"},
        "episode": {
            "episode_id": f"episode:{guid}",
            "guid": guid,
            "title": "E",
            "language": "en",
            "published_date": "2026-11-20T08:00:00+00:00",
        },
    }
    path = directory / f"{guid}.metadata.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _sync(directory: Path, guid: str) -> str:
    transcript = _write(directory, BASE + ".txt", "Hello.\n")
    return sync_episode(_metadata(directory, guid), transcript)


def _fingerprints(conn: Connection) -> list[dict[str, Any]]:  # noqa: F811
    found = conn.execute(select(audio_fingerprints)).mappings().all()
    conn.rollback()
    return [dict(r) for r in found]


def test_a_held_fingerprint_is_written_with_the_source_row_its_episode_creates(
    on_rows: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    root = str(tmp_path)
    gate.record(
        root,
        A,
        identity="guid-1",
        feed_id=FEED_URL,
        episode_title="One",
        transcript_path="run/transcripts/one.txt",
    )
    assert conn.execute(select(sources)).first() is None
    assert _fingerprints(conn) == []
    # Before its row exists, a twin under another identity is caught from the held fingerprint.
    held = gate.duplicate_of(root, A, "guid-2", feed_url=FEED_URL)
    assert held is not None and held["identity"] == "guid-1" and held["in_flight"]
    assert gate.duplicate_of(root, A, "guid-1", feed_url=FEED_URL) is None
    assert _sync(tmp_path, "guid-1") == "written"
    (row,) = _fingerprints(conn)
    item = conn.execute(select(sources.c.source_id, sources.c.raw_feed_id)).one()
    conn.rollback()
    assert (row["source_id"], row["identity"], row["episode_title"]) == (item[0], "guid-1", "One")
    assert row["item_id"] is not None
    assert item[1] == generate_feed_id(FEED_URL)
    assert rows._HELD == {}
    assert not (tmp_path / gate.INDEX_RELPATH).exists()


def test_on_rows_the_gate_reads_and_records_copies_on_the_table(
    on_rows: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    root = str(tmp_path)
    gate.record(root, A, identity="guid-1", feed_id=FEED_URL, episode_title="One")
    _sync(tmp_path, "guid-1")
    dup = gate.duplicate_of(root, A, "guid-2", feed_url=FEED_URL)
    assert dup is not None and dup["identity"] == "guid-1" and not dup.get("in_flight")
    gate.record_duplicate(root, A, "guid-2", feed_url=FEED_URL)
    gate.record_duplicate(root, A, "guid-2", feed_url=FEED_URL)
    assert gate.known_duplicate_identities(root, feed_url=FEED_URL) == {"guid-2"}
    assert rows.OUTCOMES == {"recorded": 1, "listed": 1}
    assert _fingerprints(conn)[0]["duplicates"] == ["guid-2"]
    # A feed the corpus has no row for has nothing to find and nothing to list.
    other = "https://feeds.example.com/other.xml"
    assert gate.duplicate_of(root, A, "guid-2", feed_url=other) is None
    assert gate.known_duplicate_identities(root, feed_url=other) == set()
    gate.record_duplicate(root, A, "guid-3", feed_url=other)
    assert rows.OUTCOMES["no_row"] == 1
    assert not (tmp_path / gate.INDEX_RELPATH).exists()


def test_off_rows_the_gate_keeps_the_file_and_writes_no_row(
    pipeline_db: None,  # noqa: F811
    conn: Connection,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_SERVED_FROM_ROWS, raising=False)
    rows._HELD.clear()
    root = str(tmp_path)
    gate.record(root, A, identity="guid-1", feed_id=FEED_URL, episode_title="One")
    assert (tmp_path / gate.INDEX_RELPATH).exists()
    assert rows._HELD == {}
    _sync(tmp_path, "guid-1")
    assert _fingerprints(conn) == []


def test_a_row_failure_degrades_to_no_duplicate_found(
    on_rows: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("database down")

    monkeypatch.setattr(rows, "gate_duplicate_of", broken)
    monkeypatch.setattr(rows, "gate_known_duplicates", broken)
    root = str(tmp_path)
    assert gate.duplicate_of(root, A, "guid-2", feed_url=FEED_URL) is None
    assert gate.known_duplicate_identities(root, feed_url=FEED_URL) == set()
    assert gate.duplicate_of(root, A, "guid-2", feed_url=None) is None
