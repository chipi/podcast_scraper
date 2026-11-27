"""The audio dedup gate on rows (db.audio_fingerprints): each operation, one statement each."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import event, select
from sqlalchemy.engine import Connection

from podcast_scraper.db import audio_fingerprints as fp
from podcast_scraper.db.schema import audio_fingerprints
from tests.unit.podcast_scraper.db.test_db_corpus import _item, _source, conn, url  # noqa: F401

T0 = datetime(2026, 11, 27, 9, 0, tzinfo=timezone.utc)
A = "a" * 64
B = "b" * 64


@contextlib.contextmanager
def _statements(connection: Connection) -> Iterator[list[str]]:
    seen: list[str] = []

    def count(_c: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        seen.append(statement)

    event.listen(connection, "before_cursor_execute", count)
    try:
        yield seen
    finally:
        event.remove(connection, "before_cursor_execute", count)


def _record(c: Connection, source: int, digest: str, identity: str, title: str = "One") -> None:
    fp.record(
        c,
        source,
        digest,
        identity=identity,
        episode_title=title,
        transcript_path="feeds/f/run/transcripts/one.txt",
        recorded_at=T0,
    )


def test_a_feed_finds_its_source_by_raw_feed_id(conn: Connection) -> None:  # noqa: F811
    source = _source(conn, raw_feed_id="sha256:feed")
    with _statements(conn) as seen:
        assert fp.source_for_feed(conn, "sha256:feed") == source
    assert len(seen) == 1
    assert fp.source_for_feed(conn, "sha256:other") is None
    conn.rollback()


def test_the_same_bytes_under_another_identity_are_a_duplicate_and_under_its_own_a_retry(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, raw_feed_id="sha256:feed")
    other = _source(conn, "source:b", raw_feed_id="sha256:other")
    assert fp.duplicate_of(conn, source, A, "guid-1") is None
    with _statements(conn) as seen:
        _record(conn, source, A, "guid-1")
    assert len(seen) == 1
    assert fp.duplicate_of(conn, source, A, "guid-1") is None
    with _statements(conn) as seen:
        found = fp.duplicate_of(conn, source, A, "guid-2")
    assert len(seen) == 1
    assert found == {
        "identity": "guid-1",
        "episode_title": "One",
        "transcript_path": "feeds/f/run/transcripts/one.txt",
        "recorded_at": T0,
    }
    assert fp.duplicate_of(conn, other, A, "guid-2") is None
    conn.rollback()


def test_a_record_links_its_item_and_a_re_record_keeps_the_skipped_copies(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, raw_feed_id="sha256:feed")
    item = _item(conn, source, "1", external_item_id="guid-1")
    _record(conn, source, A, "guid-1")
    assert fp.record_duplicate(conn, source, A, "guid-2")
    _record(conn, source, A, "guid-1", title="One, renamed")
    row = conn.execute(select(audio_fingerprints)).mappings().one()
    assert (row["item_id"], row["episode_title"], row["duplicates"]) == (
        item,
        "One, renamed",
        ["guid-2"],
    )
    _record(conn, source, B, "url:https://example.com/two.mp3")
    unlinked = conn.execute(
        select(audio_fingerprints.c.item_id).where(audio_fingerprints.c.digest == B)
    ).scalar_one()
    assert unlinked is None
    conn.rollback()


def test_a_skipped_copy_is_listed_once_and_only_under_a_recorded_digest(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, raw_feed_id="sha256:feed")
    other = _source(conn, "source:b", raw_feed_id="sha256:other")
    assert not fp.record_duplicate(conn, source, A, "guid-2")
    _record(conn, source, A, "guid-1")
    _record(conn, source, B, "guid-3")
    _record(conn, other, A, "guid-9")
    with _statements(conn) as seen:
        assert fp.record_duplicate(conn, source, A, "guid-2")
    assert len(seen) == 1
    assert not fp.record_duplicate(conn, source, A, "guid-2")
    assert fp.record_duplicate(conn, source, B, "guid-4")
    assert fp.record_duplicate(conn, other, A, "guid-8")
    with _statements(conn) as seen:
        assert fp.known_duplicate_identities(conn, source) == {"guid-2", "guid-4"}
    assert len(seen) == 1
    assert fp.known_duplicate_identities(conn, other) == {"guid-8"}
    conn.rollback()
