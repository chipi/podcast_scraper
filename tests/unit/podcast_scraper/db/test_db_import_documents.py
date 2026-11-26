"""A podcast item's transcript files as documents and segments (db.importers.corpus_documents)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection

from podcast_scraper.db.importers.corpus_documents import (
    DocumentRefused,
    import_documents,
    joined_body,
    transcript_files,
)
from tests.unit.podcast_scraper.db.test_db_corpus import _item, _source, conn, url  # noqa: F401

BASE = "0001 - An episode_abc"


def _write(directory: Path, name: str, body: str | bytes) -> Path:
    path = directory / name
    path.write_bytes(body.encode("utf-8") if isinstance(body, str) else body)
    return path


def _segments(directory: Path, name: str, entries: list[dict[str, object]]) -> None:
    (directory / name).write_text(json.dumps(entries), encoding="utf-8")


def _variants(directory: Path, language: str | None = None) -> dict[str, str]:
    return {f.variant: f.path.name for f in transcript_files(directory, BASE, language)}


def test_each_file_gets_its_variant_and_the_anon_copy_is_skipped(tmp_path: Path) -> None:
    for suffix in (".txt", ".cleaned.txt", ".adfree.txt", ".anon.txt"):
        _write(tmp_path, BASE + suffix, "a\n")
    _write(tmp_path, "0001 - Another_abc.txt", "other\n")
    assert _variants(tmp_path, "en") == {
        "raw": BASE + ".txt",
        "cleaned": BASE + ".cleaned.txt",
        "adfree": BASE + ".adfree.txt",
    }


def test_only_the_bodies_with_a_segments_file_carry_it(tmp_path: Path) -> None:
    _write(tmp_path, BASE + ".txt", "a\n")
    _write(tmp_path, BASE + ".cleaned.txt", "a\n")
    _segments(tmp_path, BASE + ".segments.json", [])
    found = {f.variant: f.segments_path for f in transcript_files(tmp_path, BASE, "en")}
    assert found["raw"] == tmp_path / (BASE + ".segments.json")
    assert found["cleaned"] is None


def test_a_translated_item_has_its_source_raw_and_its_english_render_en(tmp_path: Path) -> None:
    _write(tmp_path, BASE + ".txt", "Hello\n")
    _write(tmp_path, BASE + ".es.txt", "Hola\n")
    _write(tmp_path, BASE + ".translation.json", "{}")
    assert _variants(tmp_path, "es") == {"en": BASE + ".txt", "raw": BASE + ".es.txt"}


@pytest.mark.parametrize(
    "names, language, refusal",
    [
        ([".txt", ".cleaned.adfree.txt"], "en", "stacked"),
        ([".txt", ".es.txt"], "es", "with no .*translation.json"),
        ([".txt"], "es", "English render of a es item"),
        ([".txt", ".es.txt", ".fr.txt", ".translation.json"], "es", "both variant raw"),
    ],
)
def test_what_cannot_be_named_one_way_is_refused(
    tmp_path: Path, names: list[str], language: str, refusal: str
) -> None:
    for suffix in names:
        _write(tmp_path, BASE + suffix, "x\n")
    with pytest.raises(DocumentRefused, match=refusal):
        transcript_files(tmp_path, BASE, language)


def test_a_body_becomes_segments_that_join_back_byte_for_byte(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    body = "Welcome back.\r\nToday, trails.\n"
    _write(tmp_path, BASE + ".txt", body)
    _segments(
        tmp_path,
        BASE + ".segments.json",
        [
            {"start": 0.0, "end": 1.9476, "text": "Welcome back.", "char_start": 0, "char_end": 13},
            {
                "start": 1.9476,
                "end": 4.0,
                "text": "Today, trails.",
                "char_start": 15,
                "char_end": 29,
            },
        ],
    )
    _write(tmp_path, BASE + ".cleaned.txt", "Welcome.\nTrails.\n")
    item = _item(conn, _source(conn))
    imported = import_documents(conn, item, transcript_files(tmp_path, BASE, "en"))
    assert [(d.variant, d.segments, d.inserted) for d in imported] == [
        ("raw", 2, True),
        ("cleaned", 2, True),
    ]
    raw = imported[0].document_id
    assert joined_body(conn, item, raw) == body
    rows = conn.execute(
        text(
            "SELECT seq, start_ms, end_ms FROM corpus.segments WHERE document_id = :d ORDER BY seq"
        ),
        {"d": raw},
    ).all()
    assert rows == [(0, 0, 1948), (1, 1948, 4000)]
    cleaned = imported[1].document_id
    assert conn.execute(
        text("SELECT start_ms FROM corpus.segments WHERE document_id = :d"), {"d": cleaned}
    ).scalars().all() == [None, None]
    conn.rollback()


def test_a_second_import_writes_nothing_and_another_body_is_refused(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, BASE + ".txt", "one\ntwo\n")
    item = _item(conn, _source(conn))
    first = import_documents(conn, item, transcript_files(tmp_path, BASE, "en"))
    again = import_documents(conn, item, transcript_files(tmp_path, BASE, "en"))
    assert [(d.document_id, d.segments, d.inserted) for d in again] == [
        (first[0].document_id, 2, False)
    ]
    assert conn.execute(text("SELECT count(*) FROM corpus.documents")).scalar_one() == 1
    _write(tmp_path, BASE + ".txt", "one\nchanged\n")
    with pytest.raises(DocumentRefused, match="another body"):
        import_documents(conn, item, transcript_files(tmp_path, BASE, "en"))
    conn.rollback()
