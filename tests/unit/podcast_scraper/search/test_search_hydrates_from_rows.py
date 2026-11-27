"""Transcript search hits are hydrated from the corpus rows only (r-search slice 3): the text,
times, titles and speakers come from ``search_chunks`` cut from ``segments``, and a hit with no
row, or whose item is not served or is withheld, is dropped."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from sqlalchemy import insert, update
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import content_items, search_chunks, segments
from podcast_scraper.search import corpus_search
from tests.unit.podcast_scraper.db.test_db_corpus import (  # noqa: F401
    _document,
    _item,
    _source,
    conn,
    url,
)


def _hit(doc_id: str, doc_type: str = "transcript", **meta: Any) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "score": 1.0,
        "text": "index text",
        "metadata": {
            "doc_type": doc_type,
            "timestamp_start_ms": 0,
            "timestamp_end_ms": 0,
            "episode_title": "File title",
            "feed_title": "File show",
            **meta,
        },
    }


def _chunk(natural_id: str, **fields: Any) -> Dict[str, Any]:
    return {
        "natural_id": natural_id,
        "served": 1,
        "withheld": False,
        "text": "row text",
        "start_ms": 1500,
        "end_ms": 4000,
        "title": "Row title",
        "source_title": "Row show",
        "speakers": ["person:ann"],
        **fields,
    }


def test_a_transcript_hit_takes_its_rows_and_other_hits_are_untouched() -> None:
    calls: List[List[str]] = []

    def fetch(ids: List[str]) -> List[Dict[str, Any]]:
        calls.append(ids)
        return [_chunk("chunk:a:0"), _chunk("chunk:a:1", start_ms=2000, end_ms=None)]

    insight = _hit("insight:1", doc_type="insight")
    rows = [_hit("chunk:a:0"), insight, _hit("chunk:a:1"), _hit("chunk:a:0")]
    out = corpus_search._hydrate_transcript_hits(rows, fetch)
    assert calls == [["chunk:a:0", "chunk:a:1"]]
    assert [r["doc_id"] for r in out] == ["chunk:a:0", "insight:1", "chunk:a:1", "chunk:a:0"]
    first = out[0]
    assert first["text"] == "row text"
    assert first["metadata"] == {
        "doc_type": "transcript",
        "timestamp_start_ms": 1500,
        "timestamp_end_ms": 4000,
        "episode_title": "Row title",
        "feed_title": "Row show",
        "speakers": ["person:ann"],
    }
    # An untimed end: no end time rather than the index's 0.
    assert "timestamp_end_ms" not in out[2]["metadata"]
    assert out[2]["metadata"]["timestamp_start_ms"] == 2000
    assert out[1] is insight and insight["text"] == "index text"


def test_a_hit_with_no_row_an_unserved_or_a_withheld_item_is_dropped() -> None:
    def fetch(ids: List[str]) -> List[Dict[str, Any]]:
        return [
            _chunk("chunk:kept:0"),
            _chunk("chunk:unserved:0", served=0),
            _chunk("chunk:withheld:0", withheld=True),
        ]

    rows = [
        _hit("chunk:kept:0"),
        _hit("chunk:unserved:0"),
        _hit("chunk:withheld:0"),
        _hit("chunk:stale:0"),
    ]
    assert [r["doc_id"] for r in corpus_search._hydrate_transcript_hits(rows, fetch)] == [
        "chunk:kept:0"
    ]


def test_no_transcript_hit_reads_no_rows() -> None:
    def fetch(ids: List[str]) -> List[Dict[str, Any]]:
        raise AssertionError("no transcript hit, no read")

    rows = [_hit("insight:1", doc_type="insight")]
    assert corpus_search._hydrate_transcript_hits(rows, fetch) == rows


def test_the_rows_are_read_in_calls_of_at_most_500_ids(
    conn: Connection, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    source = _source(conn, title="The Show")
    item = _item(conn, source, "1", served=1)
    document = _document(conn, item)
    conn.execute(
        insert(segments).values(
            item_id=item, document_id=document, seq=0, start_ms=0, end_ms=1000, text="Hello."
        )
    )
    ids = [f"chunk:s__1:{n}" for n in range(1201)]
    conn.execute(
        insert(search_chunks),
        [
            {
                "natural_id": natural_id,
                "item_id": item,
                "document_id": document,
                "start_seq": 0,
                "start_offset": 0,
                "end_seq": 0,
                "end_offset": 5,
                "speakers": [],
            }
            for natural_id in ids
        ],
    )
    conn.execute(
        update(content_items).where(content_items.c.item_id == item).values(withheld=False)
    )

    class _Engine:
        def connect(self) -> Any:
            class _Hold:
                def __enter__(self) -> Connection:
                    return conn

                def __exit__(self, *exc: object) -> None:
                    return None

            return _Hold()

    sizes: List[int] = []
    from podcast_scraper.db import engines, intents

    real = intents.chunks_by_natural_ids

    def counted(c: Connection, batch: List[str]) -> List[Dict[str, Any]]:
        sizes.append(len(batch))
        return real(c, batch)

    monkeypatch.setattr(engines, "corpus_ro_engine", lambda: _Engine())
    monkeypatch.setattr(intents, "chunks_by_natural_ids", counted)
    found = corpus_search._chunk_rows(ids)
    assert sizes == [500, 500, 201]
    assert len(found) == 1201
    assert {r["text"] for r in found} == {"Hello"}
    assert all(r["withheld"] is False for r in found)
    conn.rollback()
