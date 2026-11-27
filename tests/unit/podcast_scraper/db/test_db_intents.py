"""The item intents (db.intents): their results, their bounds, and one statement each."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest
from sqlalchemy import event, insert
from sqlalchemy.engine import Connection

from podcast_scraper.db import intents
from podcast_scraper.db.schema import entities, item_cards, search_chunks, segments
from tests.unit.podcast_scraper.db.test_db_corpus import (  # noqa: F401
    _document,
    _item,
    _source,
    conn,
    url,
)

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


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


def test_items_by_key_return_every_served_value_and_run_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, raw_feed_id="feed-a")
    one = _item(conn, source, "1", served=1)
    two = _item(conn, source, "2", served=0)
    with _statements(conn) as seen:
        by_slug = intents.items_by_slugs(conn, ["slug-1", "slug-2", "slug-x", "slug-1"])
    assert len(seen) == 1
    assert sorted((r["item_id"], r["served"]) for r in by_slug) == [(one, 1), (two, 0)]
    with _statements(conn) as seen:
        by_id = intents.items_by_natural_ids(conn, ["item:2"])
    assert len(seen) == 1 and [r["item_id"] for r in by_id] == [two]
    with _statements(conn) as seen:
        by_pair = intents.items_by_feed_episode(conn, [("feed-a", "ext-1"), ("feed-b", "ext-2")])
    assert len(seen) == 1
    assert [(r["item_id"], r["raw_feed_id"]) for r in by_pair] == [(one, "feed-a")]
    assert "type_attrs" not in by_slug[0]
    conn.rollback()


def test_an_empty_key_list_runs_nothing_and_too_many_keys_are_refused(
    conn: Connection,  # noqa: F811
) -> None:
    with _statements(conn) as seen:
        assert intents.items_by_slugs(conn, []) == []
        assert intents.cards(conn, []) == []
    assert seen == []
    with pytest.raises(ValueError, match="more than 500"):
        intents.items_by_slugs(conn, [f"s{i}" for i in range(501)])
    with pytest.raises(ValueError, match="between 1 and 200"):
        intents.newest_items(conn, limit=201)


def test_articles_carry_their_source_title_and_url_in_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, "source:n", title="The Example Times", content_type="news_article")
    kept = _item(
        conn,
        source,
        "a",
        content_type="news_article",
        served=0,
        published_at=T0,
        type_attrs={"url": "https://www.example.com/a/1", "byline": ["A. Writer"]},
    )
    with _statements(conn) as seen:
        found = intents.articles_by_natural_ids(conn, ["item:a", "item:missing"])
    assert len(seen) == 1
    assert found == [
        {
            "item_id": kept,
            "natural_id": "item:a",
            "served": 0,
            "title": "Episode a",
            "published_at": T0,
            "source_title": "The Example Times",
            "url": "https://www.example.com/a/1",
        }
    ]
    conn.rollback()


def test_cards_are_one_row_per_item_in_one_statement(conn: Connection) -> None:  # noqa: F811
    source = _source(conn)
    one = _item(conn, source, "1")
    _item(conn, source, "2")
    conn.execute(insert(item_cards).values(item_id=one, title="Episode 1", source_title="A"))
    with _statements(conn) as seen:
        found = intents.cards(conn, [one, one])
    assert len(seen) == 1
    assert [(r["item_id"], r["source_title"]) for r in found] == [(one, "A")]
    conn.rollback()


def test_newest_items_pages_dated_then_undated_without_repeats(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn)
    other = _source(conn, "source:b")
    old = _item(conn, source, "old", served=1, published_at=T0)
    new = _item(conn, source, "new", served=1, published_at=T0 + timedelta(days=1))
    same = _item(conn, source, "same", served=1, published_at=T0)
    undated = _item(conn, source, "undated", served=1)
    _item(conn, source, "hidden", served=0, published_at=T0 + timedelta(days=2))
    _item(conn, other, "elsewhere", served=1, published_at=T0 + timedelta(days=3))
    order: list[int] = []
    after = None
    while True:
        with _statements(conn) as seen:
            page = intents.newest_items(conn, source_ids=[source], after=after, limit=2)
        assert len(seen) == 1
        order += [r["item_id"] for r in page]
        if len(page) < 2:
            break
        after = intents.cursor_of(page[-1])
    assert order == [new, max(old, same), min(old, same), undated]
    conn.rollback()


def test_newest_items_of_a_content_type(conn: Connection) -> None:  # noqa: F811
    source = _source(conn)
    podcast = _item(conn, source, "p", served=1, published_at=T0)
    _item(conn, source, "n", served=1, published_at=T0, content_type="news_article")
    found = intents.newest_items(conn, content_type="podcast")
    assert [r["item_id"] for r in found] == [podcast]
    conn.rollback()


def test_catalog_items_are_the_served_items_of_the_types_asked_in_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, raw_feed_id="feed-a")
    served = _item(
        conn,
        source,
        "1",
        served=1,
        type_attrs={"metadata_relative_path": "feeds/a/run_1/metadata/x.metadata.json"},
        source_attrs={"title": "A show"},
    )
    _item(conn, source, "2", served=0)
    _item(conn, source, "3", served=1, content_type="news_article")
    with _statements(conn) as seen:
        found = intents.catalog_items(conn, {"podcast"})
    assert len(seen) == 1
    assert [r["item_id"] for r in found] == [served]
    row = found[0]
    assert row["raw_feed_id"] == "feed-a" and row["source_attrs"] == {"title": "A show"}
    assert row["type_attrs"]["metadata_relative_path"].endswith("x.metadata.json")
    assert intents.catalog_items(conn, set()) == []
    conn.rollback()


def _entity(
    conn: Connection, natural_id: str, canonical: Optional[int] = None  # noqa: F811
) -> int:
    values: dict[str, Any] = {
        "natural_id": natural_id,
        "type": "person",
        "label": natural_id,
        "normalized_label": natural_id,
    }
    if canonical is not None:
        values["canonical_id"] = canonical
    entity_id: int = conn.execute(
        insert(entities).values(values).returning(entities.c.entity_id)
    ).scalar_one()
    return entity_id


def test_chunks_carry_their_item_range_text_times_and_canonical_speakers_in_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn, title="The Show")
    item = _item(conn, source, "1", served=1, published_at=T0)
    document = _document(conn, item)
    ann = _entity(conn, "person:ann")
    canonical = _entity(conn, "person:bob")
    member = _entity(conn, "person:bob-2", canonical=canonical)
    for seq, (body, start, end, speaker) in enumerate(
        [
            ("Hello world. ", 0, 1000, ann),
            ("Second part here. ", 1000, 2000, member),
            ("Third.", 2000, None, None),
        ]
    ):
        conn.execute(
            insert(segments).values(
                item_id=item,
                document_id=document,
                seq=seq,
                start_ms=start,
                end_ms=end,
                speaker_id=speaker,
                text=body,
            )
        )
    for natural_id, (start_seq, start_offset, end_seq, end_offset), speakers in (
        ("chunk:s__1:0", (0, 6, 0, 11), [ann]),
        ("chunk:s__1:1", (0, 6, 2, 5), [ann, member, canonical]),
        ("chunk:s__1:2", (2, 0, 2, 6), []),
    ):
        conn.execute(
            insert(search_chunks).values(
                natural_id=natural_id,
                item_id=item,
                document_id=document,
                start_seq=start_seq,
                start_offset=start_offset,
                end_seq=end_seq,
                end_offset=end_offset,
                speakers=speakers,
            )
        )
    with _statements(conn) as seen:
        found = intents.chunks_by_natural_ids(
            conn, ["chunk:s__1:0", "chunk:s__1:1", "chunk:s__1:2", "chunk:none", "chunk:s__1:0"]
        )
    assert len(seen) == 1
    by_id = {r["natural_id"]: r for r in found}
    assert set(by_id) == {"chunk:s__1:0", "chunk:s__1:1", "chunk:s__1:2"}
    one = by_id["chunk:s__1:0"]
    assert (one["text"], one["start_ms"], one["end_ms"], one["speakers"]) == (
        "world",
        0,
        1000,
        ["person:ann"],
    )
    assert one["item_natural_id"] == "item:1" and one["source_title"] == "The Show"
    assert one["published_at"] == T0 and one["served"] == 1 and one["document_id"] == document
    many = by_id["chunk:s__1:1"]
    assert many["text"] == "world. Second part here. Third"
    assert (many["start_ms"], many["end_ms"]) == (0, 2000)
    assert many["speakers"] == ["person:ann", "person:bob"]
    untimed = by_id["chunk:s__1:2"]
    assert (untimed["text"], untimed["start_ms"], untimed["end_ms"], untimed["speakers"]) == (
        "Third.",
        2000,
        None,
        [],
    )
    with _statements(conn) as seen:
        assert intents.chunks_by_natural_ids(conn, []) == []
    assert seen == []
    conn.rollback()
