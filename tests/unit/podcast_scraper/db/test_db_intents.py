"""The item intents (db.intents): their results, their bounds, and one statement each."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest
from sqlalchemy import event, insert
from sqlalchemy.engine import Connection

from podcast_scraper.db import intents
from podcast_scraper.db.schema import (
    entities,
    evidence,
    insight_support,
    insights,
    item_cards,
    item_entities,
    person_roles,
    search_chunks,
    segments,
)
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


def _typed(
    conn: Connection, natural_id: str, type_: str, label: str, **kw: Any  # noqa: F811
) -> int:
    entity_id: int = conn.execute(
        insert(entities)
        .values(
            natural_id=natural_id, type=type_, label=label, normalized_label=label.lower(), **kw
        )
        .returning(entities.c.entity_id)
    ).scalar_one()
    return entity_id


def _carry(
    conn: Connection, item: int, entity: int, role: str, published: Optional[datetime]  # noqa: F811
) -> None:
    conn.execute(
        insert(item_entities).values(
            item_id=item,
            entity_id=entity,
            role=role,
            published_at=published,
            content_type="podcast",
        )
    )


def test_entities_by_natural_ids_carry_their_own_labels_in_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    topic = _typed(conn, "topic:ai-safety", "Topic", "AI Safety")
    with _statements(conn) as seen:
        found = intents.entities_by_natural_ids(conn, ["topic:ai-safety", "topic:missing"])
    assert len(seen) == 1
    assert found == [
        {
            "entity_id": topic,
            "natural_id": "topic:ai-safety",
            "type": "Topic",
            "label": "AI Safety",
            "normalized_label": "ai safety",
        }
    ]
    conn.rollback()


def test_item_entities_of_are_one_row_per_role_in_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn)
    one, two = _item(conn, source, "1", published_at=T0), _item(conn, source, "2")
    zeta = _typed(conn, "topic:zeta", "Topic", "Zeta")
    alpha = _typed(conn, "person:alpha", "Person", "Alpha")
    _carry(conn, one, zeta, "mentions", T0)
    _carry(conn, one, zeta, "about", T0)
    _carry(conn, one, alpha, "mentions", T0)
    _carry(conn, two, zeta, "mentions", None)
    with _statements(conn) as seen:
        got = intents.item_entities_of(conn, [one, two, one])
    assert len(seen) == 1
    assert [(r["item_id"], r["role"], r["natural_id"]) for r in got] == [
        (one, "about", "topic:zeta"),
        (one, "mentions", "person:alpha"),
        (one, "mentions", "topic:zeta"),
        (two, "mentions", "topic:zeta"),
    ]
    only = intents.item_entities_of(conn, [one], roles={"about"})
    assert [r["natural_id"] for r in only] == ["topic:zeta"]
    conn.rollback()


def test_entity_items_reach_every_member_of_the_canonical_once_newest_first(
    conn: Connection,  # noqa: F811
) -> None:
    """GI and KG write one topic under two ids (CRFC-005 §3.3): asked by either, the list holds
    the items of both, each once, newest first, and pages without repeats."""
    source = _source(conn)
    kg = _typed(conn, "topic:recursive-self-improvement", "Topic", "RSI")
    gi = _typed(conn, "topic:improvement-recursive-self", "Topic", "RSI", canonical_id=kg)
    other = _typed(conn, "topic:other", "Topic", "Other")
    items = [
        _item(conn, source, str(n), served=1, published_at=T0 + timedelta(days=n)) for n in range(4)
    ]
    undated = _item(conn, source, "u", served=1)
    hidden = _item(conn, source, "h", served=0, published_at=T0)
    _carry(conn, items[0], kg, "mentions", T0)
    _carry(conn, items[0], gi, "about", T0)
    _carry(conn, items[1], gi, "about", T0 + timedelta(days=1))
    _carry(conn, items[2], kg, "mentions", T0 + timedelta(days=2))
    _carry(conn, items[3], other, "mentions", T0 + timedelta(days=3))
    _carry(conn, undated, kg, "mentions", None)
    _carry(conn, hidden, kg, "mentions", T0)
    with _statements(conn) as seen:
        first = intents.entity_items(conn, "topic:improvement-recursive-self", limit=2)
    assert len(seen) == 1
    assert [r["item_id"] for r in first] == [items[2], items[1]]
    rest = intents.entity_items(
        conn, "topic:recursive-self-improvement", after=intents.cursor_of(first[-1]), limit=10
    )
    assert [r["item_id"] for r in rest] == [items[0], undated]
    assert intents.entity_items(conn, "topic:missing") == []
    assert intents.entity_items(conn, "topic:other", content_types={"news_article"}) == []
    with pytest.raises(ValueError, match="between 1 and 200"):
        intents.entity_items(conn, "topic:other", limit=0)
    conn.rollback()


def _insight(
    conn: Connection, item: int, doc: int, n: str, rank: int, **kw: Any  # noqa: F811
) -> int:
    values: dict[str, Any] = {
        "natural_id": f"insight:{n}",
        "item_id": item,
        "document_id": doc,
        "insight_type": "claim",
        "rank": rank,
        "routing_tag": "surface",
        "text_origin": "authored",
        "text": f"Insight {n}",
        **kw,
    }
    insight_id: int = conn.execute(
        insert(insights).values(values).returning(insights.c.insight_id)
    ).scalar_one()
    return insight_id


def _quote(
    conn: Connection, item: int, doc: int, n: str, insight: int, ord_: int, **kw: Any  # noqa: F811
) -> int:
    evidence_id: int = conn.execute(
        insert(evidence)
        .values(
            natural_id=f"quote:{n}",
            item_id=item,
            document_id=doc,
            start_seq=0,
            start_offset=0,
            end_seq=0,
            end_offset=5,
            text=f"Quote {n}",
            **kw,
        )
        .returning(evidence.c.evidence_id)
    ).scalar_one()
    conn.execute(
        insert(insight_support).values(insight_id=insight, evidence_id=evidence_id, ord=ord_)
    )
    return evidence_id


def test_item_insights_carry_their_quotes_and_speakers_by_rank_in_one_statement(
    conn: Connection,  # noqa: F811
) -> None:
    """Rank order, quotes in ``ord`` order with the canonical's label as the speaker, a quoteless
    insight with ``[]``; hidden insights and those of a superseded document are left out."""
    source = _source(conn)
    item = _item(conn, source, "1", served=1)
    doc = _document(conn, item)
    newer = _document(conn, item, version=2, supersedes_id=doc)
    canon = _typed(conn, "person:ada-lovelace", "Person", "Ada Lovelace")
    alias = _typed(conn, "person:ada", "Person", "Ada", canonical_id=canon)
    second = _insight(conn, item, newer, "b", 2, salience=0.4)
    first = _insight(conn, item, newer, "a", 1, salience=0.9)
    _insight(conn, item, newer, "hidden", 0, visible=False)
    _insight(conn, item, doc, "old", 0)
    _quote(conn, item, newer, "q2", first, 2, start_ms=2000, end_ms=3000)
    _quote(conn, item, newer, "q1", first, 1, speaker_id=alias, start_ms=1000, end_ms=1500)
    with _statements(conn) as seen:
        got = intents.item_insights(conn, [item])
    assert len(seen) == 1
    assert [(r["insight_id"], r["natural_id"]) for r in got] == [
        (first, "insight:a"),
        (second, "insight:b"),
    ]
    assert [(q["natural_id"], q["speaker"], q["speaker_natural_id"]) for q in got[0]["quotes"]] == [
        ("quote:q1", "Ada Lovelace", "person:ada"),
        ("quote:q2", None, None),
    ]
    assert got[0]["quotes"][0]["start_ms"] == 1000 and got[0]["salience"] == pytest.approx(0.9)
    assert got[1]["quotes"] == []
    assert intents.item_insights(conn, []) == []
    conn.rollback()


def test_person_items_reach_every_member_once_of_the_roles_asked(
    conn: Connection,  # noqa: F811
) -> None:
    source = _source(conn)
    canon = _typed(conn, "person:ada-lovelace", "Person", "Ada Lovelace")
    alias = _typed(conn, "person:ada", "Person", "Ada", canonical_id=canon)
    old = _item(conn, source, "old", served=1, published_at=T0)
    new = _item(conn, source, "new", served=1, published_at=T0 + timedelta(days=1))
    conn.execute(
        insert(person_roles),
        [
            {"item_id": old, "person_id": canon, "role": "host"},
            {"item_id": old, "person_id": alias, "role": "guest"},
            {"item_id": new, "person_id": alias, "role": "guest"},
        ],
    )
    with _statements(conn) as seen:
        got = intents.person_items(conn, "person:ada-lovelace")
    assert len(seen) == 1
    assert [r["item_id"] for r in got] == [new, old]
    assert [r["item_id"] for r in intents.person_items(conn, "person:ada", roles={"host"})] == [old]
    conn.rollback()


def test_item_insights_apply_commons_read_time_attribution_rule(
    conn: Connection,  # noqa: F811
) -> None:
    """RFC-131 S3 on the rows: a raw-labelled speaker is no speaker; an insight whose quotes name
    nobody and include a raw label is unattributed, as is one GI marked not surfaceable; a named
    quote beside a raw one keeps the insight attributed."""
    source = _source(conn)
    item = _item(conn, source, "1", served=1)
    doc = _document(conn, item)
    raw = _typed(conn, "person:speaker-07", "Person", "SPEAKER_07")
    named = _typed(conn, "person:ada", "Person", "Ada")
    only_raw = _insight(conn, item, doc, "raw", 1)
    _quote(conn, item, doc, "r1", only_raw, 1, speaker_id=raw)
    mixed = _insight(conn, item, doc, "mixed", 2)
    _quote(conn, item, doc, "m1", mixed, 1, speaker_id=raw)
    _quote(conn, item, doc, "m2", mixed, 2, speaker_id=named)
    _insight(conn, item, doc, "unsurfaceable", 3, surfaceable=False)
    _insight(conn, item, doc, "quoteless", 4)
    got = {r["natural_id"]: r for r in intents.item_insights(conn, [item])}
    assert got["insight:raw"]["attributed"] is False
    assert got["insight:raw"]["quotes"][0]["speaker"] is None
    assert "written_label" not in got["insight:raw"]["quotes"][0]
    assert got["insight:mixed"]["attributed"] is True
    assert [q["speaker"] for q in got["insight:mixed"]["quotes"]] == [None, "Ada"]
    assert got["insight:unsurfaceable"]["attributed"] is False
    assert got["insight:quoteless"]["attributed"] is True
    conn.rollback()
