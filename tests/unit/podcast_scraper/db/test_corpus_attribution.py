"""``insight_attribution``: a strict majority of the insight's quotes' people (CRFC-004 §1)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import pytest
from sqlalchemy import create_engine, insert, select, Table, update
from sqlalchemy.engine import Connection

from podcast_scraper.db.corpus_attribution import derive_attribution
from podcast_scraper.db.schema import (
    content_items,
    documents,
    entities,
    evidence,
    insight_attribution,
    insight_support,
    insights,
    sources,
)
from podcast_scraper.db.settings import normalize_url
from tests._database import worker_database_url

pytestmark = pytest.mark.integration


@pytest.fixture
def conn() -> Iterator[Connection]:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    engine = create_engine(normalize_url(url))
    with engine.connect() as connection:
        yield connection
        connection.rollback()
    engine.dispose()


def _one(conn: Connection, table: Table, **values: object) -> int:
    key = next(iter(table.primary_key.columns))
    found: int = conn.execute(insert(table).values(values).returning(key)).scalar_one()
    return found


class Corpus:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.source = _one(conn, sources, natural_id="source:a", content_type="podcast", title="A")
        self.people: dict[str, int] = {}
        self._n = 0

    def item(self, n: str) -> tuple[int, int]:
        item = _one(
            self.conn,
            content_items,
            natural_id=f"item:{n}",
            slug=f"slug-{n}",
            source_id=self.source,
            external_item_id=f"ext-{n}",
            content_type="podcast",
            title=n,
        )
        document = _one(self.conn, documents, item_id=item, kind="transcript", variant="adfree")
        return item, document

    def entity(self, natural_id: str) -> int:
        if natural_id not in self.people:
            kind = natural_id.split(":", 1)[0]
            self.people[natural_id] = _one(
                self.conn,
                entities,
                natural_id=natural_id,
                type=kind,
                label=natural_id,
                normalized_label=natural_id,
            )
        return self.people[natural_id]

    def insight(self, doc: tuple[int, int], speakers: list[Optional[str]]) -> int:
        item, document = doc
        self._n += 1
        insight = _one(
            self.conn,
            insights,
            natural_id=f"insight:{self._n}",
            item_id=item,
            document_id=document,
            insight_type="claim",
            rank=self._n,
            routing_tag="surface",
            text_origin="authored",
            text="t",
        )
        for ord_, who in enumerate(speakers):
            quote = _one(
                self.conn,
                evidence,
                natural_id=f"quote:{self._n}.{ord_}",
                item_id=item,
                document_id=document,
                speaker_id=self.entity(who) if who else None,
                start_seq=0,
                start_offset=0,
                end_seq=0,
                end_offset=1,
                text="q",
            )
            self.conn.execute(
                insert(insight_support).values(insight_id=insight, evidence_id=quote, ord=ord_)
            )
        return insight


def _rows(conn: Connection) -> set[tuple[int, int, int]]:
    return {
        (r.insight_id, r.person_id, r.item_id)
        for r in conn.execute(select(insight_attribution)).all()
    }


@pytest.mark.parametrize(
    "speakers,who",
    [
        (["person:a"], "person:a"),
        (["person:a", "person:a", "person:b"], "person:a"),
        (["person:a", None, None], "person:a"),  # a quote with no person is ignored, not dissent
        (["person:a", "org:x", "org:x"], "person:a"),  # only persons count
        (["person:a", "person:b"], None),  # a tie
        (["person:a", "person:b", "person:c"], None),
        ([None], None),
        ([], None),
    ],
)
def test_a_strict_majority_of_the_counted_quotes(
    conn: Connection, speakers: list[Optional[str]], who: Optional[str]
) -> None:
    c = Corpus(conn)
    doc = c.item("1")
    insight = c.insight(doc, speakers)
    assert derive_attribution(conn, [doc[0]]) == (1 if who else 0)
    assert _rows(conn) == ({(insight, c.people[who], doc[0])} if who else set())


def test_speakers_count_by_canonical_id_after_a_join(conn: Connection) -> None:
    c = Corpus(conn)
    doc = c.item("1")
    insight = c.insight(doc, ["person:a", "person:a2", "person:b"])
    derive_attribution(conn)
    assert _rows(conn) == set()
    conn.execute(
        update(entities)
        .where(entities.c.entity_id == c.people["person:a2"])
        .values(canonical_id=c.people["person:a"])
    )
    assert derive_attribution(conn) == 1
    assert _rows(conn) == {(insight, c.people["person:a"], doc[0])}


def test_an_item_scope_rewrites_only_that_item(conn: Connection) -> None:
    c = Corpus(conn)
    one, two = c.item("1"), c.item("2")
    first = c.insight(one, ["person:a"])
    second = c.insight(two, ["person:b"])
    derive_attribution(conn)
    conn.execute(
        update(evidence).where(evidence.c.item_id.in_([one[0], two[0]])).values(speaker_id=None)
    )
    assert derive_attribution(conn, [one[0]]) == 0
    assert _rows(conn) == {(second, c.people["person:b"], two[0])}
    assert first not in {r[0] for r in _rows(conn)}
    assert derive_attribution(conn, []) == 0
    assert _rows(conn) == {(second, c.people["person:b"], two[0])}
