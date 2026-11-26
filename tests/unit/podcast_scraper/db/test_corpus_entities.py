"""The ``entities`` row of a written id (CRFC-005 §3.3), and that the database takes it."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import Engine

from podcast_scraper.db.corpus_entities import (
    entity_row,
    entity_type,
    normalized_label,
    NotAnEntity,
    SCOPES,
)
from podcast_scraper.db.schema import entities
from podcast_scraper.db.settings import normalize_url
from podcast_scraper.search.corpus_search import _normalize_kg_surface_text
from tests._database import worker_database_url


def test_each_type_has_its_identity_scope() -> None:
    assert [entity_row(f"{t}:x", "X")["scope"] for t in ("person", "org", "topic", "object")] == [
        "person",
        "organisation",
        "topic",
        "object",
    ]


def test_a_row_keeps_the_id_and_label_as_written() -> None:
    assert entity_row("org:the-new-york-times", "The  New York\tTimes ") == {
        "natural_id": "org:the-new-york-times",
        "type": "org",
        "scope": "organisation",
        "label": "The  New York\tTimes ",
        "normalized_label": "the new york times",
    }


@pytest.mark.parametrize(
    "label", ["Yu Shan", "  ai IN healthcare ", "urban-rural disparity", "Ägypten\nHeute", ""]
)
def test_the_lookup_key_is_the_search_dedupe_key(label: str) -> None:
    assert normalized_label(label) == _normalize_kg_surface_text(label)


@pytest.mark.parametrize("bad", ["episode:abc", "person:", "jane-doe", "", "insight:1"])
def test_an_id_of_no_entity_type_is_refused(bad: str) -> None:
    with pytest.raises(NotAnEntity):
        entity_type(bad)


@pytest.mark.parametrize("label", ["", "   ", None])
def test_a_row_without_a_label_is_refused(label: str) -> None:
    with pytest.raises(NotAnEntity):
        entity_row("person:jane-doe", label)


def test_the_scopes_are_the_identity_tables_scopes() -> None:
    from podcast_scraper.db.schema import _SCOPES

    assert sorted(SCOPES.values()) == sorted(s.strip(" '") for s in _SCOPES.split(","))


@pytest.fixture
def engine() -> Iterator[Engine]:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    made = create_engine(normalize_url(url))
    yield made
    with made.begin() as conn:
        conn.execute(entities.delete())
    made.dispose()


def test_the_row_inserts_and_is_its_own_canonical(engine: Engine) -> None:
    with engine.begin() as conn:
        eid = conn.execute(
            insert(entities)
            .values(**entity_row("topic:vibe-coding", "Vibe Coding"))
            .returning(entities.c.entity_id)
        ).scalar_one()
        row = conn.execute(select(entities).where(entities.c.entity_id == eid)).one()
    assert (row.natural_id, row.type, row.scope, row.normalized_label, row.canonical_id) == (
        "topic:vibe-coding",
        "topic",
        "topic",
        "vibe coding",
        eid,
    )
