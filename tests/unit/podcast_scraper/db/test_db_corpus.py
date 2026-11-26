"""The ``corpus`` schema's built-in history: its tables, indexes and readiness (ADR-184)."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, exc as sa_exc, func, insert, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection, make_url

from podcast_scraper.db import init_engines, migrate, reset_engines, rw_engine
from podcast_scraper.db.cli_handlers import parse_db_argv, run_db_cli
from podcast_scraper.db.engines import corpus_ro_engine, DatabaseNotReady, init_corpus_engine
from podcast_scraper.db.schema import corpus_metadata
from podcast_scraper.db.settings import CorpusSettings, DatabaseSettings, normalize_url
from tests._database import captured_statements, worker_database_url
from tests.unit.podcast_scraper.db.test_db_migrate import _shape, REPO

TABLES = {
    "alembic_version",
    "schema_history",
    "corpus_meta",
    "corpus_migrations",
    "migration_receipts",
    "processing_stages",
    "jobs",
    "job_leases",
    "ingestion_runs",
    "run_item_outcomes",
    "enrichment_runs",
    "assets",
    "sources",
    "content_items",
    "item_cards",
    "documents",
    "segments",
    "entities",
    "entity_identifiers",
    "identity_builds",
    "identity_changes",
    "identity_clusters",
    "identity_members",
}

INDEXES = {
    "migration_receipts_migration_idx",
    "ingestion_runs_created_idx",
    "ingestion_runs_source_idx",
    "run_item_outcomes_run_idx",
    "run_item_outcomes_item_idx",
    "processing_stages_run_idx",
    "enrichment_runs_created_idx",
    "jobs_created_idx",
    "job_leases_expires_idx",
    "job_leases_job_idx",
    "sources_artwork_idx",
    "content_items_published_idx",
    "content_items_source_published_idx",
    "content_items_type_published_idx",
    "content_items_title_trgm_idx",
    "item_cards_artwork_idx",
    "entities_canonical_idx",
    "entities_type_label_idx",
    "entities_label_trgm_idx",
    "identity_builds_one_published_idx",
    "identity_changes_build_idx",
    "identity_clusters_canonical_idx",
    "identity_members_member_idx",
}


@pytest.fixture
def url() -> str:
    found = worker_database_url()
    assert found, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    return found


@pytest.fixture
def conn(url: str) -> Iterator[Connection]:
    engine = create_engine(normalize_url(url))
    with engine.connect() as connection:
        yield connection
    engine.dispose()


def _source(conn: Connection, natural_id: str = "source:a", **fields: object) -> int:
    values = {"natural_id": natural_id, "content_type": "podcast", "title": "A", **fields}
    table = corpus_metadata.tables["corpus.sources"]
    source_id: int = conn.execute(
        insert(table).values(values).returning(table.c.source_id)
    ).scalar_one()
    return source_id


def _item(conn: Connection, source_id: int, n: str = "1", **fields: object) -> int:
    values = {
        "natural_id": f"item:{n}",
        "slug": f"slug-{n}",
        "source_id": source_id,
        "external_item_id": f"ext-{n}",
        "content_type": "podcast",
        "title": f"Episode {n}",
        **fields,
    }
    table = corpus_metadata.tables["corpus.content_items"]
    item_id: int = conn.execute(
        insert(table).values(values).returning(table.c.item_id)
    ).scalar_one()
    return item_id


def _partitions(conn: Connection) -> set[str]:
    """The hash partitions of ``segments``: the migration makes them, the metadata does not
    list them, so the shape comparisons leave them out."""
    return {
        r[0]
        for r in conn.execute(
            text(
                "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'corpus' AND c.relispartition AND c.relkind IN ('r', 'p')"
            )
        ).all()
    }


def _document(conn: Connection, item_id: int, **fields: object) -> int:
    values = {"item_id": item_id, "kind": "transcript", "variant": "cleaned", **fields}
    table = corpus_metadata.tables["corpus.documents"]
    document_id: int = conn.execute(
        insert(table).values(values).returning(table.c.document_id)
    ).scalar_one()
    return document_id


def test_upgrade_to_head_creates_every_table_and_index(conn: Connection) -> None:
    tables = {
        r[0]
        for r in conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'corpus'")
        ).all()
    } - _partitions(conn)
    assert tables == TABLES
    indexes = {
        r[0]
        for r in conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE schemaname = 'corpus'")
        ).all()
    }
    assert INDEXES <= indexes
    assert {i.name for t in corpus_metadata.tables.values() for i in t.indexes} == INDEXES
    # Nothing from the history landed in public.
    assert (
        conn.execute(
            text(
                "SELECT count(*) FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename = ANY(:names)"
            ),
            {"names": sorted(TABLES)},
        ).scalar()
        == 0
    )


def test_the_item_list_indexes_are_partial_on_served_and_the_title_index_is_trigram(
    conn: Connection,
) -> None:
    defs: dict[str, str] = dict(
        conn.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'corpus'")
        ).all()
    )
    for name, shape in {
        "content_items_published_idx": r"\(published_at DESC, item_id\)",
        "content_items_source_published_idx": r"\(source_id, published_at DESC\)",
        "content_items_type_published_idx": r"\(content_type, published_at DESC\)",
    }.items():
        assert re.search(shape + r" WHERE \(served > 0\)$", defs[name]), defs[name]
    assert re.search(r"USING gin \(title gin_trgm_ops\)$", defs["content_items_title_trgm_idx"])
    unique = defs["content_items_source_external_key"]
    assert (
        "UNIQUE" in unique and "(source_id, external_item_id)" in unique and "WHERE" not in unique
    )


def test_the_title_index_serves_a_substring_filter(conn: Connection) -> None:
    source = _source(conn)
    for n in range(300):
        _item(conn, source, str(n), title=f"Episode {n} about something else")
    _item(conn, source, "needle", title="The Needle in the Haystack")
    conn.execute(text("ANALYZE corpus.content_items"))
    conn.execute(text("SET LOCAL enable_seqscan = off"))
    plan = "\n".join(
        r[0]
        for r in conn.execute(
            text("EXPLAIN SELECT item_id FROM corpus.content_items WHERE title ILIKE '%needle%'")
        )
    )
    conn.rollback()
    assert "content_items_title_trgm_idx" in plan


def test_served_accepts_zero_one_two_and_refuses_three(conn: Connection) -> None:
    source = _source(conn)
    for n, served in enumerate((0, 1, 2)):
        _item(conn, source, str(n), served=served)
    with pytest.raises(sa_exc.IntegrityError, match="content_items_served_check"):
        _item(conn, source, "bad", served=3)
    conn.rollback()


def test_an_item_defaults_to_not_served_with_every_flag_false(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    row = (
        conn.execute(
            text(
                "SELECT served, has_transcript, has_gi, has_kg, has_moments, withheld, unusable, "
                "type_attrs FROM corpus.content_items WHERE item_id = :i"
            ),
            {"i": item},
        )
        .mappings()
        .one()
    )
    conn.rollback()
    assert row["served"] == 0 and row["type_attrs"] == {}
    assert not any(row[k] for k in row.keys() if k not in ("served", "type_attrs"))


@pytest.mark.parametrize(
    ("column", "constraint"),
    [
        ("natural_id", "content_items_natural_id_key"),
        ("slug", "content_items_slug_key"),
        ("external_item_id", "content_items_source_external_key"),
    ],
)
def test_an_item_is_unique_by_natural_id_slug_and_source_with_external_id(
    conn: Connection, column: str, constraint: str
) -> None:
    source = _source(conn)
    _item(conn, source, "1")
    clash = {
        column: {"natural_id": "item:1", "slug": "slug-1", "external_item_id": "ext-1"}[column]
    }
    with pytest.raises(sa_exc.IntegrityError, match=constraint):
        _item(conn, source, "2", **clash)
    conn.rollback()


def test_the_same_external_id_may_belong_to_two_sources(conn: Connection) -> None:
    a, b = _source(conn, "source:a"), _source(conn, "source:b", raw_feed_id="feed-b")
    _item(conn, a, "1", external_item_id="same")
    _item(conn, b, "2", external_item_id="same")
    conn.rollback()


def test_a_source_is_unique_by_natural_id_and_raw_feed_id(conn: Connection) -> None:
    _source(conn, "source:a", raw_feed_id="feed")
    with pytest.raises(sa_exc.IntegrityError, match="sources_natural_id_key"):
        _source(conn, "source:a")
    conn.rollback()
    _source(conn, "source:a", raw_feed_id="feed")
    with pytest.raises(sa_exc.IntegrityError, match="sources_raw_feed_id_key"):
        _source(conn, "source:b", raw_feed_id="feed")
    conn.rollback()
    _source(conn, "source:a")
    _source(conn, "source:b")
    conn.rollback()


def test_a_card_goes_with_its_item_and_an_item_needs_its_source(conn: Connection) -> None:
    source = _source(conn)
    item = _item(conn, source)
    conn.execute(
        text("INSERT INTO corpus.item_cards (item_id, title, source_title) VALUES (:i, 't', 's')"),
        {"i": item},
    )
    with pytest.raises(sa_exc.IntegrityError, match="content_items_source_id_fkey"):
        conn.execute(text("DELETE FROM corpus.sources WHERE source_id = :s"), {"s": source})
    conn.rollback()

    source = _source(conn)
    item = _item(conn, source)
    conn.execute(
        text("INSERT INTO corpus.item_cards (item_id, title, source_title) VALUES (:i, 't', 's')"),
        {"i": item},
    )
    conn.execute(text("DELETE FROM corpus.content_items WHERE item_id = :i"), {"i": item})
    assert conn.execute(text("SELECT count(*) FROM corpus.item_cards")).scalar() == 0
    conn.rollback()


def test_a_card_is_one_row_per_item(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    insert = text(
        "INSERT INTO corpus.item_cards (item_id, title, source_title) VALUES (:i, 't', 's')"
    )
    conn.execute(insert, {"i": item})
    with pytest.raises(sa_exc.IntegrityError, match="item_cards_pkey"):
        conn.execute(insert, {"i": item})
    conn.rollback()


def test_the_operations_tables_are_usable(conn: Connection) -> None:
    source = _source(conn)
    item = _item(conn, source)
    run = conn.execute(
        text(
            "INSERT INTO corpus.ingestion_runs (natural_id, source_id) "
            "VALUES ('r1', :s) RETURNING run_id"
        ),
        {"s": source},
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO corpus.run_item_outcomes (run_id, item_id, outcome) VALUES (:r, :i, 'ok')"
        ),
        {"r": run, "i": item},
    )
    conn.execute(
        text("INSERT INTO corpus.processing_stages (stage, run_id) VALUES ('gi', :r)"), {"r": run}
    )
    job = conn.execute(
        text("INSERT INTO corpus.jobs (natural_id, kind) VALUES ('j1', 'enrich') RETURNING job_id")
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO corpus.job_leases (resource, mode, holder, job_id, expires_at) "
            "VALUES ('corpus', 'exclusive', 'me', :j, now() + interval '1 minute')"
        ),
        {"j": job},
    )
    with pytest.raises(sa_exc.IntegrityError, match="job_leases_mode_check"):
        conn.execute(
            text(
                "INSERT INTO corpus.job_leases (resource, mode, holder, expires_at) "
                "VALUES ('x', 'weird', 'me', now())"
            )
        )
    conn.rollback()

    conn.execute(text("INSERT INTO corpus.corpus_meta (key, value) VALUES ('k', '1')"))
    migration = conn.execute(
        text(
            "INSERT INTO corpus.corpus_migrations (natural_id, status) VALUES ('m1', 'applied') "
            "RETURNING migration_id"
        )
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO corpus.migration_receipts (migration_id, target_table, target_key) "
            "VALUES (:m, 'content_items', '1')"
        ),
        {"m": migration},
    )
    with pytest.raises(sa_exc.IntegrityError, match="corpus_migrations_status_check"):
        conn.execute(
            text("INSERT INTO corpus.corpus_migrations (natural_id, status) VALUES ('m2', 'nope')")
        )
    conn.rollback()

    asset = conn.execute(
        text(
            "INSERT INTO corpus.assets (owner_kind, owner_id, role, path, content_type) "
            "VALUES ('source', 1, 'artwork', 'a.jpg', 'image/jpeg') RETURNING asset_id"
        )
    ).scalar_one()
    with pytest.raises(sa_exc.IntegrityError, match="assets_owner_role_key"):
        conn.execute(
            text(
                "INSERT INTO corpus.assets (owner_kind, owner_id, role, path, content_type) "
                "VALUES ('source', 1, 'artwork', 'b.jpg', 'image/jpeg')"
            )
        )
    conn.rollback()
    assert asset


def _segment(conn: Connection, item_id: int, document_id: int, seq: int, **fields: object) -> None:
    values = {
        "item_id": item_id,
        "document_id": document_id,
        "seq": seq,
        "text": f"segment {seq}",
        **fields,
    }
    conn.execute(insert(corpus_metadata.tables["corpus.segments"]).values(values))


def test_a_document_is_unique_by_item_kind_variant_and_version(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    _document(conn, item)
    _document(conn, item, variant="raw")
    _document(conn, item, kind="article")
    _document(conn, item, version=2)
    with pytest.raises(sa_exc.IntegrityError, match="documents_item_kind_variant_version_key"):
        _document(conn, item)
    conn.rollback()


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        ({"kind": "podcast"}, "documents_kind_check"),
        ({"variant": "summary"}, "documents_variant_check"),
        ({"version": 0}, "documents_version_check"),
    ],
)
def test_a_document_refuses_a_bad_kind_variant_or_version(
    conn: Connection, fields: dict[str, object], constraint: str
) -> None:
    item = _item(conn, _source(conn))
    with pytest.raises(sa_exc.IntegrityError, match=constraint):
        _document(conn, item, **fields)
    conn.rollback()


def test_a_document_version_defaults_to_one(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    document = _document(conn, item)
    assert (
        conn.execute(
            text("SELECT version FROM corpus.documents WHERE document_id = :d"), {"d": document}
        ).scalar_one()
        == 1
    )
    conn.rollback()


def test_deleting_an_item_deletes_its_documents_and_their_segments(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    document = _document(conn, item)
    _document(conn, item, version=2, supersedes_id=document)
    _segment(conn, item, document, 0)
    conn.execute(text("DELETE FROM corpus.content_items WHERE item_id = :i"), {"i": item})
    assert conn.execute(text("SELECT count(*) FROM corpus.documents")).scalar_one() == 0
    assert conn.execute(text("SELECT count(*) FROM corpus.segments")).scalar_one() == 0
    conn.rollback()


def test_the_served_document_is_the_one_nothing_supersedes(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    first = _document(conn, item)
    second = _document(conn, item, version=2, supersedes_id=first)
    served = (
        conn.execute(
            text(
                "SELECT d.document_id FROM corpus.documents d WHERE d.item_id = :i "
                "AND NOT EXISTS (SELECT 1 FROM corpus.documents s "
                "WHERE s.supersedes_id = d.document_id)"
            ),
            {"i": item},
        )
        .scalars()
        .all()
    )
    assert served == [second]
    with pytest.raises(sa_exc.IntegrityError, match="documents_supersedes_key"):
        _document(conn, item, version=3, supersedes_id=first)
    conn.rollback()


@pytest.mark.parametrize(
    "successor",
    [{"variant": "raw"}, {"kind": "article"}, {"item": "other"}],
    ids=["other variant", "other kind", "other item"],
)
def test_a_version_supersedes_only_a_document_of_its_own_series(
    conn: Connection, successor: dict[str, str]
) -> None:
    source = _source(conn)
    item = _item(conn, source)
    first = _document(conn, item)
    owner = _item(conn, source, n="2") if successor.pop("item", None) else item
    with pytest.raises(sa_exc.IntegrityError, match="documents_supersedes_fkey"):
        _document(conn, owner, version=2, supersedes_id=first, **successor)
    conn.rollback()


def test_a_segment_must_name_its_documents_own_item(conn: Connection) -> None:
    source = _source(conn)
    item, other = _item(conn, source), _item(conn, source, n="2")
    document = _document(conn, item)
    with pytest.raises(sa_exc.IntegrityError, match="segments_document_fkey"):
        _segment(conn, other, document, 0)
    conn.rollback()


def test_a_segment_lands_in_the_hash_partition_of_its_item(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    document = _document(conn, item)
    _segment(conn, item, document, 0)
    landed = conn.execute(
        text(
            "SELECT c.relname FROM corpus.segments s JOIN pg_class c ON c.oid = s.tableoid "
            "WHERE s.item_id = :i"
        ),
        {"i": item},
    ).scalar_one()
    remainder = conn.execute(
        text(
            "SELECT n FROM generate_series(0, 15) n "
            "WHERE satisfies_hash_partition('corpus.segments'::regclass, 16, n, CAST(:i AS bigint))"
        ),
        {"i": item},
    ).scalar_one()
    expected = f"segments_p{remainder:02d}"
    assert landed == expected
    conn.rollback()


def test_segments_has_sixteen_partitions_each_tuned_for_autovacuum(conn: Connection) -> None:
    rows = conn.execute(
        text(
            "SELECT c.relname, c.reloptions FROM pg_class c "
            "JOIN pg_inherits i ON i.inhrelid = c.oid "
            "WHERE i.inhparent = 'corpus.segments'::regclass ORDER BY 1"
        )
    ).all()
    assert [r[0] for r in rows] == [f"segments_p{n:02d}" for n in range(16)]
    for _name, options in rows:
        assert sorted(options) == [
            "autovacuum_analyze_scale_factor=0.005",
            "autovacuum_vacuum_scale_factor=0.01",
        ]
    assert {r[0] for r in rows} == _partitions(conn)


def test_two_documents_of_one_item_each_keep_their_own_segments(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    old = _document(conn, item)
    new = _document(conn, item, version=2, supersedes_id=old)
    _segment(conn, item, old, 0, text="old text")
    _segment(conn, item, new, 0, text="new text")
    rows = conn.execute(
        text("SELECT document_id, text FROM corpus.segments WHERE item_id = :i ORDER BY 1"),
        {"i": item},
    ).all()
    assert rows == [(old, "old text"), (new, "new text")]
    with pytest.raises(sa_exc.IntegrityError, match="segments_span_check"):
        _segment(conn, item, new, 1, start_ms=500, end_ms=100)
    conn.rollback()


# --- entities and identity builds --------------------------------------------------------------


def _entity(conn: Connection, n: str = "1", **fields: object) -> int:
    values = {
        "natural_id": f"person:p{n}",
        "type": "person",
        "label": f"Person {n}",
        "normalized_label": f"person {n}",
        **fields,
    }
    table = corpus_metadata.tables["corpus.entities"]
    entity_id: int = conn.execute(
        insert(table).values(values).returning(table.c.entity_id)
    ).scalar_one()
    return entity_id


def _build(conn: Connection, state: str = "built") -> int:
    table = corpus_metadata.tables["corpus.identity_builds"]
    build_id: int = conn.execute(
        insert(table).values({"rule_version": "r1", "state": state}).returning(table.c.build_id)
    ).scalar_one()
    return build_id


def _decision(conn: Connection, build_id: int, **fields: object) -> uuid.UUID:
    values = {
        "build_id": build_id,
        "scope": "person",
        "action": "merge",
        "subject_a": "person:p1",
        "subject_b": "person:p2",
        "basis": "rule",
        "decided_by": "test",
        "decided_at": text("now()"),
        **fields,
    }
    table = corpus_metadata.tables["corpus.identity_changes"]
    decision_id: uuid.UUID = conn.execute(
        insert(table).values(values).returning(table.c.decision_id)
    ).scalar_one()
    return decision_id


def _cluster(conn: Connection, build_id: int, cluster_id: int, canonical_id: int) -> None:
    conn.execute(
        insert(corpus_metadata.tables["corpus.identity_clusters"]).values(
            build_id=build_id,
            cluster_id=cluster_id,
            scope="person",
            canonical_id=canonical_id,
            member_count=1,
        )
    )


def _member(
    conn: Connection, build_id: int, cluster_id: int, member_id: int, **fields: object
) -> None:
    conn.execute(
        insert(corpus_metadata.tables["corpus.identity_members"]).values(
            {
                "build_id": build_id,
                "cluster_id": cluster_id,
                "member_id": member_id,
                "scope": "person",
                **fields,
            }
        )
    )


def _canonical_of(conn: Connection, entity_id: int) -> int:
    found: int = conn.execute(
        text("SELECT canonical_id FROM corpus.entities WHERE entity_id = :e"), {"e": entity_id}
    ).scalar_one()
    return found


def test_an_entity_inserted_without_a_canonical_is_its_own_canonical(conn: Connection) -> None:
    entity = _entity(conn)
    assert _canonical_of(conn, entity) == entity
    conn.rollback()


def test_the_canonical_trigger_exists_on_entities_only_for_inserts(conn: Connection) -> None:
    rows = conn.execute(
        text(
            "SELECT t.tgname, p.proname, pg_get_triggerdef(t.oid) FROM pg_trigger t "
            "JOIN pg_proc p ON p.oid = t.tgfoid "
            "WHERE t.tgrelid = 'corpus.entities'::regclass AND NOT t.tgisinternal"
        )
    ).all()
    assert [(r[0], r[1]) for r in rows] == [
        ("entities_canonical_default", "entities_canonical_default")
    ]
    assert "BEFORE INSERT" in rows[0][2]


def test_a_member_keeps_the_canonical_it_names(conn: Connection) -> None:
    canonical = _entity(conn)
    member = _entity(conn, "2", canonical_id=canonical)
    assert _canonical_of(conn, member) == canonical
    assert _canonical_of(conn, canonical) == canonical
    conn.rollback()


def test_a_canonical_naming_no_row_is_refused_at_commit(conn: Connection) -> None:
    _entity(conn, canonical_id=999_999_999)
    with pytest.raises(sa_exc.IntegrityError, match="entities_canonical_fkey"):
        conn.commit()
    conn.rollback()


def test_a_canonical_lookup_returns_the_canonical_and_its_members(conn: Connection) -> None:
    canonical = _entity(conn)
    member = _entity(conn, "2", canonical_id=canonical)
    _entity(conn, "3")
    found = conn.execute(
        text("SELECT entity_id FROM corpus.entities WHERE canonical_id = :c"), {"c": canonical}
    ).scalars()
    assert sorted(found) == sorted([canonical, member])
    conn.rollback()


def test_an_entity_is_unique_by_natural_id(conn: Connection) -> None:
    _entity(conn)
    with pytest.raises(sa_exc.IntegrityError, match="entities_natural_id_key"):
        _entity(conn, "2", natural_id="person:p1")
    conn.rollback()


def _identifier(conn: Connection, entity_id: int, scheme: str, value: str) -> None:
    conn.execute(
        insert(corpus_metadata.tables["corpus.entity_identifiers"]).values(
            entity_id=entity_id, scheme=scheme, value=value
        )
    )


def test_an_identifier_belongs_to_one_entity_and_one_value_per_scheme(conn: Connection) -> None:
    first, second = _entity(conn), _entity(conn, "2")
    _identifier(conn, first, "wikidata", "Q1")
    _identifier(conn, first, "isni", "Q1")
    _identifier(conn, second, "wikidata", "Q2")
    with pytest.raises(sa_exc.IntegrityError, match="entity_identifiers_scheme_value_key"):
        _identifier(conn, second, "isni", "Q1")
    conn.rollback()
    first = _entity(conn)
    _identifier(conn, first, "wikidata", "Q1")
    with pytest.raises(sa_exc.IntegrityError, match="entity_identifiers_pkey"):
        _identifier(conn, first, "wikidata", "Q9")
    conn.rollback()


def test_an_entitys_identifiers_go_with_it(conn: Connection) -> None:
    entity = _entity(conn)
    _identifier(conn, entity, "wikidata", "Q1")
    conn.execute(text("DELETE FROM corpus.entities WHERE entity_id = :e"), {"e": entity})
    assert conn.execute(text("SELECT count(*) FROM corpus.entity_identifiers")).scalar_one() == 0
    conn.rollback()


def test_at_most_one_identity_build_is_published(conn: Connection) -> None:
    first = _build(conn, "published")
    with pytest.raises(sa_exc.IntegrityError, match="identity_builds_one_published_idx"):
        _build(conn, "published")
    conn.rollback()
    first = _build(conn, "published")
    _build(conn, "built")
    _build(conn, "retired")
    conn.execute(
        text("UPDATE corpus.identity_builds SET state = 'retired' WHERE build_id = :b"),
        {"b": first},
    )
    _build(conn, "published")
    conn.rollback()


def test_an_identity_build_refuses_a_bad_state(conn: Connection) -> None:
    with pytest.raises(sa_exc.IntegrityError, match="identity_builds_state_check"):
        _build(conn, "draft")
    conn.rollback()


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        ({"scope": "place"}, "identity_changes_scope_check"),
        ({"action": "split"}, "identity_changes_action_check"),
        ({"basis": "guess"}, "identity_changes_basis_check"),
        ({"status": "pending"}, "identity_changes_status_check"),
    ],
)
def test_an_identity_change_refuses_a_bad_scope_action_basis_or_status(
    conn: Connection, fields: dict[str, object], constraint: str
) -> None:
    build = _build(conn)
    with pytest.raises(sa_exc.IntegrityError, match=constraint):
        _decision(conn, build, **fields)
    conn.rollback()


def test_an_identity_change_gets_a_uuid_and_is_accepted_by_default(conn: Connection) -> None:
    decision = _decision(conn, _build(conn))
    assert isinstance(decision, uuid.UUID)
    assert (
        conn.execute(
            text("SELECT status FROM corpus.identity_changes WHERE decision_id = :d"),
            {"d": decision},
        ).scalar_one()
        == "accepted"
    )
    conn.rollback()


def test_a_cluster_refuses_a_bad_scope_or_an_empty_member_count(conn: Connection) -> None:
    build, entity = _build(conn), _entity(conn)
    table = corpus_metadata.tables["corpus.identity_clusters"]
    values = {"build_id": build, "cluster_id": 1, "canonical_id": entity}
    with pytest.raises(sa_exc.IntegrityError, match="identity_clusters_scope_check"):
        conn.execute(insert(table).values(**values, scope="place", member_count=1))
    conn.rollback()
    build, entity = _build(conn), _entity(conn)
    values = {"build_id": build, "cluster_id": 1, "canonical_id": entity}
    with pytest.raises(sa_exc.IntegrityError, match="identity_clusters_member_count_check"):
        conn.execute(insert(table).values(**values, scope="person", member_count=0))
    conn.rollback()


def test_a_member_must_name_a_cluster_of_its_own_build(conn: Connection) -> None:
    first, second = _build(conn), _build(conn)
    canonical, other = _entity(conn), _entity(conn, "2")
    _cluster(conn, first, 1, canonical)
    _member(conn, first, 1, canonical)
    with pytest.raises(sa_exc.IntegrityError, match="identity_members_cluster_fkey"):
        _member(conn, second, 1, other)
    conn.rollback()


def test_a_member_is_of_its_clusters_scope(conn: Connection) -> None:
    build = _build(conn)
    canonical, other = _entity(conn), _entity(conn, "2")
    _cluster(conn, build, 1, canonical)
    with pytest.raises(sa_exc.IntegrityError, match="identity_members_cluster_fkey"):
        _member(conn, build, 1, other, scope="topic")
    conn.rollback()


def test_a_written_id_is_in_one_cluster_per_build(conn: Connection) -> None:
    build = _build(conn)
    canonical, other = _entity(conn), _entity(conn, "2")
    _cluster(conn, build, 1, canonical)
    _cluster(conn, build, 2, other)
    _member(conn, build, 1, canonical)
    with pytest.raises(sa_exc.IntegrityError, match="identity_members_pkey"):
        _member(conn, build, 2, canonical)
    conn.rollback()


def test_deleting_a_build_deletes_its_clusters_members_and_changes(conn: Connection) -> None:
    build = _build(conn)
    canonical = _entity(conn)
    decision = _decision(conn, build)
    _cluster(conn, build, 1, canonical)
    _member(conn, build, 1, canonical, decision_id=decision)
    conn.execute(text("DELETE FROM corpus.identity_builds WHERE build_id = :b"), {"b": build})
    for name in ("identity_changes", "identity_clusters", "identity_members"):
        table = corpus_metadata.tables[f"corpus.{name}"]
        assert conn.execute(select(func.count()).select_from(table)).scalar_one() == 0
    conn.rollback()


def test_a_segment_speaker_must_be_an_entity_and_is_cleared_when_it_goes(conn: Connection) -> None:
    item = _item(conn, _source(conn))
    document = _document(conn, item)
    with pytest.raises(sa_exc.IntegrityError, match="segments_speaker_fkey"):
        _segment(conn, item, document, 0, speaker_id=999_999_999)
    conn.rollback()
    item = _item(conn, _source(conn))
    document = _document(conn, item)
    speaker = _entity(conn)
    _segment(conn, item, document, 0, speaker_id=speaker)
    conn.execute(text("DELETE FROM corpus.entities WHERE entity_id = :e"), {"e": speaker})
    assert conn.execute(text("SELECT speaker_id FROM corpus.segments")).scalar_one() is None
    conn.rollback()


def test_the_corpus_history_has_one_head() -> None:
    scripts = ScriptDirectory.from_config(migrate._config(history=migrate.CORPUS))
    assert scripts.get_heads() == ["corpus_0005_entities"]
    assert [r.revision for r in reversed(list(scripts.walk_revisions()))] == [
        "corpus_0001_bookkeeping",
        "corpus_0002_operations",
        "corpus_0003_sources_and_items",
        "corpus_0004_documents",
        "corpus_0005_entities",
    ]
    assert migrate.head_revision("corpus") == "corpus_0005_entities"


def test_the_metadata_is_the_migrated_shape(conn: Connection) -> None:
    partitions = _partitions(conn)
    ctx = MigrationContext.configure(
        conn,
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "include_schemas": True,
            # schema_history is the kernel's table definition, written there by migrate; the
            # partitions of segments are the migration's, not the metadata's.
            "include_name": lambda name, kind, _parent: (
                name == "corpus"
                if kind == "schema"
                else name not in ("alembic_version", "schema_history") and name not in partitions
            ),
        },
    )
    assert compare_metadata(ctx, corpus_metadata) == []


def test_the_shape_names_every_corpus_table_and_index(conn: Connection) -> None:
    shape = _shape(conn)
    partitions = _partitions(conn)
    assert {t for (s, t, *_r) in shape["columns"] if s == "corpus"} - partitions == TABLES
    assert {i for (s, i, _d) in shape["indexes"] if s == "corpus"} >= INDEXES


def test_the_offline_sql_creates_the_corpus_tables_without_schema_or_extension() -> None:
    out = migrate.upgrade_sql(None, "player")
    corpus = out[out.index("-- schema corpus:") : out.index("-- plane player:")]
    for table in TABLES:
        assert re.search(rf'CREATE TABLE "?corpus"?\.{table} \(', corpus), table
    assert re.search(r"CREATE INDEX content_items_published_idx ON corpus\.content_items", corpus)
    assert "WHERE served > 0" in corpus
    assert "gin_trgm_ops" in corpus
    assert "PARTITION BY HASH (item_id)" in corpus
    assert "CREATE FUNCTION corpus.entities_canonical_default()" in corpus
    assert re.search(
        r"CREATE TRIGGER entities_canonical_default BEFORE INSERT ON corpus\.entities", corpus
    )
    assert "ADD CONSTRAINT segments_speaker_fkey FOREIGN KEY(speaker_id)" in corpus
    for n in range(16):
        assert re.search(
            rf"CREATE TABLE corpus\.segments_p{n:02d} PARTITION OF corpus\.segments "
            rf"FOR VALUES WITH \(MODULUS 16, REMAINDER {n}\)",
            corpus,
        ), n
    assert not re.search(r"CREATE (SCHEMA|EXTENSION)|GRANT", corpus, re.I)
    assert not re.search(r"INSERT INTO \"?corpus\"?\.schema_history", corpus)
    assert out.index("-- schema shared:") < out.index("-- schema corpus:")


# --- readiness ---------------------------------------------------------------------------------


def _set_corpus_revision(conn: Connection, revision: str) -> None:
    conn.execute(text("UPDATE corpus.alembic_version SET version_num = :r"), {"r": revision})
    conn.commit()


@pytest.fixture
def restore_corpus(conn: Connection) -> Iterator[None]:
    head = migrate.head_revision("corpus")
    yield
    conn.rollback()
    conn.execute(text("DELETE FROM corpus.schema_history WHERE revision LIKE '9999%'"))
    _set_corpus_revision(conn, head)


def test_corpus_readiness_at_head_behind_and_ahead(conn: Connection, restore_corpus: None) -> None:
    head = migrate.head_revision("corpus")
    assert migrate.readiness(conn, "corpus") == ("at_head", head, head)

    _set_corpus_revision(conn, "corpus_0002_operations")
    assert migrate.readiness(conn, "corpus") == ("behind", "corpus_0002_operations", head)

    # A later release migrated past this code: its head was applied here before.
    conn.execute(text("INSERT INTO corpus.schema_history (revision) VALUES (:r)"), {"r": head})
    _set_corpus_revision(conn, "9999_later")
    assert migrate.readiness(conn, "corpus") == ("ahead", "9999_later", head)
    conn.execute(text("DELETE FROM corpus.schema_history WHERE revision = :r"), {"r": head})
    conn.commit()
    assert migrate.readiness(conn, "corpus")[0] == "unknown"


def test_the_corpus_engine_refuses_a_corpus_behind_the_code_and_accepts_one_ahead(
    url: str, conn: Connection, restore_corpus: None
) -> None:
    settings = CorpusSettings(url=url)
    head = migrate.head_revision("corpus")
    reset_engines()
    _set_corpus_revision(conn, "corpus_0001_bookkeeping")
    with pytest.raises(DatabaseNotReady, match=r"schema 'corpus' is at revision corpus_0001"):
        init_corpus_engine(settings)

    conn.execute(text("INSERT INTO corpus.schema_history (revision) VALUES (:r)"), {"r": head})
    _set_corpus_revision(conn, "9999_later")
    init_corpus_engine(settings)
    assert corpus_ro_engine() is not None
    reset_engines()


def _check(url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> int:
    monkeypatch.setenv("PODCAST_DATABASE_CORPUS_RO_URL", url)
    code = run_db_cli(parse_db_argv(["check"]), logging.getLogger("t"))
    return code


def test_db_check_passes_at_head_and_ahead_and_fails_behind(
    url: str,
    conn: Connection,
    restore_corpus: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    head = migrate.head_revision("corpus")
    assert _check(url, monkeypatch, capsys) == 0
    assert f"schema corpus: at_head, current {head}, head {head}" in capsys.readouterr().out

    _set_corpus_revision(conn, "corpus_0002_operations")
    assert _check(url, monkeypatch, capsys) == 2
    assert "schema corpus: behind, current corpus_0002_operations" in capsys.readouterr().out

    # This code's head is in schema_history, so a later revision is a release migrated past it.
    _set_corpus_revision(conn, "9999_later")
    assert _check(url, monkeypatch, capsys) == 0
    assert "schema corpus: ahead, current 9999_later" in capsys.readouterr().out


def test_db_check_runs_as_a_role_that_may_only_read_the_corpus_version(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prod runs it as corpus_ro: USAGE on corpus and SELECT on its version tables, no write."""
    name = f"corpus_ro_probe_{uuid.uuid4().hex[:10]}"
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD 'probe'"))
            c.execute(text(f"GRANT USAGE ON SCHEMA corpus TO {name}"))
            c.execute(
                text(f"GRANT SELECT ON corpus.alembic_version, corpus.schema_history TO {name}")
            )
        role_url = (
            make_url(url).set(username=name, password="probe").render_as_string(hide_password=False)
        )
        assert _check(role_url, monkeypatch, capsys) == 0
        assert "schema corpus: at_head" in capsys.readouterr().out
    finally:
        with admin.connect() as c:
            c.execute(text(f"DROP OWNED BY {name}"))
            c.execute(text(f"DROP ROLE IF EXISTS {name}"))
        admin.dispose()


def test_the_plane_engines_start_on_a_role_with_no_corpus_privilege(url: str) -> None:
    """Prod: the plane roles hold nothing on ``corpus``; only corpus_ro reads its version."""
    name = f"plane_probe_{uuid.uuid4().hex[:10]}"
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    reset_engines()
    try:
        with admin.connect() as c:
            c.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD 'probe'"))
            for schema in ("player", "shared"):
                c.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {name}"))
                c.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {name}"))
        role_url = (
            make_url(url).set(username=name, password="probe").render_as_string(hide_password=False)
        )
        init_engines(DatabaseSettings(rw_url=role_url, ro_url=role_url))
        assert rw_engine() is not None
    finally:
        reset_engines()
        with admin.connect() as c:
            c.execute(text(f"DROP OWNED BY {name}"))
            c.execute(text(f"DROP ROLE IF EXISTS {name}"))
        admin.dispose()


def test_a_fresh_corpus_schema_writes_a_history_row_per_revision(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CL_RELEASE", "b5a243d")
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    reset_engines()
    with admin.connect() as c:
        c.execute(text("DROP SCHEMA IF EXISTS corpus CASCADE"))
    try:
        assert migrate.upgrade_corpus(url) == [
            "corpus_0001_bookkeeping",
            "corpus_0002_operations",
            "corpus_0003_sources_and_items",
            "corpus_0004_documents",
            "corpus_0005_entities",
        ]
        rows = migrate.history_rows(url, "corpus")
        assert [r["revision"] for r in rows] == [
            "corpus_0005_entities",
            "corpus_0004_documents",
            "corpus_0003_sources_and_items",
            "corpus_0002_operations",
            "corpus_0001_bookkeeping",
        ]
        assert {r["release"] for r in rows} == {"b5a243d"}
        assert migrate.upgrade_corpus(url) == []
    finally:
        admin.dispose()
        migrate.upgrade(url, "operator")


def test_an_upgrade_waits_while_another_holds_the_schemas_migration_lock(url: str) -> None:
    holder = create_engine(normalize_url(url))
    done = threading.Event()
    failed: list[BaseException] = []

    def upgrade() -> None:
        try:
            migrate.upgrade_corpus(url)
        except BaseException as exc:  # handed to the test thread below
            failed.append(exc)
        finally:
            done.set()

    try:
        with holder.connect() as c:
            c.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:name, 0))"),
                {"name": migrate.migration_lock_name("corpus")},
            )
            worker = threading.Thread(target=upgrade)
            worker.start()
            assert not done.wait(2.0), "the upgrade ran while the lock was held"
            c.rollback()
        assert done.wait(60.0)
        worker.join()
        assert not failed, failed
    finally:
        holder.dispose()


def test_two_parallel_corpus_upgrades_apply_each_revision_once(url: str) -> None:
    """Prod's plane deploys migrate in parallel; each brings corpus to head."""
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    reset_engines()
    with admin.connect() as c:
        c.execute(text("DROP SCHEMA IF EXISTS corpus CASCADE"))
    start = threading.Barrier(2)
    results: list[list[str]] = []
    failed: list[BaseException] = []

    def upgrade() -> None:
        start.wait()
        try:
            results.append(migrate.upgrade_corpus(url))
        except BaseException as exc:  # handed to the test thread below
            failed.append(exc)

    try:
        workers = [threading.Thread(target=upgrade) for _ in range(2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(120)
        assert not failed, failed
        assert sorted(len(r) for r in results) == [0, 5]
        assert len(migrate.history_rows(url, "corpus", limit=10)) == 5
    finally:
        admin.dispose()
        migrate.upgrade(url, "operator")


def test_upgrade_all_runs_corpus_between_shared_and_the_kernel(url: str) -> None:
    applied = migrate.upgrade_all(url, "player")
    assert list(applied)[:3] == ["shared", "corpus", "kernel"]


def test_every_corpus_table_names_its_schema_so_no_translate_map_is_needed(
    conn: Connection,
) -> None:
    """The ``corpus_ro`` engine carries no ``schema_translate_map``: a query built from the
    metadata must render ``corpus.<table>`` by itself, and run on a connection with no map."""

    assert {t.schema for t in corpus_metadata.tables.values()} == {"corpus"}
    for table in corpus_metadata.sorted_tables:
        sql = str(select(table).compile(dialect=postgresql.dialect()))
        assert f"FROM corpus.{table.name}" in sql, sql
        conn.execute(select(table).limit(1))
    assert not conn.get_execution_options().get("schema_translate_map")


def test_the_two_plane_upgrades_prod_runs_apply_corpus_once(url: str) -> None:
    """Prod runs ``db upgrade --plane player`` then ``--plane operator``, never a corpus step."""

    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    reset_engines()
    with admin.connect() as c:
        c.execute(text("DROP SCHEMA IF EXISTS corpus CASCADE"))
    env = {**os.environ, "PODCAST_DATABASE_URL": url, "PYTHONPATH": str(REPO / "src")}
    try:
        for plane in ("player", "operator"):
            done = subprocess.run(
                [sys.executable, "-m", "podcast_scraper.cli", "db", "upgrade", "--plane", plane],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
                check=False,
            )
            assert done.returncode == 0, done.stderr
            head = migrate.head_revision("corpus")
            assert f"schema corpus: {head} (head {head})" in done.stdout
            first = plane == "player"
            assert ("corpus_0001_bookkeeping" in done.stdout.split("schema corpus:")[1]) is first
        revisions = [r["revision"] for r in migrate.history_rows(url, "corpus", limit=10)]
        assert len(revisions) == 5
    finally:
        admin.dispose()
        migrate.upgrade(url, "operator")


def test_the_shared_schema_ahead_is_read_through_an_app_engine(url: str) -> None:
    """The engines carry a connection-level ``None -> plane`` map; the history read must still
    be the schema's own, as for corpus."""
    settings = DatabaseSettings(rw_url=url, ro_url=url)
    head = migrate.head_revision("shared")
    reset_engines()
    admin = create_engine(normalize_url(url))
    try:
        with admin.begin() as c:
            c.execute(text("UPDATE shared.alembic_version SET version_num = '9999_later'"))
            c.execute(text("INSERT INTO shared.schema_history (revision) VALUES (:r)"), {"r": head})
        init_engines(settings)
        with rw_engine().connect() as c:
            assert migrate.readiness(c, "shared") == ("ahead", "9999_later", head)
            assert migrate.readiness(c, "player")[0] == "at_head"
    finally:
        reset_engines()
        with admin.begin() as c:
            c.execute(text("DELETE FROM shared.schema_history WHERE revision = :r"), {"r": head})
            c.execute(text("UPDATE shared.alembic_version SET version_num = :r"), {"r": head})
        admin.dispose()


def test_upgrade_creates_pg_trgm_only_when_it_is_missing(url: str) -> None:
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    name = f"trgm_{os.getpid()}"
    try:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            c.execute(text(f'CREATE DATABASE "{name}"'))
        scratch = (
            make_url(normalize_url(url)).set(database=name).render_as_string(hide_password=False)
        )
        assert migrate.upgrade_corpus(scratch)[-1] == "corpus_0005_entities"
        probe = create_engine(normalize_url(scratch))
        with probe.connect() as c:
            assert c.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")).first()
        probe.dispose()
        with captured_statements() as seen:
            migrate.upgrade_corpus(scratch)
        assert not [sql for sql, _ in seen if "CREATE EXTENSION" in sql.upper()]
    finally:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
