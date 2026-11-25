"""The ``corpus`` schema's built-in history: its tables, indexes and readiness (ADR-184)."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, exc as sa_exc, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import init_engines, migrate, reset_engines, rw_engine
from podcast_scraper.db.engines import DatabaseNotReady
from podcast_scraper.db.schema import corpus_metadata
from podcast_scraper.db.settings import DatabaseSettings, normalize_url
from tests._database import worker_database_url
from tests.unit.podcast_scraper.db.test_db_migrate import _shape

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
    cols = ", ".join(values)
    binds = ", ".join(f":{k}" for k in values)
    return conn.execute(
        text(f"INSERT INTO corpus.sources ({cols}) VALUES ({binds}) RETURNING source_id"), values
    ).scalar_one()


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
    cols = ", ".join(values)
    binds = ", ".join(f":{k}" for k in values)
    return conn.execute(
        text(f"INSERT INTO corpus.content_items ({cols}) VALUES ({binds}) RETURNING item_id"),
        values,
    ).scalar_one()


def test_upgrade_to_head_creates_every_table_and_index(conn: Connection) -> None:
    tables = {
        r[0]
        for r in conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'corpus'")
        ).all()
    }
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
    defs = dict(
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


def test_the_corpus_history_has_one_head() -> None:
    scripts = ScriptDirectory.from_config(migrate._config(history=migrate.CORPUS))
    assert scripts.get_heads() == ["corpus_0003_sources_and_items"]
    assert [r.revision for r in reversed(list(scripts.walk_revisions()))] == [
        "corpus_0001_bookkeeping",
        "corpus_0002_operations",
        "corpus_0003_sources_and_items",
    ]
    assert migrate.head_revision("corpus") == "corpus_0003_sources_and_items"


def test_the_metadata_is_the_migrated_shape(conn: Connection) -> None:
    ctx = MigrationContext.configure(
        conn,
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "include_schemas": True,
            # schema_history is the kernel's table definition, written there by migrate.
            "include_name": lambda name, kind, _parent: (
                name == "corpus"
                if kind == "schema"
                else name not in ("alembic_version", "schema_history")
            ),
        },
    )
    assert compare_metadata(ctx, corpus_metadata) == []


def test_the_shape_names_every_corpus_table_and_index(conn: Connection) -> None:
    shape = _shape(conn)
    assert {t for (s, t, *_r) in shape["columns"] if s == "corpus"} == TABLES
    assert {i for (s, i, _d) in shape["indexes"] if s == "corpus"} >= INDEXES


def test_the_offline_sql_creates_the_corpus_tables_without_schema_or_extension() -> None:
    out = migrate.upgrade_sql(None, "player")
    corpus = out[out.index("-- schema corpus:") : out.index("-- plane player:")]
    for table in TABLES:
        assert re.search(rf'CREATE TABLE "?corpus"?\.{table} \(', corpus), table
    assert re.search(r"CREATE INDEX content_items_published_idx ON corpus\.content_items", corpus)
    assert "WHERE served > 0" in corpus
    assert "gin_trgm_ops" in corpus
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


def test_the_engines_refuse_a_corpus_behind_the_code_and_accept_one_ahead(
    url: str, conn: Connection, restore_corpus: None
) -> None:
    settings = DatabaseSettings(rw_url=url, ro_url=url)
    head = migrate.head_revision("corpus")
    reset_engines()
    _set_corpus_revision(conn, "corpus_0001_bookkeeping")
    with pytest.raises(DatabaseNotReady, match=r"schema 'corpus' is at revision corpus_0001"):
        init_engines(settings)

    conn.execute(text("INSERT INTO corpus.schema_history (revision) VALUES (:r)"), {"r": head})
    _set_corpus_revision(conn, "9999_later")
    init_engines(settings)
    assert rw_engine() is not None
    reset_engines()
    init_engines(settings)


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
        ]
        rows = migrate.history_rows(url, "corpus")
        assert [r["revision"] for r in rows] == [
            "corpus_0003_sources_and_items",
            "corpus_0002_operations",
            "corpus_0001_bookkeeping",
        ]
        assert {r["release"] for r in rows} == {"b5a243d"}
        assert migrate.upgrade_corpus(url) == []
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
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    assert {t.schema for t in corpus_metadata.tables.values()} == {"corpus"}
    for table in corpus_metadata.sorted_tables:
        sql = str(select(table).compile(dialect=postgresql.dialect()))
        assert f"FROM corpus.{table.name}" in sql, sql
        conn.execute(select(table).limit(1))
    assert not conn.get_execution_options().get("schema_translate_map")
