"""Run and inspect the Alembic histories (ADR-169), never at app start.

Two histories: the kernel's, run once per plane schema, and the ``shared`` schema's (the rows both
planes read and write). ``upgrade --plane P`` brings ``shared`` to its head first, in its own
transaction, then the plane; a failed plane upgrade leaves ``shared`` where it got to.
"""

from __future__ import annotations

import io
from datetime import timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, Engine, text
from sqlalchemy.schema import CreateSchema

from podcast_scraper import build_info
from podcast_scraper.db.engines import application_name_args
from podcast_scraper.db.schema import schema_history, SHARED_SCHEMA
from podcast_scraper.db.settings import normalize_url, validate_plane

_HERE = Path(__file__).parent
_PLANE_SCRIPTS = _HERE / "migrations"
_SHARED_SCRIPTS = _HERE / "migrations_shared"


def _scripts(schema: str | None) -> Path:
    return _SHARED_SCRIPTS if schema == SHARED_SCHEMA else _PLANE_SCRIPTS


def _config(schema: str | None = None, connection: Connection | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_scripts(schema)))
    if schema is not None:
        cfg.attributes["plane"] = schema
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def head_revision(schema: str | None = None) -> str:
    """The head of the shipped history: the kernel's (any plane), or ``shared``'s."""
    head = ScriptDirectory.from_config(_config(schema)).get_current_head()
    assert head is not None
    return head


def current_revision(connection: Connection, schema: str) -> str | None:
    """The schema's applied revision, or None when it has never been upgraded."""
    ctx = MigrationContext.configure(connection, opts={"version_table_schema": schema})
    return ctx.get_current_revision()


def _ensure_schema(conn: Connection, schema: str) -> None:
    """Create the schema only when it is missing.

    In prod infra pre-creates the schemas and the migrating role owns them without holding CREATE
    on the database; Postgres checks that privilege before the IF NOT EXISTS short-circuit, so an
    unconditional ``CREATE SCHEMA IF NOT EXISTS`` would fail there.
    """
    present = conn.execute(
        text("SELECT 1 FROM pg_namespace WHERE nspname = :name"), {"name": schema}
    ).first()
    if present is None:
        conn.execute(CreateSchema(schema, if_not_exists=True))


def _engine(url: str) -> Engine:
    return create_engine(normalize_url(url), connect_args=application_name_args())


def _upgrade_schema(engine: Engine, schema: str, revision: str) -> list[str]:
    """One history on one schema, with its ``schema_history`` rows, in one transaction."""
    applied: list[tuple[str, int]] = []
    with engine.begin() as conn:
        _ensure_schema(conn, schema)
    with engine.connect() as conn:
        cfg = _config(schema, conn)
        cfg.attributes["applied"] = applied
        command.upgrade(cfg, revision)
        _record_history(conn, schema, applied)
        conn.commit()
    return [rev for rev, _ in applied]


def upgrade(url: str, plane: str, revision: str = "head") -> list[str]:
    """Bring ``shared`` to its head, then the plane to ``revision`` (schemas created if missing).

    Appends one ``schema_history`` row per revision each run applied (none when already there),
    in the same transaction as that upgrade. Returns the plane's applied revisions, oldest first;
    :func:`upgrade_shared` reports ``shared``'s.
    """
    validate_plane(plane)
    engine = _engine(url)
    try:
        _upgrade_schema(engine, SHARED_SCHEMA, "head")
        return _upgrade_schema(engine, plane, revision)
    finally:
        engine.dispose()


def upgrade_shared(url: str) -> list[str]:
    """Bring ``shared`` to its head; the revisions applied, oldest first."""
    engine = _engine(url)
    try:
        return _upgrade_schema(engine, SHARED_SCHEMA, "head")
    finally:
        engine.dispose()


def _record_history(conn: Connection, schema: str, applied: list[tuple[str, int]]) -> None:
    """Insert the run's applied revisions, when the schema has a ``schema_history`` table yet."""
    if not applied or not _has_history_table(conn, schema):
        return
    release, source = build_info.release()
    conn.execute(
        schema_history.insert().execution_options(schema_translate_map={None: schema}),
        [
            {"revision": rev, "release": release, "release_source": source, "duration_ms": ms}
            for rev, ms in applied
        ],
    )


def _has_history_table(conn: Connection, schema: str) -> bool:
    found = conn.execute(
        text("SELECT to_regclass(:name)"), {"name": f'"{schema}".schema_history'}
    ).scalar()
    return found is not None


def history(conn: Connection, schema: str, limit: int = 5) -> list[dict[str, object]]:
    """The schema's last ``limit`` ``schema_history`` rows, newest first (``[]`` before any)."""
    if not _has_history_table(conn, schema):
        return []
    rows = conn.execute(
        schema_history.select()
        .order_by(schema_history.c.id.desc())
        .limit(limit)
        .execution_options(schema_translate_map={None: schema})
    ).mappings()
    return [dict(r) for r in rows]


def _status(url: str, schema: str) -> tuple[str | None, str]:
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            return current_revision(conn, schema), head_revision(schema)
    finally:
        engine.dispose()


def status(url: str, plane: str) -> tuple[str | None, str]:
    """``(current, head)`` for one plane."""
    validate_plane(plane)
    return _status(url, plane)


def shared_status(url: str) -> tuple[str | None, str]:
    """``(current, head)`` for the ``shared`` schema."""
    return _status(url, SHARED_SCHEMA)


def history_rows(url: str, schema: str, limit: int = 5) -> list[dict[str, object]]:
    """``history`` over a fresh connection to ``url``, with ``applied_at`` in UTC."""
    if schema != SHARED_SCHEMA:
        validate_plane(schema)
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            rows = history(conn, schema, limit)
    finally:
        engine.dispose()
    for row in rows:
        row["applied_at"] = row["applied_at"].astimezone(timezone.utc)  # type: ignore[attr-defined]
    return rows


def _render(schema: str, start: str | None, label: str) -> str:
    head = head_revision(schema)
    if start == head:
        return f"-- {label} {schema} is already at {head}; nothing to run\n"
    cfg = _config(schema)
    buffer = io.StringIO()
    cfg.output_buffer = buffer
    command.upgrade(cfg, f"{start or 'base'}:{head}", sql=True)
    return (
        f'-- {label} {schema}: {start or "base"} -> {head}. '
        f'The schema "{schema}" must already exist.\n' + buffer.getvalue()
    )


def upgrade_sql(url: str | None, plane: str, *, from_revision: str | None = None) -> str:
    """The SQL ``upgrade`` would run for ``plane``, as text; nothing is executed.

    ``shared`` first, then the plane, as ``upgrade`` runs them. Each starts from its current
    revision, read with a read-only query when a URL is configured, else from base;
    ``from_revision`` overrides the plane's start. Already at head: a comment line.
    """
    validate_plane(plane)
    start: str | None = None
    shared_start: str | None = None
    if url:
        engine = create_engine(
            normalize_url(url),
            connect_args={
                "options": "-c default_transaction_read_only=on",
                **application_name_args(),
            },
        )
        try:
            with engine.connect() as conn:
                start = current_revision(conn, plane)
                shared_start = current_revision(conn, SHARED_SCHEMA)
        finally:
            engine.dispose()
    if from_revision is not None:
        start = from_revision
    return _render(SHARED_SCHEMA, shared_start, "schema") + _render(plane, start, "plane")
