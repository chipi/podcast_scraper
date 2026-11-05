"""Run and inspect the Alembic history (ADR-169): once per plane schema, never at app start."""

from __future__ import annotations

import io
from datetime import timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.schema import CreateSchema

from podcast_scraper import build_info
from podcast_scraper.db.schema import schema_history
from podcast_scraper.db.settings import normalize_url, validate_plane

_SCRIPT_LOCATION = Path(__file__).parent / "migrations"


def _config(plane: str | None = None, connection: Connection | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_SCRIPT_LOCATION))
    if plane is not None:
        cfg.attributes["plane"] = plane
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def head_revision() -> str:
    """The head revision of the shipped Alembic history."""
    head = ScriptDirectory.from_config(_config()).get_current_head()
    assert head is not None
    return head


def current_revision(connection: Connection, plane: str) -> str | None:
    """The plane's applied revision, or None when the plane has never been upgraded."""
    ctx = MigrationContext.configure(connection, opts={"version_table_schema": plane})
    return ctx.get_current_revision()


def _ensure_schema(conn: Connection, plane: str) -> None:
    """Create the plane schema only when it is missing.

    In prod infra pre-creates the schemas and the migrating role owns them without holding CREATE
    on the database; Postgres checks that privilege before the IF NOT EXISTS short-circuit, so an
    unconditional ``CREATE SCHEMA IF NOT EXISTS`` would fail there.
    """
    present = conn.execute(
        text("SELECT 1 FROM pg_namespace WHERE nspname = :name"), {"name": plane}
    ).first()
    if present is None:
        conn.execute(CreateSchema(plane, if_not_exists=True))


def upgrade(url: str, plane: str, revision: str = "head") -> list[str]:
    """Create the plane's schema if missing and bring it to ``revision``.

    Appends one ``schema_history`` row per revision this run applied (none when already there),
    in the same transaction as the upgrade. Returns the applied revisions, oldest first.
    """
    validate_plane(plane)
    engine = create_engine(normalize_url(url))
    applied: list[tuple[str, int]] = []
    try:
        with engine.begin() as conn:
            _ensure_schema(conn, plane)
        with engine.connect() as conn:
            cfg = _config(plane, conn)
            cfg.attributes["applied"] = applied
            command.upgrade(cfg, revision)
            _record_history(conn, plane, applied)
            conn.commit()
    finally:
        engine.dispose()
    return [rev for rev, _ in applied]


def _record_history(conn: Connection, plane: str, applied: list[tuple[str, int]]) -> None:
    """Insert the run's applied revisions, when the plane has a ``schema_history`` table yet."""
    if not applied or not _has_history_table(conn, plane):
        return
    release, source = build_info.release()
    conn.execute(
        schema_history.insert().execution_options(schema_translate_map={None: plane}),
        [
            {"revision": rev, "release": release, "release_source": source, "duration_ms": ms}
            for rev, ms in applied
        ],
    )


def _has_history_table(conn: Connection, plane: str) -> bool:
    found = conn.execute(
        text("SELECT to_regclass(:name)"), {"name": f'"{plane}".schema_history'}
    ).scalar()
    return found is not None


def history(conn: Connection, plane: str, limit: int = 5) -> list[dict[str, object]]:
    """The plane's last ``limit`` ``schema_history`` rows, newest first; ``[]`` before 0002."""
    if not _has_history_table(conn, plane):
        return []
    rows = conn.execute(
        schema_history.select()
        .order_by(schema_history.c.id.desc())
        .limit(limit)
        .execution_options(schema_translate_map={None: plane})
    ).mappings()
    return [dict(r) for r in rows]


def status(url: str, plane: str) -> tuple[str | None, str]:
    """``(current, head)`` for one plane."""
    validate_plane(plane)
    engine = create_engine(normalize_url(url))
    try:
        with engine.connect() as conn:
            return current_revision(conn, plane), head_revision()
    finally:
        engine.dispose()


def history_rows(url: str, plane: str, limit: int = 5) -> list[dict[str, object]]:
    """``history`` over a fresh connection to ``url``, with ``applied_at`` in UTC."""
    validate_plane(plane)
    engine = create_engine(normalize_url(url))
    try:
        with engine.connect() as conn:
            rows = history(conn, plane, limit)
    finally:
        engine.dispose()
    for row in rows:
        row["applied_at"] = row["applied_at"].astimezone(timezone.utc)  # type: ignore[attr-defined]
    return rows


def upgrade_sql(url: str | None, plane: str, *, from_revision: str | None = None) -> str:
    """The SQL ``upgrade`` would run for ``plane``, as text; nothing is executed.

    Starts from ``from_revision`` when given; else from the plane's current revision, read with a
    read-only query when a URL is configured; else from base. Already at head: a comment line.
    """
    validate_plane(plane)
    start = from_revision
    if start is None and url:
        engine = create_engine(
            normalize_url(url),
            connect_args={"options": "-c default_transaction_read_only=on"},
        )
        try:
            with engine.connect() as conn:
                start = current_revision(conn, plane)
        finally:
            engine.dispose()
    head = head_revision()
    if start == head:
        return f"-- plane {plane} is already at {head}; nothing to run\n"
    cfg = _config(plane)
    buffer = io.StringIO()
    cfg.output_buffer = buffer
    command.upgrade(cfg, f"{start or 'base'}:{head}", sql=True)
    return (
        f'-- plane {plane}: {start or "base"} -> {head}. The schema "{plane}" must already exist.\n'
        + buffer.getvalue()
    )
