"""Run and inspect the Alembic history (ADR-169): once per plane schema, never at app start."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, text

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
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{plane}"'))


def upgrade(url: str, plane: str, revision: str = "head") -> None:
    """Create the plane's schema if missing and bring it to ``revision``."""
    validate_plane(plane)
    engine = create_engine(normalize_url(url))
    try:
        with engine.begin() as conn:
            _ensure_schema(conn, plane)
        with engine.connect() as conn:
            command.upgrade(_config(plane, conn), revision)
            conn.commit()
    finally:
        engine.dispose()


def status(url: str, plane: str) -> tuple[str | None, str]:
    """``(current, head)`` for one plane."""
    validate_plane(plane)
    engine = create_engine(normalize_url(url))
    try:
        with engine.connect() as conn:
            return current_revision(conn, plane), head_revision()
    finally:
        engine.dispose()
