"""Test database support (ADR-169): one throwaway database per pytest process, kernel tables
emptied after every test.

``PODCAST_TEST_DATABASE_URL`` points at a Postgres server (``make test-db-up`` starts one and
prints the line to export). Each pytest process (each xdist worker) creates its own database on it,
so parallel workers never truncate each other's rows, upgrades every plane to head, and points the
engine registry at it. With no URL set nothing here connects: a test that needs app state fails
with the "start the test database" message from ``DatabaseSettings.from_env``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from podcast_scraper.db import (
    engines as _engines,
    init_engines,
    migrate,
    PLANES,
    reset_engines,
    rw_engine,
)
from podcast_scraper.db.schema import metadata
from podcast_scraper.db.settings import DatabaseSettings, normalize_url

ENV_TEST_URL = "PODCAST_TEST_DATABASE_URL"

_state: dict[str, str] = {}


def _database_name() -> str:
    # The xdist controller is this worker's parent; a plain run is its own controller. Two pytest
    # runs against one server (unit + integration side by side) must not drop each other's database.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    controller = os.getppid() if worker else os.getpid()
    return f"podcast_test_{controller}_{worker or 'main'}"


def _create_worker_database(base_url: str) -> str:
    name = _database_name()
    admin = create_engine(normalize_url(base_url), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()
    url = make_url(normalize_url(base_url)).set(database=name)
    rendered = url.render_as_string(hide_password=False)
    for plane in PLANES:
        migrate.upgrade(rendered, plane)
    _state["admin_url"] = base_url
    _state["name"] = name
    return rendered


def worker_database_url() -> str | None:
    """The per-process test database URL, created and migrated on first use; None without a URL."""
    base = os.environ.get(ENV_TEST_URL, "").strip()
    if not base:
        return None
    if "url" not in _state:
        _state["url"] = _create_worker_database(base)
    return _state["url"]


def empty_kernel_tables() -> None:
    with rw_engine().begin() as conn:
        for plane in PLANES:
            planed = conn.execution_options(schema_translate_map={None: plane})
            for table in reversed(metadata.sorted_tables):
                planed.execute(table.delete())


def database_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # A developer's real database must never be what a test talks to.
    for var in ("PODCAST_DATABASE_URL", "PODCAST_DATABASE_RO_URL", "PODCAST_DB_PLANE"):
        monkeypatch.delenv(var, raising=False)
    url = worker_database_url()
    if url is None:
        yield
        return
    settings = DatabaseSettings(rw_url=url, ro_url=url)
    if _engines.engines_open() and _engines.current_settings() != settings:
        reset_engines()
    init_engines(settings)
    yield
    if _engines.engines_open():
        empty_kernel_tables()


def drop_worker_database() -> None:
    """Session end: dispose the engines and drop this process's database."""
    reset_engines()
    name = _state.pop("name", None)
    admin_url = _state.pop("admin_url", None)
    _state.pop("url", None)
    if not name or not admin_url:
        return
    admin = create_engine(normalize_url(admin_url), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


def register_user(user_id: str, **fields: object) -> None:
    """Insert (idempotently) a user row under a FIXED id, for tests that name their user.

    The store derives ids from ``(provider, subject)``; a test that wants a literal id (and no
    handle) writes the row directly. Stores that gate on "the account exists" then see it.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from podcast_scraper.db.schema import users

    values: dict[str, object] = {"provider": "test", "subject": user_id, **fields}
    with rw_engine().begin() as conn:
        conn.execute(pg_insert(users).values(user_id=user_id, **values).on_conflict_do_nothing())
