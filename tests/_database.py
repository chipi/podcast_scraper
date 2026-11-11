"""Test database support (ADR-169): one throwaway database per pytest process, kernel tables
emptied after every test.

``PODCAST_TEST_DATABASE_URL`` points at a Postgres server. Unset on a local run, the controller
provides one before the workers spawn (``provide_session_database``): this checkout's
``make test-db-up`` container if it is running, else an embedded PostgreSQL for the run, so a fresh
checkout needs no export and no docker. CI sets the variable where it needs it. Each pytest process
(each xdist worker) creates its own database on the server, so parallel workers never truncate each
other's rows, upgrades every plane to head, and points the engine registry at it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg.types.json  # noqa: F401
import pytest
import sqlalchemy.dialects.postgresql.psycopg  # noqa: F401
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine, make_url

from podcast_scraper.db import (
    engines as _engines,
    init_engines,
    migrate,
    PLANES,
    reset_engines,
    rw_engine,
)
from podcast_scraper.db.embedded import EmbeddedPostgres, EmbeddedPostgresMissing
from podcast_scraper.db.migrate import with_kernel
from podcast_scraper.db.schema import app_releases, metadata
from podcast_scraper.db.settings import DatabaseSettings, normalize_url

# Bound at import, not inside the cleanup: a test that strips podcast_scraper from sys.modules
# would otherwise make the cleanup re-import a second copy of the extensions module, whose
# Extension class an installed extension's cached EXTENSION is not an instance of.
from podcast_scraper.extensions import load_extensions

# ``psycopg.types.json`` and ``sqlalchemy.dialects.postgresql.psycopg`` are imported at the top
# because they load lazily, on the first JSONB bind. If that bind happens inside a test's
# ``patch.dict(sys.modules, …)`` window, the window's exit evicts them, the next use re-imports
# them, and psycopg no longer knows the new ``Jsonb`` class ("cannot adapt type 'Jsonb'"). This
# module is imported at collection (tests/conftest.py), before any window opens.


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


NO_DATABASE = (
    "no test database: install the [dev] extra (pip install -e '.[dev]') or run make test-db-up"
)

# Same settings as the make test-db-up container: the data is disposable and the suite write-heavy.
_EMBEDDED_SETTINGS = {
    "fsync": "off",
    "synchronous_commit": "off",
    "full_page_writes": "off",
    "max_connections": "300",
}

_session_server: list[EmbeddedPostgres] = []


def _container_url() -> str:
    """This checkout's ``make test-db-up`` container URL, or "" when there is none (no docker,
    daemon down, container absent or stopped)."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "test_db.sh"
    try:
        done = subprocess.run(
            ["bash", str(script), "url"],
            cwd=script.parent,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def provide_session_database() -> str | None:
    """Controller only, before the workers spawn: make sure ``PODCAST_TEST_DATABASE_URL`` is set.

    An exported URL is used as is. In CI nothing is provided (each job that needs app state runs
    its own service container and sets the URL). Otherwise this checkout's ``make test-db-up``
    container, else an embedded PostgreSQL started for this run and stopped by
    ``release_session_database``. Returns a line saying which, None when nothing was decided here.
    Raises ``EmbeddedPostgresMissing`` (message: ``NO_DATABASE``) when there is neither.
    """
    if os.environ.get(ENV_TEST_URL, "").strip() or os.environ.get("CI"):
        return None
    url = _container_url()
    where = "this checkout's make test-db-up container"
    if not url:
        pgdata = Path(tempfile.mkdtemp(prefix=f"pgtest-{os.getpid()}-"))
        server = EmbeddedPostgres(pgdata, settings=_EMBEDDED_SETTINGS, delete_on_stop=True)
        try:
            url = server.start()
        except EmbeddedPostgresMissing:
            shutil.rmtree(pgdata, ignore_errors=True)
            raise EmbeddedPostgresMissing(NO_DATABASE) from None
        _session_server.append(server)
        where = f"an embedded PostgreSQL in {pgdata}"
    os.environ[ENV_TEST_URL] = url
    return f"test database: {where}"


def release_session_database() -> None:
    """Controller only, at the end of the run: stop the embedded server if this run started one."""
    while _session_server:
        _session_server.pop().stop()


def worker_database_url() -> str | None:
    """The per-process test database URL, created and migrated on first use; None without a URL."""
    base = os.environ.get(ENV_TEST_URL, "").strip()
    if not base:
        return None
    if "url" not in _state:
        _state["url"] = _create_worker_database(base)
    return _state["url"]


def empty_kernel_tables() -> None:
    """Empty every kernel table and every installed module's tables that exist in each plane
    (a module table with no foreign key to ``users`` would otherwise outlive the test)."""

    modules = [ext.database for ext in load_extensions() if ext.database is not None]
    with rw_engine().begin() as conn:
        for plane in PLANES:
            planed = conn.execution_options(schema_translate_map={None: plane})
            for module in modules:
                if plane not in module.planes:
                    continue
                for table in reversed(with_kernel(module.metadata).sorted_tables):
                    if table.name in module.metadata.tables and _exists(planed, plane, table.name):
                        planed.execute(table.delete())
            for table in reversed(metadata.sorted_tables):
                planed.execute(table.delete())
        conn.execute(app_releases.delete())
    # In-process memory of what the emptied tables held (imported here: the server modules are
    # heavy and the window comment above applies to them too).
    from podcast_scraper.server import app_outbox_store, app_release_store

    app_release_store.clear_cache()
    app_outbox_store.reset_purge_clock()


@contextmanager
def captured_statements() -> Iterator[list[tuple[str, bool]]]:
    """Every statement any engine runs while the block is open: ``(sql, ran_in_autocommit)``."""
    seen: list[tuple[str, bool]] = []

    def grab(conn: Any, cursor: Any, statement: str, *_a: Any) -> None:
        autocommit = bool(conn.connection.driver_connection.autocommit)
        seen.append((" ".join(statement.split()), autocommit))

    event.listen(Engine, "before_cursor_execute", grab)
    try:
        yield seen
    finally:
        event.remove(Engine, "before_cursor_execute", grab)


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


def _exists(conn: object, plane: str, table: str) -> bool:
    found = conn.execute(  # type: ignore[attr-defined]
        text("SELECT to_regclass(:name)"), {"name": f'"{plane}".{table}'}
    ).scalar()
    return found is not None
