"""The per-process engine registry (ADR-169).

Engines are opened once per process, never per request: ``init_engines()`` at process start (the
API lifespan; lazily for CLI entry points), then ``rw_engine()`` / ``ro_engine()`` anywhere. Both
carry ``schema_translate_map={None: plane}`` so kernel tables, defined without a schema, land in
the process's plane schema.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from sqlalchemy import create_engine, Engine, exc as sa_exc, text
from sqlalchemy.engine import Connection

from podcast_scraper.db.settings import DatabaseSettings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseNotReady(RuntimeError):
    """The database is unreachable, or its schema is not at the head revision."""


@dataclass(frozen=True)
class _Engines:
    settings: DatabaseSettings
    rw: Engine
    ro: Engine


_lock = threading.Lock()
_engines: _Engines | None = None

ENV_APPLICATION_NAME = "PODCAST_DB_APPLICATION_NAME"


def application_name_args() -> dict[str, str]:
    """``application_name`` for the connection, so ``pg_stat_activity`` and the server log name
    the container: ``PODCAST_DB_APPLICATION_NAME``, else ``OTEL_SERVICE_NAME``; none when both are
    empty. Its own variable because services that share an OTEL name (board-summarizer reports as
    ``podcast-pipeline``) must still be told apart here."""
    name = (
        os.environ.get(ENV_APPLICATION_NAME) or os.environ.get("OTEL_SERVICE_NAME") or ""
    ).strip()
    # Postgres truncates past NAMEDATALEN - 1 (63 bytes).
    return {"application_name": name[:63]} if name else {}


def app_connect_args(settings: DatabaseSettings, *, read_only: bool) -> dict[str, object]:
    """libpq arguments for an app connection (ADR-173): the timeouts and, on the ro engine,
    read-only, in one ``options`` string so none overwrites another."""
    options = [
        f"-c statement_timeout={settings.statement_timeout_ms}",
        f"-c idle_in_transaction_session_timeout={settings.idle_tx_timeout_ms}",
    ]
    if read_only:
        options.append("-c default_transaction_read_only=on")
    return {
        "connect_timeout": "5",
        "options": " ".join(options),
        "prepare_threshold": settings.prepare_threshold,
        **application_name_args(),
    }


def unbounded_connect_args(*, read_only: bool = False) -> dict[str, str]:
    """libpq arguments for migrations and importers: both timeouts explicitly off, because the
    role the URL names may carry a role-level default (``ALTER ROLE ... SET statement_timeout``)
    that a session-level ``options`` value overrides. A migration may legitimately run long."""
    options = "-c statement_timeout=0 -c idle_in_transaction_session_timeout=0"
    if read_only:
        options += " -c default_transaction_read_only=on"
    return {"options": options, **application_name_args()}


def _make_engine(settings: DatabaseSettings, url: str, *, read_only: bool) -> Engine:
    plane = settings.plane
    connect_args = app_connect_args(settings, read_only=read_only)
    return create_engine(
        url,
        pool_pre_ping=True,
        # A failed statement's parameters (token hashes, addresses) stay out of its error text,
        # and so out of logs and GlitchTip.
        hide_parameters=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_s,
        connect_args=connect_args,
        execution_options={"schema_translate_map": {None: plane}},
    )


def check_ready(engines: _Engines) -> None:
    """Fail fast: both engines connect, and every history on the plane is at its head or ahead."""
    from podcast_scraper.db import migrate

    plane = engines.settings.plane
    for label, engine in (("rw", engines.rw), ("ro", engines.ro)):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except sa_exc.SQLAlchemyError as exc:
            raise DatabaseNotReady(f"database ({label}) is not reachable: {exc}") from exc
    from podcast_scraper.db.schema import SHARED_SCHEMA

    checks = [("schema", SHARED_SCHEMA, migrate.SHARED), ("plane", plane, migrate.KERNEL)]
    checks += [(f"module {h.name} on plane", plane, h) for h in migrate.modules(plane)]
    with engines.rw.connect() as conn:
        found = [
            (label, schema, *migrate.readiness(conn, schema, hist))
            for label, schema, hist in checks
        ]
    for label, schema, state, current, head in found:
        if state == "ahead":
            # A later release migrated past this code, expand-only: the rollback case (ADR-172).
            logger.warning(
                "%s %r is at revision %s, ahead of this code's head %s; starting",
                label,
                schema,
                current,
                head,
            )
        elif state == "behind":
            raise DatabaseNotReady(
                f"{label} {schema!r} is at revision {current or 'none'}, behind head {head}; run "
                f"`python -m podcast_scraper.cli db upgrade --plane {plane}` (prod: the dedicated "
                "migrate step, never app start)"
            )
        elif state == "unknown":
            raise DatabaseNotReady(
                f"{label} {schema!r} is at revision {current}, which this code does not know and "
                f"whose history never applied this code's head {head}; refusing to start"
            )


def init_engines(settings: DatabaseSettings | None = None, *, check: bool = True) -> None:
    """Open the process's engines once; a later call is a no-op.

    Raises ``DatabaseNotConfigured`` with no URL, ``DatabaseNotReady`` when ``check`` finds the
    database unreachable or not migrated.
    """
    global _engines
    with _lock:
        if _engines is not None:
            return
        cfg = settings or DatabaseSettings.from_env()
        opened = _Engines(
            settings=cfg,
            rw=_make_engine(cfg, cfg.rw_url, read_only=False),
            ro=_make_engine(cfg, cfg.ro_url, read_only=True),
        )
        if check:
            try:
                check_ready(opened)
            except Exception:
                opened.rw.dispose()
                opened.ro.dispose()
                raise
        _engines = opened
        logger.info("database engines ready (plane=%s)", cfg.plane)


def reset_engines() -> None:
    """Dispose the registry (process shutdown; tests that switch databases)."""
    global _engines
    with _lock:
        if _engines is None:
            return
        _engines.rw.dispose()
        _engines.ro.dispose()
        _engines = None


def engines_open() -> bool:
    """Whether the process's engines have been opened."""
    return _engines is not None


def database_available() -> bool:
    """Whether this process has app state in a database: engines open, or a URL to open them.

    False only for a process deliberately run without one (the control-plane api), where the few
    stores it touches behave as their absent file did.
    """
    from podcast_scraper.db.settings import is_configured

    return engines_open() or is_configured()


def _current() -> _Engines:
    if _engines is None:
        init_engines()
    assert _engines is not None
    return _engines


def rw_engine() -> Engine:
    """The read-write engine, opened lazily from the environment on first use."""
    return _current().rw


def ro_engine() -> Engine:
    """The read-only engine (the same database unless a replica URL is set)."""
    return _current().ro


def plane_schema() -> str:
    """The schema this process serves (``player`` or ``operator``)."""
    return _current().settings.plane


def read_connection(engine: Engine) -> Connection:
    """A connection for a read of ONE statement: autocommit, so the read costs no BEGIN and no
    ROLLBACK round trip. A read of several statements that must see one snapshot uses
    ``engine.connect()`` instead."""
    return engine.connect().execution_options(isolation_level="AUTOCOMMIT")


def read_with_retry(fn: Callable[[Engine], T], *, attempts: int = 3, backoff_s: float = 0.1) -> T:
    """Run an idempotent read against the ro engine, retrying a dropped connection.

    Only connection-level errors (a restarted server, a dead pooled connection) are retried; a
    query error is raised at once. Never wrap a write in this.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(ro_engine())
        except (sa_exc.OperationalError, sa_exc.InterfaceError) as exc:
            if attempt == attempts:
                raise
            logger.warning("database read failed (%s); retry %d/%d", exc, attempt, attempts - 1)
            time.sleep(backoff_s * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def current_settings() -> DatabaseSettings | None:
    """The settings the open engines were built from, or None when none are open."""
    return _engines.settings if _engines is not None else None
