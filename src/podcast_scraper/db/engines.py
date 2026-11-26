"""The per-process engine registry (ADR-169).

Engines are opened once per process, never per request: ``init_engines()`` at process start (the
API lifespan; lazily for CLI entry points), then ``rw_engine()`` / ``ro_engine()`` anywhere. Both
carry ``schema_translate_map={None: plane}`` so kernel tables, defined without a schema, land in
the process's plane schema. They are one engine unless the ro URL names another database.

The ``corpus_ro`` engine (CRFC-005 §7) is separate: its own URL, its own 5 s statement timeout and
pool, no schema map, opened by ``init_corpus_engine()`` or on first ``corpus_ro_engine()``. A
process with only a corpus URL never opens the plane's engines. The ``pipeline_rw`` engine is the
same shape for the corpus writers: ``init_pipeline_engine()`` or the first ``pipeline_rw_engine()``,
no statement timeout, a pool of 1+1.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from sqlalchemy import create_engine, Engine, exc as sa_exc, text
from sqlalchemy.engine import Connection

from podcast_scraper.db.settings import (
    CorpusSettings,
    DatabaseNotConfigured,
    DatabaseSettings,
    ENV_MIGRATOR_URL,
    normalize_url,
    PipelineSettings,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseNotReady(RuntimeError):
    """The database is unreachable, or its schema is not at the head revision."""


@dataclass(frozen=True)
class _Engines:
    settings: DatabaseSettings
    rw: Engine
    ro: Engine

    def dispose(self) -> None:
        self.rw.dispose()
        if self.ro is not self.rw:
            self.ro.dispose()


_lock = threading.Lock()
_engines: _Engines | None = None
_corpus: tuple[CorpusSettings, Engine] | None = None
_pipeline: tuple[PipelineSettings, Engine] | None = None

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


def app_connect_args(
    settings: DatabaseSettings | CorpusSettings | PipelineSettings, *, read_only: bool
) -> dict[str, object]:
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


def migrator_engine(url: str | None = None) -> Engine:
    """A ``migrator`` engine for one corpus migration one-shot on rows (RFC-135 §3); the caller
    disposes it.

    Only one-shots get the ``migrator`` URL (CRFC-005 §7). One connection, no statement timeout as
    for schema migrations, and a 5 s lock timeout so a row another writer holds fails the run
    instead of hanging it.
    """
    if url is None:
        url = (os.environ.get(ENV_MIGRATOR_URL) or "").strip()
        if not url:
            raise DatabaseNotConfigured(
                f"{ENV_MIGRATOR_URL} is not set; a corpus migration on rows runs as migrator"
            )
    args = unbounded_connect_args()
    args["options"] += " -c lock_timeout=5000"
    return create_engine(
        normalize_url(url),
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args=args,
    )


def _create(
    settings: DatabaseSettings | CorpusSettings | PipelineSettings,
    url: str,
    *,
    read_only: bool,
    execution_options: dict[str, object],
) -> Engine:
    return create_engine(
        url,
        pool_pre_ping=True,
        # A failed statement's parameters (token hashes, addresses) stay out of its error text,
        # and so out of logs and GlitchTip.
        hide_parameters=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_s,
        connect_args=app_connect_args(settings, read_only=read_only),
        execution_options=execution_options,
    )


def _make_engine(settings: DatabaseSettings, url: str, *, read_only: bool) -> Engine:
    return _create(
        settings,
        url,
        read_only=read_only,
        execution_options={"schema_translate_map": {None: settings.plane}},
    )


def _check_reachable(label: str, engine: Engine) -> None:
    try:
        with read_connection(engine) as conn:
            conn.execute(text("SELECT 1"))
    except sa_exc.SQLAlchemyError as exc:
        raise DatabaseNotReady(f"database ({label}) is not reachable: {exc}") from exc


def check_ready(engines: _Engines) -> None:
    """Fail fast: the engines connect, and every history on the plane is at its head or ahead."""
    from podcast_scraper.db import migrate

    plane = engines.settings.plane
    _check_reachable("rw", engines.rw)
    if engines.ro is not engines.rw:
        _check_reachable("ro", engines.ro)
    from podcast_scraper.db.schema import SHARED_SCHEMA

    checks = [("schema", SHARED_SCHEMA, migrate.SHARED), ("plane", plane, migrate.KERNEL)]
    checks += [(f"module {h.name} on plane", plane, h) for h in migrate.modules(plane)]
    with read_connection(engines.rw) as conn:
        _refuse_unless_ready(conn, checks, f"--plane {plane}")


def _refuse_unless_ready(
    conn: Connection, checks: list[tuple[str, str, Any]], plane_arg: str
) -> None:
    from podcast_scraper.db import migrate

    found = [
        (label, schema, *migrate.readiness(conn, schema, hist)) for label, schema, hist in checks
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
                f"`python -m podcast_scraper.cli db upgrade {plane_arg}` (prod: the dedicated "
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
        rw = _make_engine(cfg, cfg.rw_url, read_only=False)
        ro = rw if cfg.ro_url == cfg.rw_url else _make_engine(cfg, cfg.ro_url, read_only=True)
        opened = _Engines(settings=cfg, rw=rw, ro=ro)
        if check:
            try:
                check_ready(opened)
            except Exception:
                opened.dispose()
                raise
        _engines = opened
        logger.info("database engines ready (plane=%s)", cfg.plane)


def init_corpus_engine(settings: CorpusSettings | None = None, *, check: bool = True) -> None:
    """Open the process's ``corpus_ro`` engine once; a later call is a no-op.

    Raises ``DatabaseNotConfigured`` with no corpus URL, ``DatabaseNotReady`` when ``check`` finds
    it unreachable.
    """
    global _corpus
    with _lock:
        if _corpus is not None:
            return
        cfg = settings or CorpusSettings.from_env()
        engine = _create(cfg, cfg.url, read_only=True, execution_options={})
        if check:
            try:
                _check_reachable("corpus_ro", engine)
                _check_corpus_ready(engine)
            except Exception:
                engine.dispose()
                raise
        _corpus = (cfg, engine)
        logger.info("corpus_ro engine ready")


def _check_corpus_ready(engine: Engine) -> None:
    # Through corpus_ro itself: the plane roles hold no privilege on the corpus schema.
    from podcast_scraper.db import migrate
    from podcast_scraper.db.schema import CORPUS_SCHEMA

    with read_connection(engine) as conn:
        _refuse_unless_ready(
            conn, [("schema", CORPUS_SCHEMA, migrate.CORPUS)], "--plane <player|operator>"
        )


def corpus_ro_engine() -> Engine:
    """The ``corpus_ro`` engine, opened lazily from the environment on first use."""
    if _corpus is None:
        init_corpus_engine()
    assert _corpus is not None
    return _corpus[1]


def init_pipeline_engine(settings: PipelineSettings | None = None, *, check: bool = True) -> None:
    """Open the process's ``pipeline_rw`` engine once; a later call is a no-op.

    Raises ``DatabaseNotConfigured`` with no pipeline URL, ``DatabaseNotReady`` when ``check``
    finds it unreachable or the corpus schema not at head.
    """
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return
        cfg = settings or PipelineSettings.from_env()
        engine = _create(cfg, cfg.url, read_only=False, execution_options={})
        if check:
            try:
                _check_reachable("pipeline_rw", engine)
                _check_corpus_ready(engine)
            except Exception:
                engine.dispose()
                raise
        _pipeline = (cfg, engine)
        logger.info("pipeline_rw engine ready")


def pipeline_rw_engine() -> Engine:
    """The ``pipeline_rw`` engine, opened lazily from the environment on first use."""
    if _pipeline is None:
        init_pipeline_engine()
    assert _pipeline is not None
    return _pipeline[1]


def reset_engines() -> None:
    """Dispose the registry (process shutdown; tests that switch databases)."""
    global _engines, _corpus, _pipeline
    with _lock:
        if _engines is not None:
            _engines.dispose()
            _engines = None
        if _corpus is not None:
            _corpus[1].dispose()
            _corpus = None
        if _pipeline is not None:
            _pipeline[1].dispose()
            _pipeline = None


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
    """The read engine: the rw engine itself unless a separate ro URL (a replica) is set."""
    return _current().ro


def plane_schema() -> str:
    """The schema this process serves (``player`` or ``operator``)."""
    return _current().settings.plane


def read_connection(engine: Engine) -> Connection:
    """The connection every read uses: autocommit, so a read sends no BEGIN, no COMMIT and no
    ROLLBACK, and Postgres counts each statement as an implicit commit. A read closed by
    ``engine.connect()`` without a commit ends in a ROLLBACK, which makes the rollback counter
    (``pg_stat_database.xact_rollback``) count reads instead of failures. Each statement sees its
    own snapshot; a read that needs several statements on one snapshot is rare enough to be named
    where it happens."""
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
