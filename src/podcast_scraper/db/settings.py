"""Database settings from the environment (ADR-169)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: One schema per API plane (ADR-116). The ``corpus`` schema is not a plane.
PLANES = ("player", "operator")
DEFAULT_PLANE = "player"

ENV_URL = "PODCAST_DATABASE_URL"
ENV_RO_URL = "PODCAST_DATABASE_RO_URL"
ENV_PLANE = "PODCAST_DB_PLANE"

# The corpus schema's roles (CRFC-005 §7): one URL, one engine, one pool per role. A process
# that reads the corpus and keeps no app state (MCP, the control-plane api) sets only the
# ``corpus_ro`` URL. The writers' URLs are named here for infra to stage; the writers read them.
ENV_CORPUS_RO_URL = "PODCAST_DATABASE_CORPUS_RO_URL"
ENV_PIPELINE_RW_URL = "PODCAST_DATABASE_PIPELINE_RW_URL"
ENV_MIGRATOR_URL = "PODCAST_DATABASE_MIGRATOR_URL"
ENV_CORPUS_STATEMENT_TIMEOUT_MS = "PODCAST_DB_CORPUS_STATEMENT_TIMEOUT_MS"
ENV_CORPUS_POOL_SIZE = "PODCAST_DB_CORPUS_POOL_SIZE"
ENV_CORPUS_MAX_OVERFLOW = "PODCAST_DB_CORPUS_MAX_OVERFLOW"
ENV_PIPELINE_POOL_SIZE = "PODCAST_DB_PIPELINE_POOL_SIZE"
ENV_PIPELINE_MAX_OVERFLOW = "PODCAST_DB_PIPELINE_MAX_OVERFLOW"

# The session's timeout replaces the role's, so the engine passes corpus_ro's 5 s itself: the
# plane's 15 s would otherwise apply to corpus reads.
DEFAULT_CORPUS_STATEMENT_TIMEOUT_MS = 5_000
DEFAULT_CORPUS_POOL_SIZE = 2
DEFAULT_CORPUS_MAX_OVERFLOW = 2
# pipeline_rw has no statement timeout, since writers run long statements, and 1+1 per job: at most
# four jobs at once stay inside the role's connection limit (CRFC-005 §7).
DEFAULT_PIPELINE_STATEMENT_TIMEOUT_MS = 0
DEFAULT_PIPELINE_POOL_SIZE = 1
DEFAULT_PIPELINE_MAX_OVERFLOW = 1

# Connection behaviour of the app's engines (ADR-173). Migrations and the importers do not read
# these: a migration may legitimately run long.
ENV_STATEMENT_TIMEOUT_MS = "PODCAST_DB_STATEMENT_TIMEOUT_MS"
ENV_IDLE_TX_TIMEOUT_MS = "PODCAST_DB_IDLE_TX_TIMEOUT_MS"
ENV_POOL_SIZE = "PODCAST_DB_POOL_SIZE"
ENV_MAX_OVERFLOW = "PODCAST_DB_MAX_OVERFLOW"
ENV_POOL_TIMEOUT_S = "PODCAST_DB_POOL_TIMEOUT_S"
ENV_PREPARE_THRESHOLD = "PODCAST_DB_PREPARE_THRESHOLD"

DEFAULT_STATEMENT_TIMEOUT_MS = 15_000
DEFAULT_IDLE_TX_TIMEOUT_MS = 60_000
# pool_size + max_overflow per engine, times workers, times processes, must stay under the
# server's max_connections (or PgBouncer's pool) and each role's CONNECTION LIMIT. The plane's ro
# engine is the rw engine unless the ro URL names another database: two pools on one role would
# double its connections (CRFC-005 §7).
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 5
DEFAULT_POOL_TIMEOUT_S = 10.0
# psycopg 3's own default: a statement is prepared server-side after this many executions. Set
# the variable to ``none`` behind PgBouncer in transaction mode, which cannot carry a prepared
# statement from one transaction to the next.
DEFAULT_PREPARE_THRESHOLD = 5

#: Where the image's secrets shim reads its files (`docker/secrets-shim.sh`).
DEFAULT_SECRETS_DIR = "/run/secrets"


def _lookup(source: Mapping[str, str], name: str) -> str:
    """``name`` from the environment, else from the secret file the image's shim would export.

    ``docker exec`` skips the shim, so an admin command run that way has the secret files but not
    the variables; reading the file keeps such a command on the same database as the server.
    """
    value = (source.get(name) or "").strip()
    if value:
        return value
    path = Path(source.get("SECRETS_DIR") or DEFAULT_SECRETS_DIR) / name.lower()
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def is_configured() -> bool:
    """Whether this process was given a database URL for its plane's app state."""
    return bool(_lookup(os.environ, ENV_URL))


def corpus_is_configured() -> bool:
    """Whether this process was given a ``corpus_ro`` URL."""
    return bool(_lookup(os.environ, ENV_CORPUS_RO_URL))


def pipeline_is_configured() -> bool:
    """Whether this process was given a ``pipeline_rw`` URL."""
    return bool(_lookup(os.environ, ENV_PIPELINE_RW_URL))


class DatabaseNotConfigured(RuntimeError):
    """App state was used with no database URL configured."""


def normalize_url(url: str) -> str:
    """Pin the psycopg 3 driver so a plain ``postgresql://`` URL does not pick psycopg2."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def validate_plane(plane: str) -> str:
    """Return ``plane`` if it is a known plane, else raise ``ValueError``."""
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; expected one of {', '.join(PLANES)}")
    return plane


def _int_setting(source: Mapping[str, str], name: str, default: int, *, minimum: int) -> int:
    raw = (source.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _pool_timeout(source: Mapping[str, str]) -> float:
    raw = (source.get(ENV_POOL_TIMEOUT_S) or "").strip()
    if not raw:
        return DEFAULT_POOL_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{ENV_POOL_TIMEOUT_S} must be a number, got {raw!r}") from None
    if not value > 0:
        raise ValueError(f"{ENV_POOL_TIMEOUT_S} must be greater than 0, got {raw!r}")
    return value


def _prepare_threshold(source: Mapping[str, str]) -> int | None:
    raw = (source.get(ENV_PREPARE_THRESHOLD) or "").strip().lower()
    if raw == "none":
        return None
    return _int_setting(source, ENV_PREPARE_THRESHOLD, DEFAULT_PREPARE_THRESHOLD, minimum=0)


@dataclass(frozen=True)
class DatabaseSettings:
    """Where app state lives: a read-write URL, a read-only URL (may equal it) and the plane."""

    rw_url: str
    ro_url: str
    plane: str = DEFAULT_PLANE
    #: Per-statement ceiling on the server, 0 = none. A statement over it fails with SQLSTATE 57014.
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS
    #: A session idle inside a transaction this long is terminated, 0 = never.
    idle_tx_timeout_ms: int = DEFAULT_IDLE_TX_TIMEOUT_MS
    pool_size: int = DEFAULT_POOL_SIZE
    max_overflow: int = DEFAULT_MAX_OVERFLOW
    #: Seconds a request waits for a pooled connection before failing.
    pool_timeout_s: float = DEFAULT_POOL_TIMEOUT_S
    #: Executions before psycopg prepares a statement server-side; None disables preparing.
    prepare_threshold: int | None = DEFAULT_PREPARE_THRESHOLD

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "DatabaseSettings":
        """Read the settings from ``env`` (default ``os.environ``).

        Raises ``DatabaseNotConfigured`` when no URL is set.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        rw = _lookup(source, ENV_URL)
        if not rw:
            raise DatabaseNotConfigured(
                f"{ENV_URL} is not set, and app state (users) lives in Postgres. Set it to the "
                "database URL and run `python -m podcast_scraper.cli db upgrade --plane "
                "<player|operator>`. In tests, export PODCAST_TEST_DATABASE_URL: `make "
                "test-db-up` starts a local postgres:18 and prints the line to export."
            )
        ro = _lookup(source, ENV_RO_URL) or rw
        plane = (source.get(ENV_PLANE) or "").strip() or DEFAULT_PLANE
        return cls(
            rw_url=normalize_url(rw),
            ro_url=normalize_url(ro),
            plane=validate_plane(plane),
            statement_timeout_ms=_int_setting(
                source, ENV_STATEMENT_TIMEOUT_MS, DEFAULT_STATEMENT_TIMEOUT_MS, minimum=0
            ),
            idle_tx_timeout_ms=_int_setting(
                source, ENV_IDLE_TX_TIMEOUT_MS, DEFAULT_IDLE_TX_TIMEOUT_MS, minimum=0
            ),
            pool_size=_int_setting(source, ENV_POOL_SIZE, DEFAULT_POOL_SIZE, minimum=1),
            max_overflow=_int_setting(source, ENV_MAX_OVERFLOW, DEFAULT_MAX_OVERFLOW, minimum=0),
            pool_timeout_s=_pool_timeout(source),
            prepare_threshold=_prepare_threshold(source),
        )


@dataclass(frozen=True)
class CorpusSettings:
    """The ``corpus_ro`` engine's URL and its own timeout and pool (CRFC-005 §7).

    Independent of the plane: a process with a plane URL may have no corpus URL, and the reverse.
    The idle-in-transaction timeout, the pool timeout and the prepare threshold are the process's,
    shared with the plane's engine.
    """

    url: str
    statement_timeout_ms: int = DEFAULT_CORPUS_STATEMENT_TIMEOUT_MS
    idle_tx_timeout_ms: int = DEFAULT_IDLE_TX_TIMEOUT_MS
    pool_size: int = DEFAULT_CORPUS_POOL_SIZE
    max_overflow: int = DEFAULT_CORPUS_MAX_OVERFLOW
    pool_timeout_s: float = DEFAULT_POOL_TIMEOUT_S
    prepare_threshold: int | None = DEFAULT_PREPARE_THRESHOLD

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "CorpusSettings":
        """Read the settings from ``env`` (default ``os.environ``).

        Raises ``DatabaseNotConfigured`` when no ``corpus_ro`` URL is set.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        url = _lookup(source, ENV_CORPUS_RO_URL)
        if not url:
            raise DatabaseNotConfigured(
                f"{ENV_CORPUS_RO_URL} is not set, and corpus reads go to Postgres as the "
                "corpus_ro role. Set it to that role's URL."
            )
        return cls(
            url=normalize_url(url),
            statement_timeout_ms=_int_setting(
                source,
                ENV_CORPUS_STATEMENT_TIMEOUT_MS,
                DEFAULT_CORPUS_STATEMENT_TIMEOUT_MS,
                minimum=0,
            ),
            idle_tx_timeout_ms=_int_setting(
                source, ENV_IDLE_TX_TIMEOUT_MS, DEFAULT_IDLE_TX_TIMEOUT_MS, minimum=0
            ),
            pool_size=_int_setting(
                source, ENV_CORPUS_POOL_SIZE, DEFAULT_CORPUS_POOL_SIZE, minimum=1
            ),
            max_overflow=_int_setting(
                source, ENV_CORPUS_MAX_OVERFLOW, DEFAULT_CORPUS_MAX_OVERFLOW, minimum=0
            ),
            pool_timeout_s=_pool_timeout(source),
            prepare_threshold=_prepare_threshold(source),
        )


@dataclass(frozen=True)
class PipelineSettings:
    """The ``pipeline_rw`` engine's URL and its own pool (CRFC-005 §7).

    The pipeline and enricher jobs write corpus rows through it. No statement timeout; the
    idle-in-transaction timeout, the pool timeout and the prepare threshold are the process's.
    """

    url: str
    statement_timeout_ms: int = DEFAULT_PIPELINE_STATEMENT_TIMEOUT_MS
    idle_tx_timeout_ms: int = DEFAULT_IDLE_TX_TIMEOUT_MS
    pool_size: int = DEFAULT_PIPELINE_POOL_SIZE
    max_overflow: int = DEFAULT_PIPELINE_MAX_OVERFLOW
    pool_timeout_s: float = DEFAULT_POOL_TIMEOUT_S
    prepare_threshold: int | None = DEFAULT_PREPARE_THRESHOLD

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PipelineSettings":
        """Read the settings from ``env`` (default ``os.environ``).

        Raises ``DatabaseNotConfigured`` when no ``pipeline_rw`` URL is set.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        url = _lookup(source, ENV_PIPELINE_RW_URL)
        if not url:
            raise DatabaseNotConfigured(
                f"{ENV_PIPELINE_RW_URL} is not set, and corpus rows are written as the "
                "pipeline_rw role. Set it to that role's URL."
            )
        return cls(
            url=normalize_url(url),
            idle_tx_timeout_ms=_int_setting(
                source, ENV_IDLE_TX_TIMEOUT_MS, DEFAULT_IDLE_TX_TIMEOUT_MS, minimum=0
            ),
            pool_size=_int_setting(
                source, ENV_PIPELINE_POOL_SIZE, DEFAULT_PIPELINE_POOL_SIZE, minimum=1
            ),
            max_overflow=_int_setting(
                source, ENV_PIPELINE_MAX_OVERFLOW, DEFAULT_PIPELINE_MAX_OVERFLOW, minimum=0
            ),
            pool_timeout_s=_pool_timeout(source),
            prepare_threshold=_prepare_threshold(source),
        )
