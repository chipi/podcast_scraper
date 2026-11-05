"""Database settings from the environment (ADR-169)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: One schema per API plane (ADR-116). A later ``corpus`` schema is not a plane.
PLANES = ("player", "operator")
DEFAULT_PLANE = "player"

ENV_URL = "PODCAST_DATABASE_URL"
ENV_RO_URL = "PODCAST_DATABASE_RO_URL"
ENV_PLANE = "PODCAST_DB_PLANE"

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
    """Whether this process was given a database URL."""
    return bool(_lookup(os.environ, ENV_URL))


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


@dataclass(frozen=True)
class DatabaseSettings:
    """Where app state lives: a read-write URL, a read-only URL (may equal it) and the plane."""

    rw_url: str
    ro_url: str
    plane: str = DEFAULT_PLANE

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
        return cls(rw_url=normalize_url(rw), ro_url=normalize_url(ro), plane=validate_plane(plane))
