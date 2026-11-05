"""The released version of each client app, changeable at runtime — rows in ``shared``.

A released version is the newest app version people can install (TestFlight / Play); the native
app prompts "update available" when it is higher than its own. The deploy sets a default per app
(``APP_PLAYER_VERSION`` for the player), but a native-only release must not need a server
restart, so an admin can override it here (``PUT /api/app/admin/release``) and every request reads
it. No row means "no override" — the environment default applies.

One ``shared.app_releases`` row per app (ADR-169, the operator 2026-11-05), because the kernel
serves more than one client app (ADR-168) and because both planes need the same value (#2296): the
admin field is in the operator viewer and the phones read the player api. It used to be a file in
a dir both apis mounted (``APP_RELEASE_DIR``): ``app_releases.json``, with the player's older
``player_release.json`` beside it; the importer reads both. ``shared`` holds only this, never a
plane's user data. A process with no database (the control plane) has no override.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db import database_available, read_with_retry, rw_engine
from podcast_scraper.db.schema import app_releases

#: Dotted numeric versions only (``1``, ``1.0``, ``1.0.1``, ``1.0.1.4``): the client compares them
#: number by number, so anything else would be a prompt that fires wrongly or never.
_VERSION = re.compile(r"\d{1,4}(\.\d{1,4}){0,3}")

#: App ids are short lowercase slugs; they become JSON keys and env-var names.
_APP_ID = re.compile(r"[a-z][a-z0-9_]{0,31}")

PLAYER = "player"


#: The files the override was kept in before the database; the importer reads them.
RELEASES_FILE = "app_releases.json"
LEGACY_PLAYER_FILE = "player_release.json"


def release_dir(state: Any) -> Path | None:
    """The directory the release file was in: ``app_release_dir`` if set, else ``app_data_dir``.

    Still the argument the store functions take (their signatures are the file store's); ``None``
    means a process with no app data, which has no override.
    """
    raw = getattr(state, "app_release_dir", None) or getattr(state, "app_data_dir", None)
    return Path(raw) if raw is not None else None


def valid_player_version(value: str) -> bool:
    """Whether *value* is a version the client can compare."""
    return bool(_VERSION.fullmatch(value))


def valid_app_id(value: str) -> bool:
    """Whether *value* is a well-formed app id (``player``, ``news``)."""
    return bool(_APP_ID.fullmatch(value))


def load_released_versions(data_dir: Path | None) -> dict[str, str]:
    """Every app's runtime override (the table's CHECKs keep malformed values out)."""
    if data_dir is None or not database_available():
        return {}

    def read(engine: Any) -> dict[str, str]:
        with engine.connect() as conn:
            rows = conn.execute(
                select(app_releases.c.app, app_releases.c.version).order_by(app_releases.c.app)
            ).all()
        return {app: version for app, version in rows}

    return read_with_retry(read)


def load_released_version(data_dir: Path | None, app: str = PLAYER) -> str | None:
    """One app's runtime override, or ``None`` when there is none."""
    return load_released_versions(data_dir).get(app)


def save_released_version(data_dir: Path, version: str | None, app: str = PLAYER) -> None:
    """Persist one app's override; ``None`` clears it (back to the deploy default)."""
    if not valid_app_id(app):
        raise ValueError(f"not an app id: {app!r}")
    if version is not None and not valid_player_version(version):
        raise ValueError(f"not a dotted numeric version: {version!r}")
    with rw_engine().begin() as conn:
        if version is None:
            conn.execute(delete(app_releases).where(app_releases.c.app == app))
            return
        conn.execute(
            pg_insert(app_releases)
            .values(app=app, version=version)
            .on_conflict_do_update(
                index_elements=[app_releases.c.app],
                set_={"version": version, "updated_at": func.now()},
            )
        )


__all__ = [
    "PLAYER",
    "load_released_version",
    "load_released_versions",
    "release_dir",
    "save_released_version",
    "valid_app_id",
    "valid_player_version",
]
