"""The released version of each client app, changeable at runtime — one JSON file per instance.

A released version is the newest app version people can install (TestFlight / Play); the native
app prompts "update available" when it is higher than its own. The deploy sets a default per app
(``APP_PLAYER_VERSION`` for the player), but a native-only release must not need a server
restart, so an admin can override it here (``PUT /api/app/admin/release``) and every request reads
the file. An absent or malformed entry means "no override" — the environment default applies —
exactly as ``app_ranking_config_store`` falls back to its defaults.

One file holds every app (``app_releases.json``: ``{"versions": {"player": "1.0.2"}}``), because
the kernel serves more than one client app (ADR-158). The player's override was stored on its own
in ``player_release.json`` before that; it is still read when the new file has no player entry,
and still written alongside, so a rollback to code that only knows the old file keeps the
override.

WHERE the file lives is ``APP_RELEASE_DIR`` when set, else the instance's ``APP_DATA_DIR`` (#2296).
On prod the player and operator apis have separate data dirs, and the admin field is in the
operator viewer: it wrote the operator's copy while phones read the player's. Both apis therefore
mount one small shared dir and point ``APP_RELEASE_DIR`` at it — only this file is shared, never a
stack's user data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from podcast_scraper.server.atomic_write import atomic_write_text

#: Dotted numeric versions only (``1``, ``1.0``, ``1.0.1``, ``1.0.1.4``): the client compares them
#: number by number, so anything else would be a prompt that fires wrongly or never.
_VERSION = re.compile(r"\d{1,4}(\.\d{1,4}){0,3}")

#: App ids are short lowercase slugs; they become JSON keys and env-var names.
_APP_ID = re.compile(r"[a-z][a-z0-9_]{0,31}")

PLAYER = "player"


def release_dir(state: Any) -> Path | None:
    """The directory holding the release file: ``app_release_dir`` if set, else ``app_data_dir``."""
    raw = getattr(state, "app_release_dir", None) or getattr(state, "app_data_dir", None)
    return Path(raw) if raw is not None else None


def _releases_path(data_dir: Path) -> Path:
    return data_dir / "app_releases.json"


def _legacy_player_path(data_dir: Path) -> Path:
    return data_dir / "player_release.json"


def valid_player_version(value: str) -> bool:
    """Whether *value* is a version the client can compare."""
    return bool(_VERSION.fullmatch(value))


def valid_app_id(value: str) -> bool:
    """Whether *value* is a well-formed app id (``player``, ``news``)."""
    return bool(_APP_ID.fullmatch(value))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_released_versions(data_dir: Path | None) -> dict[str, str]:
    """Every app's runtime override (absent, unreadable or malformed entries are left out)."""
    if data_dir is None:
        return {}
    raw = _read_json(_releases_path(data_dir)).get("versions")
    versions = {
        app: value
        for app, value in (raw.items() if isinstance(raw, dict) else ())
        if isinstance(app, str)
        and valid_app_id(app)
        and isinstance(value, str)
        and valid_player_version(value)
    }
    if PLAYER not in versions:
        legacy = _read_json(_legacy_player_path(data_dir)).get("player_version")
        if isinstance(legacy, str) and valid_player_version(legacy):
            versions[PLAYER] = legacy
    return versions


def load_released_version(data_dir: Path | None, app: str = PLAYER) -> str | None:
    """One app's runtime override, or ``None`` when there is none."""
    return load_released_versions(data_dir).get(app)


def save_released_version(data_dir: Path, version: str | None, app: str = PLAYER) -> None:
    """Persist one app's override atomically; ``None`` clears it (back to the deploy default)."""
    if not valid_app_id(app):
        raise ValueError(f"not an app id: {app!r}")
    if version is not None and not valid_player_version(version):
        raise ValueError(f"not a dotted numeric version: {version!r}")
    data_dir.mkdir(parents=True, exist_ok=True)
    versions = load_released_versions(data_dir)
    if version is None:
        versions.pop(app, None)
    else:
        versions[app] = version
    atomic_write_text(_releases_path(data_dir), json.dumps({"versions": versions}, indent=2))
    if app == PLAYER:
        legacy = _legacy_player_path(data_dir)
        if version is None:
            legacy.unlink(missing_ok=True)
        else:
            atomic_write_text(legacy, json.dumps({"player_version": version}, indent=2))


__all__ = [
    "PLAYER",
    "load_released_version",
    "load_released_versions",
    "release_dir",
    "save_released_version",
    "valid_app_id",
    "valid_player_version",
]
