"""The released player-app version, changeable at runtime — one JSON file per instance.

``player_version`` is the newest app version people can install (TestFlight / Play); the native
app prompts "update available" when it is higher than its own. The deploy sets a default through
``APP_PLAYER_VERSION``, but a native-only release must not need a server restart, so an admin can
override it here (``PUT /api/app/admin/release``) and every request reads the file. An absent or
malformed file means "no override" — the environment default applies — exactly as
``app_ranking_config_store`` falls back to its defaults.

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


def release_dir(state: Any) -> Path | None:
    """The directory holding the release file: ``app_release_dir`` if set, else ``app_data_dir``."""
    raw = getattr(state, "app_release_dir", None) or getattr(state, "app_data_dir", None)
    return Path(raw) if raw is not None else None


def _release_path(data_dir: Path) -> Path:
    return data_dir / "player_release.json"


def valid_player_version(value: str) -> bool:
    """Whether *value* is a version the client can compare."""
    return bool(_VERSION.fullmatch(value))


def load_released_version(data_dir: Path | None) -> str | None:
    """The runtime override, or ``None`` when there is none (absent, unreadable or malformed)."""
    if data_dir is None:
        return None
    path = _release_path(data_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("player_version") if isinstance(data, dict) else None
    return value if isinstance(value, str) and valid_player_version(value) else None


def save_released_version(data_dir: Path, version: str | None) -> None:
    """Persist the override atomically; ``None`` clears it (back to the environment default)."""
    path = _release_path(data_dir)
    if version is None:
        path.unlink(missing_ok=True)
        return
    if not valid_player_version(version):
        raise ValueError(f"not a dotted numeric version: {version!r}")
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({"player_version": version}, indent=2))


__all__ = [
    "load_released_version",
    "release_dir",
    "save_released_version",
    "valid_player_version",
]
