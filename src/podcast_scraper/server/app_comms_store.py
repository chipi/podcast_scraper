"""Per-user comms / delivery consent — a per-TYPE × per-CHANNEL matrix (#1414 → wave-I).

The consent a user has set for every notification TYPE (``digest`` / ``new_episodes`` /
``product``) across every delivery CHANNEL (``email`` / ``push`` / ``in_app``). Same file-based
per-user overlay as ``app_user_preferences`` (RFC-098 §3): one ``comms.json`` per user,
FileLock-serialised read-modify-writes.

This store is the gate for delivery: a ``DeliveryEnvelope`` for ``(type, channel)`` is only sent
when ``types[type][channel]`` is set (see :func:`channel_enabled`). ``digest_schedule`` holds the
email-delivery cadence (cadence / day_of_week / hour / paused) — it is not per-channel, so it sits
outside the matrix. The ``unsubscribe_ref`` is an opaque, rotatable handle the delivery service
embeds in the one-click email link; :func:`unsubscribe` resolves it back to the user and disables
the digest *email* channel. ``email_verified`` is NOT stored here — it is identity-derived (from
the OAuth provider) at the route layer.

Channels: ``email`` + ``push`` are outbound and opt-in (default OFF); ``in_app`` is the in-app
inbox — cheap, silent, no OS grant — so it defaults ON. Push (OS-level, reaches you when the app
is closed) and in-app (what's waiting when you return) are distinct surfaces, not the same thing.
"""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "comms.json"

#: The notification TYPES a user can tune, each independently per CHANNEL. Adding a type is a
#: single entry here (+ its emitter + a Profile-UI row).
TYPES: tuple[str, ...] = ("digest", "new_episodes", "product")
#: The delivery CHANNELS. ``email``/``push`` are outbound (opt-in); ``in_app`` is the inbox.
CHANNELS: tuple[str, ...] = ("email", "push", "in_app")


def _default_row() -> dict[str, bool]:
    #: Opt-in for outbound channels; in-app inbox on by default (cheap, silent, no OS grant).
    return {"email": False, "push": False, "in_app": True}


DEFAULTS: dict[str, Any] = {
    "types": {ntype: _default_row() for ntype in TYPES},
    "digest_schedule": {
        "cadence": "weekly",
        "day_of_week": 6,  # Sunday (Python weekday 6)
        "hour": 13,
        "paused": False,
    },
}


def _comms_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _comms_lock(data_dir: Path, user_id: str) -> FileLock:
    path = _comms_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def _read_raw(data_dir: Path, user_id: str) -> dict[str, Any]:
    path = _comms_path(data_dir, user_id)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _merged(stored: dict[str, Any]) -> dict[str, Any]:
    """Overlay a stored payload onto a deep copy of DEFAULTS (missing type/channel keys default)."""
    out = copy.deepcopy(DEFAULTS)
    stored_types = stored.get("types")
    if isinstance(stored_types, dict):
        for ntype in TYPES:
            row = stored_types.get(ntype)
            if isinstance(row, dict):
                for ch in CHANNELS:
                    if ch in row:
                        out["types"][ntype][ch] = bool(row[ch])
    sched = stored.get("digest_schedule")
    if isinstance(sched, dict):
        out["digest_schedule"].update(
            {k: v for k, v in sched.items() if k in out["digest_schedule"]}
        )
    if isinstance(stored.get("unsubscribe_ref"), str):
        out["unsubscribe_ref"] = stored["unsubscribe_ref"]
    return out


def get_comms(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Return the user's comms settings merged onto defaults (read-only; no ref minted)."""
    if not _is_safe_user_id(user_id):
        return _merged({})
    return _merged(_read_raw(data_dir, user_id))


def channel_enabled(comms: dict[str, Any], ntype: str, channel: str) -> bool:
    """Whether ``ntype`` is allowed to deliver on ``channel`` per this user's matrix."""
    try:
        return bool(comms["types"][ntype][channel])
    except (KeyError, TypeError):
        return False


def set_comms(
    data_dir: Path,
    user_id: str,
    *,
    types: dict[str, dict[str, Any]] | None = None,
    digest_schedule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Partial-update the matrix and/or schedule; mint an ``unsubscribe_ref`` on first write.

    ``types`` is a partial ``{type: {channel: bool}}`` — only known type/channel keys are written
    (unknown keys ignored). ``digest_schedule`` is a partial of the schedule block. Returns the
    merged settings (including the ref). Raises ValueError for an unsafe user id. Called with no
    sections it still mints the ref (the digest path relies on that).
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    with _comms_lock(data_dir, user_id):
        current = _merged(_read_raw(data_dir, user_id))
        if types:
            for ntype, row in types.items():
                if ntype in current["types"] and isinstance(row, dict):
                    for ch, on in row.items():
                        if ch in current["types"][ntype]:
                            current["types"][ntype][ch] = bool(on)
        if digest_schedule:
            current["digest_schedule"].update(
                {k: v for k, v in digest_schedule.items() if k in current["digest_schedule"]}
            )
        if not current.get("unsubscribe_ref"):
            current["unsubscribe_ref"] = uuid.uuid4().hex
        path = _comms_path(data_dir, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(current, ensure_ascii=False, indent=2))
    return current


def set_channel(
    data_dir: Path, user_id: str, ntype: str, channel: str, enabled: bool
) -> dict[str, Any]:
    """Convenience: flip a single ``type × channel`` cell (push subscribe + bounce-suppress)."""
    return set_comms(data_dir, user_id, types={ntype: {channel: bool(enabled)}})


def disable_push_everywhere(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Turn the push channel off for every type — the last subscription is gone, push is dead."""
    return set_comms(data_dir, user_id, types={t: {"push": False} for t in TYPES})


def unsubscribe(data_dir: Path, ref: str) -> bool:
    """Resolve an ``unsubscribe_ref`` to its user and disable the digest EMAIL channel. No auth.

    O(users) scan (acceptable at current scale; RFC-101 OQ-1). The email one-click link governs
    the email channel only (not push / in-app). Returns True when a matching user was found and
    updated, False otherwise. Idempotent — re-hitting a used link is a no-op that still returns
    True.
    """
    if not ref:
        return False
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return False
    for child in sorted(users_dir.iterdir()):
        if not child.is_dir():
            continue
        raw = _read_raw(data_dir, child.name)
        if raw.get("unsubscribe_ref") == ref:
            with _comms_lock(data_dir, child.name):
                current = _merged(_read_raw(data_dir, child.name))
                # Re-verify under the lock — the ref could have been rotated out between the
                # unlocked scan and here; don't disable delivery for a stale/rotated ref.
                if current.get("unsubscribe_ref") != ref:
                    return False
                current["types"]["digest"]["email"] = False
                atomic_write_text(
                    _comms_path(data_dir, child.name),
                    json.dumps(current, ensure_ascii=False, indent=2),
                )
            return True
    return False
