"""USERPREFS-1 — per-user preferences persistence, on Postgres (ADR-169).

Cross-device sync for the UI's opinion-state that has lived in ``localStorage`` until now (graph
lens flags, theme choice, panel collapse state, corpus path, etc.):

- One ``user_preferences`` row per user in the process's plane schema; it goes with the account.
- Read-modify-writes take the row lock, so concurrent tab writes stay safe.
- Payload is a single free-form JSON object — the client owns the shape, so new preference keys
  don't require a server release. The server never interprets the payload; it just round-trips it.

Kept intentionally naive: no schema validation, no migrations, no versioning. The client-side
store (``useUserPreferencesStore``) treats absent keys as "use the local default" so payload shape
drift is a no-op. ``data_dir`` stays the first argument of every function (the signatures the file
store had); the database comes from the engine registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db import read_connection, read_with_retry, rw_engine
from podcast_scraper.db.schema import user_preferences

#: The file the store kept before the database; the importer reads it.
_FILE_NAME = "preferences.json"


def get_preferences(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Return the user's preferences payload, or ``{}`` when unset."""

    def read(engine: Any) -> dict[str, Any]:
        with read_connection(engine) as conn:
            payload = conn.execute(
                select(user_preferences.c.payload).where(user_preferences.c.user_id == user_id)
            ).scalar()
        return payload if isinstance(payload, dict) else {}

    return read_with_retry(read)


def _store(conn: Any, user_id: str, payload: dict[str, Any]) -> None:
    conn.execute(
        pg_insert(user_preferences)
        .values(user_id=user_id, payload=payload)
        .on_conflict_do_update(
            index_elements=[user_preferences.c.user_id],
            set_={"payload": payload, "updated_at": func.now()},
        )
    )


def replace_preferences(data_dir: Path, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace the user's preferences with ``payload``. Returns the stored value."""
    if not isinstance(payload, dict):
        raise ValueError("preferences payload must be an object")
    with rw_engine().begin() as conn:
        _store(conn, user_id, payload)
    return payload


def _identity(item: Any, match: list[str] | None, fold_case: bool) -> Any:
    def norm(v: Any) -> Any:
        # Typed: JSON's true is not 1, and Python's == says it is.
        return (type(v).__name__, v.strip().casefold() if fold_case and isinstance(v, str) else v)

    if not match:
        return norm(item)
    if not isinstance(item, dict) or any(f not in item for f in match):
        return _NO_IDENTITY
    return tuple(norm(item[f]) for f in match)


#: An item without every matched field: equal to nothing, so it is never dropped by mistake. Two
#: items both lacking a field were equal before, and a value lacking it dropped them all (review,
#: 2026-11-19).
_NO_IDENTITY = object()


def apply_ops(payload: dict[str, Any], ops: list[Any]) -> dict[str, Any]:
    """Pure: ``payload`` with each item-level op applied in order (see ``PreferenceOp``).

    A key holding the wrong type for its op (a list op on a string, say) starts over empty: the
    client owns the shape, and an unreadable value is what it would have replaced anyway.
    """
    out = dict(payload)
    for op in ops:
        kind, key = op.op, op.key
        if kind in ("push_front", "remove"):
            if op.value is None:
                raise ValueError(f"{kind} needs a value")
            current = out.get(key)
            items = list(current) if isinstance(current, list) else []
            ident = _identity(op.value, op.match, op.fold_case)
            if ident is _NO_IDENTITY:
                raise ValueError(f"{kind} with match needs an object value with every match field")
            if kind == "remove" and key not in out:
                continue
            items = [x for x in items if _identity(x, op.match, op.fold_case) != ident]
            if kind == "push_front":
                items.insert(0, op.value)
                if op.max is not None:
                    items = items[: op.max]
            out[key] = items
        elif kind == "clear":
            out[key] = []
        elif kind == "set_field":
            if op.field is None:
                raise ValueError("set_field needs a field")
            current = out.get(key)
            obj = dict(current) if isinstance(current, dict) else {}
            if op.value is None:
                obj.pop(op.field, None)
            else:
                obj[op.field] = op.value
            out[key] = obj
        elif kind == "increment":
            current = out.get(key)
            base = (
                current
                if isinstance(current, (int, float)) and not isinstance(current, bool)
                else 0
            )
            out[key] = base + op.by
    return out


def patch_preferences(
    data_dir: Path, user_id: str, updates: dict[str, Any], ops: list[Any] | None = None
) -> dict[str, Any]:
    """Shallow-merge ``updates`` into the user's preferences, apply ``ops``, return the new state.

    Keys whose value is ``None`` are DELETED from the stored payload — the client uses
    that to reset a specific preference to its local default without touching others.
    Any other value type (bool, string, number, nested object, array) is stored as-is.
    ``ops`` (item-level edits, ``apply_ops``) run after the merge under the same row lock, so two
    devices editing one list keep both changes.
    """
    if not isinstance(updates, dict):
        raise ValueError("preferences patch must be an object")
    with rw_engine().begin() as conn:
        # A row to lock even on the first write, so two first patches cannot both read "empty".
        conn.execute(
            pg_insert(user_preferences)
            .values(user_id=user_id, payload={})
            .on_conflict_do_nothing(index_elements=[user_preferences.c.user_id])
        )
        current = conn.execute(
            select(user_preferences.c.payload)
            .where(user_preferences.c.user_id == user_id)
            .with_for_update()
        ).scalar()
        merged: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
        for key, value in updates.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        if ops:
            merged = apply_ops(merged, ops)
        _store(conn, user_id, merged)
    return merged
