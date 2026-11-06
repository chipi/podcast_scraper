"""Per-user Web-Push subscriptions (#1415, PRD-046 FR1 / RFC-110 §6), on Postgres (ADR-169).

A user may have several push subscriptions (one per browser/device). Each is a W3C
``PushSubscription`` JSON (``{endpoint, expirationTime, keys:{p256dh,auth}}``), stored opaquely —
the app never inspects the keys; the infra worker signs + delivers (VAPID). One
``push_subscriptions`` row per endpoint, in the order they were added; they go with the account.

Dedupe is on ``endpoint`` (the stable identity of a subscription). A dead subscription is removed
either by the client (unsubscribe) or by the outbox suppression path on a push ``410/404`` bounce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.engine import Connection

from podcast_scraper.db import read_with_retry, rw_engine
from podcast_scraper.db.schema import push_subscriptions
from podcast_scraper.server.app_user_store import _is_safe_user_id

#: The file the store kept before the database; the importer reads it.
_FILE_NAME = "push_subscriptions.json"


def _select(conn: Connection, user_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        select(push_subscriptions.c.subscription)
        .where(push_subscriptions.c.user_id == user_id)
        .order_by(push_subscriptions.c.seq)
    ).scalars()
    return [s for s in rows if isinstance(s, dict) and s.get("endpoint")]


def list_subscriptions(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's stored push subscriptions ([] when unset / unsafe id)."""
    if not _is_safe_user_id(user_id):
        return []

    def read(engine: Any) -> list[dict[str, Any]]:
        with engine.connect() as conn:
            return _select(conn, user_id)

    return read_with_retry(read)


def add_subscription(
    data_dir: Path, user_id: str, subscription: dict[str, Any]
) -> list[dict[str, Any]]:
    """Add (or replace, keyed on ``endpoint``) a subscription. Returns the full list.

    A replaced subscription moves to the end, as it did in the file's list.
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    endpoint = str(subscription.get("endpoint") or "")
    if not endpoint:
        raise ValueError("subscription must carry an endpoint")
    with rw_engine().begin() as conn:
        conn.execute(
            delete(push_subscriptions).where(
                push_subscriptions.c.user_id == user_id,
                push_subscriptions.c.endpoint == endpoint,
            )
        )
        conn.execute(
            push_subscriptions.insert().values(
                user_id=user_id, endpoint=endpoint, subscription=subscription
            )
        )
        return _select(conn, user_id)


def remove_subscription(data_dir: Path, user_id: str, endpoint: str) -> list[dict[str, Any]]:
    """Remove the subscription with ``endpoint``. Returns the remaining list."""
    if not _is_safe_user_id(user_id):
        return []
    with rw_engine().begin() as conn:
        conn.execute(
            delete(push_subscriptions).where(
                push_subscriptions.c.user_id == user_id,
                push_subscriptions.c.endpoint == endpoint,
            )
        )
        return _select(conn, user_id)
