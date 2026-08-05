"""The personal "Your Week" digest assembler (#1415, PRD-046 FR2, RFC-110 §3-4).

Builds a per-user, **extractive** (D6: no LLM), **graph-carrying** ``DeliveryEnvelope`` from the
shipped substrate — the user's due resurfacing highlights (RFC-101 §5) — and enqueues it to the
outbox for the infra worker to render + deliver. This is NOT the corpus-wide operator digest
(RFC-068); it is scoped to one user's own captures.

Every digest item carries ``graph_refs`` (canonical person/topic ids resolved from the highlight's
episode KG) + a ``deep_link`` — the moat rule: an outbound item is a graph node, never a flat clip.
An item whose episode yields no KG entities is dropped rather than shipped flat, and a user with no
graph-carrying due items produces **no** envelope (zero-content → nothing enqueued).

Scope note: this module assembles + enqueues. Deciding *which users are due right now* (the Sunday
13:00 cadence gate) is the scheduler's job; :func:`enqueue_due_digests` enqueues the current
period for every consenting user and relies on outbox dedupe (per-period id) to be re-run-safe.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time
from pathlib import Path
from typing import Any

from podcast_scraper.server import (
    app_auto_picks,
    app_comms_store,
    app_graph_refs,
    app_outbox_store,
    app_push_store,
    app_user_state,
)
from podcast_scraper.server.app_resurfacing import select_due
from podcast_scraper.server.app_user_store import get_user, list_users, User

SCHEMA_VERSION = "1"
MAX_REVISIT_ITEMS = 5

_CADENCE_SECONDS = {"weekly": 7 * 86_400, "daily": 86_400}


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest_item(root: Path, highlight: dict[str, Any]) -> dict[str, Any] | None:
    """Build one graph-carrying digest item from a due highlight, or None if it has no graph.

    Prefers the refs persisted on the highlight at capture (#1419); falls back to episode-level
    resolution for highlights captured before graph refs were stored.
    """
    slug = str(highlight.get("episode_slug") or "")
    if not slug:
        return None
    refs = app_graph_refs.refs_for_highlight(root, highlight)
    if not refs:
        return None  # no graph → drop rather than ship a flat clip (moat rule)
    start_ms = highlight.get("start_ms")
    t_ms = int(start_ms) if isinstance(start_ms, int) else None
    item: dict[str, Any] = {
        "episode_slug": slug,
        "graph_refs": refs,
        "deep_link": f"/player/{slug}" + (f"?t={t_ms // 1000}" if t_ms is not None else ""),
        "t_ms": t_ms,
        "source": "user",
    }
    quote = highlight.get("quote_text")
    if isinstance(quote, str) and quote:
        item["quote"] = quote
    return item


def assemble_digest_payload(
    root: Path, data_dir: Path, user_id: str, now: int
) -> dict[str, Any] | None:
    """Assemble the extractive, graph-carrying digest payload, or None when there's nothing to send.

    The revisit section is the user's due captures (source='user'), topped up — when under the cap —
    with GI editor's-picks from heard-but-uncaptured episodes (source='auto', #1416). So a user who
    captured nothing but listened still gets a non-empty digest.
    """
    highlights = app_user_state.get_highlights(data_dir, user_id)
    state = app_user_state.get_resurfacing_state(data_dir, user_id)
    due = select_due(highlights, state, now)
    items: list[dict[str, Any]] = []
    for h in due:
        item = _digest_item(root, h)
        if item is not None:
            items.append(item)
        if len(items) >= MAX_REVISIT_ITEMS:
            break
    if len(items) < MAX_REVISIT_ITEMS:
        captured = {str(h.get("episode_slug")) for h in highlights}
        items += app_auto_picks.auto_pick_items(
            root, data_dir, user_id, exclude_slugs=captured, limit=MAX_REVISIT_ITEMS - len(items)
        )
    if not items:
        return None
    return {"sections": [{"kind": "revisit", "items": items}]}


def _period_key(now: int, cadence: str) -> str:
    """A stable per-period token for the envelope id (idempotent dedupe within a period)."""
    d = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    if cadence == "daily":
        return d.strftime("%Y%m%d")
    iso = d.isocalendar()
    return f"{iso.year}W{iso.week:02d}"


def build_email_envelope(
    user: User, comms: dict[str, Any], payload: dict[str, Any], now: int
) -> dict[str, Any]:
    """Wrap a payload into an email DeliveryEnvelope for ``user`` (schema v1 + ``expires_at``)."""
    cadence = str(comms["digest"]["cadence"])
    ttl = _CADENCE_SECONDS.get(cadence, _CADENCE_SECONDS["weekly"])
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"dgst_{_period_key(now, cadence)}_{user.user_id}",
        "user_id": user.user_id,
        "channel": "email",
        "template": "your-week-digest.v1",
        "recipient": {"email": user.email, "email_verified": _email_verified(user)},
        "consent_snapshot": {
            "digest_enabled": bool(comms["digest"]["enabled"]),
            "cadence": cadence,
            "unsubscribe_ref": comms.get("unsubscribe_ref") or "",
        },
        "payload": payload,
        "not_before": _iso(now),
        "expires_at": _iso(now + ttl),
        "created_at": _iso(now),
    }


def _email_verified(user: User) -> bool:
    """Identity-derived: Google-authenticated emails are verified (mirrors routes/app_comms)."""
    return user.provider == "google" and bool(user.email)


def enqueue_for_user(
    root: Path, data_dir: Path, user_id: str, now: int | None = None
) -> str | None:
    """Assemble + enqueue this user's current-period email digest. Returns the enqueued id or None.

    None when the user hasn't consented (digest off / paused / email unverified) or has no
    graph-carrying content. Safe to re-run within a period — the outbox dedupes on the envelope id.
    """
    now = int(time.time()) if now is None else now
    user = get_user(data_dir, user_id)
    if user is None:
        return None
    comms = app_comms_store.get_comms(data_dir, user_id)
    digest = comms["digest"]
    if not digest["enabled"] or digest["paused"] or not _email_verified(user):
        return None
    # Mint the unsubscribe_ref (first-save side effect) so the envelope always carries one.
    if not comms.get("unsubscribe_ref"):
        comms = app_comms_store.set_comms(data_dir, user_id, digest={})
    payload = assemble_digest_payload(root, data_dir, user_id, now)
    if payload is None:
        return None
    envelope = build_email_envelope(user, comms, payload, now)
    app_outbox_store.enqueue(data_dir, envelope)
    return str(envelope["id"])


def build_push_envelope(
    user: User,
    comms: dict[str, Any],
    subscription: dict[str, Any],
    payload: dict[str, Any],
    now: int,
) -> dict[str, Any]:
    """Wrap a nudge payload into a push DeliveryEnvelope for one subscription (schema v1)."""
    # One envelope per subscription; a stable hash of the FULL endpoint keeps ids distinct across
    # subscriptions (a prefix could collide) + idempotent per period.
    endpoint = str(subscription.get("endpoint") or "")
    sub_key = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16] if endpoint else "sub"
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"ndg_{_period_key(now, 'daily')}_{user.user_id}_{sub_key}",
        "user_id": user.user_id,
        "channel": "push",
        "template": "resurface-nudge.v1",
        "recipient": {"push_subscription": subscription},
        "consent_snapshot": {
            "digest_enabled": bool(comms["digest"]["enabled"]),
            "cadence": str(comms["digest"]["cadence"]),
            "unsubscribe_ref": comms.get("unsubscribe_ref") or "",
        },
        "payload": payload,
        "not_before": _iso(now),
        "expires_at": _iso(now + _CADENCE_SECONDS["daily"]),
        "created_at": _iso(now),
    }


def _nudge_payload(revisit_items: list[dict[str, Any]]) -> dict[str, Any]:
    """A resurface-nudge payload: a count + the single most-overdue lead item."""
    return {"highlight_count": len(revisit_items), "lead": revisit_items[0]}


def enqueue_push_for_user(
    root: Path, data_dir: Path, user_id: str, now: int | None = None
) -> list[str]:
    """Enqueue a push nudge to each of the user's subscriptions. Returns the enqueued ids.

    Gated on ``push.enabled`` + at least one stored subscription + graph-carrying due content.
    """
    now = int(time.time()) if now is None else now
    user = get_user(data_dir, user_id)
    if user is None:
        return []
    comms = app_comms_store.get_comms(data_dir, user_id)
    if not comms["push"]["enabled"]:
        return []
    subs = app_push_store.list_subscriptions(data_dir, user_id)
    if not subs:
        return []
    payload = assemble_digest_payload(root, data_dir, user_id, now)
    if payload is None:
        return []
    nudge = _nudge_payload(payload["sections"][0]["items"])
    enqueued: list[str] = []
    for sub in subs:
        envelope = build_push_envelope(user, comms, sub, nudge, now)
        app_outbox_store.enqueue(data_dir, envelope)
        enqueued.append(str(envelope["id"]))
    return enqueued


def enqueue_due_digests(root: Path, data_dir: Path, now: int | None = None) -> list[str]:
    """Enqueue the current-period email digest + push nudges for every consenting user.

    The cadence *timing* gate (is it this user's send slot?) belongs to the scheduler; this batch
    enqueues the current period and leans on per-period dedupe to stay idempotent across re-runs.
    """
    now = int(time.time()) if now is None else now
    enqueued: list[str] = []
    for user in list_users(data_dir):
        eid = enqueue_for_user(root, data_dir, user.user_id, now)
        if eid is not None:
            enqueued.append(eid)
        enqueued.extend(enqueue_push_for_user(root, data_dir, user.user_id, now))
    return enqueued
