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
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from podcast_scraper.server import (
    app_auto_picks,
    app_comms_store,
    app_digest_sections,
    app_graph_refs,
    app_outbox_store,
    app_push_store,
    app_user_state,
)
from podcast_scraper.server.app_resurfacing import select_due
from podcast_scraper.server.app_user_store import get_user, list_users, User
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow

logger = logging.getLogger(__name__)

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
    # `revisit` on the link is what lets a ladder ADVANCE from outside the inbox (#35). Arriving at
    # the player carrying it marks the highlight surfaced, so a user who consumes revisit through
    # Your Week or the digest email progresses exactly like one who uses the inbox. Without it, the
    # only advance path in the product was the inbox's dismiss button — so an email-only reader was
    # sent the same five items every week, for ever, and "spaced" repetition never spaced.
    highlight_id = str(highlight.get("id") or "")
    params = []
    if t_ms is not None:
        params.append(f"t={t_ms // 1000}")
    if highlight_id:
        params.append(f"revisit={quote_plus(highlight_id)}")
    item: dict[str, Any] = {
        "episode_slug": slug,
        "graph_refs": refs,
        "deep_link": f"/player/{slug}" + (f"?{'&'.join(params)}" if params else ""),
        "t_ms": t_ms,
        "source": "user",
    }
    if highlight_id:
        # The in-app card builds its own route object rather than parsing deep_link, so it needs
        # the id as data. Auto-picks deliberately carry none: there is no ladder to advance.
        item["highlight_id"] = highlight_id
    quote = highlight.get("quote_text")
    if isinstance(quote, str) and quote:
        item["quote"] = quote
    return item


def assemble_digest_payload(
    root: Path,
    data_dir: Path,
    user_id: str,
    now: int,
    *,
    catalog: list[CatalogEpisodeRow] | None = None,
) -> dict[str, Any] | None:
    """Assemble the extractive, graph-carrying digest payload, or None when there's nothing to send.

    The revisit section is the user's due captures (source='user'), topped up — when under the cap —
    with GI editor's-picks from heard-but-uncaptured episodes (source='auto', #1416). So a user who
    captured nothing but listened still gets a non-empty digest.

    ``catalog`` is an optional pre-built catalog the caller can pass so a single request does one
    corpus scan (the /your-week route shares it with its item enrichment); the email path omits it.
    """
    highlights = app_user_state.get_highlights(data_dir, user_id)
    state = app_user_state.get_resurfacing_state(data_dir, user_id)
    # Pacing pause is a control on RESURFACING (RFC-101 §5: "frequency, pause, dismiss are per-user
    # settings"), not on one tab. It used to be passed only by GET /resurfacing, so a user who
    # paused pacing still had their captures resurfaced through Your Week, the digest email and the
    # push nudge — three surfaces ignoring the setting the UI presents as switching it off. The
    # separate comms.digest.paused consent gate governs whether the EMAIL is sent at all; this
    # governs whether resurfacing CONTENT exists to send.
    paused = bool(app_user_state.get_resurfacing_settings(data_dir, user_id).get("paused"))
    due = select_due(highlights, state, now, paused=paused)
    items: list[dict[str, Any]] = []
    for h in due:
        item = _digest_item(root, h)
        if item is not None:
            items.append(item)
        if len(items) >= MAX_REVISIT_ITEMS:
            break
    # Auto-picks are resurfacing too — GI editor's-picks standing in for captures the user does not
    # have. Topping up while paused would reintroduce exactly what the pause suppresses.
    if len(items) < MAX_REVISIT_ITEMS and not paused:
        captured = {str(h.get("episode_slug")) for h in highlights}
        items += app_auto_picks.auto_pick_items(
            root, data_dir, user_id, exclude_slugs=captured, limit=MAX_REVISIT_ITEMS - len(items)
        )
    sections: list[dict[str, Any]] = []
    if items:
        sections.append({"kind": "revisit", "items": items})
    new_in_follows = app_digest_sections.new_in_follows_items(
        root, data_dir, user_id, limit=MAX_REVISIT_ITEMS, catalog=catalog
    )
    if new_in_follows:
        sections.append({"kind": "new_in_follows", "items": new_in_follows})
    # Materialise topic/person follows (#1836) — recent unheard episodes about a followed topic or
    # featuring a followed person, deterministic + flag-independent (companion to show follows).
    new_in_interests = app_digest_sections.new_in_interests_items(
        root, data_dir, user_id, limit=MAX_REVISIT_ITEMS, catalog=catalog
    )
    if new_in_interests:
        sections.append({"kind": "new_in_interests", "items": new_in_interests})
    trending = app_digest_sections.trending_items(root, data_dir, user_id, limit=MAX_REVISIT_ITEMS)
    if trending:
        sections.append({"kind": "trending_in_your_corpus", "items": trending})
    if not sections:
        return None
    return {"sections": sections}


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
    # Select the revisit section by KIND, not by position. The revisit section is only appended
    # when non-empty, so sections[0] could be new_in_follows or trending — and this template is
    # "resurface-nudge.v1": a nudge whose highlight_count actually counted new episodes and whose
    # lead was not a highlight. With the pause fix above, a paused user reaches this path with no
    # revisit section at all, which made the mismatch easy to hit rather than a corner case.
    revisit = next((s for s in payload["sections"] if s.get("kind") == "revisit"), None)
    if revisit is None or not revisit.get("items"):
        return []
    nudge = _nudge_payload(revisit["items"])
    enqueued: list[str] = []
    for sub in subs:
        envelope = build_push_envelope(user, comms, sub, nudge, now)
        app_outbox_store.enqueue(data_dir, envelope)
        enqueued.append(str(envelope["id"]))
    return enqueued


def _is_due_slot(comms: dict[str, Any], now: int) -> bool:
    """Whether ``now`` (UTC) matches the user's chosen cadence slot (day_of_week + hour / hour).

    UTC only for v1 — per-user timezone is RFC-110's open question (needs a profile ``timezone``
    field). Pairs with an hourly digest cron: the per-period envelope id keeps it idempotent, so a
    user gets exactly one digest at their slot even if the cron fires every hour.
    """
    when = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    digest = comms["digest"]
    if digest["cadence"] == "daily":
        return int(when.hour) == int(digest["hour"])
    return (when.weekday(), int(when.hour)) == (int(digest["day_of_week"]), int(digest["hour"]))


def enqueue_due_digests(root: Path, data_dir: Path, now: int | None = None) -> list[str]:
    """Enqueue the email digest + push nudges for every consenting user *at their cadence slot*.

    Honours each user's ``day_of_week``/``hour``/``cadence`` (UTC). Per-period dedupe keeps it
    idempotent across re-runs, so an hourly cron is safe.
    """
    now = int(time.time()) if now is None else now
    enqueued: list[str] = []
    for user in list_users(data_dir):
        # Isolate each user: a lock timeout / assembler error on one must not skip the rest of
        # the roster (the loop is order-sensitive, and a hourly cron/sidecar drives it). The
        # per-period envelope id makes the next cycle idempotently retry the failed user.
        try:
            if not _is_due_slot(app_comms_store.get_comms(data_dir, user.user_id), now):
                continue
            eid = enqueue_for_user(root, data_dir, user.user_id, now)
            if eid is not None:
                enqueued.append(eid)
            enqueued.extend(enqueue_push_for_user(root, data_dir, user.user_id, now))
        except Exception:  # noqa: BLE001 — one bad user must never abort the whole run
            logger.exception("digest: enqueue failed for user %s; skipping", user.user_id)
    return enqueued
