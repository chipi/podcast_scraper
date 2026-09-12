"""The daily post-episode recap digest (RFC-122 #2039).

An end-of-day email recapping the episodes a listener FINISHED that day — the email counterpart of
the in-app recap end-card (#2038), rendering the SAME per-episode recap model via the shared
:mod:`app_recap_view` builder so the two surfaces cannot drift.

Shape mirrors :mod:`app_digest_personal`: this module ASSEMBLES + ENQUEUES a ``DeliveryEnvelope``
to the outbox; the infra worker (#1412) renders ``daily-recap.v1`` + sends via Resend. Deciding
*which users are due this hour* is the scheduler's job (:func:`enqueue_due_daily_recaps`), kept
idempotent by a per-day envelope id.

The 0/1/many rule: **zero episodes finished today → no envelope** (nothing to recap, no email). One
or more → one digest carrying each finished episode's recap. The email template renders it
adaptively (full for a single episode, a compact stack for several).

Bridge-only (PRD-035 Principle 4): transcript-derived text + KG metadata + artwork only.

Timezone note: "today" is the UTC day, and the send slot is a fixed UTC hour — same limitation as
the weekly digest. Landing the recap at each recipient's LOCAL end of day needs a per-user timezone
and is the tracked follow-up; v1 ships on the existing fixed-slot mechanism.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from podcast_scraper.server import app_comms_store, app_outbox_store, app_recap_view, app_user_state
from podcast_scraper.server.app_digest_common import (
    email_verified as _email_verified,
    iso as _iso,
    local_day as _local_day,
    local_now as _local_now,
)
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_store import get_user, list_users, User
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.schemas import AppEpisodeRecap

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
#: Cap the episodes recapped in one email — a listener who binges a whole day still gets a scannable
#: digest, not an endless page. The most-recently-finished win.
MAX_RECAP_EPISODES = 5
_DAY_SECONDS = 86_400


def _finished_today(data_dir: Path, user_id: str, now: int, tz: str | None = None) -> list[str]:
    """Slugs the user FINISHED today (in THEIR timezone), most-recently-finished first, capped.

    "Today" is the user's local day (``tz``; UTC fallback) so the recap's contents match its
    local-evening send time (#2041). Keyed on the authoritative finish timestamp —
    ``listening.finished_at[slug]``, the epoch the episode was FIRST marked finished (set-once,
    #1914) — NOT ``playback.updated_at``, which a later resume/re-open would bump. So an episode
    re-opened tomorrow still counts on the day it was actually finished.
    """
    today = _local_day(now, tz)
    finished_at = app_user_state.get_listening(data_dir, user_id).get("finished_at", {})
    todays = [
        (str(slug), int(ts))
        for slug, ts in finished_at.items()
        if isinstance(ts, int) and _local_day(int(ts), tz) == today
    ]
    todays.sort(key=lambda item: item[1], reverse=True)  # most-recently-finished first
    return [slug for slug, _ in todays[:MAX_RECAP_EPISODES]]


def _recap_email_item(recap: AppEpisodeRecap) -> dict[str, Any]:
    """Project a recap model to the email item shape (the worker renders it adaptively).

    Carries every field the template needs for BOTH layouts (full for one episode, compact for
    many); the template decides how much to show from ``count``. Bridge-only — text + refs only.
    """
    quote = recap.signature_quote
    return {
        "slug": recap.slug,
        "title": recap.title,
        "podcast_title": recap.podcast_title,
        "artwork_url": recap.artwork_url,
        "key_points": list(recap.key_points[:3]),
        "signature_quote": ({"text": quote.text, "speaker": quote.speaker} if quote else None),
        "insights": [ins.text for ins in recap.insights[:3]],
        "topics": [{"id": t.id, "label": t.label} for t in recap.topics[:4]],
        "storylines": [{"id": s.id, "label": s.label} for s in recap.storylines[:2]],
        "deep_link": f"/player/{recap.slug}",
    }


def assemble_daily_recap_payload(
    root: Path,
    data_dir: Path,
    user_id: str,
    now: int,
    *,
    tz: str | None = None,
    catalog: list[CatalogEpisodeRow] | None = None,  # accepted for parity; resolve is per-slug
) -> dict[str, Any] | None:
    """Assemble the daily recap payload, or None when the user finished nothing today.

    ``tz`` is the user's IANA timezone (UTC fallback) — "today" is their local day. None (→ no
    envelope, no email) is the 0-listens case. Otherwise: one recap item per finished episode,
    newest first, via the shared recap builder.
    """
    slugs = _finished_today(data_dir, user_id, now, tz)
    if not slugs:
        return None
    episodes: list[dict[str, Any]] = []
    for slug in slugs:
        row = resolve_slug(root, slug)
        if row is None:
            continue  # the episode left the corpus since it was heard — skip, don't fail
        recap = app_recap_view.build_episode_recap(root, row, slug, limit=3)
        episodes.append(_recap_email_item(recap))
    if not episodes:
        return None
    return {"day": _local_day(now, tz), "count": len(episodes), "episodes": episodes}


def build_email_envelope(
    user: User, comms: dict[str, Any], payload: dict[str, Any], now: int
) -> dict[str, Any]:
    """Wrap a payload into a ``daily_recap`` email DeliveryEnvelope (schema v1).

    The top-level ``type: "daily_recap"`` is what the worker reads to build the one-click
    List-Unsubscribe (``…/comms/unsubscribe?ref=<ref>&type=daily_recap``), so an unsub silences the
    recap only, not the weekly digest. ``consent_snapshot`` keeps the shared shape (digest_enabled
    here = the recap-email consent snapshot; cadence is always daily).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"drcp_{_local_day(now, comms.get('timezone')).replace('-', '')}_{user.user_id}",
        "user_id": user.user_id,
        "type": "daily_recap",
        "channel": "email",
        "template": "daily-recap.v1",
        "recipient": {"email": user.email, "email_verified": _email_verified(user)},
        "consent_snapshot": {
            "digest_enabled": app_comms_store.channel_enabled(comms, "daily_recap", "email"),
            "cadence": "daily",
            "unsubscribe_ref": comms.get("unsubscribe_ref") or "",
        },
        "payload": payload,
        "not_before": _iso(now),
        "expires_at": _iso(now + _DAY_SECONDS),
        "created_at": _iso(now),
    }


def enqueue_for_user(
    root: Path, data_dir: Path, user_id: str, now: int | None = None
) -> str | None:
    """Assemble + enqueue this user's daily recap. Returns the enqueued id or None.

    None when the user hasn't consented (daily_recap email off / paused / email unverified) or
    finished nothing today. Safe to re-run within a day — the outbox dedupes on the per-day id.
    """
    now = int(time.time()) if now is None else now
    user = get_user(data_dir, user_id)
    if user is None:
        return None
    comms = app_comms_store.get_comms(data_dir, user_id)
    if (
        not app_comms_store.channel_enabled(comms, "daily_recap", "email")
        or comms["daily_recap_schedule"]["paused"]
        or not _email_verified(user)
    ):
        return None
    payload = assemble_daily_recap_payload(root, data_dir, user_id, now, tz=comms.get("timezone"))
    if payload is None:
        return None
    # Mint the unsubscribe_ref (first-save side effect) so the envelope always carries one.
    if not comms.get("unsubscribe_ref"):
        comms = app_comms_store.set_comms(data_dir, user_id)
    envelope = build_email_envelope(user, comms, payload, now)
    app_outbox_store.enqueue(data_dir, envelope)
    return str(envelope["id"])


def _is_due_slot(comms: dict[str, Any], now: int) -> bool:
    """Whether ``now`` matches the user's daily-recap hour in THEIR timezone (#2041).

    The configured hour is LOCAL to the user's ``timezone`` (IANA; UTC fallback). Pairs with the
    hourly cron: the per-day envelope id keeps it idempotent, so a user gets exactly one recap at
    their local slot even though the cron fires every hour.
    """
    when = _local_now(now, comms.get("timezone"))
    return int(when.hour) == int(comms["daily_recap_schedule"]["hour"])


def enqueue_due_daily_recaps(root: Path, data_dir: Path, now: int | None = None) -> list[str]:
    """Enqueue the daily recap for every consenting user at their slot. Idempotent per day."""
    now = int(time.time()) if now is None else now
    enqueued: list[str] = []
    for user in list_users(data_dir):
        try:
            if not _is_due_slot(app_comms_store.get_comms(data_dir, user.user_id), now):
                continue
            eid = enqueue_for_user(root, data_dir, user.user_id, now)
            if eid is not None:
                enqueued.append(eid)
        except Exception:  # noqa: BLE001 — one bad user must never abort the whole run
            logger.exception("daily-recap: enqueue failed for user %s; skipping", user.user_id)
    return enqueued
