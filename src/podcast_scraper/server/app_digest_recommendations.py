"""The monthly "recommendations" digest assembler (wave-H).

A SECOND email type alongside the weekly "Your Week" (``app_digest_personal``). Where Your Week is
personal (your revisit ladder + your follows), the recommendations digest is DISCOVERY — what is
rising in the corpus this month — so a user who has settled into their follows still gets a nudge
toward something new.

Same seam as Your Week: this module **assembles + enqueues** a ``DeliveryEnvelope`` (template
``recommendations-digest.v1``) to the outbox; the infra delivery worker (#1412) renders + sends it
(the email HTML + branding/logo live there, not here). Extractive + graph-carrying like every
envelope — each item is a topic graph node with a deep link, never a flat clip.

Cadence is a FIXED monthly (not the user's Your-Week ``digest_schedule.cadence``, which stays
weekly/daily); it rides the same ``digest`` × ``email`` consent cell + the schedule ``paused`` flag
+ email-verified gate. Zero content → no envelope.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path
from typing import Any

from podcast_scraper.server import (
    app_comms_store,
    app_digest_personal,
    app_digest_sections,
    app_outbox_store,
)
from podcast_scraper.server.app_digest_common import email_verified, iso
from podcast_scraper.server.app_user_store import get_user, list_users, User

logger = logging.getLogger(__name__)

MAX_ITEMS = 8
#: Envelope validity window. A "monthly" digest fired on the 1st stays valid ~1 month; 30 days is a
#: deliberate approximation (calendar months are 28–31 days) — it only bounds how long the infra
#: worker may still render a not-yet-sent envelope, so an exact month boundary is not required.
_MONTHLY_TTL_S = 30 * 86_400


def _period_key(now: int) -> str:
    """Stable per-month token for the envelope id (idempotent within a calendar month)."""
    return dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y%m")


def assemble_recommendations_payload(
    root: Path, data_dir: Path, user_id: str, now: int
) -> dict[str, Any] | None:
    """Discovery payload — rising topics in the user's corpus + new-in-interests. None when empty.

    Reuses the shipped extractive section builders (``app_digest_sections``), so it is graph-
    carrying and airgap-clean by construction. Distinct from Your Week: no revisit ladder, no
    show-follows — it leads with what is HEATING UP."""
    sections: list[dict[str, Any]] = []
    trending = app_digest_sections.trending_items(root, data_dir, user_id, limit=MAX_ITEMS)
    if trending:
        sections.append({"kind": "trending_in_your_corpus", "items": trending})
    new_in_interests = app_digest_sections.new_in_interests_items(
        root, data_dir, user_id, limit=MAX_ITEMS
    )
    if new_in_interests:
        sections.append({"kind": "new_in_interests", "items": new_in_interests})
    if not sections:
        return None
    return {"sections": sections}


def build_recommendations_envelope(
    user: User, comms: dict[str, Any], payload: dict[str, Any], now: int
) -> dict[str, Any]:
    """Wrap a discovery payload into a ``recommendations-digest.v1`` email envelope (monthly)."""
    return {
        "schema_version": app_digest_personal.SCHEMA_VERSION,
        "id": f"rec_{_period_key(now)}_{user.user_id}",
        "user_id": user.user_id,
        "type": "digest",
        "channel": "email",
        "template": "recommendations-digest.v1",
        "recipient": {
            "email": user.email,
            "email_verified": email_verified(user),
        },
        "consent_snapshot": {
            "digest_enabled": app_comms_store.channel_enabled(comms, "digest", "email"),
            "cadence": "monthly",
            "unsubscribe_ref": comms.get("unsubscribe_ref") or "",
        },
        "payload": payload,
        "not_before": iso(now),
        "expires_at": iso(now + _MONTHLY_TTL_S),
        "created_at": iso(now),
    }


def enqueue_recommendations_for_user(
    root: Path, data_dir: Path, user_id: str, now: int | None = None
) -> str | None:
    """Assemble + enqueue this user's monthly recommendations digest. Returns the id or None.

    None when the user hasn't consented (digest email off / paused / unverified) or has no
    discovery content. Safe to re-run within a month — the outbox dedupes on the per-month id."""
    now = int(time.time()) if now is None else now
    user = get_user(data_dir, user_id)
    if user is None:
        return None
    comms = app_comms_store.get_comms(data_dir, user_id)
    if (
        not app_comms_store.channel_enabled(comms, "digest", "email")
        or comms["digest_schedule"]["paused"]
        or not email_verified(user)
    ):
        return None
    if not comms.get("unsubscribe_ref"):
        comms = app_comms_store.set_comms(data_dir, user_id)  # mint the ref for the envelope
    payload = assemble_recommendations_payload(root, data_dir, user_id, now)
    if payload is None:
        return None
    envelope = build_recommendations_envelope(user, comms, payload, now)
    app_outbox_store.enqueue(data_dir, envelope)
    return str(envelope["id"])


def is_monthly_slot(comms: dict[str, Any], now: int) -> bool:
    """Whether ``now`` (UTC) is the user's monthly slot — the 1st of the month at their digest hour.

    Reuses the Your-Week ``digest_schedule.hour`` so a user's two emails land at the same time of
    day; per-month dedupe (the envelope id) makes an hourly cron safe.

    UTC only for v1 — matches ``app_digest_personal._is_due_slot``'s caveat: per-user timezone is
    RFC-110's open question. If the deployment runs the cron in a non-UTC zone, ``hour`` is compared
    against UTC and the slot fires at the wrong offset; tracked with the Your-Week gate."""
    when = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    return when.day == 1 and int(when.hour) == int(comms["digest_schedule"]["hour"])


def enqueue_due_recommendations(root: Path, data_dir: Path, now: int | None = None) -> list[str]:
    """Enqueue the monthly recommendations digest for every consenting user at their slot."""
    now = int(time.time()) if now is None else now
    enqueued: list[str] = []
    for user in list_users(data_dir):
        try:
            if not is_monthly_slot(app_comms_store.get_comms(data_dir, user.user_id), now):
                continue
            eid = enqueue_recommendations_for_user(root, data_dir, user.user_id, now)
            if eid is not None:
                enqueued.append(eid)
        except Exception:  # noqa: BLE001 — one bad user must never abort the whole run
            logger.exception("recommendations: enqueue failed for %s; skipping", user.user_id)
    return enqueued
