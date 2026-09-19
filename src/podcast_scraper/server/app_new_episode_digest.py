"""New-episode alerts on the OUTBOUND channels — email (#2124) and push (#2125).

WHY THIS EXISTS
---------------
The Profile UI has offered ``New episodes`` with Email / Push / In-app checkboxes since the type
shipped. Only **in-app** was ever implemented (``app_new_episode_alerts``). Email was assumed to
be covered by the weekly digest's *new-in-follows* section, and push was written down as "a
separate follow-up". Neither assumption reached the user, who saw three checkboxes and ticked
email.

Folding this into the weekly digest is not the same product: a listener who wants to know an
episode dropped, and does not want a weekly roundup, got nothing — and one who had both heard
about it up to seven days late.

EVENT-DRIVEN, NOT A SLOT
------------------------
``digest`` and ``daily_recap`` fire at a configured hour. This type does not, deliberately
(operator decision 2026-09-19): it fires when an unheard episode appears in a followed show, and
each episode is announced **exactly once ever**. A fixed hour would make it a second daily
digest.

Being event-driven means the usual per-period envelope id cannot be the idempotency key — there
is no period. Three mechanisms replace it:

* **Per-episode dedupe** — an announced-slug ledger per user. An episode announced yesterday is
  never announced again, even if it is still unheard.
* **A rate floor** — ``floor_minutes`` between sends. Following twenty active shows must not
  produce a dozen notifications in one morning; that is how a channel gets muted forever.
* **Quiet hours** — local ``quiet_start``..``quiet_end``. Anything due inside the window is HELD,
  not dropped, and rolls into the next allowed send. This matters far more for push than email.

The rate floor also bounds the blast radius of a bug: the worst this can do is one notification
per ``floor_minutes`` per user.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from podcast_scraper.server import (
    app_comms_store,
    app_digest_sections,
    app_outbox_store,
    app_push_store,
)
from podcast_scraper.server.app_user_store import get_user, list_users, User

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
EMAIL_TEMPLATE = "new-episodes.v1"
PUSH_TEMPLATE = "new-episodes.v1"

#: Per-user ledger of episodes already announced on the outbound channels. Separate from the
#: in-app notification store's ``dedupe_key`` so muting one channel never silences the other.
_LEDGER_NAME = "new_episode_alerts.json"

#: Most episodes in a single notification. Beyond this the body is unreadable and the point
#: ("something new is out") is already made; the rest roll into the next send.
_MAX_ITEMS = 5

#: Outbound alerts are worthless once stale — a shorter TTL than the weekly digest's.
_TTL_S = 12 * 3600

#: NO AGE FILTER, deliberately, and this is a limitation worth knowing about.
#: ``app_digest_sections.new_in_follows_items`` returns
#: ``{episode_slug, episode_title, graph_refs, deep_link}`` — there is NO publication timestamp
#: on the item, so "only alert about episodes newer than N days" cannot be expressed here without
#: changing a function the weekly digest also depends on.
#:
#: The first-run seed below is what protects against the real risk (a backlog blast); an old
#: unheard episode that appears later is announced once, which is acceptable — the user follows
#: that show and has not heard it.


def _ledger_path(data_dir: Path, user_id: str) -> Path:
    return Path(data_dir) / "users" / user_id / _LEDGER_NAME


def read_ledger(data_dir: Path, user_id: str) -> dict[str, Any]:
    """``{"announced": {slug: ts}, "last_send_ts": int}``; ``{}`` when absent or unreadable."""
    try:
        loaded = json.loads(_ledger_path(data_dir, user_id).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent/corrupt ledger is a fresh start, not a failure
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_ledger(data_dir: Path, user_id: str, doc: dict[str, Any]) -> None:
    """Atomic. A torn ledger would re-announce episodes — the one failure users would notice."""
    path = _ledger_path(data_dir, user_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{_LEDGER_NAME}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError:
        logger.exception("new_episodes: ledger write failed for %s", user_id)


def _local_hour(now: int, tz: str | None) -> int:
    import datetime
    import zoneinfo

    try:
        zone = zoneinfo.ZoneInfo(tz) if tz else zoneinfo.ZoneInfo("UTC")
    except Exception:  # noqa: BLE001 — unknown/invalid IANA name: UTC, same as the other cadences
        zone = zoneinfo.ZoneInfo("UTC")
    return datetime.datetime.fromtimestamp(now, tz=zone).hour


def in_quiet_hours(sched: dict[str, Any], now: int, tz: str | None) -> bool:
    """Whether ``now`` falls inside the user's local quiet window.

    Handles the wrap case (22->8 spans midnight), which is the common configuration. Equal
    start/end disables quiet hours entirely.
    """
    start = int(sched.get("quiet_start", 22))
    end = int(sched.get("quiet_end", 8))
    if start == end:
        return False
    hour = _local_hour(now, tz)
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps midnight


def _rate_limited(ledger: dict[str, Any], sched: dict[str, Any], now: int) -> bool:
    last = ledger.get("last_send_ts")
    if not last:
        return False
    return (now - int(last)) < int(sched.get("floor_minutes", 240)) * 60


def assemble_new_episodes_payload(
    root: Path,
    data_dir: Path,
    user_id: str,
    now: int,
    *,
    announced: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Unheard, not-yet-announced episodes in followed shows. ``None`` when there are none.

    ``announced`` is the per-user slug ledger; anything in it is skipped permanently. There is no
    age filter — the source items carry no publication timestamp (see the module note). The
    first-run seed is what prevents a backlog blast.
    """
    announced = announced or {}
    try:
        items = app_digest_sections.new_in_follows_items(root, data_dir, user_id, limit=50)
    except Exception:  # noqa: BLE001 — a catalog failure must not break the dispatch pass
        logger.exception("new_episodes: could not build the new-in-follows delta for %s", user_id)
        return None

    fresh: list[dict[str, Any]] = []
    for item in items:
        # ``episode_slug`` is the key the shared section builder emits — NOT ``slug``.
        slug = str(item.get("episode_slug") or "")
        if not slug or slug in announced:
            continue
        fresh.append(item)
        if len(fresh) >= _MAX_ITEMS:
            break

    if not fresh:
        return None
    return {
        "count": len(fresh),
        "episodes": fresh,
        "slugs": [str(i["episode_slug"]) for i in fresh],
    }


def seed_ledger_without_sending(root: Path, data_dir: Path, user_id: str, now: int) -> int:
    """First run: mark everything currently unheard as already-announced, and send NOTHING.

    Without this, a user who has followed shows for months and then ticks the box gets alerted
    about a backlog — the single most effective way to make someone disable notifications
    permanently. Alerts should only ever cover what appears AFTER opting in.

    Returns how many slugs were seeded.
    """
    try:
        items = app_digest_sections.new_in_follows_items(root, data_dir, user_id, limit=500)
    except Exception:  # noqa: BLE001
        logger.exception("new_episodes: seed failed for %s", user_id)
        return 0
    announced = {str(i["episode_slug"]): now for i in items if i.get("episode_slug")}
    _write_ledger(data_dir, user_id, {"announced": announced, "last_send_ts": None, "seeded": now})
    return len(announced)


def _email_verified(user: User) -> bool:
    """Mirrors ``routes/app_comms._email_verified`` — Google-authenticated addresses only."""
    return getattr(user, "provider", None) == "google" and bool(getattr(user, "email", None))


def _iso(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _envelope_id(user_id: str, slugs: list[str], now: int) -> str:
    """Keyed on the EPISODE SET, not a calendar period.

    There is no period for an event-driven type, and a per-day id would silently drop a second
    batch. Hashing the slugs makes a re-run with the same content idempotent while allowing a
    genuinely different set through.
    """
    import hashlib

    digest = hashlib.sha256("|".join(sorted(slugs)).encode("utf-8")).hexdigest()[:12]
    return f"newep_{digest}_{user_id}"


def build_email_envelope(
    user: User, comms: dict[str, Any], payload: dict[str, Any], now: int
) -> dict[str, Any]:
    """Wrap a payload into a ``new_episodes`` email DeliveryEnvelope (schema v1).

    The top-level ``type`` is what the worker reads to build the one-click List-Unsubscribe
    (``…/comms/unsubscribe?ref=<ref>&type=new_episodes``), so an unsub silences ONLY this type.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "id": _envelope_id(user.user_id, payload["slugs"], now),
        "user_id": user.user_id,
        "type": "new_episodes",
        "channel": "email",
        "template": EMAIL_TEMPLATE,
        "recipient": {"email": user.email, "email_verified": _email_verified(user)},
        "consent_snapshot": {
            "new_episodes_enabled": app_comms_store.channel_enabled(comms, "new_episodes", "email"),
            "unsubscribe_ref": comms.get("unsubscribe_ref") or "",
        },
        "payload": payload,
        "not_before": _iso(now),
        "expires_at": _iso(now + _TTL_S),
        "created_at": _iso(now),
    }


def build_push_envelope(
    user: User, subscription: dict[str, Any], payload: dict[str, Any], now: int
) -> dict[str, Any]:
    """One envelope per subscription — a user may have both an iPhone and a browser registered.

    The id includes the endpoint, or the second device is silently deduped away.
    """
    import hashlib

    endpoint = str(subscription.get("endpoint") or "")
    sub_key = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:8]
    base = _envelope_id(user.user_id, payload["slugs"], now)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"{base}_{sub_key}",
        "user_id": user.user_id,
        "type": "new_episodes",
        "channel": "push",
        "template": PUSH_TEMPLATE,
        "subscription": subscription,
        "payload": payload,
        "not_before": _iso(now),
        "expires_at": _iso(now + _TTL_S),
        "created_at": _iso(now),
    }


def enqueue_for_user(root: Path, data_dir: Path, user_id: str, now: int | None = None) -> list[str]:
    """Enqueue new-episode alerts on every consenting channel. Returns the enqueued ids.

    Empty when: no user, paused, quiet hours, rate-limited, nothing new, or no channel consented.
    The slug ledger is advanced only when at least one envelope is actually enqueued — otherwise
    a quiet-hours hold would silently burn the episodes it was supposed to defer.
    """
    now = int(time.time()) if now is None else now
    user = get_user(data_dir, user_id)
    if user is None:
        return []

    comms = app_comms_store.get_comms(data_dir, user_id)
    sched = comms.get("new_episodes_schedule", {})
    if sched.get("paused"):
        return []

    email_on = app_comms_store.channel_enabled(comms, "new_episodes", "email") and _email_verified(
        user
    )
    push_on = app_comms_store.channel_enabled(comms, "new_episodes", "push")
    if not email_on and not push_on:
        return []

    ledger = read_ledger(data_dir, user_id)

    # FIRST RUN for this user: seed the backlog as already-announced and send nothing. Alerts
    # cover what appears AFTER opting in — blasting someone about months of unheard episodes is
    # how a channel gets disabled for good.
    if not ledger:
        n = seed_ledger_without_sending(root, data_dir, user_id, now)
        logger.info("new_episodes: seeded %d existing slug(s) for %s; no alert sent", n, user_id)
        return []

    if _rate_limited(ledger, sched, now):
        return []
    if in_quiet_hours(sched, now, comms.get("timezone")):
        return []  # HELD, not dropped — the ledger is untouched so these roll into the next send

    payload = assemble_new_episodes_payload(
        root, data_dir, user_id, now, announced=ledger.get("announced", {})
    )
    if payload is None:
        return []

    if not comms.get("unsubscribe_ref"):
        comms = app_comms_store.set_comms(data_dir, user_id)

    enqueued: list[str] = []

    if email_on:
        env = build_email_envelope(user, comms, payload, now)
        app_outbox_store.enqueue(data_dir, env)
        enqueued.append(str(env["id"]))

    if push_on:
        for sub in app_push_store.list_subscriptions(data_dir, user_id):
            env = build_push_envelope(user, sub, payload, now)
            app_outbox_store.enqueue(data_dir, env)
            enqueued.append(str(env["id"]))

    if enqueued:
        announced = dict(ledger.get("announced", {}))
        for slug in payload["slugs"]:
            announced[slug] = now
        _write_ledger(data_dir, user_id, {"announced": announced, "last_send_ts": now})

    return enqueued


def enqueue_due_new_episodes(root: Path, data_dir: Path, now: int | None = None) -> list[str]:
    """Driven by ``app_digest_dispatch.ENQUEUERS`` — one entry wires both schedulers (#2119)."""
    now = int(time.time()) if now is None else now
    enqueued: list[str] = []
    for user in list_users(data_dir):
        # Per-user isolation: one bad user must never abort the roster.
        try:
            enqueued.extend(enqueue_for_user(root, data_dir, user.user_id, now))
        except Exception:  # noqa: BLE001
            logger.exception("new_episodes: enqueue failed for user %s; skipping", user.user_id)
    return enqueued
