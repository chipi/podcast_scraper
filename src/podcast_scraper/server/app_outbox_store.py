"""The delivery outbox — the app side of the app↔infra seam (#1415, RFC-110 §2, ADR-145).

The app enqueues ``DeliveryEnvelope``s here; the infra delivery worker (#1412) drains them over
``/internal/outbox/*`` (see ``routes/internal_outbox``), renders + delivers, and reports terminal
status back. This store is the **single source of truth + the only suppression authority** — there
is no external queue (Listmonk was dropped, ADR-144 (not in this repository) revision).

Layout: one ``outbox`` row per envelope (ADR-169), keyed by the envelope ``id``: the envelope, its
status, detail and last change. ``enqueue`` dedupes on the id (re-running the same digest period is
a no-op). Terminal statuses are final. Before the database it was one file per envelope at
``<data_dir>/outbox/<sha256(id)>.json``; the importer reads those.

Seam v1.1 amendments implemented here:
- **(1) idempotent status** — a repeated terminal status is a no-op (``record_status``).
- **(2) current-consent filtering** — ``list_pending`` re-reads live ``comms`` (the enqueue-time
  ``consent_snapshot`` is informational only) and drops since-unsubscribed + expired envelopes.
- **(4) ``failed`` is always-terminal** — in ``_TERMINAL``; the worker dead-letters as ``failed``.
- **(5) push ``410/404`` → ``bounced``** — a ``bounced`` push terminal disables ``push.enabled``;
  an email ``bounced``/``complaint`` disables ``digest.enabled`` (the suppression write-back).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from podcast_scraper.db import read_with_retry, rw_engine
from podcast_scraper.db.schema import comms, outbox
from podcast_scraper.server import app_comms_store
from podcast_scraper.server.app_user_store import get_user

#: Statuses that end an envelope's life. ``failed`` is always-terminal (v1.1 amendment 4): the
#: worker dead-letters as ``failed`` after N retries, so the app never re-offers it.
_TERMINAL: frozenset[str] = frozenset({"delivered", "bounced", "complaint", "suppressed", "failed"})

#: Terminal statuses that mean "stop sending to this recipient on this channel" (v1.1 amendment 5).
_SUPPRESSING: frozenset[str] = frozenset({"bounced", "complaint", "suppressed"})

#: How long a FINISHED envelope is kept (#2273). Kept forever before this: every digest sent, with
#: the recipient's address and push endpoint in it. Long enough to answer "did my digest go out last
#: week"; a sign-in envelope, which also carries the (by then expired) link, goes after a day.
_RETENTION_S = 30 * 86400
_AUTH_LINK_RETENTION_S = 86400

#: How often the retention purge runs. It rode every poll (the delivery worker polls each few
#: seconds) and deleted nothing almost every time, with JSONB filters no index serves. Retention is
#: a day at the shortest, so a purge a few minutes late is invisible.
_PURGE_EVERY_S = 600

_purge_lock = threading.Lock()
_last_purge: int | None = None


def reset_purge_clock() -> None:
    """Make the next poll purge (tests that swap or empty the database)."""
    global _last_purge
    with _purge_lock:
        _last_purge = None


def _purge_due(now: int) -> bool:
    """Whether this poll should purge, and if so note it. ``now`` is the poll's clock, so a clock
    that moved back (or a test that jumps it) purges rather than waiting out the interval."""
    global _last_purge
    with _purge_lock:
        if _last_purge is not None and 0 <= now - _last_purge < _PURGE_EVERY_S:
            return False
        _last_purge = now
        return True


#: Message types that are TRANSACTIONAL: sent because the recipient asked for them, in the moment,
#: and meaningless to gate on consent.
#:
#: There is exactly one, and it is auth. A magic-link email cannot be consent-gated because the
#: recipient HAS NO ACCOUNT YET — that is the entire point of it — so there is no ``user_id`` to
#: look a consent record up by, and no consent that could have been given. Requiring consent before
#: being allowed to let someone in is circular.
#:
#: This is a deliberate hole in a privacy gate, so it is a CLOSED SET rather than a flag on the
#: envelope. An envelope cannot opt itself out of consent by setting a field; it has to be one of
#: these named types. Adding to this set means deciding that a message may reach someone who never
#: agreed to hear from us — which is correct for "here is the link you just requested" and wrong for
#: essentially everything else, including anything resembling a notification, a digest or a nudge.
TRANSACTIONAL_TYPES = frozenset({"auth_link"})


#: The last second a ``datetime`` holds (9999-12-31T23:59:59Z); a later ``now`` is clamped to it.
_MAX_EPOCH = 253402300799


def _at(epoch: int) -> datetime:
    return datetime.fromtimestamp(min(epoch, _MAX_EPOCH), tz=timezone.utc)


def envelope_columns(envelope: dict[str, Any]) -> dict[str, Any]:
    """The columns copied out of an envelope: what the poll filters on and deletion matches."""
    raw_recipient = envelope.get("recipient")
    recipient: dict[str, Any] = raw_recipient if isinstance(raw_recipient, dict) else {}
    channel = envelope.get("channel")
    return {
        "channel": channel if isinstance(channel, str) else None,
        "user_id": str(envelope.get("user_id") or "") or None,
        "recipient_email": str(recipient.get("email") or "").strip().lower() or None,
    }


def purge_rows(conn: Connection, user_id: str, email: str) -> int:
    """Delete every envelope for this account (#2273) — pending AND finished. Returns the count.

    Matched by ``user_id`` and ALSO by recipient address: a sign-in envelope is written before an
    account exists and carries no ``user_id`` but the plaintext address, so the id alone would
    leave exactly those behind. An empty ``email`` matches by ``user_id`` alone, which account
    deletion passes when another account has the same address. Runs in the caller's transaction.
    """
    wanted_email = email.strip().lower()
    matches = []
    if user_id:
        matches.append(outbox.c.user_id == user_id)
    if wanted_email:
        matches.append(outbox.c.recipient_email == wanted_email)
    if not matches:
        return 0
    return int(conn.execute(delete(outbox).where(or_(*matches))).rowcount)


def enqueue(data_dir: Path, envelope: dict[str, Any]) -> bool:
    """Write an envelope to the outbox as ``pending``. Idempotent on ``envelope['id']``.

    Returns True when a new record was written, False when the id already existed (dedupe).
    """
    envelope_id = str(envelope.get("id") or "")
    if not envelope_id:
        raise ValueError("envelope must carry a non-empty id")
    with rw_engine().begin() as conn:
        written = conn.execute(
            pg_insert(outbox)
            .values(
                envelope_id=envelope_id,
                envelope=envelope,
                status="pending",
                detail=None,
                **envelope_columns(envelope),
            )
            .on_conflict_do_nothing(index_elements=[outbox.c.envelope_id])
            .returning(outbox.c.envelope_id)
        ).first()
    return written is not None


def _consent_allows(comms_doc: dict[str, Any], envelope: dict[str, Any]) -> bool:
    """Whether the user's *current* consent still permits delivering this envelope (amend. 2).

    Gated on the per-type × per-channel matrix (wave-I). The digest email additionally honours the
    schedule ``paused`` flag (pause applies to the digest only).

    Transactional types bypass the matrix entirely — see :data:`TRANSACTIONAL_TYPES`."""
    channel = envelope.get("channel")
    ntype = str(envelope.get("type") or "digest")
    if ntype in TRANSACTIONAL_TYPES:
        # Still channel-checked: a transactional type must not become a way to push.
        return channel == "email"
    if channel == "email":
        if not app_comms_store.channel_enabled(comms_doc, ntype, "email"):
            return False
        if ntype == "digest" and bool(comms_doc["digest_schedule"]["paused"]):
            return False
        return True
    if channel == "push":
        return app_comms_store.channel_enabled(comms_doc, ntype, "push")
    return False


def _purge_past_retention(conn: Connection, now: int) -> None:
    """Retention rides the worker's poll (at most every :data:`_PURGE_EVERY_S`): no scheduler."""
    transactional = outbox.c.envelope["type"].astext.in_(sorted(TRANSACTIONAL_TYPES))
    conn.execute(
        delete(outbox).where(
            outbox.c.status.in_(sorted(_TERMINAL)),
            or_(
                and_(
                    transactional,
                    outbox.c.updated_at < _at(now) - timedelta(seconds=_AUTH_LINK_RETENTION_S),
                ),
                and_(
                    ~transactional | outbox.c.envelope["type"].astext.is_(None),
                    outbox.c.updated_at < _at(now) - timedelta(seconds=_RETENTION_S),
                ),
            ),
        )
    )


def list_pending(
    data_dir: Path, *, channel: str, limit: int = 50, now: int | None = None
) -> list[dict[str, Any]]:
    """Pending envelopes for a channel — oldest-first, current-consent-filtered, non-expired.

    Excludes any envelope whose user has since unsubscribed (live ``comms`` re-check) and any whose
    ``expires_at`` has passed (so a homelab-down window can't flush stale digests on recovery). One
    query reads the pending envelopes together with each one's consent row.
    """
    now = int(time.time()) if now is None else now
    if _purge_due(now):
        try:
            with rw_engine().begin() as conn:
                _purge_past_retention(conn, now)
        except Exception:
            reset_purge_clock()
            raise
    query = (
        select(outbox.c.envelope, comms.c.settings, comms.c.unsubscribe_ref)
        .select_from(outbox.outerjoin(comms, comms.c.user_id == outbox.c.user_id))
        .where(outbox.c.status == "pending", outbox.c.channel == channel)
        .order_by(outbox.c.updated_at, outbox.c.envelope_id)
    )

    def read(engine: Any) -> list[Any]:
        with engine.connect() as conn:
            return list(conn.execute(query).all())

    out: list[dict[str, Any]] = []
    for row in read_with_retry(read):
        envelope = row.envelope
        if not isinstance(envelope, dict):
            continue
        expires_at = envelope.get("expires_at")
        if isinstance(expires_at, str) and _expired(expires_at, now):
            continue
        consent = app_comms_store._merged(
            app_comms_store.stored_document(row.settings, row.unsubscribe_ref)
        )
        if not _consent_allows(consent, envelope):
            continue
        out.append(envelope)
        if len(out) >= max(0, limit):
            break
    return out


def _expired(expires_at: str, now: int) -> bool:
    try:
        dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False  # unparsable TTL → treat as non-expiring rather than silently drop
    return dt.timestamp() < now


def record_status(data_dir: Path, envelope_id: str, status: str, detail: str | None = None) -> str:
    """Record a terminal status for an envelope. Idempotent per id (v1.1 amendment 1).

    A repeated terminal status is a no-op that returns the already-stored status (the worker may
    retry a status-write after a send whose ack failed). On a suppressing terminal status the
    matching consent channel is disabled — the app stops enqueuing to that recipient (amendment 5).
    Returns the effective (stored) status. Unknown ids return ``"unknown"``.
    """
    if status not in _TERMINAL:
        raise ValueError(f"not a terminal status: {status!r}")
    with rw_engine().begin() as conn:
        row = conn.execute(
            select(outbox.c.status, outbox.c.envelope)
            .where(outbox.c.envelope_id == envelope_id)
            .with_for_update()
        ).first()
        if row is None:
            return "unknown"
        if row.status in _TERMINAL:
            return str(row.status)  # idempotent: already terminal
        conn.execute(
            update(outbox)
            .where(outbox.c.envelope_id == envelope_id)
            .values(status=status, detail=detail, updated_at=func.now())
        )
        envelope = row.envelope if isinstance(row.envelope, dict) else {}
    if status in _SUPPRESSING:
        _suppress(data_dir, envelope)
    return status


def _suppress(data_dir: Path, envelope: dict[str, Any]) -> None:
    """Disable the exact type×channel the terminal status arrived on (bounce/complaint → stop)."""
    user_id = str(envelope.get("user_id") or "")
    if not user_id:
        return
    # A bounce that arrives after the account was deleted must not write consent for it (#2273).
    if get_user(data_dir, user_id) is None:
        return
    channel = envelope.get("channel")
    if channel not in ("email", "push"):
        return
    # A push bounce is about ONE device's endpoint (FCM 404 UNREGISTERED, APNs 400 BadDeviceToken).
    # Drop that subscription; only when none is left is there nothing to deliver to, and the channel
    # goes off as before. Prod 2026-10-07: two stale tokens switched new-episode push off for the
    # whole account, the two working phones included, and stayed to do it again on the next send.
    if channel == "push":
        endpoint = str(
            ((envelope.get("recipient") or {}).get("push_subscription") or {}).get("endpoint") or ""
        )
        if endpoint:
            from podcast_scraper.server import app_push_store

            if app_push_store.remove_subscription(data_dir, user_id, endpoint):
                return
    raw_type = envelope.get("type")
    try:
        # When the envelope carries no ``type`` (a pre-matrix envelope), we don't know which type
        # bounced. For push that means the endpoint is bad — disable push across ALL types rather
        # than guess ``digest`` and leave e.g. new_episodes still trying the dead endpoint.
        if not isinstance(raw_type, str) or not raw_type:
            if channel == "push":
                app_comms_store.disable_push_everywhere(data_dir, user_id)
            else:
                app_comms_store.set_channel(data_dir, user_id, "digest", "email", False)
            return
        app_comms_store.set_channel(data_dir, user_id, raw_type, channel, False)
    except ValueError:
        return
