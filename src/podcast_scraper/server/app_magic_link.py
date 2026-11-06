"""Email magic-link sign-in — a second way into the platform, for people without a Google account.

Why this exists
───────────────
Consumer auth was Google OAuth and nothing else. That is a hard wall for a beta tester who will not
create a Google account, and "create one on your existing address" is a real answer only for people
willing to do it. This gives the platform a second front door that needs no third party.

What it is NOT: a password. There is still no credential stored anywhere. The proof of identity is
the same one a password reset relies on — demonstrated control of the mailbox — just without the
password that reset flow exists to recover.

The identity it produces
────────────────────────
``get_or_create_user(provider="email", subject=<normalised email>)``. No change was needed in the
user store for this: ``user_id = sha256(provider || subject)`` already namespaces by provider, so an
email identity cannot collide with the Google identity for the same address. They are deliberately
DIFFERENT accounts — the same person signing in both ways gets two, which is the honest outcome
given we cannot prove they are the same person without trusting one of the two providers to tell us.

Three properties, and why each is enforced here
───────────────────────────────────────────────
1. **Short-lived.** A link that works for a day is a credential sitting in an inbox. 15 minutes.
2. **Single-use.** A signature proves authenticity, NOT freshness-of-use — a forwarded or
   re-fetched link would otherwise work again and again until it expired. Email is replayed
   constantly by scanners, prefetchers and corporate link-rewriters, so this is not hypothetical.
   Enforcing it needs server state, which is the one thing the stateless session design avoided; it
   is a small, self-pruning table (``magic_link_used``) rather than a session store.
3. **Purpose-bound.** The payload carries ``purpose``, so a session cookie can never be presented as
   a magic token or vice versa, even though both are signed with the same secret.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, exc as sa_exc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from podcast_scraper.db import rw_engine
from podcast_scraper.db.schema import magic_link_requests, magic_link_used
from podcast_scraper.server import app_sessions

logger = logging.getLogger(__name__)

#: Marks a token as a sign-in link. Checked on verify so a session cookie — signed with the SAME
#: secret — can never be replayed as a magic token, nor the reverse.
PURPOSE = "magic_link"

#: How long a link stays usable. Short on purpose: the link IS the credential while it lives.
TOKEN_TTL_SECONDS = 15 * 60

#: Where consumed token ids were recorded before the database, under the app data dir (the
#: importer reads them), and the send-throttle markers beside them.
_USED_DIR = "magic_link_used"
_RECENT_DIR = "magic_link_recent"

#: Consumed-id records older than this are pruned. Must exceed TOKEN_TTL_SECONDS — a record only has
#: to outlive the token it blocks, since an expired token is rejected on age anyway.
_USED_RETENTION_SECONDS = 24 * 3600


_TOKEN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _at(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def normalise_email(raw: str) -> str:
    """Lowercase and strip. The subject must be stable, or one person gets several accounts.

    Deliberately NOT doing provider-specific canonicalisation (stripping Gmail dots or ``+tags``):
    that would silently merge addresses their own provider treats as distinct, and guessing wrong
    joins two people's accounts. Case and whitespace are safe because every mail system agrees on
    them for the domain, and in practice for the local part too.
    """
    return (raw or "").strip().lower()


def issue(email: str, secret: str, *, now: int | None = None) -> tuple[str, str]:
    """Mint a signed, single-use sign-in token for ``email``. Returns ``(token, token_id)``.

    ``token_id`` is random rather than derived from the email: it is written to disk on use, and a
    derived id would turn that directory into a list of who signed in and when.
    """
    token_id = secrets.token_urlsafe(16)
    payload: dict[str, Any] = {
        "purpose": PURPOSE,
        "email": normalise_email(email),
        "jti": token_id,
        "iat": int(time.time()) if now is None else now,
    }
    return app_sessions.sign(payload, secret), token_id


def parse(token: str | None, secret: str) -> dict[str, Any] | None:
    """Validate signature, purpose and age. Returns the payload, or ``None`` when unusable.

    Does NOT consume the token — single use is a separate, atomic step (:func:`consume`), because a
    caller that parsed successfully may still fail the access policy, and burning the token in that
    case would be wrong: the person is not at fault for the operator's allowlist.
    """
    payload = app_sessions.verify(token, secret, max_age=TOKEN_TTL_SECONDS)
    if payload is None:
        return None
    if payload.get("purpose") != PURPOSE:
        return None
    email = normalise_email(str(payload.get("email") or ""))
    if not email or "@" not in email:
        return None
    if not str(payload.get("jti") or ""):
        return None
    payload["email"] = email
    return payload


def consume(data_dir: Path, token_id: str, *, now: int | None = None) -> bool:
    """Claim ``token_id`` exactly once. True on the first call, False on every later one.

    One ``INSERT … ON CONFLICT DO NOTHING``: of two clicks arriving together exactly one inserts the
    row. A double-click on a link in a mail client is the ordinary case here, not an attack, and it
    must not produce two sessions.

    Fails CLOSED: if the row cannot be written, this returns False and the sign-in is refused. A
    single-use guarantee that silently degrades to unlimited-use on a database error is not one.
    """
    token_id = str(token_id or "")
    # Only the shape ``issue`` mints (``secrets.token_urlsafe``); anything else was never ours.
    if not _TOKEN_ID.fullmatch(token_id):
        return False
    stamp = int(time.time()) if now is None else now
    try:
        with rw_engine().begin() as conn:
            claimed = conn.execute(
                pg_insert(magic_link_used)
                .values(token_id=token_id, used_at=_at(stamp))
                .on_conflict_do_nothing(index_elements=[magic_link_used.c.token_id])
                .returning(magic_link_used.c.token_id)
            ).first()
    except sa_exc.SQLAlchemyError:
        logger.warning(
            "magic link: could not record token use; refusing the sign-in", exc_info=True
        )
        return False
    if claimed is None:
        return False
    _prune(now=stamp)
    return True


def _prune(*, now: int) -> None:
    """Drop consumed-id records older than the retention window. Best-effort, never raises.

    Without this the table grows forever — one row per successful sign-in, kept to block a replay
    that stopped being possible the moment the token expired.
    """
    try:
        with rw_engine().begin() as conn:
            conn.execute(
                delete(magic_link_used).where(
                    magic_link_used.c.used_at < _at(now - _USED_RETENTION_SECONDS)
                )
            )
    except sa_exc.SQLAlchemyError:
        logger.warning("magic link: pruning used tokens failed", exc_info=True)


def email_hash(email: str) -> str:
    """The throttle's key for an address: a hash, never the address, so the table is not a list
    of everyone who asked for a link (people who never had an account included)."""
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:32]


def request_throttled(email: str, *, now: int, min_interval_s: int) -> bool:
    """True when a link was already sent to this address within ``min_interval_s``; else records
    this request and returns False.

    One upsert that only moves the time forward once the window has passed, so two requests at once
    cannot both get through. Best-effort: failing to record it must not block a legitimate sign-in.
    """
    stamp = _at(now)
    try:
        with rw_engine().begin() as conn:
            recorded = conn.execute(
                pg_insert(magic_link_requests)
                .values(email_hash=email_hash(email), requested_at=stamp)
                .on_conflict_do_update(
                    index_elements=[magic_link_requests.c.email_hash],
                    set_={"requested_at": stamp},
                    where=magic_link_requests.c.requested_at
                    <= stamp - timedelta(seconds=min_interval_s),
                )
                .returning(magic_link_requests.c.email_hash)
            ).first()
    except sa_exc.SQLAlchemyError:
        logger.warning("magic link: could not record the request; not throttling", exc_info=True)
        return False
    return recorded is None


def forget_requests(conn: Connection, email: str) -> int:
    """Remove the throttle row for this address, both spellings the route may have used."""
    hashes = {email_hash(email), email_hash(normalise_email(email))}
    return int(
        conn.execute(
            delete(magic_link_requests).where(magic_link_requests.c.email_hash.in_(hashes))
        ).rowcount
    )


__all__ = [
    "PURPOSE",
    "TOKEN_TTL_SECONDS",
    "consume",
    "email_hash",
    "forget_requests",
    "issue",
    "normalise_email",
    "parse",
    "request_throttled",
]
