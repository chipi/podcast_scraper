"""Delete an account and everything that names it (#2273) — App Store 5.1.1(v), Play data deletion.

The ``users`` row's foreign keys take every per-user kernel table with it (comms, push, preferences,
the Apple token, graph events), and ``users/<id>/`` holds the stores that are still files. That is
not the whole account: an audit (2026-10-04) found these elsewhere, never pruned —

* the delivery **outbox** — every digest and push ever queued, with the address and push endpoint,
  and every sign-in email with the plaintext address (written before an account exists, so its
  ``user_id`` is empty and only the address matches);
* the **magic-link throttle marker**, named by a hash of the address;
* the **MCP personal-token index** (``{token_hash: user_id}``) and the **MCP OAuth server's**
  grants, consents and last-use records — a live refresh token there kept working for 30 days after
  the account was gone. These belong to an extension (ADR-168) and are removed by its
  ``account_deleted`` hook, so an installed extension cannot leave records behind.

:func:`delete_account` is the ONE entry point that removes all of it, used by the self-service
route, the admin route and the CLI, so they cannot drift apart. Deliberately left alone:

* ``audit.jsonl`` — a security trail a person can delete themselves out of is not one. It holds the
  user id, not the address.
* the sign-in allowlist — keeping the address there is the operator's decision (2026-10-04), so the
  person can sign up again.
* logs already shipped to VictoriaLogs — outside the app's reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db import read_with_retry, rw_engine
from podcast_scraper.db.schema import apple_tokens, users
from podcast_scraper.extensions import load_extensions
from podcast_scraper.server import app_magic_link, app_outbox_store
from podcast_scraper.server.app_user_store import _is_safe_user_id, remove_user_files, User

#: The file the token was kept in before the database; the importer reads it.
_APPLE_TOKEN_FILE = "apple_token.json"


def store_apple_refresh_token(data_dir: Path, user_id: str, token: str) -> None:
    """Keep the account's Apple refresh token — only so deletion can revoke it (Apple requires it).

    An ``apple_tokens`` row, which goes with the account.
    """
    if not token or not _is_safe_user_id(user_id):
        return
    with rw_engine().begin() as conn:
        conn.execute(
            pg_insert(apple_tokens)
            .values(user_id=user_id, refresh_token=token)
            .on_conflict_do_update(
                index_elements=[apple_tokens.c.user_id],
                set_={"refresh_token": token, "updated_at": func.now()},
            )
        )


def apple_refresh_token(data_dir: Path, user_id: str) -> str | None:
    """Return the stored Sign in with Apple refresh token for ``user_id``, or None.

    None when the id is not a user id or no token is stored.
    """
    if not _is_safe_user_id(user_id):
        return None

    def read(engine: Any) -> str | None:
        with engine.connect() as conn:
            token = conn.execute(
                select(apple_tokens.c.refresh_token).where(apple_tokens.c.user_id == user_id)
            ).scalar()
        return str(token) if token else None

    return read_with_retry(read)


@dataclass(frozen=True)
class DeletionReport:
    """What was removed — counts only, for the log event; never an address or a token."""

    outbox_envelopes: int
    throttle_markers: int
    user_dir: bool
    #: Counts reported by extensions' ``account_deleted`` hooks, e.g. ``mcp_token_index``.
    extensions: dict[str, int] = field(default_factory=dict)


def _address_shared(conn: Any, user: User) -> bool:
    """Whether another account has this account's email address."""
    address = user.email.strip().lower()
    if not address:
        return False
    other = conn.execute(
        select(users.c.user_id)
        .where(func.lower(func.trim(users.c.email)) == address, users.c.user_id != user.user_id)
        .limit(1)
    ).first()
    return other is not None


def delete_account(data_dir: Path, user: User) -> DeletionReport:
    """Remove the account and every record that names it. Idempotent.

    Extensions first: their hooks revoke credentials (an MCP refresh token), and if one fails the
    account is still there to retry the deletion from. Then ONE transaction: the outbox envelopes
    and the throttle row (no foreign key: a sign-in envelope predates its account; matched by
    address only when no other account has the address) and the
    ``users`` row, whose foreign keys cascade to every kernel table that names the account. The
    directory of the stores that are still files goes last.
    """
    counts: dict[str, int] = {}
    for ext in load_extensions():
        for hook in ext.account_deleted:
            counts.update(hook(data_dir, user))
    if not _is_safe_user_id(user.user_id):
        return DeletionReport(0, 0, False, counts)
    with rw_engine().begin() as conn:
        # Another account on the same address (another provider) still owns the records matched by
        # address alone: its sign-in envelopes and the address's throttle row.
        address = "" if _address_shared(conn, user) else user.email
        outbox = app_outbox_store.purge_rows(conn, user.user_id, address)
        throttle = app_magic_link.forget_requests(conn, address) if address else 0
        deleted = conn.execute(delete(users).where(users.c.user_id == user.user_id)).rowcount > 0
    had_files = remove_user_files(data_dir, user.user_id)
    return DeletionReport(
        outbox_envelopes=outbox,
        throttle_markers=throttle,
        user_dir=deleted or had_files,
        extensions=counts,
    )
