"""Delete an account and everything that names it (#2273) — App Store 5.1.1(v), Play data deletion.

``app_user_store.delete_user`` removes ``users/<id>/``, which holds every per-user store. It is not
the whole account: an audit (2026-10-04) found these OUTSIDE that directory, never pruned —

* the delivery **outbox** — every digest and push ever queued, with the address and push endpoint,
  and every sign-in email with the plaintext address (written before an account exists, so its
  ``user_id`` is empty and only the address matches);
* the **magic-link throttle marker**, named by a hash of the address;
* the **MCP personal-token index** (``{token_hash: user_id}``) and the **MCP OAuth server's**
  grants, consents and last-use records — a live refresh token there kept working for 30 days after
  the account was gone. These belong to an extension (ADR-162) and are removed by its
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

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from podcast_scraper.extensions import load_extensions
from podcast_scraper.server import app_magic_link, app_outbox_store
from podcast_scraper.server.app_user_store import _is_safe_user_id, delete_user, User

_APPLE_TOKEN_FILE = "apple_token.json"


def store_apple_refresh_token(data_dir: Path, user_id: str, token: str) -> None:
    """Keep the account's Apple refresh token — only so deletion can revoke it (Apple requires it).

    Inside ``users/<id>/``, so it goes with the account; owner-readable only.
    """
    if not token or not _is_safe_user_id(user_id):
        return
    path = data_dir / "users" / user_id / _APPLE_TOKEN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"refresh_token": token}), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def apple_refresh_token(data_dir: Path, user_id: str) -> str | None:
    """Return the stored Sign in with Apple refresh token for ``user_id``, or None.

    None when the id is not a safe path segment, no token file exists, or it cannot be read.
    """
    if not _is_safe_user_id(user_id):
        return None
    path = data_dir / "users" / user_id / _APPLE_TOKEN_FILE
    try:
        token = json.loads(path.read_text(encoding="utf-8")).get("refresh_token")
    except (OSError, ValueError, AttributeError):
        return None
    return str(token) if token else None


def _forget_throttle(data_dir: Path, email: str) -> int:
    """Remove the magic-link throttle marker for this address (both spellings the route may use)."""
    removed = 0
    for form in {email, app_magic_link.normalise_email(email)}:
        digest = hashlib.sha256(form.encode("utf-8")).hexdigest()[:32]
        marker = data_dir / "magic_link_recent" / f"{digest}.txt"
        try:
            marker.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


@dataclass(frozen=True)
class DeletionReport:
    """What was removed — counts only, for the log event; never an address or a token."""

    outbox_envelopes: int
    throttle_markers: int
    user_dir: bool
    #: Counts reported by extensions' ``account_deleted`` hooks, e.g. ``mcp_token_index``.
    extensions: dict[str, int] = field(default_factory=dict)


def delete_account(data_dir: Path, user: User) -> DeletionReport:
    """Remove the account and every record outside its directory that names it. Idempotent.

    The directory goes LAST: the outbox purge matches on the address, which is read from the
    profile the caller already holds, and nothing after this point may need it.
    """
    outbox = app_outbox_store.purge_for_user(data_dir, user.user_id, user.email)
    throttle = _forget_throttle(data_dir, user.email)
    counts: dict[str, int] = {}
    for ext in load_extensions():
        for hook in ext.account_deleted:
            counts.update(hook(data_dir, user))
    return DeletionReport(
        outbox_envelopes=outbox,
        throttle_markers=throttle,
        user_dir=delete_user(data_dir, user.user_id),
        extensions=counts,
    )
