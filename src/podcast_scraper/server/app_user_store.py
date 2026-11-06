"""Per-user identity store for the consumer platform (#1063/#1064), on Postgres.

Users live in the ``users`` table of the process's plane schema (ADR-169). The user id is derived
deterministically from the OAuth identity ``(provider, subject)``, so ``get_or_create_user`` is
idempotent by primary key. The functions keep the signatures they had as a file store: ``data_dir``
stays the first argument because the other per-user stores are still files under
``<data_dir>/users/<user_id>/``; the database comes from the engine registry
(``podcast_scraper.db``), not from that argument.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, RowMapping, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from podcast_scraper.db import read_with_retry, rw_engine
from podcast_scraper.db.schema import users
from podcast_scraper.server import app_roles


@dataclass(frozen=True)
class User:
    """A platform user (identity overlay; not a corpus artifact)."""

    user_id: str
    email: str
    name: str
    provider: str
    subject: str
    #: Immutable public handle (like @x / @insta), auto-derived at creation from the OAuth identity
    #: and deduped for uniqueness (#2004 Area E). Never chosen at registration, never changed.
    username: str = ""
    #: Effective avatar (Area E): the OAuth provider's picture captured at creation, later
    #: overridden by a user upload. None when the provider gave none and nothing was uploaded.
    image: str | None = None
    disabled: bool = False
    role: str = app_roles.DEFAULT_ROLE
    #: (#1471) May connect an external agent to the MCP server. Orthogonal to ``role`` (a
    #: listener may have it, an admin may not), not a rank. **Default on** since 2026-10-16
    #: (operator: every user gets MCP; a beta user who joined the day before had no access).
    #: An admin can still revoke it per user, and an explicit ``false`` stays revoked.
    mcp_access: bool = True
    #: Pseudonymous analytics identity (#2265). A random UUIDv4 minted once, at account creation,
    #: and never rotated.
    #:
    #: Deliberately NOT ``user_id`` and NOT derived from the email. ``user_id`` is
    #: ``sha256(provider || subject)``, so anyone holding the analytics data and a candidate
    #: identity could confirm the match by recomputing it. This id is random, so the only route
    #: back to a person is the operator's own private sheet.
    #:
    #: Empty on accounts written before this field existed; ``get_or_create_user`` backfills one
    #: at the next sign-in rather than leaving an existing account permanently unmeasurable.
    analytics_id: str = ""


#: The only shape ``user_id_for`` ever produces: ``u_`` + 24 lowercase hex chars.
_USER_ID_RE = re.compile(r"u_[0-9a-f]{24}")


def user_id_for(provider: str, subject: str) -> str:
    """Stable, opaque user id from the OAuth identity ``(provider, subject)``."""
    digest = hashlib.sha256(f"{provider}\x00{subject}".encode("utf-8")).hexdigest()
    return f"u_{digest[:24]}"


def new_analytics_id() -> str:
    """Mint a fresh pseudonymous analytics id (#2265).

    Random, not derived. The contrast with :func:`user_id_for` directly above is the point: that
    one is a digest of the OAuth identity, which is exactly what an analytics id must not be. A
    derived id would let anyone holding the analytics export test a guessed identity against it;
    a random one cannot be reversed, so the mapping lives only in the operator's private sheet.
    """
    return str(uuid.uuid4())


def _is_safe_user_id(user_id: str) -> bool:
    """Whether ``user_id`` is the opaque ``user_id_for`` shape (``u_`` + 24 hex).

    Defence-in-depth: the id flows into ``<data_dir>/users/<user_id>/`` file paths (and
    ``delete_user`` ``rmtree``) of the stores that are still files. Public lookups/mutations
    early-return the graceful "unknown user" result for a non-conforming id, so ``..`` / ``/`` /
    empty can never reach the filesystem sink.
    """
    return _USER_ID_RE.fullmatch(user_id) is not None


#: Public alias — other modules (e.g. the avatar route) validate a path-bound id with this rather
#: than reaching for the underscore-private name (advisor N2).
is_safe_user_id = _is_safe_user_id


def row_to_user(row: RowMapping) -> User:
    """A ``users`` row as a :class:`User` (NULL handle and analytics id read as '')."""
    return User(
        user_id=row["user_id"],
        email=row["email"],
        name=row["name"],
        # NULL in the table: accounts older than the handle have none.
        username=row["username"] or "",
        image=row["image"] or None,
        provider=row["provider"],
        subject=row["subject"],
        disabled=bool(row["disabled"]),
        role=row["role"],
        # NULL until backfilled.
        analytics_id=row["analytics_id"] or "",
        mcp_access=bool(row["mcp_access"]),
    )


def user_to_row(user: User) -> dict[str, Any]:
    """A :class:`User` as column values ('' handle and analytics id stored as NULL)."""
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "username": user.username or None,
        "image": user.image,
        "provider": user.provider,
        "subject": user.subject,
        "disabled": user.disabled,
        "role": user.role,
        "mcp_access": user.mcp_access,
        "analytics_id": user.analytics_id or None,
    }


def _select_user(conn: Connection, user_id: str) -> User | None:
    row = conn.execute(select(users).where(users.c.user_id == user_id)).mappings().first()
    return row_to_user(row) if row is not None else None


def get_user(data_dir: Path, user_id: str) -> User | None:
    """Load a user by id, or ``None`` when absent."""
    if not _is_safe_user_id(user_id):
        return None

    def read(engine: Any) -> User | None:
        with engine.connect() as conn:
            return _select_user(conn, user_id)

    return read_with_retry(read)


def _backfill_analytics_id(user: User) -> User:
    """Give an existing account an ``analytics_id`` if it has none (#2265).

    One conditional UPDATE: a concurrent sign-in (a double-fired OAuth callback, two devices at
    once) matches ``analytics_id IS NULL`` for exactly one of them, so two ids are never minted and
    the id is never rotated, which would silently split one participant into two in every report.

    **Never raises.** A failure here returns the user unchanged, so a database problem in the
    analytics backfill cannot stop someone signing in. The cost of failing is one account measured
    from its next sign-in instead of this one; the cost of raising would be a locked-out listener.
    """
    try:
        with rw_engine().begin() as conn:
            row = (
                conn.execute(
                    update(users)
                    .where(users.c.user_id == user.user_id, users.c.analytics_id.is_(None))
                    .values(analytics_id=new_analytics_id(), updated_at=func.now())
                    .returning(*users.c)
                )
                .mappings()
                .first()
            )
            if row is not None:
                return row_to_user(row)
            current = _select_user(conn, user.user_id)
            return current if current is not None else user
    except Exception:
        return user


_HANDLE_MAX = 30
_HANDLE_STRIP_RE = re.compile(r"[^a-z0-9_]+")
_HANDLE_COLLAPSE_RE = re.compile(r"_+")
#: Handle allocation loses a race only to a concurrent signup taking the same candidate; each
#: retry re-reads the taken set, so a handful of attempts is far beyond any real contention.
_HANDLE_ATTEMPTS = 20


def _sanitize_handle(seed: str) -> str:
    """A seed string → a bare handle: lowercase, [a-z0-9_], collapsed/trimmed, bounded."""
    base = _HANDLE_STRIP_RE.sub("_", (seed or "").strip().lower())
    base = _HANDLE_COLLAPSE_RE.sub("_", base).strip("_")[:_HANDLE_MAX].strip("_")
    return base or "user"


def _derive_username(email: str, name: str, taken: set[str]) -> str:
    """Auto-derive an immutable handle from the OAuth identity, deduped against ``taken`` with a
    numeric suffix. Prefers the email local-part, falls back to the display name."""
    local = email.split("@", 1)[0] if "@" in email else email
    base = _sanitize_handle(local or name)
    if base not in taken:
        return base
    i = 2
    # Keep base+suffix within the bound (advisor L6): trim the base to make room for the digits.
    while True:
        suffix = str(i)
        candidate = f"{base[: _HANDLE_MAX - len(suffix)].strip('_') or 'user'}{suffix}"
        if candidate not in taken:
            return candidate
        i += 1


def _taken_usernames(conn: Connection) -> set[str]:
    """The handles in use. Only a hint: the UNIQUE constraint decides, a lost race retries."""
    stmt = select(users.c.username).where(users.c.username.is_not(None))
    return {u for (u,) in conn.execute(stmt)}


def get_or_create_user(
    data_dir: Path,
    *,
    provider: str,
    subject: str,
    email: str,
    name: str,
    image: str | None = None,
    role: str | None = None,
    on_created: Callable[[User], None] | None = None,
) -> User:
    """Return the existing user for ``(provider, subject)`` or create it (idempotent).

    ``role`` only sets the role **on first creation**; an existing user's role is left untouched
    here (use :func:`set_role` to change it). When omitted, new users default to ``listener``.

    Also backfills a missing ``analytics_id`` on an existing account (#2265). Minting only at
    creation would leave every account that predates the field permanently unmeasurable — which is
    every account that exists today, including the operator's own and the e2e identities. This is
    the right place for it because it runs on each sign-in, and it is a one-time write per account:
    once set, the branch never fires again, and the id never rotates.

    ``on_created`` is called EXACTLY ONCE per real creation, after the row is committed, with the
    new user — the hook the `account_created` analytics event needs (#2266). It is here rather
    than in the caller because the caller cannot tell a creation from a return without a second
    read, and that read races: two concurrent first-logins for the same identity would both see
    "absent" and both report a signup, for one account. ``INSERT … ON CONFLICT DO NOTHING
    RETURNING`` yields a row for exactly one of them. A raising callback must not fail the
    sign-in, so it is guarded.
    """
    uid = user_id_for(provider, subject)
    existing = get_user(data_dir, uid)
    if existing is not None:
        if existing.analytics_id:
            return existing
        return _backfill_analytics_id(existing)

    for _ in range(_HANDLE_ATTEMPTS):
        with rw_engine().begin() as conn:
            user = User(
                user_id=uid,
                email=email,
                name=name,
                username=_derive_username(email, name, _taken_usernames(conn)),
                image=image,
                provider=provider,
                subject=subject,
                role=app_roles.normalize_role(role),
                analytics_id=new_analytics_id(),
            )
            inserted = conn.execute(
                pg_insert(users)
                .values(**user_to_row(user))
                .on_conflict_do_nothing()
                .returning(users.c.user_id)
            ).first()
            if inserted is None:
                # Nothing inserted: the account exists (a racing first-login won), or the handle
                # was taken by a concurrent signup between the read and the insert.
                raced = _select_user(conn, uid)
                if raced is not None:
                    return raced
                continue
        if on_created is not None:
            try:
                on_created(user)
            except Exception:  # noqa: BLE001 — a metric must never fail a signup.
                pass
        return user
    raise RuntimeError("could not allocate a unique handle")


def create_user(
    data_dir: Path, *, provider: str, subject: str, email: str, name: str, role: str
) -> User:
    """Write a fresh user row (overwrites) and return it. Callers check for prior existence.

    Mints the immutable handle too (advisor M4), so admin-provisioned / seeded users get one just
    like an OAuth first-login — the UNIQUE constraint holds uniqueness; a lost race retries."""
    uid = user_id_for(provider, subject)
    for attempt in range(_HANDLE_ATTEMPTS):
        try:
            with rw_engine().begin() as conn:
                user = User(
                    user_id=uid,
                    email=email,
                    name=name,
                    username=_derive_username(email, name, _taken_usernames(conn)),
                    provider=provider,
                    subject=subject,
                    role=app_roles.normalize_role(role),
                    analytics_id=new_analytics_id(),
                )
                values = user_to_row(user)
                conn.execute(
                    pg_insert(users)
                    .values(**values)
                    .on_conflict_do_update(
                        index_elements=[users.c.user_id],
                        set_={**values, "updated_at": func.now()},
                    )
                )
            return user
        except IntegrityError:
            if attempt == _HANDLE_ATTEMPTS - 1:
                raise
    raise RuntimeError("could not allocate a unique handle")  # pragma: no cover


def list_users(data_dir: Path) -> list[User]:
    """List all users (id-sorted); empty when there are none."""

    def read(engine: Any) -> list[User]:
        with engine.connect() as conn:
            rows = conn.execute(select(users).order_by(users.c.user_id.collate("C"))).mappings()
            return [row_to_user(r) for r in rows]

    return read_with_retry(read)


def list_user_ids(data_dir: Path) -> list[str]:
    """Every account id, sorted: the account list for cross-user work. Accounts are rows; a
    directory under ``users/`` is per-user file state, which may not exist yet."""

    def read(engine: Any) -> list[str]:
        with engine.connect() as conn:
            return list(
                conn.execute(select(users.c.user_id).order_by(users.c.user_id.collate("C")))
                .scalars()
                .all()
            )

    return read_with_retry(read)


def _update_user(user_id: str, **values: Any) -> bool:
    """Set columns on one user; False when the id is unsafe or unknown."""
    if not _is_safe_user_id(user_id):
        return False
    with rw_engine().begin() as conn:
        result = conn.execute(
            update(users).where(users.c.user_id == user_id).values(**values, updated_at=func.now())
        )
    return result.rowcount > 0


def set_image(data_dir: Path, user_id: str, image: str | None) -> bool:
    """Set a user's avatar URL (Area E upload override). Returns False for unknown users."""
    return _update_user(user_id, image=image)


#: Longest display name accepted. Generous for real names; short enough that a masthead, a share
#: card and an email greeting never have to truncate mid-word.
NAME_MAX_CHARS = 60


def normalize_display_name(raw: str) -> str | None:
    """A display name as stored: whitespace collapsed and trimmed, control characters refused.

    Returns None when nothing usable is left or it is too long. Deliberately permissive otherwise —
    names are not validated against an alphabet, because any such rule rejects someone's real name.
    """
    if not isinstance(raw, str):
        return None
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in raw):
        return None
    name = " ".join(raw.split())
    if not name or len(name) > NAME_MAX_CHARS:
        return None
    return name


def set_name(data_dir: Path, user_id: str, name: str) -> bool:
    """Set a user's display name. Returns False for unknown users or an unusable name.

    The person's own choice, for every provider. It is never overwritten by a later sign-in:
    `get_or_create_user` returns an existing account untouched, so Google re-sending its name does
    not undo a rename. The handle (`username`) is NOT derived from this and does not change.
    """
    clean = normalize_display_name(name)
    if clean is None:
        return False
    return _update_user(user_id, name=clean)


def set_disabled(data_dir: Path, user_id: str, disabled: bool) -> bool:
    """Enable/disable a user (disabled users fail auth). Returns False for unknown users."""
    return _update_user(user_id, disabled=bool(disabled))


def set_role(data_dir: Path, user_id: str, role: str) -> bool:
    """Set a user's role (coerced to a known role). Returns False for unknown users."""
    return _update_user(user_id, role=app_roles.normalize_role(role))


def set_mcp_access(data_dir: Path, user_id: str, mcp_access: bool) -> bool:
    """Grant/revoke a user's MCP-access entitlement. Returns False for unknown users."""
    return _update_user(user_id, mcp_access=bool(mcp_access))


def delete_user(data_dir: Path, user_id: str) -> bool:
    """Remove a user: the row, and the directory holding the stores that are still files (GDPR
    hard delete). Returns True if anything was removed."""
    if not _is_safe_user_id(user_id):
        return False
    with rw_engine().begin() as conn:
        deleted = conn.execute(delete(users).where(users.c.user_id == user_id)).rowcount > 0
    return remove_user_files(data_dir, user_id) or deleted


def remove_user_files(data_dir: Path, user_id: str) -> bool:
    """Remove ``users/<id>/``, the stores that are still files; whether it existed."""
    if not _is_safe_user_id(user_id):
        return False
    udir = data_dir / "users" / user_id
    had_dir = udir.is_dir()
    if had_dir:
        shutil.rmtree(udir, ignore_errors=True)
    return had_dir
