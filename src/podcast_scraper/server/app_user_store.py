"""File-based per-user identity store for the consumer platform (#1063/#1064, RFC-098 §3).

Per-user state as **plain files** (no DB) — the foundation #1065 extends. Each user is a
directory ``<data_dir>/users/<user_id>/`` holding ``profile.json``. The user id is derived
deterministically from the OAuth identity ``(provider, subject)``, so lookup is a direct
path probe and ``get_or_create_user`` is idempotent (racing creates write identical bytes).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from filelock import FileLock

from podcast_scraper.server import app_roles
from podcast_scraper.server.atomic_write import atomic_write_text

#: Read-modify-write profile mutations serialize on a per-user lock so concurrent admin
#: actions (role / activate flips) can't lose updates (mirrors the per-user file locks in
#: ``app_user_state``). Distinct users never contend; same-user writes are ordered.
_LOCK_TIMEOUT_S = 15.0


def _profile_lock(data_dir: Path, user_id: str) -> FileLock:
    path = _profile_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(".profile.lock")), timeout=_LOCK_TIMEOUT_S)


def _handles_lock(data_dir: Path) -> FileLock:
    """Store-GLOBAL lock for minting handles — uniqueness is a cross-user invariant the per-user
    profile lock can't hold, so the scan-taken-then-write must be serialized across users."""
    users_dir = data_dir / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(users_dir / ".handles.lock"), timeout=_LOCK_TIMEOUT_S)


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
    #: RFC-112 (#1471): may connect an external agent to the MCP server. Orthogonal to ``role`` (a
    #: listener may have it, an admin may not) — admin-granted, not a rank. Default off.
    mcp_access: bool = False


#: The only shape ``user_id_for`` ever produces: ``u_`` + 24 lowercase hex chars.
_USER_ID_RE = re.compile(r"u_[0-9a-f]{24}")


def user_id_for(provider: str, subject: str) -> str:
    """Stable, opaque user id from the OAuth identity ``(provider, subject)``."""
    digest = hashlib.sha256(f"{provider}\x00{subject}".encode("utf-8")).hexdigest()
    return f"u_{digest[:24]}"


def _is_safe_user_id(user_id: str) -> bool:
    """Whether ``user_id`` is the opaque ``user_id_for`` shape (``u_`` + 24 hex).

    Defence-in-depth: the id flows into ``<data_dir>/users/<user_id>/`` file paths
    (and ``delete_user`` ``rmtree``). Public lookups/mutations early-return the
    graceful "unknown user" result for a non-conforming id, so ``..`` / ``/`` / empty
    can never reach the filesystem sink — even though ids only ever originate from the
    HMAC-signed session, never raw request input.
    """
    return _USER_ID_RE.fullmatch(user_id) is not None


def _profile_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / "profile.json"


def _write_profile(data_dir: Path, user: User) -> None:
    path = _profile_path(data_dir, user.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps(
            {
                "email": user.email,
                "name": user.name,
                "username": user.username,
                "image": user.image,
                "provider": user.provider,
                "subject": user.subject,
                "disabled": user.disabled,
                "role": user.role,
                "mcp_access": user.mcp_access,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def get_user(data_dir: Path, user_id: str) -> User | None:
    """Load a user by id, or ``None`` when absent/unreadable."""
    if not _is_safe_user_id(user_id):
        return None
    path = _profile_path(data_dir, user_id)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return User(
        user_id=user_id,
        email=str(doc.get("email", "")),
        name=str(doc.get("name", "")),
        # Profiles written before Area E have no ``username``.
        username=str(doc.get("username", "")),
        image=(str(doc["image"]) if doc.get("image") else None),
        provider=str(doc.get("provider", "")),
        subject=str(doc.get("subject", "")),
        disabled=bool(doc.get("disabled", False)),
        # Profiles written before #1128 have no ``role`` → default to listener.
        role=app_roles.normalize_role(doc.get("role")),
        # Profiles written before #1471 have no ``mcp_access`` → default off.
        mcp_access=bool(doc.get("mcp_access", False)),
    )


_HANDLE_MAX = 30
_HANDLE_STRIP_RE = re.compile(r"[^a-z0-9_]+")
_HANDLE_COLLAPSE_RE = re.compile(r"_+")


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


def get_or_create_user(
    data_dir: Path,
    *,
    provider: str,
    subject: str,
    email: str,
    name: str,
    image: str | None = None,
    role: str | None = None,
) -> User:
    """Return the existing user for ``(provider, subject)`` or create it (idempotent).

    ``role`` only sets the role **on first creation**; an existing user's role is left untouched
    here (use :func:`set_role` to change it). When omitted, new users default to ``listener``.
    """
    uid = user_id_for(provider, subject)
    existing = get_user(data_dir, uid)
    if existing is not None:
        return existing
    # Serialize the mint under the store-global handle lock so two racing first-logins can't derive
    # the same handle from `taken` (uniqueness is cross-user; advisor H1). Re-check existence inside
    # the lock to also collapse a double-fired OAuth callback for the same (provider, subject).
    with _handles_lock(data_dir):
        existing = get_user(data_dir, uid)
        if existing is not None:
            return existing
        taken = {u.username for u in list_users(data_dir) if u.username}
        user = User(
            user_id=uid,
            email=email,
            name=name,
            username=_derive_username(email, name, taken),
            image=image,
            provider=provider,
            subject=subject,
            role=app_roles.normalize_role(role),
        )
        _write_profile(data_dir, user)
    return user


def create_user(
    data_dir: Path, *, provider: str, subject: str, email: str, name: str, role: str
) -> User:
    """Write a fresh user profile (overwrites) and return it. Callers check for prior existence.

    Mints the immutable handle too (advisor M4), so admin-provisioned / seeded users get one just
    like an OAuth first-login — under the store-global lock for uniqueness."""
    with _handles_lock(data_dir):
        taken = {u.username for u in list_users(data_dir) if u.username}
        user = User(
            user_id=user_id_for(provider, subject),
            email=email,
            name=name,
            username=_derive_username(email, name, taken),
            provider=provider,
            subject=subject,
            role=app_roles.normalize_role(role),
        )
        _write_profile(data_dir, user)
    return user


def list_users(data_dir: Path) -> list[User]:
    """List all users (id-sorted); empty when the store is absent."""
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    out: list[User] = []
    for child in sorted(users_dir.iterdir()):
        if child.is_dir():
            user = get_user(data_dir, child.name)
            if user is not None:
                out.append(user)
    return out


def set_image(data_dir: Path, user_id: str, image: str | None) -> bool:
    """Set a user's avatar URL (Area E upload override). Returns False for unknown users."""
    if not _is_safe_user_id(user_id):
        return False
    with _profile_lock(data_dir, user_id):
        user = get_user(data_dir, user_id)
        if user is None:
            return False
        _write_profile(data_dir, replace(user, image=image))
    return True


def set_disabled(data_dir: Path, user_id: str, disabled: bool) -> bool:
    """Enable/disable a user (disabled users fail auth). Returns False for unknown users."""
    if not _is_safe_user_id(user_id):
        return False
    with _profile_lock(data_dir, user_id):
        user = get_user(data_dir, user_id)
        if user is None:
            return False
        _write_profile(data_dir, replace(user, disabled=bool(disabled)))
    return True


def set_role(data_dir: Path, user_id: str, role: str) -> bool:
    """Set a user's role (coerced to a known role). Returns False for unknown users."""
    if not _is_safe_user_id(user_id):
        return False
    with _profile_lock(data_dir, user_id):
        user = get_user(data_dir, user_id)
        if user is None:
            return False
        _write_profile(data_dir, replace(user, role=app_roles.normalize_role(role)))
    return True


def set_mcp_access(data_dir: Path, user_id: str, mcp_access: bool) -> bool:
    """Grant/revoke a user's MCP-access entitlement (RFC-112). Returns False for unknown users."""
    if not _is_safe_user_id(user_id):
        return False
    with _profile_lock(data_dir, user_id):
        user = get_user(data_dir, user_id)
        if user is None:
            return False
        _write_profile(data_dir, replace(user, mcp_access=bool(mcp_access)))
    return True


def delete_user(data_dir: Path, user_id: str) -> bool:
    """Remove a user's directory (GDPR hard delete). Returns True if anything was removed."""
    if not _is_safe_user_id(user_id):
        return False
    udir = data_dir / "users" / user_id
    if not udir.is_dir():
        return False
    shutil.rmtree(udir, ignore_errors=True)
    return True
