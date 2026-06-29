"""File-based per-user identity store for the consumer platform (#1063/#1064, RFC-098 §3).

Per-user state as **plain files** (no DB) — the foundation #1065 extends. Each user is a
directory ``<data_dir>/users/<user_id>/`` holding ``profile.json``. The user id is derived
deterministically from the OAuth identity ``(provider, subject)``, so lookup is a direct
path probe and ``get_or_create_user`` is idempotent (racing creates write identical bytes).
"""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True)
class User:
    """A platform user (identity overlay; not a corpus artifact)."""

    user_id: str
    email: str
    name: str
    provider: str
    subject: str
    disabled: bool = False
    role: str = app_roles.DEFAULT_ROLE


def user_id_for(provider: str, subject: str) -> str:
    """Stable, opaque user id from the OAuth identity ``(provider, subject)``."""
    digest = hashlib.sha256(f"{provider}\x00{subject}".encode("utf-8")).hexdigest()
    return f"u_{digest[:24]}"


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
                "provider": user.provider,
                "subject": user.subject,
                "disabled": user.disabled,
                "role": user.role,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def get_user(data_dir: Path, user_id: str) -> User | None:
    """Load a user by id, or ``None`` when absent/unreadable."""
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
        provider=str(doc.get("provider", "")),
        subject=str(doc.get("subject", "")),
        disabled=bool(doc.get("disabled", False)),
        # Profiles written before #1128 have no ``role`` → default to listener.
        role=app_roles.normalize_role(doc.get("role")),
    )


def get_or_create_user(
    data_dir: Path,
    *,
    provider: str,
    subject: str,
    email: str,
    name: str,
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
    user = User(
        user_id=uid,
        email=email,
        name=name,
        provider=provider,
        subject=subject,
        role=app_roles.normalize_role(role),
    )
    _write_profile(data_dir, user)
    return user


def create_user(
    data_dir: Path, *, provider: str, subject: str, email: str, name: str, role: str
) -> User:
    """Write a fresh user profile (overwrites) and return it. Callers check for prior existence."""
    user = User(
        user_id=user_id_for(provider, subject),
        email=email,
        name=name,
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


def set_disabled(data_dir: Path, user_id: str, disabled: bool) -> bool:
    """Enable/disable a user (disabled users fail auth). Returns False for unknown users."""
    with _profile_lock(data_dir, user_id):
        user = get_user(data_dir, user_id)
        if user is None:
            return False
        _write_profile(data_dir, replace(user, disabled=bool(disabled)))
    return True


def set_role(data_dir: Path, user_id: str, role: str) -> bool:
    """Set a user's role (coerced to a known role). Returns False for unknown users."""
    with _profile_lock(data_dir, user_id):
        user = get_user(data_dir, user_id)
        if user is None:
            return False
        _write_profile(data_dir, replace(user, role=app_roles.normalize_role(role)))
    return True


def delete_user(data_dir: Path, user_id: str) -> bool:
    """Remove a user's directory (GDPR hard delete). Returns True if anything was removed."""
    udir = data_dir / "users" / user_id
    if not udir.is_dir():
        return False
    shutil.rmtree(udir, ignore_errors=True)
    return True
