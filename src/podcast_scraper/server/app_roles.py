"""Platform role vocabulary for the consumer + viewer apps (#1128).

A single identity store spans both apps; each user carries one ``role``:

- ``listener`` — the Learning Player only (the pre-#1128 default; no viewer access).
- ``creator`` — viewer access (KG curation: digest / library / graph / dashboard).
- ``admin`` — everything ``creator`` has, plus ops, configuration, and user management.

Roles are totally ordered ``listener < creator < admin`` so "at least creator" checks are a
simple rank comparison. ``admin`` bootstrap is by email allowlist (``APP_ADMIN_EMAILS``); there is
no self-service role assignment (an admin grants roles via the user-management surface).
"""

from __future__ import annotations

import os

LISTENER = "listener"
CREATOR = "creator"
ADMIN = "admin"

#: Valid roles, lowest-privilege first.
ROLES: tuple[str, ...] = (LISTENER, CREATOR, ADMIN)

DEFAULT_ROLE = LISTENER

_RANK = {role: i for i, role in enumerate(ROLES)}


def is_role(value: str) -> bool:
    """True when ``value`` is one of the known roles."""
    return value in _RANK


def normalize_role(value: str | None, *, default: str = DEFAULT_ROLE) -> str:
    """Coerce an arbitrary string to a known role, falling back to ``default``."""
    candidate = (value or "").strip().lower()
    return candidate if candidate in _RANK else default


def rank(role: str) -> int:
    """Privilege rank of ``role`` (unknown roles rank as the lowest)."""
    return _RANK.get(normalize_role(role), 0)


def at_least(role: str, minimum: str) -> bool:
    """True when ``role`` is at least as privileged as ``minimum``."""
    return rank(role) >= rank(minimum)


def can_use_viewer(role: str) -> bool:
    """True when ``role`` may use the viewer at all (creator or admin)."""
    return at_least(role, CREATOR)


def is_admin(role: str) -> bool:
    """True when ``role`` is admin."""
    return normalize_role(role) == ADMIN


def admin_emails_from_env() -> frozenset[str]:
    """Bootstrap-admin email allowlist from ``APP_ADMIN_EMAILS`` (CSV, lowercased)."""
    raw = os.environ.get("APP_ADMIN_EMAILS", "")
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def resolve_login_role(
    current: str,
    *,
    email: str,
    grant: str | None,
    admin_emails: frozenset[str],
) -> str:
    """Effective role for a user signing in — never *downgrades* an existing role.

    Precedence: the ``APP_ADMIN_EMAILS`` allowlist wins (→ ``admin``); otherwise a ``creator``
    grant (the viewer's login hint) promotes a ``listener`` up to ``creator``. Only ``creator`` is
    ever granted this way — ``admin`` is bootstrap-only, never via a request hint.
    """
    current = normalize_role(current)
    if (email or "").strip().lower() in admin_emails:
        return ADMIN
    if normalize_role(grant, default="") == CREATOR and rank(current) < rank(CREATOR):
        return CREATOR
    return current
