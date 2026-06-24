"""Sign-in access control for the consumer platform (#1064, RFC-098).

Default-deny: an allowlist of permitted emails (and/or domains) gates account creation,
so the platform isn't open to the whole internet once deployed. ``open`` mode disables the
gate (any successful OAuth login is allowed). Configured from env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AccessPolicy:
    """Who may sign in. ``allowlist`` (default) = only listed emails/domains; ``open`` = all."""

    mode: str
    allowed_emails: frozenset[str]
    allowed_domains: frozenset[str]

    def is_allowed(self, email: str) -> bool:
        """True if ``email`` may sign in under this policy."""
        if self.mode == "open":
            return True
        normalized = (email or "").strip().lower()
        if not normalized:
            return False
        if normalized in self.allowed_emails:
            return True
        domain = normalized.rpartition("@")[2]
        return bool(domain) and domain in self.allowed_domains


def _split_csv(raw: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def policy_from_env() -> AccessPolicy:
    """Build the access policy from the ``APP_SIGNUP_MODE`` / ``APP_ALLOWED_*`` env vars."""
    mode = os.environ.get("APP_SIGNUP_MODE", "allowlist").strip().lower() or "allowlist"
    if mode not in ("allowlist", "open"):
        mode = "allowlist"
    return AccessPolicy(
        mode=mode,
        allowed_emails=_split_csv(os.environ.get("APP_ALLOWED_EMAILS", "")),
        allowed_domains=_split_csv(os.environ.get("APP_ALLOWED_DOMAINS", "")),
    )
