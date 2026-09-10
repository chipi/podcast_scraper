"""Shared helpers for the outbound digest assemblers (personal + recommendations).

Both the weekly/daily personal digest (:mod:`app_digest_personal`) and the monthly recommendations
digest (:mod:`app_digest_recommendations`) build :class:`DeliveryEnvelope`-shaped payloads and gate
on the same consent identity rules. These helpers are the shared surface so neither assembler has
to reach into the other's private module internals.
"""

from __future__ import annotations

import datetime as dt

from podcast_scraper.server.app_user_store import User


def iso(ts: int) -> str:
    """A UTC ``YYYY-MM-DDTHH:MM:SSZ`` timestamp for an envelope's time fields."""
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def email_verified(user: User) -> bool:
    """Identity-derived: Google-authenticated emails are verified (mirrors routes/app_comms)."""
    return user.provider == "google" and bool(user.email)
