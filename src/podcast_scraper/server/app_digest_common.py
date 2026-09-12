"""Shared helpers for the outbound digest assemblers (personal + recommendations).

Both the weekly/daily personal digest (:mod:`app_digest_personal`) and the monthly recommendations
digest (:mod:`app_digest_recommendations`) build :class:`DeliveryEnvelope`-shaped payloads and gate
on the same consent identity rules. These helpers are the shared surface so neither assembler has
to reach into the other's private module internals.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from podcast_scraper.server.app_user_store import User


def iso(ts: int) -> str:
    """A UTC ``YYYY-MM-DDTHH:MM:SSZ`` timestamp for an envelope's time fields."""
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def email_verified(user: User) -> bool:
    """Identity-derived: Google-authenticated emails are verified (mirrors routes/app_comms)."""
    return user.provider == "google" and bool(user.email)


def resolve_tz(tz_name: str | None) -> dt.tzinfo:
    """The user's IANA timezone as a ``tzinfo`` (DST-safe), or UTC when unset/invalid (#2041).

    A bad or empty name falls back to UTC — the prior behavior — so a garbage tz can never crash a
    digest run or send at the wrong-and-unexplained time; it just reverts to server time.
    """
    if not tz_name:
        return dt.timezone.utc
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return dt.timezone.utc


def local_now(now_ts: int, tz_name: str | None) -> dt.datetime:
    """``now_ts`` (epoch) as an aware datetime in the user's timezone (UTC fallback)."""
    return dt.datetime.fromtimestamp(int(now_ts), dt.timezone.utc).astimezone(resolve_tz(tz_name))


def local_day(ts: int, tz_name: str | None) -> str:
    """The user's LOCAL calendar day (YYYY-MM-DD) for an epoch (UTC fallback)."""
    return local_now(int(ts), tz_name).date().isoformat()
