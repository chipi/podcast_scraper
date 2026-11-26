"""Times as the corpus stores them: UTC, aware, written with ``Z`` (operator, 2026-11-26).

Every timestamp the pipeline writes goes through :func:`utc_iso`; local time is the viewer's job,
at display. A naive value is UTC: ``email.utils.parsedate_to_datetime`` returns one for RFC 5322's
``-0000`` ("UTC, local offset unknown") and for a date with no zone, and the containers' clock is
UTC, so reading it as UTC is right, not a guess (the 2026-11-26 audit: 953 of 958 stored naive
publish dates equal the live feed's ``-0000`` pubDate).

Only what is written is normalized. ``rss.parser`` keeps the parsed value as it was, because
``metadata_generation.generate_episode_id`` hashes ``published_date.isoformat()`` for an episode
with no GUID: converting there would change those ids.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime


def as_utc(value: datetime) -> datetime:
    """``value`` as an aware UTC datetime; a naive one is taken as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def utc_iso(value: datetime) -> str:
    """``value`` as ISO 8601 in UTC with ``Z``: ``2026-11-03T08:00:00Z``."""
    return as_utc(value).isoformat().replace("+00:00", "Z")


def utc_now() -> datetime:
    """The current time, aware, in UTC."""
    return datetime.now(timezone.utc)


def rfc2822_from_iso(stored: str) -> str:
    """A stored ISO 8601 time as an RSS ``pubDate`` that ``parsedate_to_datetime`` reads back to
    the same value: a naive one as ``-0000``, an aware one with its own offset."""
    moment = datetime.fromisoformat(stored.strip().replace("Z", "+00:00"))
    return format_datetime(moment)
