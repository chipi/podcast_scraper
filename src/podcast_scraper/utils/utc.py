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

from datetime import date, datetime, timezone
from email.utils import format_datetime
from typing import Optional


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


def parse_stored(raw: object) -> Optional[datetime]:
    """A stored time with a time of day (ISO 8601, any offset, ``Z``, or naive = UTC) as an aware
    UTC datetime; ``None`` for a date with no time, a missing value or an unreadable one."""
    if isinstance(raw, datetime):
        return as_utc(raw)
    if isinstance(raw, str) and len(raw.strip()) > 10:
        try:
            return as_utc(datetime.fromisoformat(raw.strip().replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def utc_day(raw: object) -> Optional[str]:
    """The UTC day (``YYYY-MM-DD``) of a stored time, or a stored date with no time as it is; the
    form to compare as text against a ``since`` day. Not the day in the value's own offset."""
    moment = parse_stored(raw)
    if moment is not None:
        return moment.date().isoformat()
    if isinstance(raw, str) and len(raw.strip()) >= 10:
        try:
            return date.fromisoformat(raw.strip()[:10]).isoformat()
        except ValueError:
            return None
    return None


def utc_sort_key(raw: object) -> str:
    """A stored time as text that sorts in time order whatever form it was stored in: the UTC
    instant with ``Z`` and a fixed six-digit fraction (``.`` sorts below ``Z``, so a variable width
    would put ``00.5Z`` before ``00Z``), else the stored date, else ``""`` (sorts first)."""
    moment = parse_stored(raw)
    if moment is not None:
        return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return utc_day(raw) or ""


def rfc2822_from_iso(stored: str) -> str:
    """A stored ISO 8601 time as an RSS ``pubDate`` that ``parsedate_to_datetime`` reads back to
    the same value: a naive one as ``-0000``, an aware one with its own offset."""
    moment = datetime.fromisoformat(stored.strip().replace("Z", "+00:00"))
    return format_datetime(moment)
