"""Listening analytics (PRD-043 / RFC-102) computed from per-user files — no DB, no LLM.

Two surfaces:

* :func:`compute_user_stats` — the signed-in user's own listening (Profile panel): episodes/shows
  opened, an estimate of time invested, active-day streak, and a daily opens sparkline.
* :func:`compute_episode_stats` — cross-user reach for one episode (Player corner): how many people
  opened it and a daily opens sparkline, aggregated by scanning every user's listen log.

All series are zero-filled day buckets in UTC so the client can render a sparkline directly. Time is
injected (``now``) so the functions are deterministic and testable without a clock.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.server import app_user_state

#: How many days of history the sparklines cover (inclusive of today).
SERIES_DAYS = 14


def _ts_to_date(ts: Any) -> date | None:
    # Accept the legacy epoch-int ts AND the canonical ISO-8601 string (ADR-119
    # emit_event envelope) — a listen log can hold both old and new events.
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), timezone.utc).date()
    except (TypeError, ValueError, OSError, OverflowError):
        pass
    try:
        return datetime.fromisoformat(str(ts)).astimezone(timezone.utc).date()
    except (TypeError, ValueError):
        return None


def _today(now: int | None) -> date:
    base = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
    return base.date()


def _daily_series(dates: list[date], today: date, days: int = SERIES_DAYS) -> list[dict[str, Any]]:
    """A zero-filled ``[{date, count}]`` series for the ``days`` window ending today (UTC)."""
    window = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
    counts = {d: 0 for d in window}
    for d in dates:
        if d in counts:
            counts[d] += 1
    return [{"date": d.isoformat(), "count": counts[d]} for d in window]


def _day_streak(active: set[date], today: date) -> int:
    """Length of the current consecutive-day run (anchored at today, or yesterday if idle today)."""
    anchor: date | None = None
    if today in active:
        anchor = today
    elif (today - timedelta(days=1)) in active:
        anchor = today - timedelta(days=1)
    if anchor is None:
        return 0
    streak = 0
    cursor = anchor
    while cursor in active:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def compute_user_stats(data_dir: Path, user_id: str, *, now: int | None = None) -> dict[str, Any]:
    """The signed-in user's own listening summary (single scores + daily opens sparkline)."""
    events = app_user_state.list_listen_events(data_dir, user_id)
    playback = app_user_state.list_playback(data_dir, user_id)
    today = _today(now)

    event_dates = [d for d in (_ts_to_date(e.get("ts")) for e in events) if d is not None]
    # Episodes/shows = the union of what they've opened (events) and what they have a position for
    # (playback predates the event log), so the scores are meaningful from day one.
    episode_slugs = {str(e.get("slug")) for e in events} | {str(p.get("slug")) for p in playback}
    show_ids = {str(e.get("feed_id")) for e in events if e.get("feed_id")}
    listening_seconds = sum(float(p.get("position_seconds", 0.0)) for p in playback)

    return {
        "episodes": len(episode_slugs),
        "shows": len(show_ids),
        "listening_seconds": listening_seconds,
        "active_days": len(set(event_dates)),
        "day_streak": _day_streak(set(event_dates), today),
        "daily": _daily_series(event_dates, today),
    }


#: Minimum distinct listeners before a cross-user count is reported at all (#1923).
#:
#: The reach endpoint is deliberately public, on the reasoning that an aggregate count carries no
#: user identity. That holds at scale and FAILS at small N: with one user in the system,
#: ``listeners: 1`` on an episode says that user listened to it, and anyone who can reach the
#: endpoint can walk the catalogue and reconstruct a large part of one person's listening history.
#: Listening history is sensitive, and the current user count is what makes this acute rather than
#: theoretical.
#:
#: Below the floor the count is withheld (``None``) rather than rounded or zeroed: a zero would be
#: a lie, and a rounded number still leaks by changing.
K_ANONYMITY_MIN_LISTENERS = 5


def compute_episode_stats(data_dir: Path, slug: str, *, now: int | None = None) -> dict[str, Any]:
    """Cross-user reach for one episode: distinct listeners, total opens, daily opens sparkline.

    Counts below :data:`K_ANONYMITY_MIN_LISTENERS` are withheld — see that constant.
    """
    today = _today(now)
    listeners = 0
    open_dates: list[date] = []
    total_opens = 0
    # DISTINCT listeners per day, which is what a per-day privacy floor has to be measured in.
    # The first version of this floor compared the day's OPEN count against a listeners threshold —
    # different units — so one person replaying an episode five times in a day published a
    # single-person bucket with its exact count, the precise thing the floor exists to stop
    # (advisor-2 #4).
    listeners_by_day: dict[date, set[str]] = {}
    for uid in app_user_state.iter_user_ids(data_dir):
        user_opens = [
            e for e in app_user_state.list_listen_events(data_dir, uid) if e.get("slug") == slug
        ]
        if not user_opens:
            continue
        listeners += 1
        total_opens += len(user_opens)
        for event in user_opens:
            day = _ts_to_date(event.get("ts"))
            if day is None:
                continue
            open_dates.append(day)
            listeners_by_day.setdefault(day, set()).add(uid)

    if listeners < K_ANONYMITY_MIN_LISTENERS:
        # Withhold the whole shape, not just the headline: `opens` and the daily series are just
        # as re-identifying when they describe one person's week.
        return {"listeners": None, "opens": None, "daily": []}

    # Clearing the episode-level floor is NOT enough for the per-day series: an episode with 40
    # listeners still has days only one person listened on, and a count of 1 on a known date is the
    # same per-event leak one level down. A day is published only when enough DISTINCT people
    # listened on it.
    series = [
        (
            point
            if len(listeners_by_day.get(date.fromisoformat(point["date"]), set()))
            >= K_ANONYMITY_MIN_LISTENERS
            else {**point, "count": 0}
        )
        for point in _daily_series(open_dates, today)
    ]
    # `opens` is the sum over the WHOLE history, and publishing it exact hands back everything the
    # series withheld: one subtraction gives the suppressed total, and polling reconstructs each
    # day. It is reported only when it agrees with what the series already shows.
    published = sum(int(point["count"]) for point in series)
    opens = total_opens if published == total_opens else None
    return {
        "listeners": listeners,
        "opens": opens,
        "daily": series,
    }
