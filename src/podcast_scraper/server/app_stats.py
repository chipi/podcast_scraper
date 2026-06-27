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
    try:
        return datetime.fromtimestamp(int(ts), timezone.utc).date()
    except (TypeError, ValueError, OSError, OverflowError):
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


def compute_episode_stats(data_dir: Path, slug: str, *, now: int | None = None) -> dict[str, Any]:
    """Cross-user reach for one episode: distinct listeners, total opens, daily opens sparkline."""
    today = _today(now)
    listeners = 0
    open_dates: list[date] = []
    total_opens = 0
    for uid in app_user_state.iter_user_ids(data_dir):
        user_opens = [
            e for e in app_user_state.list_listen_events(data_dir, uid) if e.get("slug") == slug
        ]
        if not user_opens:
            continue
        listeners += 1
        total_opens += len(user_opens)
        open_dates.extend(
            d for d in (_ts_to_date(e.get("ts")) for e in user_opens) if d is not None
        )

    return {
        "listeners": listeners,
        "opens": total_opens,
        "daily": _daily_series(open_dates, today),
    }
