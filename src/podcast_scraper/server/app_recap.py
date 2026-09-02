"""Listening recaps — week, month, year (#1914).

Reads only. Everything here aggregates what ``app_user_state`` already records:

* ``listening_daily`` — seconds actually accrued, bucketed by the LISTENER'S day (Phase 0).
* ``listen_events`` — one line per episode start, carrying the moment it happened.
* ``playback`` — ``finished_at``, so "finished in March" is answerable at all.

Two rules this module exists to enforce, both learned from the issue:

1. **Never fabricate the headline.** ``app_stats`` reports "hours listened" as
   ``sum(position_seconds)`` — a lifetime snapshot of furthest-position-reached that cannot be
   windowed. A recap must not lead with it. Every window here reports ``days_recorded`` and
   ``coverage_from`` alongside the number, so a caller can always tell how much of the window we
   actually have. A recap over a window we only partly recorded is a lie of omission otherwise.
2. **Windows are the LISTENER'S.** Day keys are already local (see ``_day_key``), so a window is
   just a string range over them — no timezone maths here, and no chance of this module and the
   recorder disagreeing about when Tuesday was.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from podcast_scraper.server import app_user_state

Window = Literal["week", "month", "year"]

#: How many days each window covers, counting back from and including `today`.
WINDOW_DAYS: dict[str, int] = {"week": 7, "month": 30, "year": 365}


def _day_range(today: date, days: int) -> list[str]:
    """The window's day keys, oldest first. Inclusive of ``today``."""
    return [(today - timedelta(days=n)).isoformat() for n in range(days - 1, -1, -1)]


def _local_today(now_ts: int, tz_offset_minutes: int) -> date:
    offset = app_user_state.clamp_tz_offset(tz_offset_minutes)
    return datetime.fromtimestamp(int(now_ts) + offset * 60, timezone.utc).date()


def _event_day(rec: dict[str, Any], tz_offset_minutes: int) -> str | None:
    """The listener's day an event happened on; None when its timestamp is unusable.

    Listen events store ISO-8601 or epoch (both are written; ``app_stats`` parses both), so this
    accepts both rather than assuming the newer shape and silently dropping older history.
    """
    raw = rec.get("ts")
    if raw is None:
        return None
    try:
        ts = int(raw)
    except (TypeError, ValueError):
        try:
            ts = int(datetime.fromisoformat(str(raw)).timestamp())
        except ValueError:
            return None
    return app_user_state._day_key(ts, tz_offset_minutes)


def build_recap(
    data_dir: Path,
    user_id: str,
    window: Window,
    now_ts: int,
    tz_offset_minutes: int = 0,
    top_n: int = 5,
) -> dict[str, Any]:
    """One window's recap. Pure aggregation — no corpus reads, no network."""
    days = WINDOW_DAYS[window]
    today = _local_today(now_ts, tz_offset_minutes)
    keys = _day_range(today, days)
    in_window = set(keys)

    listening = app_user_state.get_listening(data_dir, user_id)
    recorded = listening.get("days", {})
    by_day = {k: round(float(recorded.get(k, 0.0)), 1) for k in keys}
    total = round(sum(by_day.values()), 1)

    # Episodes STARTED in the window, most-played first. Counted from the listen log rather than
    # from playback records, because playback holds one row per episode ever — it cannot tell us
    # what happened this month, and a re-listen would be invisible.
    starts: Counter[str] = Counter()
    for rec in app_user_state.list_listen_events(data_dir, user_id):
        day = _event_day(rec, tz_offset_minutes)
        if day and day in in_window and rec.get("slug"):
            starts[str(rec["slug"])] += 1

    finished_at = listening.get("finished_at", {}) or {}
    finished = [
        slug
        for slug, ts in finished_at.items()
        if isinstance(ts, int) and app_user_state._day_key(ts, tz_offset_minutes) in in_window
    ]

    # How much of the window we actually have. Recording started when Phase 0 shipped, so a "year"
    # asked for today covers weeks — saying so is the difference between a recap and a fiction.
    recorded_days = sorted(k for k in recorded if k in in_window)
    first_ever = listening.get("first_listened_at")

    return {
        "window": window,
        "from_day": keys[0],
        "to_day": keys[-1],
        "listening_seconds": total,
        "by_day": by_day,
        "episodes_started": sum(starts.values()),
        "distinct_episodes": len(starts),
        "top_episodes": [
            {"slug": slug, "starts": n} for slug, n in starts.most_common(top_n)
        ],
        "episodes_finished": len(finished),
        # The honesty fields. A caller that ignores these can still render something true; a
        # caller that uses them can say "since you started listening in August" instead of
        # implying a full year.
        "days_recorded": len(recorded_days),
        "days_in_window": days,
        "coverage_from": recorded_days[0] if recorded_days else None,
        "first_listened_at": first_ever,
    }
