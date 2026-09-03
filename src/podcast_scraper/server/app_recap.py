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

from podcast_scraper.server import app_corpus_strength, app_user_corpus, app_user_state
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_slugs import slug_for_row

Window = Literal["week", "month", "year", "ytd"]

#: How many days each FIXED-length window covers, counting back from and including `today`.
WINDOW_DAYS: dict[str, int] = {"week": 7, "month": 30, "year": 365}


def window_length(window: Window, today: date) -> int:
    """Days in a window, counting back from and including ``today``.

    ``ytd`` is the odd one and the reason this is a function: "your 2026 so far" is 1 January to
    today, which is a different length every day and on 1 January is a single day. The other
    windows are rolling — "the last 7 days", not "this calendar week" — because a Monday recap of
    a calendar week would be one day long, which is nobody's idea of a week.
    """
    if window == "ytd":
        return (today - date(today.year, 1, 1)).days + 1
    return WINDOW_DAYS[window]


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


def _exposure_counts(
    data_dir: Path, user_id: str, in_window: set[str], tz: int
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], str]]:
    """Per-episode topic/person counts from the RECORDED exposure log (#1923).

    Preferred over re-deriving from the corpus because it is what was true at the time: a
    re-enrichment can change an episode's KG, and deriving would silently rewrite the listener's
    history. Counted per EPISODE, so a topic that ran through five listens outranks one mentioned
    forty times in a single episode.
    """
    counts: Counter[tuple[str, str]] = Counter()
    labels: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str, str]] = set()
    for rec in app_user_state.list_topic_exposure(data_dir, user_id):
        day = _event_day(rec, tz)
        if not day or day not in in_window:
            continue
        kind, ent_id, slug = str(rec.get("kind")), str(rec.get("id")), str(rec.get("slug"))
        # One exposure per (episode, entity) even if the log holds duplicates from a re-listen:
        # the count is "episodes it appeared in", not "times it was written".
        marker = (kind, ent_id, slug)
        if marker in seen:
            continue
        seen.add(marker)
        counts[(kind, ent_id)] += 1
        labels.setdefault((kind, ent_id), str(rec.get("label") or ent_id))
    return counts, labels


def _themes_for(root: Path, slugs: set[str], top_n: int) -> dict[str, list[dict[str, Any]]]:
    """Topics and people across the episodes heard in the window, most-recurring first.

    #1914 asks for the taste signal "with the 90-day decay disabled for a recap window". Rather
    than add a mode to ``derived_interest_counts`` — one function that exists precisely because it
    used to be three that disagreed — the WINDOW does the job decay does: it already restricts to
    the period being recapped, so there is nothing left to discount. Counting inside it keeps that
    function single-purpose and this one honest about what it measured.

    Counted per EPISODE, so a topic mentioned forty times in one episode does not outrank one that
    ran through five of them: "what kept coming up" is about recurrence across listens.
    """
    if not slugs:
        return {"topics": [], "people": []}
    rows = {slug_for_row(r): r for r in cached_catalog(root)}
    counts: Counter[tuple[str, str]] = Counter()
    labels: dict[tuple[str, str], str] = {}
    for slug in slugs:
        row = rows.get(slug)
        if row is None:
            continue
        seen: set[tuple[str, str]] = set()
        for kind, ent_id, label in app_user_corpus._episode_entities(root, row):
            key = (kind, ent_id)
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            labels.setdefault(key, label or ent_id)

    return _ranked(counts, labels, top_n, Counter())


def _ranked(
    counts: Counter[tuple[str, str]],
    labels: dict[tuple[str, str], str],
    top_n: int,
    previous: Counter[tuple[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Top topics and people, each carrying its change against the previous window.

    `delta` is what turns a flat list into a story: the same three labels every week say nothing,
    "systems thinking, up two" says what changed. `is_new` is separate from a positive delta on
    purpose — arriving from nothing reads differently from growing, and a UI wants to say "new"
    rather than "+3".
    """

    def top(kind: str) -> list[dict[str, Any]]:
        ranked = sorted(
            ((k, n) for k, n in counts.items() if k[0] == kind),
            # Count descending, then id ascending: deterministic, so a tie does not reorder
            # between reads and a recap looks the same each time it is opened.
            key=lambda kv: (-kv[1], kv[0][1]),
        )[:top_n]
        return [
            {
                "token": app_user_corpus.interest_token(k[0], k[1]),
                "label": labels[k],
                "episodes": n,
                "delta": n - previous.get(k, 0),
                "is_new": previous.get(k, 0) == 0,
            }
            for k, n in ranked
        ]

    return {"topics": top("topic"), "people": top("person")}


def _best_line(data_dir: Path, user_id: str, in_window: set[str], tz: int) -> dict[str, Any] | None:
    """The longest verbatim line the listener saved in the window, or None.

    The one part of a recap that is an ARTIFACT rather than a statistic — highlights persist
    ``quote_text``, so this is something they actually chose to keep. Longest is a deliberate
    proxy for "most substantial": we have no quality signal, and picking the newest would make the
    headline depend on when the recap happens to be opened.
    """
    best: dict[str, Any] | None = None
    for h in app_user_state.get_highlights(data_dir, user_id):
        quote = (h.get("quote_text") or "").strip()
        created = h.get("created_at")
        if not quote or not isinstance(created, int):
            continue
        if app_user_state._day_key(created, tz) not in in_window:
            continue
        if best is None or len(quote) > len(best["quote_text"]):
            best = {
                "quote_text": quote,
                "episode_slug": h.get("episode_slug"),
                "start_ms": h.get("start_ms"),
                "created_at": created,
            }
    return best


def build_recap(
    data_dir: Path,
    user_id: str,
    window: Window,
    now_ts: int,
    tz_offset_minutes: int = 0,
    top_n: int = 5,
    root: Path | None = None,
) -> dict[str, Any]:
    """One window's recap.

    ``root`` is OPTIONAL: without a corpus the totals, the day series and the saved line still
    render, and only the themes and the strength ranking are empty. A recap must not 500 because
    the corpus is briefly unavailable — the same rule capture already follows.
    """
    today = _local_today(now_ts, tz_offset_minutes)
    days = window_length(window, today)
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

    # The interesting half: what this app can say that a play-count cannot (#1914).
    heard = set(starts)
    themes: dict[str, list[dict[str, Any]]] = {"topics": [], "people": []}
    top_by_strength: list[dict[str, Any]] = []

    # The RECORDED exposure is the preferred source (#1923) — it is what was true at the time, and
    # it needs no corpus. The previous window of the same length gives every theme its trend.
    previous_keys = set(_day_range(today - timedelta(days=days), days))
    counts, labels = _exposure_counts(data_dir, user_id, in_window, tz_offset_minutes)
    if counts:
        prior, _ = _exposure_counts(data_dir, user_id, previous_keys, tz_offset_minutes)
        themes = _ranked(counts, labels, top_n, prior)

    if root is not None:
        try:
            # Fall back to deriving from the corpus for history recorded BEFORE the exposure log
            # existed. No trend in that case: the previous window has nothing to compare against,
            # and inventing one would be worse than omitting it.
            if not counts:
                themes = _themes_for(root, heard, top_n)
            # Ranked by STRENGTH — heard-fraction, captures, favourites, relistens (RFC-114) —
            # rather than by starts. "The episode you kept coming back to" is a better sentence
            # than "the episode you pressed play on most", and it is the one this corpus can say.
            strengths = app_corpus_strength.episode_strengths(root, data_dir, user_id)
            top_by_strength = [
                {"slug": slug, "strength": round(strengths[slug], 4)}
                for slug in sorted(
                    (s for s in heard if s in strengths),
                    key=lambda s: (-strengths[s], s),
                )[:top_n]
            ]
        except Exception:  # noqa: BLE001 — a missing/partial corpus must not fail the recap.
            themes = {"topics": [], "people": []}
            top_by_strength = []

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
        "topics": themes["topics"],
        "people": themes["people"],
        "top_by_strength": top_by_strength,
        "best_line": _best_line(data_dir, user_id, in_window, tz_offset_minutes),
        "days_recorded": len(recorded_days),
        "days_in_window": days,
        "coverage_from": recorded_days[0] if recorded_days else None,
        "first_listened_at": first_ever,
    }
