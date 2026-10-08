"""Shows to suggest in Home's guided start (operator 2026-10-08).

The guided start's "follow a few shows" step showed the first 8 of ``/podcasts``, which is sorted
by feed id: alphabetical, not good. The operator's order instead:

1. **Active** — the show published in the last month. A show that stopped a year ago is a poor
   first follow, however loved.
2. **Loved** — how many listeners follow it, favourited it or its episodes, or listened to it.

Within that, a show carrying the interests the listener just chose in step 1 moves up, shows they
already follow are left out, and no category takes more than its share of the first rail.

Pure over its inputs (``rank_guided_shows``); ``guided_show_signals`` gathers the cross-user
counts from the per-user state files.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow

ACTIVE_DAYS = 30
# One follow is a stronger statement than one listen; a favourite sits between.
FOLLOW_WEIGHT = 3
SHOW_FAVORITE_WEIGHT = 3
EPISODE_FAVORITE_WEIGHT = 1
LISTENER_WEIGHT = 1
# Per episode of the show that carries a just-chosen interest, capped so one prolific show cannot
# outrank every loved one on matches alone.
INTEREST_WEIGHT = 2
INTEREST_CAP = 5
MAX_PER_CATEGORY = 2


@dataclass(frozen=True)
class ShowSignals:
    """Cross-user love for each show, keyed by ``feed_id``."""

    followers: Counter[str]
    show_favorites: Counter[str]
    episode_favorites: Counter[str]
    listeners: Counter[str]


def guided_show_signals(data_dir: Path, rows: Iterable[CatalogEpisodeRow]) -> ShowSignals:
    """Count, per show, the listeners who follow it, favourited it or its episodes, or played it.

    Each listener counts once per show per signal, so one heavy listener cannot make a show loved.
    """
    feed_by_slug = {slug_for_row(r): r.feed_id for r in rows}
    followers: Counter[str] = Counter()
    show_favs: Counter[str] = Counter()
    ep_favs: Counter[str] = Counter()
    listeners: Counter[str] = Counter()
    for uid in app_user_state.iter_user_ids(data_dir):
        followers.update(
            {str(x.get("feed_id") or "") for x in app_user_state.get_library(data_dir, uid)} - {""}
        )
        favs = app_user_state.get_favorites(data_dir, uid)
        show_favs.update({str(f["ref"]) for f in favs if f.get("kind") == "show"})
        ep_favs.update(
            {
                feed_by_slug[str(f["ref"])]
                for f in favs
                if f.get("kind") == "episode" and str(f["ref"]) in feed_by_slug
            }
        )
        listeners.update(
            {
                feed_by_slug[p["slug"]]
                for p in app_user_state.list_playback(data_dir, uid)
                if p["slug"] in feed_by_slug
            }
        )
    return ShowSignals(followers, show_favs, ep_favs, listeners)


def _love(fid: str, s: ShowSignals) -> int:
    return (
        FOLLOW_WEIGHT * s.followers[fid]
        + SHOW_FAVORITE_WEIGHT * s.show_favorites[fid]
        + EPISODE_FAVORITE_WEIGHT * s.episode_favorites[fid]
        + LISTENER_WEIGHT * s.listeners[fid]
    )


def _published(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def rank_guided_shows(
    rows: Iterable[CatalogEpisodeRow],
    *,
    signals: ShowSignals,
    now: datetime,
    followed: set[str],
    matching_relpaths: set[str],
    limit: int,
) -> list[str]:
    """``feed_id``s in the order the guided start should offer them (see the module docstring)."""
    cutoff = now - timedelta(days=ACTIVE_DAYS)
    latest: dict[str, datetime] = {}
    matches: Counter[str] = Counter()
    category: dict[str, str] = {}
    for r in rows:
        if not r.feed_id or r.feed_id in followed:
            continue
        category.setdefault(r.feed_id, (r.feed_category or "").strip().lower())
        when = _published(r.publish_date)
        if (
            when is not None
            and when <= now
            and when > latest.get(r.feed_id, datetime.min.replace(tzinfo=timezone.utc))
        ):
            latest[r.feed_id] = when
        if r.metadata_relative_path in matching_relpaths:
            matches[r.feed_id] += 1

    def key(fid: str) -> tuple[bool, int, datetime]:
        when = latest.get(fid)
        score = _love(fid, signals) + INTEREST_WEIGHT * min(matches[fid], INTEREST_CAP)
        return (
            when is not None and when >= cutoff,
            score,
            when or datetime.min.replace(tzinfo=timezone.utc),
        )

    ordered = sorted(category, key=key, reverse=True)
    # Mix categories: at most MAX_PER_CATEGORY of one in the rail, unless there is nothing else.
    picked: list[str] = []
    held: list[str] = []
    per_cat: defaultdict[str, int] = defaultdict(int)
    for fid in ordered:
        cat = category[fid]
        if cat and per_cat[cat] >= MAX_PER_CATEGORY:
            held.append(fid)
            continue
        per_cat[cat] += 1
        picked.append(fid)
    return (picked + held)[:limit]
