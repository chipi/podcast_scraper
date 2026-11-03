"""Continue the thread — Discover arc Step 2.3 (ADR-163; competitive scan, "storylines").

A listener who follows a storyline (an interest token ``thc:…``) gets the episodes that carry it
on: episodes about any of the storyline's topics that they have not heard, captured or saved, and
published AFTER their last listen in that storyline (all of them while they have not listened in it
yet). Newest first; storylines take turns, so one busy storyline does not fill the rail; one card
per episode. Each card says which storyline it continues, over a quote from the episode about the
topic (topic perspectives, ``app_reasons`` — extractive only). Built from what is on disk: no LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from podcast_scraper.server.app_captured_rail import topic_quotes
from podcast_scraper.server.app_rail_turns import take_turns
from podcast_scraper.server.app_reasons import Reason, ReasonQuote, ReasonSubject
from podcast_scraper.server.app_trending_episodes import TrendingEpisodeCard

#: Storylines considered, in the order the listener followed them.
MAX_STORYLINES = 8


def followed_storylines(data_dir: Path, user_id: str) -> list[str]:
    """The storylines the listener follows (``thc:`` interest tokens), oldest follow first."""
    from podcast_scraper.server.app_user_state import get_interests

    return [t for t in get_interests(data_dir, user_id) if t.startswith("thc:")][:MAX_STORYLINES]


def storyline_members(root: Path) -> dict[str, tuple[str, list[str]]]:
    """``{storyline id: (label, [member topic ids, sorted])}`` from the storylines artifact."""
    from podcast_scraper.search.storylines import storyline_map_by_topic

    out: dict[str, tuple[str, list[str]]] = {}
    for tid, info in storyline_map_by_topic(root).items():
        sid = str(info.get("storyline_id") or "")
        if sid:
            label = str(info.get("storyline_label") or sid)
            out.setdefault(sid, (label, []))[1].append(tid)
    return {sid: (label, sorted(ts)) for sid, (label, ts) in out.items()}


def last_listen_dates(
    root: Path, data_dir: Path, user_id: str, members: dict[str, tuple[str, list[str]]]
) -> dict[str, str]:
    """``{storyline id: YYYY-MM-DD}`` — the day of the listener's last listen to an episode about
    one of the storyline's topics. A storyline they never listened in is absent."""
    from podcast_scraper.server.app_rail_turns import episode_entity_ids
    from podcast_scraper.server.app_stats import _ts_to_date
    from podcast_scraper.server.app_user_state import list_listen_events
    from podcast_scraper.server.slugs import resolve_slug

    last_by_slug: dict[str, str] = {}
    for ev in list_listen_events(data_dir, user_id):
        when = _ts_to_date(ev.get("ts"))  # a log holds epoch and ISO stamps alike
        if when is None:
            continue
        day = when.isoformat()
        slug = str(ev["slug"])
        if day > last_by_slug.get(slug, ""):
            last_by_slug[slug] = day
    out: dict[str, str] = {}
    for slug, day in last_by_slug.items():
        row = resolve_slug(root, slug)
        if row is None:
            continue
        ids = episode_entity_ids(root, row.kg_relative_path if row.has_kg else None)
        for sid, (_label, topics) in members.items():
            if ids.intersection(topics) and day > out.get(sid, ""):
                out[sid] = day
    return out


def thread_rail(
    root: Path, data_dir: Path, user_id: str, *, limit: int = 6
) -> list[TrendingEpisodeCard]:
    """New episodes in the storylines the listener follows (see the module docstring)."""
    from podcast_scraper.server.app_user_corpus import experienced_episode_set, saved_episode_set
    from podcast_scraper.server.slugs import resolve_slug

    followed = followed_storylines(data_dir, user_id)
    if not followed:
        return []
    members = storyline_members(root)
    followed = [s for s in followed if s in members]
    if not followed:
        return []
    since = last_listen_dates(root, data_dir, user_id, members)
    excluded = experienced_episode_set(root, data_dir, user_id) | saved_episode_set(
        data_dir, user_id
    )

    def published(slug: str) -> Optional[str]:
        row = resolve_slug(root, slug)
        if row is None or not row.publish_date:
            return None
        return row.publish_date[:10]

    per_storyline: list[list[tuple[ReasonSubject, ReasonQuote]]] = []
    for sid in followed:
        label, topics = members[sid]
        subject = ReasonSubject("storyline", sid, label)
        after = since.get(sid)
        found: dict[str, tuple[str, ReasonQuote]] = {}
        for tid in topics:
            for q in topic_quotes(root, tid):
                slug = q.episode_slug
                if slug in excluded or slug in found:
                    continue
                day = published(slug)
                if after is not None and (day is None or day <= after):
                    continue  # not new since they last listened in this storyline
                found[slug] = (day or "", q)
        newest = sorted(found.items(), key=lambda kv: (kv[1][0], kv[0]), reverse=True)
        per_storyline.append([(subject, q) for _slug, (_day, q) in newest])

    picked = take_turns(per_storyline, limit=limit, key=lambda f: f[1].episode_slug)
    return [
        TrendingEpisodeCard(
            slug=q.episode_slug, why=Reason(kind="thread", quote=q, subject=subject)
        )
        for subject, q in picked
    ]
