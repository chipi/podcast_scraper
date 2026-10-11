"""People you keep hearing — Discover arc Step 3 (ADR-163; competitive scan #5).

The people who keep turning up in what the listener heard or captured from — as a host or guest in
two or more of those episodes, never someone merely mentioned (mentions are noise) — recommended on
OTHER shows: their episodes on shows the listener neither follows nor heard them on, not yet heard,
captured or saved. The person heard most leads; people take turns, so one prolific guest does not
fill the rail; one card per episode. Each card says who, over a quote that person says in the
recommended episode (``app_reasons``, extractive only). Built from what is on disk: no LLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from podcast_scraper.server.app_captured_rail import person_quotes
from podcast_scraper.server.app_reasons import Reason, ReasonQuote, ReasonSubject
from podcast_scraper.server.app_trending_episodes import TrendingEpisodeCard

#: A person counts as "kept hearing" from this many of the listener's episodes.
MIN_EPISODES = 2
#: People considered, most heard first.
MAX_PEOPLE = 8
#: Speaking roles. ``mentioned`` (and no role) is not hearing someone.
SPEAKING_ROLES = frozenset({"host", "guest"})


@dataclass(frozen=True)
class KeptHearing:
    """One person the listener keeps hearing, and where."""

    person_id: str
    name: str
    episodes: int  # of the listener's episodes they speak in
    shows: frozenset[str]  # the feeds the listener heard them on


def people_kept_hearing(root: Path, data_dir: Path, user_id: str) -> list[KeptHearing]:
    """Speakers in at least ``MIN_EPISODES`` of the listener's episodes, most heard first."""
    from podcast_scraper.identity.bare_name_scope import is_scoped_person_id
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_kg_view import entities_from_kg
    from podcast_scraper.server.app_slugs import resolve_slug
    from podcast_scraper.server.app_user_corpus import experienced_episode_set

    counts: Counter[str] = Counter()
    names: dict[str, str] = {}
    shows: dict[str, set[str]] = {}
    for slug in sorted(experienced_episode_set(root, data_dir, user_id)):
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        persons, _orgs, _topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        for p in persons:
            # An episode-scoped id names someone within one episode only; it cannot recur.
            if p.role not in SPEAKING_ROLES or is_scoped_person_id(p.id):
                continue
            counts[p.id] += 1
            names.setdefault(p.id, p.name)
            shows.setdefault(p.id, set()).add(row.feed_id or "")
    ranked = sorted(
        (pid for pid, n in counts.items() if n >= MIN_EPISODES),
        key=lambda pid: (-counts[pid], names[pid]),
    )
    return [
        KeptHearing(pid, names[pid], counts[pid], frozenset(shows[pid]))
        for pid in ranked[:MAX_PEOPLE]
    ]


def people_rail(
    root: Path, data_dir: Path, user_id: str, *, limit: int = 6
) -> list[TrendingEpisodeCard]:
    """Episodes on other shows with the people the listener keeps hearing (module docstring)."""
    from podcast_scraper.server.app_slugs import resolve_slug
    from podcast_scraper.server.app_user_corpus import (
        experienced_episode_set,
        followed_show_ids,
        saved_episode_set,
    )

    people = people_kept_hearing(root, data_dir, user_id)
    if not people:
        return []
    followed = followed_show_ids(data_dir, user_id)
    excluded = experienced_episode_set(root, data_dir, user_id) | saved_episode_set(
        data_dir, user_id
    )

    per_person: list[list[tuple[ReasonSubject, int, ReasonQuote]]] = []
    for person in people:
        subject = ReasonSubject("person", person.person_id, person.name)
        found = []
        for q in person_quotes(root, person.person_id, person.name):
            row = resolve_slug(root, q.episode_slug)
            feed = (row.feed_id or "") if row is not None else ""
            if row is None or q.episode_slug in excluded:
                continue
            if feed in followed or feed in person.shows:
                continue  # discovery is another show, not more of one they already hear
            found.append((subject, person.episodes, q))
        per_person.append(found)

    cards: list[TrendingEpisodeCard] = []
    taken: set[str] = set()
    depth = 0
    while len(cards) < limit and any(depth < len(f) for f in per_person):
        for found in per_person:
            if depth < len(found) and len(cards) < limit:
                subject, heard_in, q = found[depth]
                if q.episode_slug in taken:
                    continue
                taken.add(q.episode_slug)
                cards.append(
                    TrendingEpisodeCard(
                        slug=q.episode_slug,
                        why=Reason(kind="person", quote=q, subject=subject, events=heard_in),
                    )
                )
        depth += 1
    return cards
