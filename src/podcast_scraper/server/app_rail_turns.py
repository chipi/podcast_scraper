"""Helpers the Discover rails share: lists taking turns, and an episode's people and topics.

Under Mine every card on Discover is the listener's own (operator 2026-10-11): there is no share of
Everyone's cards mixed in. Breadth is the Everyone scope, one tap away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


def episode_entity_ids(root: Path, kg_relative_path: Optional[str]) -> set[str]:
    """The person and topic ids an episode's KG names (empty without a KG)."""
    if not kg_relative_path:
        return set()
    from podcast_scraper.server.corpus_access import load_json_artifact
    from podcast_scraper.server.kg_view import entities_from_kg

    persons, _orgs, topics = entities_from_kg(load_json_artifact(root, kg_relative_path))
    return {e.id for e in [*persons, *topics] if getattr(e, "id", None)}


def take_turns(
    lists: Sequence[Sequence[T]],
    *,
    limit: int,
    key: Callable[[T], str],
    per_list: Optional[int] = None,
) -> list[T]:
    """At most ``limit`` items, the lists taking turns in order, one item per ``key`` (an episode).

    On its turn a list gives its next item not already taken — an item another list took is passed
    over, not the turn — so overlapping lists still fill the rail. ``per_list`` caps what any one
    list gives.
    """
    out: list[T] = []
    taken: set[str] = set()
    pos = [0] * len(lists)
    given = [0] * len(lists)
    progressed = True
    while len(out) < limit and progressed:
        progressed = False
        for i, items in enumerate(lists):
            if len(out) >= limit or (per_list is not None and given[i] >= per_list):
                continue
            while pos[i] < len(items) and key(items[pos[i]]) in taken:
                pos[i] += 1
            if pos[i] < len(items):
                item = items[pos[i]]
                pos[i] += 1
                taken.add(key(item))
                out.append(item)
                given[i] += 1
                progressed = True
    return out
