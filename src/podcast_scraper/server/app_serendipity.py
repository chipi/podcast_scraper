"""The serendipity share every Discover rail uses (ADR-163 §7; Discover arc Step 1.4).

About one item in five on a rail comes from OUTSIDE the listener's habits, and it is labelled
(``Reason.outside_habits``, rendered by the app as "Outside your usual"). Breadth is part of every
rail, not a separate one people learn to skip.

**One definition of "outside"** (ADR-162's world): an episode is outside the listener's habits when
its show is not in their shows (:func:`personal_show_ids`) AND none of its people or topics is in
their world (:func:`personal_entity_ids`). Under the Everyone scope there are no habits — nothing
is labelled outside; callers pass ``habits=None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

#: One item in this many on a rail is from outside the listener's habits (ADR-163 §7).
OUTSIDE_EVERY = 5

T = TypeVar("T")


@dataclass(frozen=True)
class Habits:
    """The listener's usual: their shows (feed ids) and the entities of their world."""

    shows: frozenset[str]
    entities: frozenset[str]


def listener_habits(root: Path, data_dir: Path, user_id: str) -> Habits:
    """The listener's habits, from the one definition of their world (ADR-162)."""
    from podcast_scraper.server.app_user_corpus import personal_entity_ids, personal_show_ids

    return Habits(
        shows=frozenset(personal_show_ids(root, data_dir, user_id)),
        entities=frozenset(personal_entity_ids(root, data_dir, user_id)),
    )


def episode_entity_ids(root: Path, kg_relative_path: Optional[str]) -> set[str]:
    """The person and topic ids an episode's KG names (empty without a KG)."""
    if not kg_relative_path:
        return set()
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_kg_view import entities_from_kg

    persons, _orgs, topics = entities_from_kg(load_json_artifact(root, kg_relative_path))
    return {e.id for e in [*persons, *topics] if getattr(e, "id", None)}


def is_outside_habits(
    feed_id: Optional[str], entity_ids: set[str], habits: Optional[Habits]
) -> bool:
    """Outside the listener's habits: not their show, and none of its people or topics theirs.

    Always ``False`` under the Everyone scope (``habits=None``) — there is no "usual" to leave.
    """
    if habits is None:
        return False
    if feed_id and feed_id in habits.shows:
        return False
    return not (entity_ids & habits.entities)


def interleave_outside(
    on_habit: Sequence[T],
    outside: Sequence[T],
    *,
    limit: int,
    mark: Callable[[T], T],
    every: int = OUTSIDE_EVERY,
) -> list[T]:
    """A rail of at most ``limit`` items where every ``every``-th slot (5th, 10th, …) is taken from
    ``outside`` while it lasts; otherwise each list fills the other's gaps, in its own order.

    Every item drawn from ``outside`` passes through ``mark`` (e.g. setting its reason's
    ``outside_habits``), so an outside item can never reach the rail unlabelled.
    """
    out: list[T] = []
    on_i = out_i = 0
    while len(out) < limit and (on_i < len(on_habit) or out_i < len(outside)):
        slot_is_outside = (len(out) + 1) % every == 0
        if (slot_is_outside and out_i < len(outside)) or on_i >= len(on_habit):
            out.append(mark(outside[out_i]))
            out_i += 1
        else:
            out.append(on_habit[on_i])
            on_i += 1
    return out


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
