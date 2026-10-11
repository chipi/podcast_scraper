"""The serendipity share (ADR-163 §7): one definition of "outside the listener's habits", and about
one item in five from outside, always labelled."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from podcast_scraper.server.app_reasons import Reason
from podcast_scraper.server.app_serendipity import (
    Habits,
    interleave_outside,
    is_outside_habits,
    OUTSIDE_EVERY,
    take_turns,
)

pytestmark = [pytest.mark.unit]

HABITS = Habits(shows=frozenset({"feed:mine"}), entities=frozenset({"topic:rates", "person:ann"}))


def test_outside_means_not_their_show_and_none_of_its_people_or_topics_theirs() -> None:
    assert is_outside_habits("feed:other", {"topic:whales"}, HABITS) is True
    assert is_outside_habits("feed:mine", {"topic:whales"}, HABITS) is False  # their show
    assert is_outside_habits("feed:other", {"topic:whales", "person:ann"}, HABITS) is False
    assert is_outside_habits(None, set(), HABITS) is True  # nothing ties it to them


def test_under_everyone_nothing_is_outside() -> None:
    assert is_outside_habits("feed:other", {"topic:whales"}, None) is False


def _mark(r: Reason) -> Reason:
    return replace(r, outside_habits=True)


def test_every_fifth_slot_is_from_outside_and_labelled() -> None:
    on = [Reason(kind="rising_topic", events=i) for i in range(10)]
    out = [Reason(kind="rising_topic", events=100 + i) for i in range(3)]
    rail = interleave_outside(on, out, limit=10, mark=_mark)
    assert OUTSIDE_EVERY == 5
    assert [r.outside_habits for r in rail] == [False] * 4 + [True] + [False] * 4 + [True]
    assert [r.events for r in rail if r.outside_habits] == [100, 101]
    assert all(not r.outside_habits for r in on + out)  # the inputs are not mutated


def test_each_list_fills_the_others_gaps() -> None:
    few_on = [Reason(kind="thread", events=1)]
    many_out = [Reason(kind="thread", events=100 + i) for i in range(4)]
    rail = interleave_outside(few_on, many_out, limit=5, mark=_mark)
    assert [r.events for r in rail] == [1, 100, 101, 102, 103]
    assert [r.outside_habits for r in rail] == [False, True, True, True, True]
    assert interleave_outside([], [], limit=5, mark=_mark) == []
    assert len(interleave_outside(few_on * 9, many_out, limit=6, mark=_mark)) == 6


def test_an_episodes_people_and_topics_come_from_its_kg(tmp_path: Path) -> None:
    import json

    from podcast_scraper.server.app_serendipity import episode_entity_ids

    kg = {
        "nodes": [
            {
                "id": "topic:interest-rates",
                "type": "Topic",
                "properties": {"label": "Interest rates"},
            },
            {"id": "person:ann-lee", "type": "Person", "properties": {"name": "Ann Lee"}},
            {"id": "org:fed", "type": "Organization", "properties": {"name": "The Fed"}},
        ],
        "edges": [],
    }
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "ep.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    assert episode_entity_ids(tmp_path, "metadata/ep.kg.json") == {
        "topic:interest-rates",
        "person:ann-lee",
    }
    assert episode_entity_ids(tmp_path, None) == set()


def test_take_turns_passes_over_an_item_another_list_took_not_the_turn() -> None:
    """Lists sharing episodes still fill the rail: on its turn a list gives its next free item."""
    a = ["ep-1", "ep-2", "ep-3"]
    b = ["ep-2", "ep-1", "ep-4"]
    assert take_turns([a, b], limit=10, key=lambda s: s) == ["ep-1", "ep-2", "ep-3", "ep-4"]


def test_take_turns_caps_each_list_and_the_rail() -> None:
    a, b = ["a1", "a2", "a3"], ["b1"]
    assert take_turns([a, b], limit=10, key=lambda s: s, per_list=2) == ["a1", "b1", "a2"]
    assert take_turns([a, b], limit=2, key=lambda s: s) == ["a1", "b1"]
    assert take_turns([], limit=3, key=lambda s: s) == []
