"""The rails' shared helpers: lists taking turns, and an episode's people and topics."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.app_rail_turns import take_turns

pytestmark = [pytest.mark.unit]


def test_an_episodes_people_and_topics_come_from_its_kg(tmp_path: Path) -> None:
    import json

    from podcast_scraper.server.app_rail_turns import episode_entity_ids

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
