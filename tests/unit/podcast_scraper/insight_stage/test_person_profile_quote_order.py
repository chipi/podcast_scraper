"""A person's quotes come out in the GI file's node order, the same in every process.

The quote ids were collected in a set and iterated, so the order followed Python's per-process
hash seed: the same request listed a person's quotes differently after every restart (found by the
release-1 route comparison, 2026-11-09: /api/persons/{id}/brief differed between two runs of the
same code and was identical with PYTHONHASHSEED=0).
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_scraper.insight_stage import cil_queries


class _ReversedSet(set):
    """A set that iterates in reverse insertion order, standing in for an unlucky hash seed."""

    def __init__(self, items: list[str]) -> None:
        super().__init__(items)
        self._order = list(items)

    def __iter__(self):  # type: ignore[override]
        return iter(reversed(self._order))


def _gi(quote_ids: list[str]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [{"id": "person:a", "type": "Person"}]
    nodes += [{"id": q, "type": "Quote", "properties": {"text": q}} for q in quote_ids]
    edges = [{"type": "SPOKEN_BY", "from": q, "to": "person:a"} for q in quote_ids]
    return {"nodes": nodes, "edges": edges}


def test_quotes_follow_the_file_order_whatever_the_set_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = [f"quote:{i:02d}" for i in range(12)]
    monkeypatch.setattr(
        cil_queries, "_quote_ids_spoken_by_person", lambda gi, ids: _ReversedSet(order)
    )
    all_quotes: list[dict[str, Any]] = []
    cil_queries._person_profile_append_for_episode(
        {"episode_id": "ep-1"}, _gi(order), {"person:a"}, {}, all_quotes
    )
    assert [q["quote"]["id"] for q in all_quotes] == order


def test_a_repeated_node_id_is_listed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    gi = _gi(["quote:a", "quote:b"])
    gi["nodes"].append({"id": "quote:a", "type": "Quote", "properties": {"text": "again"}})
    all_quotes: list[dict[str, Any]] = []
    cil_queries._person_profile_append_for_episode(
        {"episode_id": "ep-1"}, gi, {"person:a"}, {}, all_quotes
    )
    assert [q["quote"]["id"] for q in all_quotes] == ["quote:a", "quote:b"]
    assert all_quotes[0]["quote"]["properties"]["text"] == "quote:a"  # the first node, as before
