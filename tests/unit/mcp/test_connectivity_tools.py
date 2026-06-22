"""Unit tests for the connectivity MCP tools (#1054) — no MCP SDK / artifact loading.

A hand-built typed graph (alice + bob both speak ABOUT topic:ai) exercises the one-call
``entity_neighborhood`` plus ``person_topics`` / ``co_occurring_entities`` and the uniform
``{ok, kind, subject, data, note}`` envelope.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import connectivity as conn
from podcast_scraper.search.corpus_graph import Node

pytestmark = pytest.mark.unit


class _FakeGraph:
    def __init__(self, nodes, edges) -> None:
        self._nodes = {nid: Node(id=nid, type=t, payload=p) for nid, (t, p) in nodes.items()}
        self._typed: Dict[str, List[Tuple[str, str]]] = {}
        for a, b, et in edges:
            self._typed.setdefault(a, []).append((b, et))
            self._typed.setdefault(b, []).append((a, et))

    def get_node(self, node_id: Optional[str]) -> Optional[Node]:
        return self._nodes.get(node_id) if node_id else None

    def typed_neighbors(self, node_id: str, edge_type: str) -> List[str]:
        return sorted({n for n, e in self._typed.get(node_id, ()) if e == edge_type})


def _fixture() -> _FakeGraph:
    nodes = {
        "person:alice": ("person", {"name": "Alice"}),
        "person:bob": ("person", {"name": "Bob"}),
        "org:acme": ("org", {"name": "Acme"}),
        "topic:ai": ("topic", {"label": "AI"}),
        "insight:1": ("insight", {"text": "Alice on AI", "show_id": "show1"}),
        "insight:2": ("insight", {"text": "Bob on AI", "show_id": "show2"}),
        "episode:e1": ("episode", {}),
        "episode:e2": ("episode", {}),
        "podcast:show1": ("podcast", {"name": "Show One"}),
        "podcast:show2": ("podcast", {"name": "Show Two"}),
    }
    edges = [
        ("person:alice", "insight:1", "STATES"),
        ("person:bob", "insight:2", "STATES"),
        ("insight:1", "topic:ai", "ABOUT"),
        ("insight:2", "topic:ai", "ABOUT"),
        ("insight:1", "org:acme", "MENTIONS"),
        ("episode:e1", "insight:1", "HAS_INSIGHT"),
        ("episode:e2", "insight:2", "HAS_INSIGHT"),
        ("podcast:show1", "episode:e1", "HAS_EPISODE"),
        ("podcast:show2", "episode:e2", "HAS_EPISODE"),
    ]
    return _FakeGraph(nodes, edges)


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    graph = _fixture()
    monkeypatch.setattr(
        "podcast_scraper.search.corpus_graph.get_corpus_graph",
        lambda *a, **k: graph,
    )
    return CorpusContext.from_path(tmp_path)


# ── entity_neighborhood ─────────────────────────────────────────────────────


def test_neighborhood_person_one_call_connectivity(ctx) -> None:
    out = conn.entity_neighborhood(ctx, "person:alice")
    assert out["ok"] is True
    assert out["kind"] == "person"
    assert out["subject"]["label"] == "Alice"
    d = out["data"]
    assert [n["id"] for n in d["stated"]] == ["insight:1"]
    # the dead-end fix: person -> topics in the SAME call.
    assert [t["id"] for t in d["topics"]] == ["topic:ai"]
    # person -> co-people (shared topic).
    assert [p["id"] for p in d["co_speakers"]] == ["person:bob"]


def test_neighborhood_topic_facets(ctx) -> None:
    out = conn.entity_neighborhood(ctx, "topic:ai")
    assert out["ok"] is True and out["kind"] == "topic"
    assert set(out["data"]["speakers"]) == {"person:alice", "person:bob"}
    assert "org:acme" in [e["id"] for e in out["data"]["entities"]]


def test_neighborhood_missing_entity_is_not_ok_with_note(ctx) -> None:
    out = conn.entity_neighborhood(ctx, "person:nobody")
    assert out["ok"] is False
    assert "resolve_entity" in out["note"]


def test_neighborhood_org(ctx) -> None:
    out = conn.entity_neighborhood(ctx, "org:acme")
    assert out["ok"] is True and out["kind"] == "org"
    assert [n["id"] for n in out["data"]["mentioned_in"]] == ["insight:1"]


# ── person_topics / co_occurring_entities ───────────────────────────────────


def test_person_topics(ctx) -> None:
    out = conn.person_topics(ctx, "person:alice")
    assert out["ok"] is True
    assert [t["id"] for t in out["data"]["topics"]] == ["topic:ai"]


def test_person_topics_empty_has_explaining_note(ctx) -> None:
    out = conn.person_topics(ctx, "person:nobody")
    assert out["data"]["topics"] == []
    assert out["note"]  # explains WHY it's empty


def test_co_occurring_entities_person(ctx) -> None:
    out = conn.co_occurring_entities(ctx, "person:alice")
    assert [p["id"] for p in out["data"]["co_occurring"]] == ["person:bob"]


def test_co_occurring_entities_rejects_non_person(ctx) -> None:
    out = conn.co_occurring_entities(ctx, "topic:ai")
    assert out["ok"] is False
    assert "person:" in out["note"]
