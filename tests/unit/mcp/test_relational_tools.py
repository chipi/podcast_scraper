"""Unit tests for the relational MCP tools (RFC-095 slice 2) — no MCP SDK required.

Uses a hand-built typed graph (mirrors ``tests/integration/server/test_viewer_relational``)
so the tools' traversal + serialization are tested without artifact loading. The hybrid
re-rank degrades to structural order here (no index under tmp_path), which is what we assert.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import relational as rel
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


def _fixture():
    nodes = {
        "person:alice": ("person", {"name": "Alice"}),
        "org:acme": ("org", {"name": "Acme"}),
        "topic:ai": ("topic", {"label": "AI"}),
        "insight:1": ("insight", {"text": "Alice on AI"}),
        "insight:2": ("insight", {"text": "Bob on AI"}),
        "episode:e1": ("episode", {}),
        "podcast:show1": ("podcast", {"name": "Show One"}),
    }
    edges = [
        ("person:alice", "insight:1", "STATES"),
        ("insight:1", "topic:ai", "ABOUT"),
        ("insight:2", "topic:ai", "ABOUT"),
        ("insight:1", "org:acme", "MENTIONS"),
        ("episode:e1", "insight:1", "HAS_INSIGHT"),
        ("podcast:show1", "episode:e1", "HAS_EPISODE"),
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


def test_person_positions(ctx) -> None:
    out = rel.person_positions(ctx, "person:alice")
    assert out["subject"] == "person:alice"
    assert [r["id"] for r in out["results"]] == ["insight:1"]


def test_insights_about_entity(ctx) -> None:
    out = rel.insights_about_entity(ctx, "org:acme")
    assert [r["id"] for r in out["results"]] == ["insight:1"]


def test_related_insights(ctx) -> None:
    out = rel.related_insights(ctx, "insight:1")
    assert [r["id"] for r in out["results"]] == ["insight:2"]


def test_topic_entities(ctx) -> None:
    out = rel.topic_entities(ctx, "topic:ai")
    assert [r["id"] for r in out["results"]] == ["org:acme"]


def test_show_episodes(ctx) -> None:
    out = rel.show_episodes(ctx, "podcast:show1")
    assert [r["id"] for r in out["results"]] == ["episode:e1"]


def test_who_said_about_topic_groups_by_person(ctx) -> None:
    out = rel.who_said_about_topic(ctx, "topic:ai")
    assert set(out["groups"]) == {"person:alice"}
    assert [r["id"] for r in out["groups"]["person:alice"]] == ["insight:1"]


def test_cross_show_synthesis_groups_by_show(ctx) -> None:
    out = rel.cross_show_synthesis(ctx, "topic:ai")
    assert set(out["groups"]) == {"podcast:show1"}
    assert [r["id"] for r in out["groups"]["podcast:show1"]] == ["insight:1"]


def test_node_serialization_shape(ctx) -> None:
    row = rel.person_positions(ctx, "person:alice")["results"][0]
    assert set(row) == {"id", "type", "text", "show_id", "episode_id"}
    assert row["type"] == "insight"
