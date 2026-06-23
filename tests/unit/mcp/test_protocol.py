"""MCP protocol round-trip tests (RFC-095) — invoke tools through FastMCP, not directly.

Complements the per-tool unit tests: this layer proves the *wiring* — input-schema
generation, dispatch, and result serialization — by calling ``server.call_tool`` (the
same path an agent client drives). Heavy deps (resolver / search / graph) are mocked, so
no real corpus is needed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from podcast_scraper.mcp.server import build_server

pytestmark = pytest.mark.unit


def _call(server, name: str, arguments: dict) -> dict:
    """Invoke a tool through the MCP protocol and parse its JSON result."""
    blocks = asyncio.run(server.call_tool(name, arguments))
    # FastMCP serializes a dict result into a TextContent JSON block.
    text = next(b.text for b in blocks if getattr(b, "type", None) == "text")
    loaded = json.loads(text)
    assert isinstance(loaded, dict)
    return loaded


def test_resolve_entity_roundtrip(tmp_path, monkeypatch) -> None:
    class _Detail:
        id = "person:jane"
        score = 0.9
        method = "alias"

    class _Resolver:
        registry = type(
            "R", (), {"records": {"person:jane": {"type": "person", "display_name": "Jane"}}}
        )()

        def resolve_detail(self, text):
            return _Detail()

    monkeypatch.setattr(
        "podcast_scraper.identity.resolver.get_entity_resolver",
        lambda corpus_dir: _Resolver(),
    )
    out = _call(build_server(tmp_path), "resolve_entity", {"name": "jane"})
    # #1054 uniform envelope: every tool returns {ok, data, note}.
    assert out["ok"] is True
    assert out["data"]["candidates"][0]["id"] == "person:jane"
    assert out["data"]["candidates"][0]["kind"] == "person"


def test_search_corpus_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "podcast_scraper.search.capability.structured_corpus_search",
        lambda root, query, **kw: {
            "query_type": "entity_lookup",
            "results": [
                {
                    "doc_id": "i1",
                    "source_tier": "insight",
                    "score": 0.9,
                    "text": "x",
                    "metadata": {},
                }
            ],
            "error": None,
            "lift_stats": None,
        },
    )
    out = _call(build_server(tmp_path), "search_corpus", {"query": "Jane Doe", "tier": "insight"})
    assert out["ok"] is True
    assert out["data"]["query_type"] == "entity_lookup"
    assert out["data"]["results"][0]["source_tier"] == "insight"


def test_relational_tool_roundtrip(tmp_path, monkeypatch) -> None:
    from podcast_scraper.search.corpus_graph import Node

    class _Graph:
        _nodes = {
            "person:p": Node(id="person:p", type="person", payload={"name": "P"}),
            "insight:1": Node(id="insight:1", type="insight", payload={"text": "pos"}),
        }
        _typed = {"person:p": [("insight:1", "STATES")], "insight:1": [("person:p", "STATES")]}

        def get_node(self, nid):
            return self._nodes.get(nid) if nid else None

        def typed_neighbors(self, nid, etype):
            return sorted({n for n, e in self._typed.get(nid, ()) if e == etype})

    monkeypatch.setattr(
        "podcast_scraper.search.corpus_graph.get_corpus_graph",
        lambda *a, **k: _Graph(),
    )
    out = _call(build_server(tmp_path), "person_positions", {"person_id": "person:p"})
    assert out["ok"] is True
    assert [r["id"] for r in out["data"]["results"]] == ["insight:1"]


def test_unknown_tool_raises(tmp_path) -> None:
    with pytest.raises(Exception):
        asyncio.run(build_server(tmp_path).call_tool("no_such_tool", {}))


def test_uniform_envelope_and_error_path(tmp_path, monkeypatch) -> None:
    # every tool returns {ok, data, note}; a tool that raises becomes ok=False (not a crash).
    def _boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("podcast_scraper.search.corpus_graph.get_corpus_graph", _boom)
    out = _call(build_server(tmp_path), "person_positions", {"person_id": "person:x"})
    assert out["ok"] is False
    assert "kaboom" in out["note"]
    assert out["data"] == {}
