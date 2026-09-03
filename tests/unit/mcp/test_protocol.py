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


# Every cross-surface-refresh tool, invoked through the FastMCP protocol against an empty
# corpus: proves the registration wrapper dispatches and returns the uniform envelope (never
# crashes the harness), even when the corpus has no index / graph / artifacts.
_REFRESH_TOOL_CALLS = [
    ("corpus_trending", {}),
    ("insight_detail", {"insight_id": "insight:x"}),
    ("topic_conversation_arc", {"topic_id": "topic:x"}),
    ("topic_perspective_leaders", {}),
    ("explore_insights", {}),
    ("episode_insights", {"metadata_path": "nope.metadata.json"}),
    ("compare_subjects", {"subject_a": "topic:a", "subject_b": "topic:b"}),
    ("cluster_search", {"query": "x"}),
    ("consensus_search", {"query": "x"}),
    ("corpus_enrichment_signals", {}),
    ("episode_enrichment_signals", {"metadata_path": "nope.metadata.json"}),
    ("episode_speaker_roster", {"metadata_path": "nope.metadata.json"}),
    ("ego_network", {"entity_id": "person:x"}),
    ("topic_clusters", {"topic_id": "topic:x"}),
    ("entity_dossier", {"entity_id": "topic:x"}),
    ("episode_digest", {"metadata_path": "nope.metadata.json"}),
]


@pytest.mark.parametrize("name,args", _REFRESH_TOOL_CALLS)
def test_refresh_tool_dispatches(tmp_path, name, args) -> None:
    out = _call(build_server(tmp_path), name, args)
    # Uniform envelope always; ok may be True (empty result) or False (clean error) — the
    # point is the wrapper dispatched the tool without crashing the protocol layer.
    assert isinstance(out.get("ok"), bool)


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
    # Only the exception CLASS is surfaced to the MCP client — the message
    # (which can carry host paths / corpus layout) is not leaked (review low/mcp-leak).
    assert out["note"] == "RuntimeError"
    assert "kaboom" not in out["note"]
    assert out["data"] == {}


# --- corpus WRITES require the write scope (#1916) -----------------------------


@pytest.mark.parametrize("tool", ["reenrich", "reindex"])
def test_a_corpus_write_is_refused_with_a_read_only_token(tmp_path, tool: str) -> None:
    """The defect this closes: `reenrich` and `reindex` mutate the corpus and were gated by
    nothing but the `mcp_access` entitlement, so any entitled user's agent could trigger a
    corpus-wide reindex with a READ-ONLY token. The scope was on the wire the whole time.

    The refusal arrives as `ok: false` rather than as a raise: `_safe` turns every tool error into
    the uniform envelope, and deliberately sends only the exception CLASS — a full message can
    carry absolute host paths. `McpScopeError` still tells an agent to present a different token
    rather than to retry.
    """
    from podcast_scraper.mcp import auth

    token = auth.current_mcp_scopes.set(frozenset({"mcp:read"}))
    try:
        out = _call(build_server(tmp_path), tool, {})
        assert out["ok"] is False
        assert out["note"] == "McpScopeError"
        assert out["data"] == {}
    finally:
        auth.current_mcp_scopes.reset(token)


@pytest.mark.parametrize("tool", ["reenrich", "reindex"])
def test_a_corpus_write_is_refused_when_the_token_granted_nothing(tmp_path, tool: str) -> None:
    from podcast_scraper.mcp import auth

    token = auth.current_mcp_scopes.set(frozenset())
    try:
        assert _call(build_server(tmp_path), tool, {})["note"] == "McpScopeError"
    finally:
        auth.current_mcp_scopes.reset(token)


def test_reads_are_unaffected_by_the_gate(tmp_path) -> None:
    # Phase 0 adds no capability and removes none from readers: `mcp:read` is what every existing
    # token carries, and the read tools never ask for more.
    from podcast_scraper.mcp import auth

    token = auth.current_mcp_scopes.set(frozenset({"mcp:read"}))
    try:
        out = _call(build_server(tmp_path), "corpus_status", {})
        assert out["ok"] is not False
    finally:
        auth.current_mcp_scopes.reset(token)


def test_stdio_can_still_write_because_it_is_local_trust(tmp_path) -> None:
    # No HTTP auth context at all → no token → the same local trust that lets stdio run
    # unauthenticated in the first place. Asserted so a future "tighten everything" change has to
    # confront this case explicitly rather than break local tooling silently.
    from podcast_scraper.mcp import auth

    assert auth.current_mcp_scopes.get() is None
    assert _call(build_server(tmp_path), "reenrich", {})["note"] != "McpScopeError"
