"""Unit tests for the MCP tool logic (RFC-095 slice 1) — no MCP SDK required."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools.resolve import resolve_entity
from podcast_scraper.mcp.tools.search import search_corpus

pytestmark = pytest.mark.unit


def test_context_rejects_non_dir(tmp_path) -> None:
    with pytest.raises(ValueError):
        CorpusContext.from_path(tmp_path / "does-not-exist")


def test_context_resolves_dir(tmp_path) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    assert ctx.corpus_dir == tmp_path.resolve()


def test_resolve_entity_empty_name(tmp_path) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    assert resolve_entity(ctx, "   ")["candidates"] == []


def test_resolve_entity_shapes_candidate(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)

    class _Detail:
        id = "person:jane-doe"
        score = 0.92
        method = "alias"

    class _Registry:
        records = {"person:jane-doe": {"type": "person", "display_name": "Jane Doe"}}

    class _Resolver:
        registry = _Registry()

        def resolve_detail(self, text):
            return _Detail()

    monkeypatch.setattr(
        "podcast_scraper.identity.resolver.get_entity_resolver",
        lambda corpus_dir: _Resolver(),
    )
    out = resolve_entity(ctx, "jane")
    assert out["query"] == "jane"
    cand = out["candidates"][0]
    assert cand == {
        "id": "person:jane-doe",
        "kind": "person",
        "display_name": "Jane Doe",
        "score": 0.92,
        "method": "alias",
    }


def test_resolve_entity_no_match(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)

    class _Resolver:
        registry = type("R", (), {"records": {}})()

        def resolve_detail(self, text):
            return None

    monkeypatch.setattr(
        "podcast_scraper.identity.resolver.get_entity_resolver",
        lambda corpus_dir: _Resolver(),
    )
    assert resolve_entity(ctx, "nobody")["candidates"] == []


def test_search_corpus_empty_query(tmp_path) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    out = search_corpus(ctx, "  ")
    assert out["error"] == "empty_query"
    assert out["results"] == []


def test_search_corpus_maps_tier_and_clamps_top_k(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    captured: dict = {}

    def fake_struct(root, query, **kwargs):
        captured.update(kwargs)
        captured["root"] = root
        captured["query"] = query
        return {"query_type": "semantic", "results": [], "error": None, "lift_stats": None}

    monkeypatch.setattr("podcast_scraper.search.capability.structured_corpus_search", fake_struct)
    search_corpus(ctx, "climate", tier="segment", top_k=999, grounded_only=True)
    assert captured["root"] == tmp_path.resolve()
    assert captured["query"] == "climate"
    assert captured["doc_types"] == ["transcript"]  # tier → doc_types
    assert captured["top_k"] == 100  # clamped to [1, 100]
    assert captured["grounded_only"] is True
