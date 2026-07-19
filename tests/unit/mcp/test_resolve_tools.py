"""Unit tests for the resolve_entity MCP tool (RFC-095 slice 1) — resolver mocked."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import resolve

pytestmark = pytest.mark.unit


class _Detail:
    def __init__(self, id_: str, score: float = 0.9, method: str = "exact") -> None:
        self.id = id_
        self.score = score
        self.method = method


class _Registry:
    def __init__(self, records: dict) -> None:
        self.records = records


class _Resolver:
    def __init__(self, detail, records) -> None:
        self._detail = detail
        self.registry = _Registry(records)

    def resolve_detail(self, name: str):
        return self._detail


def _patch(monkeypatch, detail, records) -> None:
    monkeypatch.setattr(
        "podcast_scraper.identity.resolver.get_entity_resolver",
        lambda *a, **k: _Resolver(detail, records),
    )


def test_resolve_entity_returns_real_person(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    _patch(
        monkeypatch,
        _Detail("person:jane"),
        {"person:jane": {"type": "person", "display_name": "Jane"}},
    )
    out = resolve.resolve_entity(ctx, "Jane")
    assert [c["id"] for c in out["candidates"]] == ["person:jane"]


def test_resolve_entity_drops_speaker_placeholder(tmp_path, monkeypatch) -> None:
    # #1193: a name must never resolve to an unresolved diarization placeholder.
    ctx = CorpusContext.from_path(tmp_path)
    _patch(
        monkeypatch,
        _Detail("person:speaker-05"),
        {"person:speaker-05": {"type": "person", "display_name": "SPEAKER_05"}},
    )
    out = resolve.resolve_entity(ctx, "Speaker 05")
    assert out == {"query": "Speaker 05", "candidates": []}
