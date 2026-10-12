"""Search's "Also about" never names an umbrella topic (``app_umbrellas``; operator 2026-10-12)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from podcast_scraper.server import query_enricher_helper
from podcast_scraper.server.schemas import SearchHitModel


class _Registry:
    async def run_chain(self, *, envelope: Any, request_id: str) -> Any:
        related = [
            {"topic_id": "topic:umbrella", "label": "on everything"},
            {"topic_id": "topic:dive-planning", "label": "dive planning"},
        ]
        return SimpleNamespace(
            hits=[{"doc_id": h["doc_id"], "related_topics": related} for h in envelope.hits]
        )


def test_related_topics_leave_out_the_umbrella(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        query_enricher_helper, "umbrella_tokens", lambda root: frozenset({"topic:umbrella"})
    )
    state = SimpleNamespace(query_enricher_registry=_Registry(), query_enricher_corpus_root=None)
    request: Any = SimpleNamespace(app=SimpleNamespace(state=state))
    hits = [SearchHitModel(doc_id="d1", score=1.0)]
    asyncio.run(query_enricher_helper.apply_query_enrichers(request, tmp_path, "q", hits))
    related = hits[0].metadata["query_enrichments"]["related_topics"]
    assert [r["topic_id"] for r in related] == ["topic:dive-planning"]
