"""Unit tests for the chunk-5 QueryEnricher protocol + chain runner + concrete enricher."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.protocol import EnricherTier
from podcast_scraper.enrichment.query_enrichers import (
    ALL_DETERMINISTIC_QUERY_ENRICHER_IDS,
    QueryTopicRelatednessEnricher,
    register_deterministic_query_enrichers,
)
from podcast_scraper.enrichment.query_protocol import (
    make_request_ctx,
    QueryEnricher,
    QueryEnricherManifest,
    QueryResultEnvelope,
)
from podcast_scraper.enrichment.query_registry import QueryEnricherRegistry


def _seed_topic_similarity(corpus_root: Path, topics_data: list[dict[str, Any]]) -> None:
    out = corpus_root / "enrichments"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "enricher_id": "topic_similarity",
        "data": {"topic_count": len(topics_data), "topics": topics_data},
    }
    (out / "topic_similarity.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# make_request_ctx
# ---------------------------------------------------------------------------


def test_make_request_ctx_uses_request_id_for_run_and_job() -> None:
    ctx = make_request_ctx(request_id="req-42", enricher_id="x", tier="deterministic")
    assert ctx.run_id == "req-42"
    assert ctx.job_id == "req-42"
    assert ctx.enricher_id == "x"
    assert ctx.attempt == 1


# ---------------------------------------------------------------------------
# QueryEnricherRegistry — sequential chain + error tolerance
# ---------------------------------------------------------------------------


class _Tagger:
    """Tiny test query enricher that adds a tag to annotations."""

    def __init__(self, tag: str) -> None:
        self._tag = tag
        self.manifest = QueryEnricherManifest(
            id=tag, version="1", tier=EnricherTier.DETERMINISTIC, description=tag
        )

    async def enrich_query_result(
        self, *, envelope: QueryResultEnvelope, config: dict, ctx: Any
    ) -> QueryResultEnvelope:
        envelope.annotations.setdefault("tags", []).append(self._tag)
        return envelope


def test_registry_register_and_get() -> None:
    reg = QueryEnricherRegistry()
    a = _Tagger("a")
    reg.register(a)
    assert reg.all_ids() == ["a"]
    assert reg.get("a") is a


def test_registry_register_duplicate_raises() -> None:
    reg = QueryEnricherRegistry()
    reg.register(_Tagger("a"))
    with pytest.raises(ValueError):
        reg.register(_Tagger("a"))


def test_run_chain_runs_all_registered_by_default() -> None:
    reg = QueryEnricherRegistry()
    reg.register(_Tagger("a"))
    reg.register(_Tagger("b"))
    env = QueryResultEnvelope(query="q")
    out = asyncio.run(reg.run_chain(envelope=env, request_id="r"))
    assert out.annotations["tags"] == ["a", "b"]


def test_run_chain_runs_only_named_ids() -> None:
    reg = QueryEnricherRegistry()
    reg.register(_Tagger("a"))
    reg.register(_Tagger("b"))
    out = asyncio.run(
        reg.run_chain(envelope=QueryResultEnvelope(query="q"), request_id="r", enricher_ids=["b"])
    )
    assert out.annotations["tags"] == ["b"]


def test_run_chain_skips_unknown_ids() -> None:
    reg = QueryEnricherRegistry()
    reg.register(_Tagger("a"))
    out = asyncio.run(
        reg.run_chain(
            envelope=QueryResultEnvelope(query="q"),
            request_id="r",
            enricher_ids=["a", "missing"],
        )
    )
    assert out.annotations["tags"] == ["a"]


def test_run_chain_recovers_from_raising_enricher() -> None:
    class Boomer:
        manifest = QueryEnricherManifest(
            id="boom", version="1", tier=EnricherTier.DETERMINISTIC, description="x"
        )

        async def enrich_query_result(self, *, envelope, config, ctx):
            raise RuntimeError("kaboom")

    reg = QueryEnricherRegistry()
    reg.register(Boomer())
    reg.register(_Tagger("a"))
    # The chain must not raise; the raising enricher's envelope is passed
    # through to the next enricher (a).
    out = asyncio.run(reg.run_chain(envelope=QueryResultEnvelope(query="q"), request_id="r"))
    assert out.annotations["tags"] == ["a"]


# ---------------------------------------------------------------------------
# QueryTopicRelatednessEnricher
# ---------------------------------------------------------------------------


def test_query_topic_relatedness_decorates_matching_hits(tmp_path: Path) -> None:
    _seed_topic_similarity(
        tmp_path,
        [
            {
                "topic_id": "topic:ai",
                "topic_label": "AI",
                "top_k": [
                    {"topic_id": "topic:ml", "topic_label": "ML", "similarity": 0.92},
                    {"topic_id": "topic:safety", "topic_label": "Safety", "similarity": 0.81},
                ],
            }
        ],
    )
    enricher = QueryTopicRelatednessEnricher(corpus_root_provider=lambda: tmp_path)
    env = QueryResultEnvelope(
        query="ai",
        hits=[
            {"topic_id": "topic:ai", "score": 0.5},
            {"topic_id": "topic:unrelated", "score": 0.3},
        ],
    )
    ctx = make_request_ctx(
        request_id="r", enricher_id="query_topic_relatedness", tier="deterministic"
    )
    out = asyncio.run(enricher.enrich_query_result(envelope=env, config={}, ctx=ctx))
    # First hit decorated; second left alone.
    assert "related_topics" in out.hits[0]
    assert out.hits[0]["related_topics"][0]["topic_id"] == "topic:ml"
    assert "related_topics" not in out.hits[1]
    assert out.annotations["query_topic_relatedness"]["available"] is True
    assert out.annotations["query_topic_relatedness"]["decorated_hits"] == 1


def test_query_topic_relatedness_no_similarity_file_passes_through(tmp_path: Path) -> None:
    enricher = QueryTopicRelatednessEnricher(corpus_root_provider=lambda: tmp_path)
    env = QueryResultEnvelope(query="x", hits=[{"topic_id": "topic:a"}])
    ctx = make_request_ctx(
        request_id="r", enricher_id="query_topic_relatedness", tier="deterministic"
    )
    out = asyncio.run(enricher.enrich_query_result(envelope=env, config={}, ctx=ctx))
    assert "related_topics" not in out.hits[0]
    assert out.annotations["query_topic_relatedness"]["available"] is False


def test_query_topic_relatedness_max_per_hit_config_override(tmp_path: Path) -> None:
    _seed_topic_similarity(
        tmp_path,
        [
            {
                "topic_id": "topic:a",
                "topic_label": "A",
                "top_k": [
                    {"topic_id": f"topic:n{i}", "similarity": 1 - i * 0.1} for i in range(10)
                ],
            }
        ],
    )
    enricher = QueryTopicRelatednessEnricher(corpus_root_provider=lambda: tmp_path, max_per_hit=10)
    env = QueryResultEnvelope(query="q", hits=[{"topic_id": "topic:a"}])
    ctx = make_request_ctx(
        request_id="r", enricher_id="query_topic_relatedness", tier="deterministic"
    )
    out = asyncio.run(
        enricher.enrich_query_result(envelope=env, config={"max_per_hit": 3}, ctx=ctx)
    )
    assert len(out.hits[0]["related_topics"]) == 3


def test_query_topic_relatedness_reads_kg_topic_source_id_from_api_shape(
    tmp_path: Path,
) -> None:
    """The shipped ``/api/search`` response for ``doc_type: kg_topic`` puts
    the topic id at ``metadata.source_id`` (not ``metadata.topic_id``).
    Enricher must decorate those hits too. Bug caught 2026-07-22 while
    exploratory-testing prod-v2 — S5 hero never rendered because 0 hits
    matched the old ``metadata.topic_id`` lookup."""
    _seed_topic_similarity(
        tmp_path,
        [
            {
                "topic_id": "topic:compute",
                "topic_label": "Compute",
                "top_k": [
                    {"topic_id": "topic:policy", "similarity": 0.9},
                ],
            }
        ],
    )
    enricher = QueryTopicRelatednessEnricher(corpus_root_provider=lambda: tmp_path)
    env = QueryResultEnvelope(
        query="compute",
        hits=[
            {
                "doc_id": "kg_topic:sha256_abc:topic:compute",
                "metadata": {"doc_type": "kg_topic", "source_id": "topic:compute"},
            },
        ],
    )
    ctx = make_request_ctx(
        request_id="r", enricher_id="query_topic_relatedness", tier="deterministic"
    )
    out = asyncio.run(enricher.enrich_query_result(envelope=env, config={}, ctx=ctx))
    assert out.hits[0]["related_topics"][0]["topic_id"] == "topic:policy"
    assert out.annotations["query_topic_relatedness"]["decorated_hits"] == 1


def test_query_topic_relatedness_source_id_not_treated_as_topic_when_doc_type_wrong(
    tmp_path: Path,
) -> None:
    """``source_id`` on a non-kg_topic hit (e.g. kg_entity) may look like
    ``topic:…`` but is semantically different — don't decorate it as a
    topic hit. Guardrail against over-eager fallback."""
    _seed_topic_similarity(
        tmp_path,
        [{"topic_id": "topic:x", "topic_label": "X", "top_k": [{"topic_id": "topic:y"}]}],
    )
    enricher = QueryTopicRelatednessEnricher(corpus_root_provider=lambda: tmp_path)
    env = QueryResultEnvelope(
        query="x",
        hits=[
            {"metadata": {"doc_type": "kg_entity", "source_id": "topic:x"}},
        ],
    )
    ctx = make_request_ctx(
        request_id="r", enricher_id="query_topic_relatedness", tier="deterministic"
    )
    out = asyncio.run(enricher.enrich_query_result(envelope=env, config={}, ctx=ctx))
    assert "related_topics" not in out.hits[0]


def test_query_topic_relatedness_validation() -> None:
    with pytest.raises(ValueError):
        QueryTopicRelatednessEnricher(corpus_root_provider=lambda: Path("/tmp"), max_per_hit=0)


def test_query_topic_relatedness_satisfies_protocol(tmp_path: Path) -> None:
    enricher = QueryTopicRelatednessEnricher(corpus_root_provider=lambda: tmp_path)
    assert isinstance(enricher, QueryEnricher)


def test_register_deterministic_query_enrichers(tmp_path: Path) -> None:
    reg = QueryEnricherRegistry()
    register_deterministic_query_enrichers(reg, corpus_root_provider=lambda: tmp_path)
    assert sorted(reg.all_ids()) == sorted(ALL_DETERMINISTIC_QUERY_ENRICHER_IDS)


def test_query_enricher_chain_with_topic_relatedness(tmp_path: Path) -> None:
    _seed_topic_similarity(
        tmp_path,
        [
            {
                "topic_id": "topic:a",
                "top_k": [{"topic_id": "topic:b", "similarity": 0.7}],
            }
        ],
    )
    reg = QueryEnricherRegistry()
    register_deterministic_query_enrichers(reg, corpus_root_provider=lambda: tmp_path)
    env = QueryResultEnvelope(query="a", hits=[{"topic_id": "topic:a"}])
    out = asyncio.run(reg.run_chain(envelope=env, request_id="r-1"))
    assert out.hits[0]["related_topics"][0]["topic_id"] == "topic:b"
