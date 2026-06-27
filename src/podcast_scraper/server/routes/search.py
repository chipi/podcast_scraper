"""GET /api/search — semantic corpus search (viewer API)."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Query, Request

from podcast_scraper.enrichment.query_enrichers import (
    register_deterministic_query_enrichers,
)
from podcast_scraper.enrichment.query_protocol import QueryResultEnvelope
from podcast_scraper.enrichment.query_registry import QueryEnricherRegistry
from podcast_scraper.search.capability import structured_corpus_search
from podcast_scraper.search.query_log import append_query_event
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    CorpusSearchApiResponse,
    CorpusSearchLiftStatsModel,
    SearchHitModel,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


@router.get("/search", response_model=CorpusSearchApiResponse)
async def search_corpus(
    request: Request,
    q: str = Query(min_length=1, description="Natural-language query."),
    path: str | None = Query(
        default=None,
        description="Corpus output dir (contains search/). Omit to use server default.",
    ),
    type_filters: list[str] | None = Query(
        default=None,
        alias="type",
        description=(
            "Restrict to doc_type (insight, quote, …). Repeat param or comma-separated values."
        ),
    ),
    feed: str | None = Query(default=None, description="Substring match on feed_id."),
    since: str | None = Query(default=None, description="Publish date on/after (YYYY-MM-DD)."),
    speaker: str | None = Query(default=None, description="Speaker substring (quotes / insights)."),
    grounded_only: bool = Query(default=False),
    top_k: int = Query(default=10, ge=1, le=100),
    embedding_model: str | None = Query(
        default=None,
        description="Optional override; should match index model for reliable scores.",
    ),
    dedupe_kg_surfaces: bool = Query(
        default=True,
        description=(
            "When true, collapse duplicate kg_entity/kg_topic rows with the same embedded text "
            "(best score kept; metadata lists merged episode ids)."
        ),
    ),
    enrich_results: bool = Query(
        default=False,
        description=(
            "RFC-088 Phase 4: run the QueryEnricher chain over results. Currently runs "
            "query_topic_relatedness only (decorates hits with topic_similarity ranks "
            "when the corpus has an enrichments/topic_similarity.json output). Passing "
            "through unmodified when no enrichment output is present."
        ),
    ),
) -> CorpusSearchApiResponse:
    """Semantic corpus search.

    Routes through the two-tier hybrid ``RetrievalLayer`` over the LanceDB index — the
    single search path (RFC-090 Phase 2 / ADR-099, #995; FAISS retired). When no usable
    index exists the response carries ``error="no_index"``.
    """
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        return CorpusSearchApiResponse(query=q, error="no_corpus_path")

    doc_types: list[str] | None = None
    if type_filters:
        flat: list[str] = []
        for item in type_filters:
            for part in item.split(","):
                p = part.strip()
                if p:
                    flat.append(p)
        doc_types = flat or None

    outcome = structured_corpus_search(
        root,
        q,
        doc_types=doc_types,
        feed=feed,
        since=since,
        speaker=speaker,
        grounded_only=grounded_only,
        top_k=top_k,
        embedding_model=embedding_model,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
    )

    if outcome["error"]:
        return CorpusSearchApiResponse(
            query=q,
            results=[],
            error=outcome["error"],
            detail=outcome["detail"],
        )

    hits = [
        SearchHitModel(
            doc_id=str(r["doc_id"]),
            score=float(r["score"]),
            metadata=dict(r["metadata"] or {}),
            text=str(r["text"]),
            source_tier=str(r["source_tier"]),
            supporting_quotes=r["supporting_quotes"],
            lifted=r["lifted"],
        )
        for r in outcome["results"]
    ]
    stats_raw = outcome["lift_stats"]
    lift_stats: CorpusSearchLiftStatsModel | None = None
    if isinstance(stats_raw, dict):
        try:
            th = int(stats_raw.get("transcript_hits_returned", 0))
            la = int(stats_raw.get("lift_applied", 0))
            lift_stats = CorpusSearchLiftStatsModel(
                transcript_hits_returned=max(0, th),
                lift_applied=max(0, la),
            )
        except (TypeError, ValueError):
            lift_stats = None
    query_type = str(outcome["query_type"])
    # FR6.2 — record search activity (timestamp + intent only). Best-effort.
    append_query_event(root, query_type)
    if enrich_results:
        # Run the chunk-5 QueryEnricher chain. Failures inside the chain
        # never break the response — the chain swallows + logs them.
        try:
            corpus_root: Path = root
            registry = QueryEnricherRegistry()
            register_deterministic_query_enrichers(
                registry, corpus_root_provider=lambda: corpus_root
            )
            envelope = QueryResultEnvelope(
                query=q,
                hits=[{"doc_id": h.doc_id, "metadata": dict(h.metadata)} for h in hits],
            )
            decorated = await registry.run_chain(envelope=envelope, request_id=str(uuid.uuid4()))
            # Re-merge decorations back into the typed hits via metadata.
            for hit, decorated_hit in zip(hits, decorated.hits):
                related = decorated_hit.get("related_topics")
                if related is not None:
                    hit.metadata.setdefault("query_enrichments", {})["related_topics"] = related
        except Exception as exc:  # noqa: BLE001 — never break /api/search
            logger.warning("query enricher chain failed: %s", exc)
    return CorpusSearchApiResponse(
        query=q, results=hits, query_type=query_type, lift_stats=lift_stats
    )
