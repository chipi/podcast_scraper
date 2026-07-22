"""GET /api/search — semantic corpus search (viewer API)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from podcast_scraper.search.capability import structured_corpus_search
from podcast_scraper.search.compare import (
    BriefingPack as CompareBriefingPack,
    compare_subjects,
    SubjectRef as CompareSubjectRef,
)
from podcast_scraper.search.operators import cluster_hits, consensus_pairs_for_hits
from podcast_scraper.search.query_log import append_query_event
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.query_enricher_helper import apply_query_enrichers
from podcast_scraper.server.schemas import (
    CompareBriefingPackModel,
    CompareSubjectRefModel,
    CorpusSearchApiResponse,
    CorpusSearchLiftStatsModel,
    SearchClusterGroupModel,
    SearchCompareRequest,
    SearchCompareResponse,
    SearchConsensusPairModel,
    SearchHitModel,
)

# Valid values for the ``operator`` query param — anything else is treated as
# a no-op (endpoint returns the plain top-k page and ``operator=null``).
_VALID_OPERATORS = frozenset({"cluster", "consensus"})

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
    topic: str | None = Query(
        default=None,
        description=(
            "Topic substring (kg_topic id/text; insight ABOUT edges). Server-side "
            "counterpart to SearchTopicChip; retires the S1 client-side fallback."
        ),
    ),
    episode_id: str | None = Query(
        default=None,
        description=(
            "Exact episode_id scope (Search v3 §S6). Enables the 'Search within "
            "this episode' rail launcher on EpisodeDetailPanel — every hit's "
            "``metadata.episode_id`` must equal this value. Corpus-stable id "
            "match (not substring); rail always knows the exact target."
        ),
    ),
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
    operator: str | None = Query(
        default=None,
        description=(
            "Search v3 §S4b result-set operator: ``cluster`` (group by topic / theme "
            "cluster) or ``consensus`` (read enrichments/topic_consensus.json and "
            "filter pairs to topics in the hit set). Both run Python-side AFTER the "
            "hybrid pipeline returns — no new native combine site. Anything else is "
            "treated as no operator (plain top-k response, ``operator=null``)."
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
        topic=topic,
        episode_id=episode_id,
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
        await apply_query_enrichers(request, root, q, hits)
    # ------------------------------------------------------------------
    # Search v3 §S4b — result-set operators (cluster / consensus).
    #
    # Both surfaces are additive and Python-side only. Errors here NEVER
    # break the response — a failed aggregation logs + degrades to null
    # (so the caller renders the plain top-k page).
    # ------------------------------------------------------------------
    operator_key: str | None = None
    clusters: list[SearchClusterGroupModel] | None = None
    consensus_pairs: list[SearchConsensusPairModel] | None = None
    if operator and operator.strip().lower() in _VALID_OPERATORS:
        operator_key = operator.strip().lower()
        hit_dicts = [{"doc_id": h.doc_id, "metadata": dict(h.metadata)} for h in hits]
        try:
            if operator_key == "cluster":
                cluster_rows = cluster_hits(hit_dicts, root)
                clusters = [SearchClusterGroupModel(**row) for row in cluster_rows]
            elif operator_key == "consensus":
                pairs = consensus_pairs_for_hits(hit_dicts, root)
                consensus_pairs = [SearchConsensusPairModel(**p) for p in pairs]
        except Exception as exc:  # noqa: BLE001 — never break /api/search
            logger.warning("operator %r failed: %s", operator_key, exc)
            if operator_key == "cluster":
                clusters = []
            else:
                consensus_pairs = []
    return CorpusSearchApiResponse(
        query=q,
        results=hits,
        query_type=query_type,
        lift_stats=lift_stats,
        operator=operator_key,
        clusters=clusters,
        consensus_pairs=consensus_pairs,
    )


def _pack_to_model(pack: Any) -> CompareBriefingPackModel:
    """Adapt ``search.compare.BriefingPack`` (dataclass) → Pydantic model."""
    return CompareBriefingPackModel(
        subject=CompareSubjectRefModel(
            kind=pack.subject.kind,
            id=pack.subject.id,
            label=pack.subject.label,
        ),
        query=pack.query,
        query_type=pack.query_type,
        rendered=pack.rendered,
        token_count=pack.token_count,
        max_tokens=pack.max_tokens,
        top_insight_id=pack.top_insight_id,
        top_insight_text=pack.top_insight_text,
        supporting_segment_ids=list(pack.supporting_segment_ids),
        supporting_segment_texts=list(pack.supporting_segment_texts),
        coverage_summary=dict(pack.coverage_summary),
        confidence_p50=float(pack.confidence_p50),
        result_count=int(pack.result_count),
        grounded=bool(pack.grounded),
    )


@router.post("/search/compare", response_model=SearchCompareResponse)
async def search_compare(
    request: Request,
    body: SearchCompareRequest,
) -> SearchCompareResponse:
    """Search v3 §S8 — Compare 2 subjects.

    Wraps ``search.compare.compare_subjects`` (which itself wraps the RFC-093
    ``build_briefing_pack`` twice — one call per side). Judge summary is
    deterministic (no LLM) and muted when either side is ungrounded.
    """

    def _empty_pack(subject_model: CompareSubjectRefModel) -> CompareBriefingPack:
        return CompareBriefingPack(
            subject=CompareSubjectRef(
                kind=subject_model.kind,
                id=subject_model.id,
                label=subject_model.label,
            ),
            query=body.q,
            max_tokens=body.max_tokens,
        )

    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(body.path, fallback)
    if root is None:
        return SearchCompareResponse(
            pack_a=_pack_to_model(_empty_pack(body.subject_a)),
            pack_b=_pack_to_model(_empty_pack(body.subject_b)),
            judge_summary=None,
            error="no_corpus_path",
        )

    outcome = compare_subjects(
        root,
        CompareSubjectRef(
            kind=body.subject_a.kind, id=body.subject_a.id, label=body.subject_a.label
        ),
        CompareSubjectRef(
            kind=body.subject_b.kind, id=body.subject_b.id, label=body.subject_b.label
        ),
        q=body.q,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
    )
    return SearchCompareResponse(
        pack_a=_pack_to_model(outcome.pack_a),
        pack_b=_pack_to_model(outcome.pack_b),
        judge_summary=outcome.judge_summary,
        error=outcome.error,
        detail=outcome.detail,
    )
