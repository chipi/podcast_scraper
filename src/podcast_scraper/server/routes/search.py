"""GET /api/search — semantic corpus search (RFC-062)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request

from podcast_scraper.search.corpus_search import run_corpus_search
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import CorpusSearchApiResponse, SearchHitModel

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
) -> CorpusSearchApiResponse:
    """Semantic search via FAISS + sentence embeddings."""
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

    outcome = run_corpus_search(
        root,
        q,
        doc_types=doc_types,
        feed=feed,
        since=since,
        speaker=speaker,
        grounded_only=grounded_only,
        top_k=top_k,
        index_path=None,
        embedding_model=embedding_model,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
    )

    if outcome.error:
        return CorpusSearchApiResponse(
            query=q,
            results=[],
            error=outcome.error,
            detail=outcome.detail,
        )

    hits = [
        SearchHitModel(
            doc_id=str(r.get("doc_id", "")),
            score=float(r.get("score", 0.0)),
            metadata=dict(r.get("metadata") or {}),
            text=str(r.get("text") or ""),
            supporting_quotes=r.get("supporting_quotes"),
        )
        for r in outcome.results
    ]
    return CorpusSearchApiResponse(query=q, results=hits)
