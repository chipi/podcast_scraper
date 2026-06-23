"""GET /api/app/search — library-wide grounded search for the consumer platform (#1068).

Extractive grounded retrieval over the existing hybrid index (RFC-090); **no request-time
LLM** (D6). Spans the whole shared corpus for now — scoped to the signed-in user's library
once auth + library land (#1063/#1064). Episode-scoped search lives in ``app_episodes``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.search.capability import structured_corpus_search
from podcast_scraper.search.query_log import append_query_event
from podcast_scraper.server.app_search_view import build_search_response
from podcast_scraper.server.schemas import CorpusSearchApiResponse

router = APIRouter(tags=["app"])


def _corpus_root(request: Request) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    if anchor is None:
        raise HTTPException(status_code=503, detail="No corpus configured for the platform API.")
    return Path(anchor)


@router.get("/search", response_model=CorpusSearchApiResponse)
async def app_search(
    request: Request,
    q: str = Query(min_length=1, description="Natural-language query."),
    top_k: int = Query(default=10, ge=1, le=100),
    grounded_only: bool = Query(default=False),
) -> CorpusSearchApiResponse:
    """Grounded library-wide search (extractive grounded passages; no request-time LLM)."""
    root = _corpus_root(request)
    outcome = structured_corpus_search(root, q, top_k=top_k, grounded_only=grounded_only)
    if not outcome.get("error"):
        append_query_event(root, str(outcome.get("query_type") or ""))
    return build_search_response(q, outcome)
