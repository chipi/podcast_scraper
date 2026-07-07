"""GET /api/corpus/trending — operator global momentum view (RFC-103).

The operator's bird's-eye "what's hot corpus-wide": top momentum entities **per kind** in one
response, each with its weekly ``series`` (per-kind sparkline). Corpus scope only (no per-user);
backed by the same momentum capability the consumer ``/api/app/trending`` and the discover ranker
use, so "hot" means one thing across every surface.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_momentum import MomentumConfig, resolve_as_of_week, trending
from podcast_scraper.server.schemas import AppCorpusTrendingResponse, AppTrendingEntity

router = APIRouter(tags=["corpus"])

_KINDS = ("topic", "cluster", "storyline", "person", "episode", "show", "insight")


@router.get("/corpus/trending", response_model=AppCorpusTrendingResponse)
async def corpus_trending(
    request: Request,
    limit_per_kind: int = Query(default=8, ge=1, le=50, description="Top entities per kind."),
) -> AppCorpusTrendingResponse:
    """Top momentum entities per kind, corpus-wide (operator global view)."""
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    data_dir = Path(raw_dir) if raw_dir is not None else None
    cfg = MomentumConfig.from_dict(getattr(request.app.state, "momentum_config", None))
    kinds = {
        kind: [
            AppTrendingEntity(**vars(r))
            for r in trending(
                root, data_dir, kind=kind, scope="corpus", limit=limit_per_kind, config=cfg
            )
        ]
        for kind in _KINDS
    }
    return AppCorpusTrendingResponse(as_of_week=resolve_as_of_week(), kinds=kinds)
