"""Consumer personalized-discovery routes (``/api/app/clusters``, ``/api/app/discover``).

The interests picker reads the corpus's top clusters; the discovery feed re-ranks the catalog by
the signed-in user's interests when ``APP_PERSONALIZED_RANKING`` is enabled (PRD-043 FR4 / #1098).
Both are read-only over the shared corpus; ``/discover`` reads per-user interests when signed in
and otherwise (or when the flag is off) returns recency — the default, unchanged behaviour.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from podcast_scraper.search.topic_clusters import top_clusters_by_member_count
from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_discover_view import rank_discover
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative
from podcast_scraper.server.routes.app_auth import get_optional_user
from podcast_scraper.server.schemas import (
    AppEpisodesResponse,
    AppInterestCluster,
    AppInterestClustersResponse,
)

router = APIRouter(tags=["app"])


@router.get("/clusters", response_model=AppInterestClustersResponse)
async def top_clusters(
    request: Request,
    limit: int = Query(default=12, ge=1, le=50, description="Max clusters (by prevalence)."),
) -> AppInterestClustersResponse:
    """Top interest clusters by corpus prevalence — the picker's choices (PRD-043 FR4)."""
    root = corpus_root_or_503(request)
    items = [AppInterestCluster(**c) for c in top_clusters_by_member_count(root, limit)]
    return AppInterestClustersResponse(items=items)


@router.get("/discover", response_model=AppEpisodesResponse)
async def discover(
    request: Request,
    limit: int = Query(default=8, ge=1, le=50, description="Episodes to return."),
    user: User | None = Depends(get_optional_user),
) -> AppEpisodesResponse:
    """Home discovery feed: interest-ranked when enabled + signed-in, else recency (the default).

    Personalization is gated by ``app.state.personalized_ranking`` (env
    ``APP_PERSONALIZED_RANKING``, default off) AND requires the signed-in user to have saved
    interests; otherwise the feed is newest-first, identical to the catalog.
    """
    root = corpus_root_or_503(request)
    interests: list[str] = []
    personalized = bool(getattr(request.app.state, "personalized_ranking", False))
    if personalized and user is not None:
        data_dir = getattr(request.app.state, "app_data_dir", None)
        if data_dir is not None:
            interests = app_user_state.get_interests(Path(data_dir), user.user_id)

    rows = build_catalog_rows_cumulative(root)
    rows.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    pool = rows[: max(limit * 4, limit)]
    items = rank_discover(root, interests, pool, limit=limit)
    return AppEpisodesResponse(
        items=items, page=1, page_size=limit, total=len(items), has_more=False
    )
