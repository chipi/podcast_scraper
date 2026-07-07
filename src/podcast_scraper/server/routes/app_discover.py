"""Consumer personalized-discovery routes (``/api/app/clusters``, ``/api/app/discover``).

The interests picker reads the corpus's top clusters; the discovery feed re-ranks the catalog by
the signed-in user's interests when ``APP_PERSONALIZED_RANKING`` is enabled (PRD-043 FR4 / #1098).
Both are read-only over the shared corpus; ``/discover`` reads per-user interests when signed in
and otherwise (or when the flag is off) returns recency — the default, unchanged behaviour.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response

from podcast_scraper.search.theme_clusters import top_theme_clusters_by_member_count
from podcast_scraper.search.topic_clusters import top_clusters_by_member_count
from podcast_scraper.server import (
    app_ranking_config_store,
    app_ranking_telemetry,
    app_user_state,
)
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_discover_view import rank_discover
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    ranking_config_from_dict,
    ranking_config_to_dict,
)
from podcast_scraper.server.app_user_corpus import derive_interests
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative
from podcast_scraper.server.routes.app_auth import get_admin_user, get_optional_user
from podcast_scraper.server.schemas import (
    AppDiscoverClickBody,
    AppEpisodesResponse,
    AppInterestCluster,
    AppInterestClustersResponse,
    AppStoryline,
    AppStorylinesResponse,
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


@router.get("/theme-clusters", response_model=AppStorylinesResponse)
async def top_storylines(
    request: Request,
    limit: int = Query(default=12, ge=1, le=50, description="Max storylines (by member count)."),
) -> AppStorylinesResponse:
    """Top storylines (theme clusters — topics discussed together) for the Home rail + picker.

    Complementary to ``/clusters`` (semantic): these group co-occurring topics. Each is followable
    as a ``thc:`` interest and carries an ``anchor_topic_id`` so the client can open a card that
    shows the whole storyline. Empty (never 404) when the theme-cluster artifact is absent.
    """
    root = corpus_root_or_503(request)
    items = [AppStoryline(**s) for s in top_theme_clusters_by_member_count(root, limit)]
    return AppStorylinesResponse(items=items)


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
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    data_dir = Path(raw_dir) if raw_dir is not None else None
    interests: list[str] = []
    personalized = bool(getattr(request.app.state, "personalized_ranking", False))
    if personalized and user is not None and data_dir is not None:
        interests = app_user_state.get_interests(data_dir, user.user_id)
        # #1139: also fold in interests derived from what the user has heard/captured, so a user
        # who never used the picker still gets personalized discovery. Explicit follows lead;
        # derived tokens fill in behind them.
        if bool(getattr(request.app.state, "derived_interests", False)):
            derived = derive_interests(root, data_dir, user.user_id)
            interests = list(dict.fromkeys([*interests, *derived]))

    # B2 — the active ranking config (operator-tuned via the admin endpoint), else the default.
    config = (
        app_ranking_config_store.load_ranking_config(data_dir)
        if data_dir is not None
        else DEFAULT_RANKING_CONFIG
    )
    rows = build_catalog_rows_cumulative(root)
    rows.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    pool = rows[: max(limit * 4, limit)]
    items = rank_discover(root, interests, pool, limit=limit, config=config)

    # #11 telemetry: log what the feed showed (slugs in rank order) + the effective variant, so
    # clicks can later be compared against the configured rank. Signed-in only; best-effort.
    if user is not None and data_dir is not None:
        variant = "personalized" if (personalized and interests) else "recency"
        app_ranking_telemetry.record_impressions(
            data_dir,
            user.user_id,
            shown=[it.slug for it in items],
            variant=variant,
            ts=int(time.time()),
        )
    return AppEpisodesResponse(
        items=items, page=1, page_size=limit, total=len(items), has_more=False
    )


@router.get("/ranking-config")
async def get_ranking_config(
    request: Request, _admin: User = Depends(get_admin_user)
) -> dict[str, Any]:
    """The active discovery ranking-signal config (admin only) — the #11 'manage in one place'."""
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    config = (
        app_ranking_config_store.load_ranking_config(Path(raw_dir))
        if raw_dir is not None
        else DEFAULT_RANKING_CONFIG
    )
    return ranking_config_to_dict(config)


@router.put("/ranking-config")
async def put_ranking_config(
    request: Request,
    body: dict[str, Any] = Body(...),
    _admin: User = Depends(get_admin_user),
) -> dict[str, Any]:
    """Replace the ranking-signal config (admin only). Parsing is total — a malformed body merges
    onto the defaults rather than emptying ranking. Returns the stored config."""
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    if raw_dir is None:
        raise HTTPException(status_code=503, detail="No app data dir configured.")
    config = ranking_config_from_dict(body)
    app_ranking_config_store.save_ranking_config(Path(raw_dir), config)
    return ranking_config_to_dict(config)


@router.post("/discover/click", status_code=204)
async def discover_click(
    request: Request,
    body: AppDiscoverClickBody,
    user: User | None = Depends(get_optional_user),
) -> Response:
    """Record a click on a discovery-feed episode for ranking telemetry (#11).

    No-op (still 204) when signed out or without a data dir, so the client can fire-and-forget.
    """
    data_dir = getattr(request.app.state, "app_data_dir", None)
    if user is not None and data_dir is not None:
        variant = (
            "personalized"
            if bool(getattr(request.app.state, "personalized_ranking", False))
            else "recency"
        )
        app_ranking_telemetry.record_click(
            Path(data_dir),
            user.user_id,
            slug=body.slug,
            position=body.position,
            variant=variant,
            ts=int(time.time()),
        )
    return Response(status_code=204)
