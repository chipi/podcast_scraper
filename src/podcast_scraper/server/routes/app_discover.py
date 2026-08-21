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
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_discover_view import build_discover_pool, rank_discover
from podcast_scraper.server.app_momentum import MomentumConfig, resolve_as_of_week, trending
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    ranking_config_from_dict,
    ranking_config_to_dict,
)
from podcast_scraper.server.app_user_corpus import derive_interests
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_admin_user, get_optional_user
from podcast_scraper.server.schemas import (
    AppDiscoverClickBody,
    AppEpisodesResponse,
    AppInterestCluster,
    AppInterestClustersResponse,
    AppStoryline,
    AppStorylinesResponse,
    AppTrendingEntity,
    AppTrendingResponse,
)

# Every kind the momentum layer can rank (RFC-103). Namespaced ids per kind.
_TRENDING_KINDS = ("topic", "cluster", "storyline", "person", "episode", "show", "insight")


def _momentum_config(request: Request) -> MomentumConfig:
    """Momentum config from ``app.state.momentum_config`` (dict), else the packaged defaults."""
    return MomentumConfig.from_dict(getattr(request.app.state, "momentum_config", None))


router = APIRouter(tags=["app"])


@router.get("/clusters", response_model=AppInterestClustersResponse)
def top_clusters(
    request: Request,
    limit: int = Query(default=12, ge=1, le=50, description="Max clusters (by prevalence)."),
) -> AppInterestClustersResponse:
    """Top interest clusters by corpus prevalence — the picker's choices (PRD-043 FR4)."""
    root = corpus_root_or_503(request)
    items = [AppInterestCluster(**c) for c in top_clusters_by_member_count(root, limit)]
    return AppInterestClustersResponse(items=items)


@router.get("/theme-clusters", response_model=AppStorylinesResponse)
def top_storylines(
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


@router.get("/trending", response_model=AppTrendingResponse)
def app_trending(
    request: Request,
    kind: str = Query(default="topic", description=f"One of {_TRENDING_KINDS}."),
    scope: str = Query(default="corpus", description="corpus (all) | mine (per-user; needs auth)."),
    limit: int = Query(default=12, ge=1, le=50),
    user: User | None = Depends(get_optional_user),
) -> AppTrendingResponse:
    """Trending entities of ``kind`` — read-time momentum (velocity + volume) anchored to today.

    Blends the corpus content series (mentions/appearances) with engagement (saves/plays/opens/
    follows), per-kind. ``scope=mine`` ranks the signed-in user's own engagement; corpus otherwise.
    """
    if kind not in _TRENDING_KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {_TRENDING_KINDS}.")
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    data_dir = Path(raw_dir) if raw_dir is not None else None
    eff_scope = "mine" if (scope == "mine" and user is not None) else "corpus"
    uid = user.user_id if (eff_scope == "mine" and user is not None) else None
    rows = trending(
        root,
        data_dir,
        kind=kind,
        scope=eff_scope,
        user_id=uid,
        limit=limit,
        config=_momentum_config(request),
    )
    return AppTrendingResponse(
        kind=kind,
        scope=eff_scope,
        as_of_week=resolve_as_of_week(),
        items=[AppTrendingEntity(**vars(r)) for r in rows],
    )


@router.get("/discover", response_model=AppEpisodesResponse)
def discover(
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
    derived: list[str] = []
    personalized = bool(getattr(request.app.state, "personalized_ranking", False))
    if personalized and user is not None and data_dir is not None:
        interests = app_user_state.get_interests(data_dir, user.user_id)
        # #1139: also fold in interests derived from what the user has heard/captured, so a user
        # who never used the picker still gets personalized discovery. Explicit follows lead;
        # derived tokens fill in behind them.
        # Kept SEPARATE from explicit follows, not merged (#19). Merging pooled them into one
        # affinity denominator, so enabling this flag DROPPED a 2-follow user's per-match boost
        # from 0.5 to 0.2 — switching implicit personalisation on made the user's own follows
        # count for less. The ranker now weights an inference below a stated preference.
        if bool(getattr(request.app.state, "derived_interests", False)):
            derived = derive_interests(root, data_dir, user.user_id)

    # B2 — the active ranking config (operator-tuned via the admin endpoint), else the default.
    config = (
        app_ranking_config_store.load_ranking_config(data_dir)
        if data_dir is not None
        else DEFAULT_RANKING_CONFIG
    )
    rows = cached_catalog(root)
    rows.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    # Shared with the offline eval — see build_discover_pool. Inlining the slice here is what let
    # the eval score the full catalog while production scored this window. `interests` + `root`
    # let the pool include older episodes that MATCH, so a niche follow is not starved out by
    # recency on a large corpus.
    # `config` reaches the POOL too, not only the scoring. Admission is the one parameter no
    # weight can compensate for — an episode the pool excluded cannot be promoted — so it has to
    # be as tunable as everything else (#1795).
    pool = build_discover_pool(
        rows, limit=limit, interests=[*interests, *derived], root=root, config=config
    )
    items = rank_discover(
        root, interests, pool, limit=limit, config=config, derived_interests=derived
    )

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
def get_ranking_config(request: Request, _admin: User = Depends(get_admin_user)) -> dict[str, Any]:
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
        # The variant of the feed that PRODUCED this click, read back off the impression log
        # rather than recomputed. Recomputing used only the personalized_ranking flag, while the
        # impression side also required the user to actually have interests — so a flag-on user
        # with no interests logged `recency` impressions and `personalized` clicks, corrupting any
        # CTR-by-variant comparison before an experiment could start. The impression variant also
        # depends on DERIVED interests, which cost real KG loads to recompute; this beacon must
        # stay cheap. Falling back to the flag keeps a click loggable when no impression precedes
        # it (deep link, cleared log).
        variant = app_ranking_telemetry.last_impression_variant(Path(data_dir), user.user_id) or (
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
