"""Consumer personalized-discovery routes (``/api/app/themes``, ``/api/app/discover``).

The interests picker reads the corpus's top themes; the discovery feed re-ranks the catalog by
the signed-in user's interests when ``APP_PERSONALIZED_RANKING`` is enabled (PRD-043 FR4 / #1098).
Both are read-only over the shared corpus; ``/discover`` reads per-user interests when signed in
and otherwise (or when the flag is off) returns recency — the default, unchanged behaviour.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from podcast_scraper import perf_cache
from podcast_scraper.search.storylines import top_storylines_by_member_count
from podcast_scraper.search.topic_clusters import top_themes_by_member_count
from podcast_scraper.server import (
    app_ranking_config_store,
    app_user_state,
)
from podcast_scraper.server.app_another_view_rail import engaged_topics
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_discover_rails import (
    FIRST,
    first_pages,
    MINE_RAILS,
    page_of,
    rail_cards,
    RAIL_MAX,
    RailContext,
    rails_for,
)
from podcast_scraper.server.app_discover_view import (
    build_discover_pool,
    interest_relpaths,
    rank_discover,
    whats_new_rows,
)
from podcast_scraper.server.app_goal_reel import goal_reel
from podcast_scraper.server.app_momentum import (
    MomentumConfig,
    resolve_as_of_week,
    trending,
    WINDOW_MONTHS,
)
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    ranking_config_from_dict,
    ranking_config_to_dict,
)
from podcast_scraper.server.app_relational_view import hosted_photo_urls, search_interests
from podcast_scraper.server.app_slugs import resolve_slug, slug_for_row
from podcast_scraper.server.app_trending_episodes import TrendingEpisodeCard
from podcast_scraper.server.app_user_corpus import (
    derive_interests,
    personal_entity_ids,
    user_episode_set,
)
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import (
    get_admin_user,
    get_current_user,
    get_optional_user,
)
from podcast_scraper.server.schemas import (
    AppDiscoverRail,
    AppDiscoverRailsResponse,
    AppEpisodesResponse,
    AppInterestCluster,
    AppInterestClustersResponse,
    AppInterestSearchResponse,
    AppReason,
    AppRecommendedResponse,
    AppReelClip,
    AppReelTopic,
    AppReelTopicsResponse,
    AppStoryline,
    AppStorylinesResponse,
    AppSuggestLessBody,
    AppSuggestLessItem,
    AppSuggestLessResponse,
    AppTopicReelResponse,
    AppTrendingEntity,
    AppTrendingEpisodeCard,
    AppTrendingResponse,
    AppWhatsNewResponse,
)

# Every kind the momentum layer can rank (RFC-103). Namespaced ids per kind. `organization` (#2031)
# is accepted so the surface can request it, but stays honest-empty until the content-series
# enricher emits an org series (the org card/browse do not depend on trending).
_TRENDING_KINDS = (
    "topic",
    "theme",
    "storyline",
    "person",
    "organization",
    "episode",
    "show",
    "insight",
)


def _momentum_config(request: Request) -> MomentumConfig:
    """Momentum config from ``app.state.momentum_config`` (dict), else the packaged defaults."""
    return MomentumConfig.from_dict(getattr(request.app.state, "momentum_config", None))


router = APIRouter(tags=["app"])


@router.get("/themes", response_model=AppInterestClustersResponse)
def top_themes(
    request: Request,
    limit: int = Query(default=12, ge=1, le=50, description="Max themes (by prevalence)."),
    _user: User = Depends(get_current_user),
) -> AppInterestClustersResponse:
    """Top themes by corpus prevalence — the picker's choices (PRD-043 FR4)."""
    root = corpus_root_or_503(request)
    items = [AppInterestCluster(**c) for c in top_themes_by_member_count(root, limit)]
    return AppInterestClustersResponse(items=items)


@router.get("/storylines", response_model=AppStorylinesResponse)
def top_storylines(
    request: Request,
    limit: int = Query(default=12, ge=1, le=50, description="Max storylines (by member count)."),
    _user: User = Depends(get_current_user),
) -> AppStorylinesResponse:
    """Top storylines (topics discussed together) for the Home rail + picker.

    Complementary to ``/themes`` (semantic): these group co-occurring topics. Each is followable
    as a ``thc:`` interest and carries an ``anchor_topic_id`` so the client can open a card that
    shows the whole storyline. Empty (never 404) when the storyline artifact is absent.

    Floored at ``storylines.DEFAULT_MIN_STORYLINE_MEMBERS`` (4): a storyline is somewhere a
    listener is SENT, and a 2-member theme is a single co-occurrence pair, not a destination. The
    artifact keeps every theme — this is a surfacing decision, shared with the operator overlay so
    the two cannot drift.
    """
    root = corpus_root_or_503(request)
    items = [AppStoryline(**s) for s in top_storylines_by_member_count(root, limit)]
    return AppStorylinesResponse(items=items)


@router.get("/interests/search", response_model=AppInterestSearchResponse)
async def interest_search(
    request: Request,
    kind: Literal["topic", "person", "theme", "storyline"] = Query(
        description="Which kind of followable to search."
    ),
    q: str = Query(min_length=1, max_length=100, description="Text the label must contain."),
    limit: int = Query(default=20, ge=1, le=50),
    _user: User = Depends(get_current_user),
) -> AppInterestSearchResponse:
    """Followables of ``kind`` whose label contains ``q`` — each Interests section's search box.

    ``/themes`` and ``/storylines`` stop at 50 and ``/entities/search`` returns at most one exact
    match, so without this a listener could only follow what a list happened to show them.
    """
    root = corpus_root_or_503(request)
    items = await asyncio.to_thread(search_interests, root, kind, q, limit)
    return AppInterestSearchResponse(query=q, kind=kind, items=items)


_TRENDING_NS = "app_trending_corpus"


@router.get("/trending", response_model=AppTrendingResponse)
def app_trending(
    request: Request,
    kind: str = Query(default="topic", description=f"One of {_TRENDING_KINDS}."),
    scope: str = Query(default="corpus", description="corpus (all) | mine (per-user; needs auth)."),
    window: str = Query(default="3m", description="Trend window: 1m | 3m | 6m | 1y (RFC-103 R2)."),
    limit: int = Query(default=12, ge=1, le=50),
    user: User = Depends(get_current_user),
) -> AppTrendingResponse:
    """Trending entities of ``kind`` — monthly momentum over the selected ``window`` (RFC-103 R2).

    Velocity is recent-window rate ÷ prior-history rate, anchored to the corpus's latest content
    month; the list is floored at ``min_total`` and ranked by velocity × volume. Blends corpus
    content (mentions) with engagement (saves/plays/opens/follows); ``scope=mine`` ranks the
    signed-in user's own engagement.
    """
    if kind not in _TRENDING_KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {_TRENDING_KINDS}.")
    if window not in WINDOW_MONTHS:
        raise HTTPException(
            status_code=422, detail=f"window must be one of {sorted(WINDOW_MONTHS)}."
        )
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    data_dir = Path(raw_dir) if raw_dir is not None else None
    eff_scope = "mine" if (scope == "mine" and user is not None) else "corpus"
    uid = user.user_id if (eff_scope == "mine" and user is not None) else None
    # "Mine" is the listener's own world only — not their engagement blended into everyone's.
    mine = (
        personal_entity_ids(root, data_dir, uid)
        if (uid is not None and data_dir is not None)
        else None
    )
    config = _momentum_config(request)

    def compute() -> list[Any]:
        return trending(
            root,
            data_dir,
            kind=kind,
            scope=eff_scope,
            user_id=uid,
            limit=limit,
            window=window,
            config=config,
            restrict_to=mine,
        )

    if eff_scope == "corpus":
        # Corpus-wide trends are the same for everyone and change slowly, but each call rebuilt the
        # series from the corpus AND every user's engagement: ~0.6 s on prod, four at once on
        # Profile → Interests (measured 2026-10-08). Cached until the corpus changes or 5 minutes
        # pass, whichever is first — engagement moves without the corpus changing.
        token = float(hash((perf_cache.corpus_mtime(root), int(time.time() // 300))))
        key = (str(root), kind, window, limit, repr(config))
        rows = perf_cache.get_or_compute(_TRENDING_NS, key, token, compute)
    else:
        rows = compute()
    items = [AppTrendingEntity(**vars(r)) for r in rows]
    if kind == "person":
        # Hydrate the person avatar so the people-browse chips can show a face (value + space).
        photos = hosted_photo_urls(root)
        items = [
            (
                it.model_copy(update={"image_url": photos[it.entity_id]})
                if it.entity_id in photos
                else it
            )
            for it in items
        ]
    return AppTrendingResponse(
        kind=kind,
        scope=eff_scope,
        as_of_week=resolve_as_of_week(),
        window=window,
        items=items,
    )


def _rail_context(request: Request, scope: str, user: User) -> RailContext:
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    return RailContext(
        root=root,
        data_dir=Path(raw_dir) if raw_dir is not None else None,
        user_id=user.user_id if user is not None else None,
        scope="mine" if scope == "mine" else "corpus",
        config=_momentum_config(request),
    )


def _project(root: Path, cards: list[TrendingEpisodeCard]) -> list[AppTrendingEpisodeCard]:
    items: list[AppTrendingEpisodeCard] = []
    for card in cards:
        row = resolve_slug(root, card.slug)
        if row is not None:
            items.append(
                AppTrendingEpisodeCard(
                    why=AppReason.model_validate(card.why.to_dict()),
                    episode=row_to_summary(root, row),
                )
            )
    return items


@router.get("/discover/rails", response_model=AppDiscoverRailsResponse)
def app_discover_rails(
    request: Request,
    scope: str = Query(default="corpus", description="corpus (Everyone) | mine."),
    first: int = Query(default=FIRST, ge=1, le=RAIL_MAX),
    user: User = Depends(get_current_user),
) -> AppDiscoverRailsResponse:
    """Discover's rails (ADR-163), first page each, in page order: the quote cards, then — under
    Mine — more from what you captured, continue the thread, people you keep hearing, hear another
    view. One card per episode across the page (the highest rail keeps it); a rail left with
    nothing is not returned. ``total`` is what "Show more" can page through
    (``/discover/rails/{rail}``). Under Mine with an empty world, the quote rail is Everyone's,
    ``fallback``."""
    ctx = _rail_context(request, scope, user)
    pages = first_pages(ctx, first)
    return AppDiscoverRailsResponse(
        scope="mine" if ctx.mine else "corpus",
        rails=[
            AppDiscoverRail(
                rail=p.rail, items=_project(ctx.root, p.items), total=p.total, fallback=p.fallback
            )
            for p in pages
        ],
    )


@router.get("/discover/rails/{rail}", response_model=AppDiscoverRail)
def app_discover_rail_page(
    request: Request,
    rail: str,
    scope: str = Query(default="corpus"),
    offset: int = Query(default=0, ge=0, le=RAIL_MAX),
    limit: int = Query(default=4, ge=1, le=RAIL_MAX),
    exclude: list[str] = Query(default=[], max_length=RAIL_MAX * len(MINE_RAILS)),
    user: User = Depends(get_current_user),
) -> AppDiscoverRail:
    """One page of one rail — what "Show more" asks for. ``exclude`` is the episodes the rails
    above it showed first, so the page stays one card per episode; ``offset`` counts after them."""
    ctx = _rail_context(request, scope, user)
    if rail not in rails_for(ctx):
        raise HTTPException(status_code=404, detail=f"No rail {rail!r} under this scope.")
    cards, fallback = rail_cards(ctx, rail)
    items, total = page_of(cards, exclude, offset, limit)
    return AppDiscoverRail(
        rail=rail, items=_project(ctx.root, items), total=total, fallback=fallback
    )


def _topic_label(root: Path, topic_id: str, slugs: list[str]) -> str:
    """The topic's display label, from the KG of an episode that names it; else the id's words."""
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_kg_view import entities_from_kg

    for slug in slugs[:3]:
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        _p, _o, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        label = next((t.label for t in topics if t.id == topic_id and t.label), None)
        if label:
            return label
    return topic_id.split(":", 1)[-1].replace("-", " ")


@router.get("/discover/reel", response_model=AppTopicReelResponse)
def app_discover_reel(
    request: Request,
    topic: str = Query(min_length=7, max_length=200, pattern=r"^topic:"),
    minutes: int = Query(default=15, ge=1, le=60),
    _user: User = Depends(get_current_user),
) -> AppTopicReelResponse:
    """N minutes on a topic (ADR-163 §8 amendment; Discover arc Step 3, goal-shaped listening): the
    strongest clips ABOUT it across episodes, breadth first, within ``minutes``. ``total_seconds``
    is what it actually runs — a small corpus gives less than the budget, and the client says so."""
    root = corpus_root_or_503(request)
    config = getattr(request.app.state, "moments_config", None)
    clips = goal_reel(root, topic, minutes=minutes, config=config)
    out: list[AppReelClip] = []
    for c in clips:
        row = resolve_slug(root, c.episode_slug)
        if row is None:
            continue
        m = c.moment
        out.append(
            AppReelClip(
                episode=row_to_summary(root, row),
                insight_id=m.insight_id,
                text=m.text,
                quote_text=m.quote_text,
                speaker=m.speaker,
                start_ms=m.start_ms,
                end_ms=m.end_ms,
            )
        )
    return AppTopicReelResponse(
        topic_id=topic,
        topic_label=_topic_label(root, topic, [c.episode.slug for c in out]),
        minutes=minutes,
        total_seconds=round(sum(c.end_ms - c.start_ms for c in out) / 1000.0, 1),
        clips=out,
    )


#: Topics on Discover's "15 minutes on…" row, and how many candidates are tried to find them.
REEL_TOPICS = 3
REEL_TOPIC_CANDIDATES = 8
#: A reel of one clip is not "minutes on" anything.
REEL_MIN_CLIPS = 2


@router.get("/discover/reel-topics", response_model=AppReelTopicsResponse)
def app_discover_reel_topics(
    request: Request,
    scope: str = Query(default="corpus", description="corpus | mine"),
    user: User = Depends(get_current_user),
) -> AppReelTopicsResponse:
    """The topics Discover offers "15 minutes on": under Mine the listener's own top topics
    (umbrellas skipped), under Everyone the rising ones — each kept only if its reel has at least
    ``REEL_MIN_CLIPS`` clips, so a chip never opens an empty reel."""
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    data_dir = Path(raw_dir) if raw_dir is not None else None
    eff = "mine" if scope == "mine" and data_dir is not None else "corpus"
    if eff == "mine" and data_dir is not None:
        candidates = engaged_topics(root, data_dir, user.user_id)[:REEL_TOPIC_CANDIDATES]
    else:
        rising = trending(
            root,
            data_dir,
            kind="topic",
            scope="corpus",
            limit=REEL_TOPIC_CANDIDATES,
            config=_momentum_config(request),
        )
        candidates = [(t.entity_id, t.label) for t in rising]
    # A topic the listener asked to see less of is not offered either (ADR-163 §6): the row sits on
    # the same page as the rails that already leave it out.
    if eff == "mine" and data_dir is not None:
        hidden = app_user_state.suggest_less_tokens(data_dir, user.user_id)
        candidates = [(tid, label) for tid, label in candidates if tid not in hidden]
    config = getattr(request.app.state, "moments_config", None)
    items: list[AppReelTopic] = []
    for tid, label in candidates:
        clips = goal_reel(root, tid, minutes=15, config=config)
        if len(clips) >= REEL_MIN_CLIPS:
            seconds = sum(c.moment.end_ms - c.moment.start_ms for c in clips) / 1000.0
            items.append(
                AppReelTopic(
                    topic_id=tid,
                    topic_label=label,
                    clips=len(clips),
                    total_seconds=round(seconds, 1),
                )
            )
        if len(items) >= REEL_TOPICS:
            break
    return AppReelTopicsResponse(scope=eff, items=items)


def _data_dir(request: Request) -> Path:
    # get_current_user has already guaranteed app_data_dir is configured.
    return Path(request.app.state.app_data_dir)


def _valid_less_token(token: str) -> str:
    token = token.strip()
    if len(token) > 200 or not token.startswith(app_user_state.SUGGEST_LESS_PREFIXES):
        raise HTTPException(status_code=422, detail="Not a topic, person, storyline or theme.")
    return token


@router.get("/discover/suggest-less", response_model=AppSuggestLessResponse)
def app_get_suggest_less(
    request: Request, user: User = Depends(get_current_user)
) -> AppSuggestLessResponse:
    """What the listener asked Discover to suggest less of (ADR-163 §6), for Profile to list."""
    rows = app_user_state.get_suggest_less(_data_dir(request), user.user_id)
    return AppSuggestLessResponse(items=[AppSuggestLessItem(**r) for r in rows])


@router.post("/discover/suggest-less/{token}", response_model=AppSuggestLessResponse)
def app_add_suggest_less(
    request: Request,
    token: str,
    body: AppSuggestLessBody,
    user: User = Depends(get_current_user),
) -> AppSuggestLessResponse:
    """ "Suggest less" on a card: hide the subject its reason names from every rail. Idempotent."""
    rows = app_user_state.add_suggest_less(
        _data_dir(request), user.user_id, _valid_less_token(token), body.label, int(time.time())
    )
    return AppSuggestLessResponse(items=[AppSuggestLessItem(**r) for r in rows])


@router.delete("/discover/suggest-less/{token}", response_model=AppSuggestLessResponse)
def app_remove_suggest_less(
    request: Request, token: str, user: User = Depends(get_current_user)
) -> AppSuggestLessResponse:
    """Undo a "suggest less" (Profile › Interests, or the card's Undo). No-op if absent."""
    rows = app_user_state.remove_suggest_less(_data_dir(request), user.user_id, token)
    return AppSuggestLessResponse(items=[AppSuggestLessItem(**r) for r in rows])


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
    if user is None:
        limit = min(limit, 8)
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
    return AppEpisodesResponse(
        items=items, page=1, page_size=limit, total=len(items), has_more=False
    )


@router.get("/whats-new", response_model=AppWhatsNewResponse)
def whats_new(
    request: Request,
    limit: int = Query(default=5, ge=1, le=20, description="Episodes to return."),
    user: User = Depends(get_current_user),
) -> AppWhatsNewResponse:
    """Home's What's new: the newest episodes from what the listener follows (operator 2026-10-07).

    Shows they follow, plus episodes carrying a followed topic, person, theme or storyline, newest
    first. With nothing followed — or nothing matching — it is the newest across every show, and
    ``scope`` says which, so the section can label itself honestly. Recommended is the
    relevance-first section; this one is ordered by time only.
    """
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    data_dir = Path(raw_dir) if raw_dir is not None else None
    feeds: list[str] = []
    matching: set[str] = set()
    if data_dir is not None:
        feeds = [
            str(x.get("feed_id") or "") for x in app_user_state.get_library(data_dir, user.user_id)
        ]
        matching = interest_relpaths(root, app_user_state.get_interests(data_dir, user.user_id))
    rows = cached_catalog(root)
    rows.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    picked, personal = whats_new_rows(
        rows, followed_feeds=feeds, matching_relpaths=matching, limit=limit
    )
    return AppWhatsNewResponse(
        items=[row_to_summary(root, r) for r in picked], scope="yours" if personal else "all"
    )


@router.get("/recommended", response_model=AppRecommendedResponse)
def recommended(
    request: Request,
    limit: int = Query(default=8, ge=1, le=20, description="Episodes to return."),
    user: User = Depends(get_current_user),
) -> AppRecommendedResponse:
    """Home's Recommended when there is no listen to base it on (operator 2026-10-07).

    Episodes carrying what the listener follows (topics, people, themes, storylines) or what their
    listening implies, ranked by the discover ranker, minus what they already played. Independent of
    ``APP_PERSONALIZED_RANKING``, which gates the shared discovery FEED: a section called
    Recommended has to be personal or absent. Nothing to base it on → ``basis: none``, no items, and
    Home does not show the section.
    """
    root = corpus_root_or_503(request)
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    if raw_dir is None:
        return AppRecommendedResponse(items=[], basis="none")
    data_dir = Path(raw_dir)
    explicit = app_user_state.get_interests(data_dir, user.user_id)
    derived = derive_interests(root, data_dir, user.user_id)
    matching = interest_relpaths(root, [*explicit, *derived])
    if not matching:
        return AppRecommendedResponse(items=[], basis="none")
    heard = user_episode_set(root, data_dir, user.user_id)
    rows = cached_catalog(root)
    rows.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    candidates = [
        r for r in rows if r.metadata_relative_path in matching and slug_for_row(r) not in heard
    ]
    config = app_ranking_config_store.load_ranking_config(data_dir)
    items = rank_discover(
        root, explicit, candidates, limit=limit, config=config, derived_interests=derived
    )
    return AppRecommendedResponse(items=items, basis="interests")


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
def put_ranking_config(
    request: Request,
    body: dict[str, Any] = Body(...),
    admin: User = Depends(get_admin_user),
) -> dict[str, Any]:
    """Replace the ranking-signal config (admin only). Parsing is total — a malformed body merges
    onto the defaults rather than emptying ranking. Returns the stored config.

    Audited like the other admin writes (`app_admin._audit`): who, and the config before and after.
    It changes what every listener's Discover shows, and was the one admin write with no record.
    """
    raw_dir = getattr(request.app.state, "app_data_dir", None)
    if raw_dir is None:
        raise HTTPException(status_code=503, detail="No app data dir configured.")
    before = ranking_config_to_dict(app_ranking_config_store.load_ranking_config(Path(raw_dir)))
    config = ranking_config_from_dict(body)
    app_ranking_config_store.save_ranking_config(Path(raw_dir), config)
    after = ranking_config_to_dict(config)
    append_audit(
        getattr(request.app.state, "audit_path", None),
        {"action": "ranking_config_set", "by": admin.user_id, "before": before, "after": after},
    )
    return after
