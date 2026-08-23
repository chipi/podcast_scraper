"""Personal-corpus routes — the one unified definition every surface reads (RFC-114 Phase 1, #1470).

Auth-gated, read-time. Surfaces faceted membership (`experienced` vs `saved`), the revision +
change log (adds + tombstones) that RFC-113's incremental export consumes, and a top-entities
summary. `experienced` is the set recall / `scope=mine` already read (now corrected to exclude
whole-episode favorites — RFC-114 §1.1). No request-time LLM (D6).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from podcast_scraper.server import app_corpus_revision, app_corpus_strength, app_user_corpus
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    CorpusChangesResponse,
    CorpusFacetEpisodesResponse,
    CorpusRankedEpisode,
    CorpusRankedResponse,
    CorpusSummary,
    DerivedInterest,
)

router = APIRouter(tags=["app"])

_TOP_ENTITIES = 8


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def _top_entities(root: Path, data_dir: Path, user_id: str) -> list[DerivedInterest]:
    """The user's top derived interests, from the ONE definition (app_user_corpus).

    This used to re-derive them here over ``sorted(slugs)[:_MAX_ENTITY_SCAN]`` — the alphabetical
    freeze #18 fixed for ``/discover`` and nowhere else. Slugs are ``{feed-slug}-{hash}``, so that
    sort grouped by SHOW: past 40 episodes this screen only ever read the alphabetically-first
    shows, and told the user they were into whatever those happened to cover. Same concept, second
    implementation, second answer.
    """
    return [
        DerivedInterest(**row)
        for row in app_user_corpus.derived_interest_counts(root, data_dir, user_id)[:_TOP_ENTITIES]
    ]


@router.get("/corpus", response_model=CorpusSummary)
async def corpus_summary(request: Request, user: User = Depends(get_current_user)) -> CorpusSummary:
    """Faceted membership counts + current revision + top entities."""
    root = corpus_root_or_503(request)
    data_dir = _data_dir(request)
    experienced = app_user_corpus.experienced_episode_set(root, data_dir, user.user_id)
    saved = app_user_corpus.saved_episode_set(data_dir, user.user_id) - experienced
    revision = app_corpus_revision.current(root, data_dir, user.user_id)
    return CorpusSummary(
        revision=revision,
        experienced_count=len(experienced),
        saved_count=len(saved),
        top_entities=_top_entities(root, data_dir, user.user_id),
    )


@router.get("/corpus/episodes", response_model=CorpusFacetEpisodesResponse)
async def corpus_episodes(
    request: Request,
    facet: str = Query(default="experienced", pattern="^(experienced|saved)$"),
    user: User = Depends(get_current_user),
) -> CorpusFacetEpisodesResponse:
    """The episode slugs in a facet (`experienced` or `saved`)."""
    root = corpus_root_or_503(request)
    data_dir = _data_dir(request)
    experienced = app_user_corpus.experienced_episode_set(root, data_dir, user.user_id)
    if facet == "experienced":
        slugs = experienced
        return CorpusFacetEpisodesResponse(facet="experienced", slugs=sorted(slugs))
    slugs = app_user_corpus.saved_episode_set(data_dir, user.user_id) - experienced
    return CorpusFacetEpisodesResponse(facet="saved", slugs=sorted(slugs))


@router.get("/corpus/changes", response_model=CorpusChangesResponse)
async def corpus_changes(
    request: Request,
    since: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
) -> CorpusChangesResponse:
    """The change-log delta (adds + tombstones) after `since`; `truncated` → do a full re-export."""
    root = corpus_root_or_503(request)
    result = app_corpus_revision.changes_since(root, _data_dir(request), user.user_id, since)
    return CorpusChangesResponse(**result)


@router.get("/corpus/ranked", response_model=CorpusRankedResponse)
async def corpus_ranked(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    user: User = Depends(get_current_user),
) -> CorpusRankedResponse:
    """Experienced episodes ranked by corpus strength, strongest first (RFC-114 Phase 2)."""
    root = corpus_root_or_503(request)
    scores = app_corpus_strength.episode_strengths(root, _data_dir(request), user.user_id)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return CorpusRankedResponse(items=[CorpusRankedEpisode(slug=s, strength=v) for s, v in ordered])
