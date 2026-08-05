"""Personal-corpus routes — the one unified definition every surface reads (RFC-114 Phase 1, #1470).

Auth-gated, read-time. Surfaces faceted membership (`experienced` vs `saved`), the revision +
change log (adds + tombstones) that RFC-113's incremental export consumes, and a top-entities
summary. `experienced` is the set recall / `scope=mine` already read (now corrected to exclude
whole-episode favorites — RFC-114 §1.1). No request-time LLM (D6).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from podcast_scraper.server import app_corpus_revision, app_user_corpus
from podcast_scraper.server.app_corpus_access import corpus_root_or_503, load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_resurfacing import derive_interest_signals
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    CorpusChangesResponse,
    CorpusFacetEpisodesResponse,
    CorpusSummary,
    DerivedInterest,
)

router = APIRouter(tags=["app"])

_TOP_ENTITIES = 8
_MAX_ENTITY_SCAN = 40


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def _top_entities(root: Path, slugs: set[str]) -> list[DerivedInterest]:
    """Rank person/topic occurrences across the experienced episodes (bounded scan)."""
    occurrences: list[tuple[str, str, str]] = []
    for slug in sorted(slugs)[:_MAX_ENTITY_SCAN]:
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        persons, _orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        occurrences += [("person", p.id, p.name) for p in persons]
        occurrences += [("topic", t.id, t.label) for t in topics]
    signals = derive_interest_signals(occurrences)[:_TOP_ENTITIES]
    return [DerivedInterest(**s) for s in signals]


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
        top_entities=_top_entities(root, experienced),
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
