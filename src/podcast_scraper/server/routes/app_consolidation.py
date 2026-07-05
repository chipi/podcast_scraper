"""P3 Consolidation routes — spaced resurfacing + derived interests (#1123, RFC-101 §5-6).

All auth-gated and read-time (no scheduler, no request-time LLM). Resurfacing surfaces the user's
own highlights on a spaced ladder; derived interests rank the people/topics across the user's
heard∪captured corpus as *implicit* signals beside their explicit follows.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_corpus_access import corpus_root_or_503, load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_resurfacing import (
    derive_interest_signals,
    reflection_prompt,
    select_due,
)
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_corpus import user_episode_set
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    DerivedInterest,
    DerivedInterestsResponse,
    Highlight,
    ResurfacingItem,
    ResurfacingResponse,
    ResurfacingSettings,
)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


@router.get("/resurfacing", response_model=ResurfacingResponse)
async def resurfacing(
    request: Request, user: User = Depends(get_current_user)
) -> ResurfacingResponse:
    """Highlights due to resurface (most-overdue first) + a reflection prompt; honours pacing."""
    data_dir = _data_dir(request)
    settings = app_user_state.get_resurfacing_settings(data_dir, user.user_id)
    paused = bool(settings["paused"])
    highlights = app_user_state.get_highlights(data_dir, user.user_id)
    state = app_user_state.get_resurfacing_state(data_dir, user.user_id)
    due = select_due(highlights, state, int(time.time()), paused=paused)
    items = [
        ResurfacingItem(highlight=Highlight(**h), reflection_prompt=reflection_prompt(str(h["id"])))
        for h in due
    ]
    return ResurfacingResponse(items=items, paused=paused)


@router.post("/resurfacing/{highlight_id}/surfaced", status_code=204)
async def mark_surfaced(
    request: Request, highlight_id: str, user: User = Depends(get_current_user)
) -> None:
    """Record that the user has just seen a resurfaced highlight (advances its ladder step)."""
    app_user_state.mark_surfaced(_data_dir(request), user.user_id, highlight_id, int(time.time()))


@router.get("/resurfacing/settings", response_model=ResurfacingSettings)
async def get_settings(
    request: Request, user: User = Depends(get_current_user)
) -> ResurfacingSettings:
    """Return the user's resurfacing pacing settings."""
    return ResurfacingSettings(
        **app_user_state.get_resurfacing_settings(_data_dir(request), user.user_id)
    )


@router.put("/resurfacing/settings", response_model=ResurfacingSettings)
async def put_settings(
    request: Request, body: ResurfacingSettings, user: User = Depends(get_current_user)
) -> ResurfacingSettings:
    """Update pacing (pause/resume)."""
    stored = app_user_state.set_resurfacing_settings(
        _data_dir(request), user.user_id, paused=body.paused
    )
    return ResurfacingSettings(**stored)


@router.get("/interests/derived", response_model=DerivedInterestsResponse)
async def derived_interests(
    request: Request, user: User = Depends(get_current_user)
) -> DerivedInterestsResponse:
    """Implicit interests ranked by occurrence across the user's heard∪captured episodes."""
    root = corpus_root_or_503(request)
    mine = user_episode_set(root, _data_dir(request), user.user_id)
    occurrences: list[tuple[str, str, str]] = []
    for slug in mine:
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        persons, _orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        occurrences += [("person", p.id, p.name) for p in persons]
        occurrences += [("topic", t.id, t.label) for t in topics]
    signals = derive_interest_signals(occurrences)
    return DerivedInterestsResponse(items=[DerivedInterest(**s) for s in signals])
