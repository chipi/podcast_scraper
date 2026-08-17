"""P3 Consolidation routes — spaced resurfacing + derived interests (#1123, RFC-101 §5-6).

All auth-gated and read-time (no scheduler, no request-time LLM). Resurfacing surfaces the user's
own highlights on a spaced ladder; derived interests rank the people/topics across the user's
heard∪captured corpus as *implicit* signals beside their explicit follows.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from podcast_scraper.server import app_graph_refs, app_user_corpus, app_user_state
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_resurfacing import reflection_prompt, select_due
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


@router.get("/resurfacing", response_model=ResurfacingResponse)
async def resurfacing(
    request: Request, user: User = Depends(get_current_user)
) -> ResurfacingResponse:
    """Highlights due to resurface (most-overdue first) + a reflection prompt; honours pacing.

    Graph-gated, exactly like Your Week and the digest email (#38). Until now this route had NO
    such requirement while the digest assembler dropped refless items, so the Revisit tab listed
    moments the other two silently withheld — the same capture, present on one surface and absent
    from the other two. The email is meant to be a reminder of the page you would see anyway; three
    surfaces answering one question three ways is the opposite of that.
    """
    data_dir = _data_dir(request)
    root = corpus_root_or_503(request)
    settings = app_user_state.get_resurfacing_settings(data_dir, user.user_id)
    paused = bool(settings["paused"])
    highlights = app_user_state.get_highlights(data_dir, user.user_id)
    state = app_user_state.get_resurfacing_state(data_dir, user.user_id)
    due = select_due(highlights, state, int(time.time()), paused=paused)

    items: list[ResurfacingItem] = []
    withheld = 0
    for h in due:
        if not app_graph_refs.carries_the_graph(root, h):
            # An episode with no KG is a PIPELINE defect, not a normal state — corpus validation
            # fails the build on it. Logged per occurrence so it is visible in a running system
            # too: the silent drop is precisely why an empty Your Week went unexplained.
            withheld += 1
            logger.warning(
                "resurfacing: withholding highlight %s (episode %s) — no graph refs; "
                "the episode is missing its KG artifact",
                h.get("id"),
                h.get("episode_slug"),
            )
            continue
        items.append(
            ResurfacingItem(
                highlight=Highlight(**h), reflection_prompt=reflection_prompt(str(h["id"]))
            )
        )
    if withheld:
        logger.warning(
            "resurfacing: %d of %d due highlights withheld for missing graph refs (user %s)",
            withheld,
            len(due),
            user.user_id,
        )
    return ResurfacingResponse(items=items, paused=paused)


@router.post("/resurfacing/{highlight_id}/surfaced", status_code=204)
async def mark_surfaced(
    request: Request, highlight_id: str, user: User = Depends(get_current_user)
) -> None:
    """Record that the user has just seen a resurfaced highlight (advances its ladder step).

    404s on an id the caller does not own. This route used to write whatever key it was handed
    (#39): the id was never checked for existence or ownership, so ``resurfacing.json`` accumulated
    arbitrary entries. ``select_due`` iterates HIGHLIGHTS and looks state up, so junk keys were
    never read back — the file simply grew, unboundedly, at the caller's discretion.

    Cheap hygiene before; load-bearing now. Since #35 the mark is triggered by a ``?revisit=``
    query parameter on the player URL, so any string a user can type into their address bar reaches
    this handler.
    """
    data_dir = _data_dir(request)
    owned = any(
        h.get("id") == highlight_id for h in app_user_state.get_highlights(data_dir, user.user_id)
    )
    if not owned:
        # 404 rather than 403: whether some other user holds this id is not this caller's business.
        raise HTTPException(status_code=404, detail="highlight not found")
    app_user_state.mark_surfaced(data_dir, user.user_id, highlight_id, int(time.time()))


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
    # The ONE definition (app_user_corpus.derived_interest_counts). This used to re-derive them
    # here over EVERY episode in the user's set — no bound at all, so a heavy listener paid an
    # unbounded number of KG loads on a page load, and the list was ranked over a different set of
    # episodes than /discover and /corpus each used. Three implementations, three answers.
    signals = app_user_corpus.derived_interest_counts(root, _data_dir(request), user.user_id)
    return DerivedInterestsResponse(items=[DerivedInterest(**s) for s in signals])
