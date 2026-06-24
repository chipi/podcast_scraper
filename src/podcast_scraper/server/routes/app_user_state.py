"""Per-user state routes — playback, queue, library (#1065, RFC-098 §3).

All gated by ``get_current_user`` (401 otherwise) and scoped to the signed-in user's
plain files under ``<data_dir>/users/<id>/``. No DB; the personal overlay only.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    LibraryAdd,
    LibraryItem,
    LibraryResponse,
    PlaybackListResponse,
    PlaybackPosition,
    PlaybackUpdate,
    QueueResponse,
    QueueUpdate,
)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    # get_current_user has already guaranteed app_data_dir is configured.
    return Path(request.app.state.app_data_dir)


def _library_items(rows: list[dict]) -> list[LibraryItem]:
    return [
        LibraryItem(
            feed_id=str(r.get("feed_id", "")),
            feed_url=r.get("feed_url"),
            title=r.get("title"),
            added_at=r.get("added_at"),
        )
        for r in rows
        if r.get("feed_id")
    ]


@router.get("/playback", response_model=PlaybackListResponse)
async def list_playback(
    request: Request, user: User = Depends(get_current_user)
) -> PlaybackListResponse:
    """All saved playback positions, newest-updated first (Home 'Continue listening')."""
    rows = app_user_state.list_playback(_data_dir(request), user.user_id)
    return PlaybackListResponse(
        items=[
            PlaybackPosition(
                slug=r["slug"],
                position_seconds=float(r["position_seconds"]),
                updated_at=r.get("updated_at"),
            )
            for r in rows
        ]
    )


@router.get("/playback/{slug}", response_model=PlaybackPosition)
async def get_playback(
    request: Request, slug: str, user: User = Depends(get_current_user)
) -> PlaybackPosition:
    """Return the saved playback position for an episode (0 when none)."""
    rec = app_user_state.get_playback(_data_dir(request), user.user_id, slug) or {}
    return PlaybackPosition(
        slug=slug,
        position_seconds=float(rec.get("position_seconds", 0.0)),
        updated_at=rec.get("updated_at"),
    )


@router.put("/playback/{slug}", response_model=PlaybackPosition)
async def put_playback(
    request: Request, slug: str, body: PlaybackUpdate, user: User = Depends(get_current_user)
) -> PlaybackPosition:
    """Save the playback position for an episode."""
    rec = app_user_state.set_playback(
        _data_dir(request), user.user_id, slug, body.position_seconds, int(time.time())
    )
    return PlaybackPosition(
        slug=slug, position_seconds=float(rec["position_seconds"]), updated_at=rec["updated_at"]
    )


@router.get("/queue", response_model=QueueResponse)
async def get_queue(request: Request, user: User = Depends(get_current_user)) -> QueueResponse:
    """Return the user's play queue."""
    return QueueResponse(items=app_user_state.get_queue(_data_dir(request), user.user_id))


@router.put("/queue", response_model=QueueResponse)
async def put_queue(
    request: Request, body: QueueUpdate, user: User = Depends(get_current_user)
) -> QueueResponse:
    """Replace the user's play queue (ordered slugs)."""
    return QueueResponse(
        items=app_user_state.set_queue(_data_dir(request), user.user_id, body.items)
    )


@router.get("/library", response_model=LibraryResponse)
async def get_library(request: Request, user: User = Depends(get_current_user)) -> LibraryResponse:
    """Return the user's subscribed podcasts."""
    return LibraryResponse(
        items=_library_items(app_user_state.get_library(_data_dir(request), user.user_id))
    )


@router.post("/library", response_model=LibraryResponse)
async def add_library(
    request: Request, body: LibraryAdd, user: User = Depends(get_current_user)
) -> LibraryResponse:
    """Subscribe to a podcast (idempotent on feed_id)."""
    item = {
        "feed_id": body.feed_id,
        "feed_url": body.feed_url,
        "title": body.title,
        "added_at": int(time.time()),
    }
    rows = app_user_state.add_subscription(_data_dir(request), user.user_id, item)
    return LibraryResponse(items=_library_items(rows))


@router.delete("/library/{feed_id}", response_model=LibraryResponse)
async def remove_library(
    request: Request, feed_id: str, user: User = Depends(get_current_user)
) -> LibraryResponse:
    """Unsubscribe from a podcast (no-op if absent)."""
    rows = app_user_state.remove_subscription(_data_dir(request), user.user_id, feed_id)
    return LibraryResponse(items=_library_items(rows))
