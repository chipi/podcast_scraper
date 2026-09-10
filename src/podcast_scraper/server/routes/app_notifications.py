"""In-app notification inbox routes ``/api/app/notifications`` (wave-I).

Auth-gated — a user reads and marks their own inbox. The inbox is the ``in_app`` delivery channel
(what's waiting when you open the app); OS push and email are separate channels handled elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request

from podcast_scraper.server import app_notifications_store
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    MarkReadResponse,
    NotificationsResponse,
)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


@router.get("/notifications", response_model=NotificationsResponse)
async def list_notifications(
    request: Request, user: User = Depends(get_current_user)
) -> NotificationsResponse:
    """The user's inbox (newest-first) + the unread count for the bell badge."""
    data_dir = _data_dir(request)
    items = app_notifications_store.list_notifications(data_dir, user.user_id)
    unread = app_notifications_store.unread_count(data_dir, user.user_id)
    # model_validate coerces the stored dicts (which carry store-only keys like dedupe_key/meta)
    # into NotificationItem, dropping the extras.
    return NotificationsResponse.model_validate({"items": items, "unread": unread})


@router.post("/notifications/{notif_id}/read", response_model=MarkReadResponse)
async def mark_read(
    notif_id: str, request: Request, user: User = Depends(get_current_user)
) -> MarkReadResponse:
    """Mark one notification read (idempotent). Returns the fresh unread count."""
    data_dir = _data_dir(request)
    app_notifications_store.mark_read(data_dir, user.user_id, notif_id)
    return MarkReadResponse(unread=app_notifications_store.unread_count(data_dir, user.user_id))


@router.post("/notifications/read-all", response_model=MarkReadResponse)
async def mark_all_read(
    request: Request, user: User = Depends(get_current_user)
) -> MarkReadResponse:
    """Mark every notification read. Returns the fresh unread count (0)."""
    data_dir = _data_dir(request)
    app_notifications_store.mark_all_read(data_dir, user.user_id)
    return MarkReadResponse(unread=app_notifications_store.unread_count(data_dir, user.user_id))
