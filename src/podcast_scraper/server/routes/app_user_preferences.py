"""USERPREFS-1 — GET + PATCH /api/app/preferences.

Cross-device sync for UI opinion-state (graph lens flags, theme, panel state,
corpus path, etc.). All gated by ``get_current_user`` (401 otherwise) and
scoped to the signed-in user's per-user file overlay under
``<data_dir>/users/<id>/preferences.json``.

Payload is a free-form JSON object — client owns the shape, server round-trips
without interpretation. New preference keys don't require a server release.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status

from podcast_scraper.server import app_user_preferences
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import UserPreferencesPatch, UserPreferencesResponse

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    # get_current_user has already guaranteed app_data_dir is configured.
    return Path(request.app.state.app_data_dir)


@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    request: Request, user: User = Depends(get_current_user)
) -> UserPreferencesResponse:
    """Return the current user's preferences payload ({} when unset)."""
    payload = app_user_preferences.get_preferences(_data_dir(request), user.user_id)
    return UserPreferencesResponse(preferences=payload)


@router.put("/preferences", response_model=UserPreferencesResponse)
async def replace_user_preferences(
    request: Request, body: UserPreferencesPatch, user: User = Depends(get_current_user)
) -> UserPreferencesResponse:
    """Replace the ENTIRE preferences payload with the request body.

    Prefer PATCH for single-key changes to avoid last-write-wins races between
    concurrent tabs; PUT exists for the "reset to a specific full state" path
    (e.g. import from another device).
    """
    try:
        payload = app_user_preferences.replace_preferences(
            _data_dir(request), user.user_id, body.preferences
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return UserPreferencesResponse(preferences=payload)


@router.patch("/preferences", response_model=UserPreferencesResponse)
async def patch_user_preferences(
    request: Request, body: UserPreferencesPatch, user: User = Depends(get_current_user)
) -> UserPreferencesResponse:
    """Shallow-merge the request body into the stored payload.

    Keys with value ``null`` are DELETED; any other value is stored as-is
    (bool, string, number, nested object, array). Returns the merged state.
    """
    try:
        payload = app_user_preferences.patch_preferences(
            _data_dir(request), user.user_id, body.preferences
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return UserPreferencesResponse(preferences=payload)
