"""Delivery consent routes — the "Your Week" digest + Web-Push nudge settings (#1414).

PRD-046 FR1 / RFC-110 §3.1. ``GET``/``PUT`` are auth-gated (the user manages their own
consent); ``POST /comms/unsubscribe`` is **public** (no auth) — it is the one-click link
embedded in the digest email, resolved by an opaque ``ref`` rather than a session.

``email_verified`` is identity-derived here (the OAuth provider), not stored: Google-issued
emails are verified, so email delivery is gated on ``provider == "google"``. The delivery
service (#1412) still suppresses on hard bounce regardless.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from podcast_scraper.server import app_comms_store
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import CommsSettings, CommsUpdate

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def _email_verified(user: User) -> bool:
    """Identity-derived: Google-authenticated emails are verified."""
    return user.provider == "google" and bool(user.email)


def _to_settings(stored: dict, *, email_verified: bool) -> CommsSettings:
    return CommsSettings(
        digest=stored["digest"],
        push=stored["push"],
        email_verified=email_verified,
        unsubscribe_ref=stored.get("unsubscribe_ref"),
    )


@router.get("/comms", response_model=CommsSettings)
async def get_comms(request: Request, user: User = Depends(get_current_user)) -> CommsSettings:
    """The user's delivery consent + cadence (defaults, off, when never set)."""
    stored = app_comms_store.get_comms(_data_dir(request), user.user_id)
    return _to_settings(stored, email_verified=_email_verified(user))


@router.put("/comms", response_model=CommsSettings)
async def put_comms(
    request: Request, body: CommsUpdate, user: User = Depends(get_current_user)
) -> CommsSettings:
    """Update whichever section(s) the client sends; mints the unsubscribe ref on first save."""
    stored = app_comms_store.set_comms(
        _data_dir(request),
        user.user_id,
        digest=body.digest.model_dump() if body.digest is not None else None,
        push=body.push.model_dump() if body.push is not None else None,
    )
    return _to_settings(stored, email_verified=_email_verified(user))


@router.post("/comms/unsubscribe")
async def unsubscribe(request: Request, ref: str = Query(..., min_length=1)) -> dict[str, bool]:
    """Public one-click unsubscribe: disable the digest for the user behind ``ref``.

    No auth — the ref *is* the capability. Idempotent; unknown/used refs return
    ``{"unsubscribed": false}`` without leaking whether the ref ever existed.
    """
    ok = app_comms_store.unsubscribe(_data_dir(request), ref)
    return {"unsubscribed": ok}
