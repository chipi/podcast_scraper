"""Delivery consent routes — the "Your Week" digest + Web-Push nudge settings (#1414).

PRD-046 FR1 / RFC-110 §3.1. ``GET``/``PUT`` are auth-gated (the user manages their own
consent); ``POST /comms/unsubscribe`` is **public** (no auth) — it is the one-click link
embedded in the digest email, resolved by an opaque ``ref`` rather than a session.

``email_verified`` is identity-derived here (the OAuth provider), not stored: Google-issued
emails are verified, so email delivery is gated on ``provider == "google"``. The delivery
service (#1412) still suppresses on hard bounce regardless.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from podcast_scraper.server import app_comms_store, app_push_store
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    CommsSettings,
    CommsUpdate,
    PushSubscription,
    PushSubscriptionsResponse,
    PushUnsubscribeBody,
    VapidKeyResponse,
)

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


@router.get("/comms/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page(ref: str = Query(..., min_length=1)) -> HTMLResponse:
    """The email-link landing page. A GET MUST NOT mutate — email clients + link scanners
    prefetch links, which would silently unsubscribe the user. So this only renders a confirm
    button that POSTs (the actual mutation). Complements the RFC-8058 one-click POST below.
    """
    safe_ref = html.escape(ref, quote=True)
    page = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=robots content=noindex><title>Unsubscribe</title>"
        "<body style='font-family:system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
        "<h1>Unsubscribe from the weekly digest?</h1>"
        "<p>You'll stop receiving the “Your Week” email. Re-enable it anytime in the app.</p>"
        f"<form method=post action='/api/app/comms/unsubscribe?ref={safe_ref}'>"
        "<button type=submit style='padding:.6rem 1.2rem;font-size:1rem'>Unsubscribe</button>"
        "</form></body></html>"
    )
    return HTMLResponse(page)


@router.post("/comms/unsubscribe")
async def unsubscribe(request: Request, ref: str = Query(..., min_length=1)) -> dict[str, bool]:
    """Public one-click unsubscribe: disable the digest for the user behind ``ref``.

    No auth — the ref *is* the capability. Serves both the confirm-page form POST (above) and the
    RFC-8058 ``List-Unsubscribe-Post`` one-click header. Idempotent; unknown/used refs return
    ``{"unsubscribed": false}`` without leaking whether the ref ever existed.
    """
    ok = app_comms_store.unsubscribe(_data_dir(request), ref)
    return {"unsubscribed": ok}


# --- Web Push subscriptions (RFC-110 §6) — the browser registers here so the worker can nudge. ---


@router.get("/push/vapid-key", response_model=VapidKeyResponse)
async def vapid_key(request: Request, user: User = Depends(get_current_user)) -> VapidKeyResponse:
    """The public VAPID key the browser needs to subscribe. 503 when push isn't configured."""
    key = getattr(request.app.state, "vapid_public_key", "") or ""
    if not key:
        raise HTTPException(status_code=503, detail="push not configured")
    return VapidKeyResponse(key=key)


@router.post("/push/subscribe", response_model=PushSubscriptionsResponse)
async def subscribe_push(
    request: Request, body: PushSubscription, user: User = Depends(get_current_user)
) -> PushSubscriptionsResponse:
    """Store a browser push subscription and enable the push channel for this user."""
    subs = app_push_store.add_subscription(
        _data_dir(request), user.user_id, body.model_dump(exclude_none=True)
    )
    app_comms_store.set_comms(_data_dir(request), user.user_id, push={"enabled": True})
    return PushSubscriptionsResponse(count=len(subs))


@router.delete("/push/subscribe", response_model=PushSubscriptionsResponse)
async def unsubscribe_push(
    request: Request, body: PushUnsubscribeBody, user: User = Depends(get_current_user)
) -> PushSubscriptionsResponse:
    """Remove a subscription; disable the push channel when the last one is gone."""
    subs = app_push_store.remove_subscription(_data_dir(request), user.user_id, body.endpoint)
    if not subs:
        app_comms_store.set_comms(_data_dir(request), user.user_id, push={"enabled": False})
    return PushSubscriptionsResponse(count=len(subs))
