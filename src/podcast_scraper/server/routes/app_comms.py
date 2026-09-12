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
        types=stored["types"],
        digest_schedule=stored["digest_schedule"],
        email_verified=email_verified,
        timezone=stored.get("timezone", ""),
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
    """Update the matrix and/or schedule the client sends; mints the unsubscribe ref on first save.

    The client PUTs the FULL ``types`` matrix (its current state) — a partial matrix resets the
    omitted cells (see CommsUpdate)."""
    stored = app_comms_store.set_comms(
        _data_dir(request),
        user.user_id,
        types=body.types.model_dump() if body.types is not None else None,
        digest_schedule=(
            body.digest_schedule.model_dump() if body.digest_schedule is not None else None
        ),
        timezone=body.timezone,
    )
    return _to_settings(stored, email_verified=_email_verified(user))


# Human labels for the unsubscribe page, per email type. An unknown type falls back to the digest
# copy (the store rejects it anyway), so the page never leaks an internal type name.
_UNSUB_LABELS = {
    "digest": ("the weekly digest", "the “Your Week” email"),
    "daily_recap": ("the daily recap", "the end-of-day recap email"),
}


@router.get("/comms/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page(
    ref: str = Query(..., min_length=1),
    ntype: str = Query("digest", alias="type"),
) -> HTMLResponse:
    """The email-link landing page. A GET MUST NOT mutate — email clients + link scanners
    prefetch links, which would silently unsubscribe the user. So this only renders a confirm
    button that POSTs (the actual mutation). Complements the RFC-8058 one-click POST below.

    ``type`` selects which email the link came from (``digest`` / ``daily_recap``), so the copy +
    the pref it flips match the email the reader clicked from.
    """
    safe_ref = html.escape(ref, quote=True)
    safe_type = html.escape(ntype, quote=True)
    what, email_name = _UNSUB_LABELS.get(ntype, _UNSUB_LABELS["digest"])
    page = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=robots content=noindex><title>Unsubscribe</title>"
        "<body style='font-family:system-ui;max-width:32rem;margin:4rem auto;padding:0 1rem'>"
        f"<h1>Unsubscribe from {html.escape(what)}?</h1>"
        f"<p>You'll stop receiving {html.escape(email_name)}. Re-enable it anytime in the app.</p>"
        f"<form method=post action='/api/app/comms/unsubscribe?ref={safe_ref}&type={safe_type}'>"
        "<button type=submit style='padding:.6rem 1.2rem;font-size:1rem'>Unsubscribe</button>"
        "</form></body></html>"
    )
    return HTMLResponse(page)


@router.post("/comms/unsubscribe")
async def unsubscribe(
    request: Request,
    ref: str = Query(..., min_length=1),
    ntype: str = Query("digest", alias="type"),
) -> dict[str, bool]:
    """Public one-click unsubscribe: disable ONE email type for the user behind ``ref``.

    No auth — the ref *is* the capability. ``type`` (``digest`` default / ``daily_recap``) is which
    email it came from, so it flips only that type's email channel. Serves both the confirm-page
    form POST (above) and the RFC-8058 ``List-Unsubscribe-Post`` one-click header. Idempotent;
    unknown/used refs return ``{"unsubscribed": false}`` without leaking whether the ref existed.
    """
    ok = app_comms_store.unsubscribe(_data_dir(request), ref, ntype)
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
    """Register a browser push subscription (the endpoint the worker delivers to).

    Registration and *consent* are separate gates now: which types push is a per-type toggle in
    the matrix (PUT /comms). A subscription without any push type enabled simply never receives —
    the delivery path needs both an endpoint AND ``types[type].push``. The client ensures a
    subscription exists before it flips a push toggle on.
    """
    subs = app_push_store.add_subscription(
        _data_dir(request), user.user_id, body.model_dump(exclude_none=True)
    )
    return PushSubscriptionsResponse(count=len(subs))


@router.delete("/push/subscribe", response_model=PushSubscriptionsResponse)
async def unsubscribe_push(
    request: Request, body: PushUnsubscribeBody, user: User = Depends(get_current_user)
) -> PushSubscriptionsResponse:
    """Remove a subscription; when the last is gone, disable push everywhere (unreachable)."""
    subs = app_push_store.remove_subscription(_data_dir(request), user.user_id, body.endpoint)
    if not subs:
        app_comms_store.disable_push_everywhere(_data_dir(request), user.user_id)
    return PushSubscriptionsResponse(count=len(subs))
