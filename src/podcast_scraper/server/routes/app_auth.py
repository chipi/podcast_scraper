"""Consumer platform auth routes + ``get_current_user`` (#1063, RFC-098 §2).

``/api/app/auth/{login,callback,logout}`` runs a single-provider OAuth code flow and sets
a stdlib HMAC-signed session cookie; ``get_current_user`` is the dependency that gates the
per-user routes. Provider, session secret, and per-user data dir come from ``app.state``
(set in ``create_app`` from env) so tests can inject a stub provider + temp data dir.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app_oauth import OAuthError, OAuthProvider
from podcast_scraper.server.app_user_store import get_or_create_user, get_user, User

router = APIRouter(tags=["app"])


def _secret(request: Request) -> str:
    return getattr(request.app.state, "session_secret", "") or ""


def _data_dir(request: Request) -> Path | None:
    raw = getattr(request.app.state, "app_data_dir", None)
    return Path(raw) if raw is not None else None


def _provider(request: Request) -> OAuthProvider | None:
    return getattr(request.app.state, "oauth_provider", None)


def _secure(request: Request) -> bool:
    return bool(getattr(request.app.state, "session_cookie_secure", False))


def _callback_uri(request: Request) -> str:
    return str(request.url_for("app_auth_callback"))


def get_current_user(request: Request) -> User:
    """Resolve the signed session cookie to a ``User``; raise 401 otherwise."""
    secret = _secret(request)
    data_dir = _data_dir(request)
    if not secret or data_dir is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    payload = app_sessions.verify(request.cookies.get(app_sessions.SESSION_COOKIE), secret)
    user_id = payload.get("user_id") if payload else None
    user = get_user(data_dir, str(user_id)) if user_id else None
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


@router.get("/auth/login")
async def app_auth_login(request: Request) -> RedirectResponse:
    """Begin the OAuth flow: redirect to the provider with a CSRF state cookie."""
    provider = _provider(request)
    secret = _secret(request)
    if provider is None or not secret:
        raise HTTPException(status_code=503, detail="Auth is not configured.")
    state = secrets.token_urlsafe(24)
    url = provider.authorization_url(state=state, redirect_uri=_callback_uri(request))
    resp = RedirectResponse(url, status_code=307)
    resp.set_cookie(
        app_sessions.STATE_COOKIE,
        app_sessions.sign({"state": state, "iat": int(time.time())}, secret),
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=_secure(request),
    )
    return resp


@router.get("/auth/callback", name="app_auth_callback")
async def app_auth_callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code."),
    state: str = Query(..., description="CSRF state echoed by the provider."),
) -> RedirectResponse:
    """Complete the OAuth flow: verify state, exchange code, upsert user, set session."""
    provider = _provider(request)
    secret = _secret(request)
    data_dir = _data_dir(request)
    if provider is None or not secret or data_dir is None:
        raise HTTPException(status_code=503, detail="Auth is not configured.")
    saved = app_sessions.verify(request.cookies.get(app_sessions.STATE_COOKIE), secret, max_age=600)
    if not saved or saved.get("state") != state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    try:
        identity = provider.exchange_code(code=code, redirect_uri=_callback_uri(request))
    except OAuthError as exc:
        raise HTTPException(status_code=502, detail="OAuth exchange failed.") from exc
    user = get_or_create_user(
        data_dir,
        provider=identity.provider,
        subject=identity.subject,
        email=identity.email,
        name=identity.name,
    )
    resp = RedirectResponse("/", status_code=307)
    resp.set_cookie(
        app_sessions.SESSION_COOKIE,
        app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, secret),
        max_age=app_sessions.DEFAULT_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_secure(request),
    )
    resp.delete_cookie(app_sessions.STATE_COOKIE)
    return resp


@router.post("/auth/logout")
async def app_auth_logout() -> Response:
    """Clear the session cookie."""
    resp = Response(status_code=204)
    resp.delete_cookie(app_sessions.SESSION_COOKIE)
    return resp


@router.get("/me")
async def app_me(user: User = Depends(get_current_user)) -> dict[str, str]:
    """Return the signed-in user's basic profile (401 when not authenticated)."""
    return {"user_id": user.user_id, "email": user.email, "name": user.name}
