"""Consumer platform auth routes + ``get_current_user`` (#1063, RFC-098 §2).

``/api/app/auth/{login,callback,logout}`` runs a single-provider OAuth code flow and sets
a stdlib HMAC-signed session cookie; ``get_current_user`` is the dependency that gates the
per-user routes. Provider, session secret, and per-user data dir come from ``app.state``
(set in ``create_app`` from env) so tests can inject a stub provider + temp data dir.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from podcast_scraper.server import app_roles, app_sessions
from podcast_scraper.server.app_oauth import OAuthError, OAuthProvider
from podcast_scraper.server.app_user_store import get_or_create_user, get_user, set_role, User

router = APIRouter(tags=["app"])

# Custom URL scheme the native shell registers for the OAuth deep-link callback (#1310). The app
# opens login in an external browser; on success the callback redirects here with the signed token.
NATIVE_AUTH_SCHEME = "closelistening"


def _native_scheme(request: Request) -> str:
    return getattr(request.app.state, "native_auth_scheme", None) or NATIVE_AUTH_SCHEME


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


def _bearer_token(request: Request) -> str | None:
    """The token from an ``Authorization: Bearer <token>`` header, or ``None``.

    The native shell (#1310) can't use the session cookie (OAuth completes in an external browser
    whose cookie jar the WebView can't see), so it carries the SAME signed session token as a Bearer
    header instead. Web clients keep sending the cookie; both verify identically.
    """
    header = request.headers.get("Authorization") or request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer":
        return value.strip() or None
    return None


def get_current_user(request: Request) -> User:
    """Resolve the signed session (cookie OR Bearer token) to a ``User``; raise 401 otherwise."""
    secret = _secret(request)
    data_dir = _data_dir(request)
    if not secret or data_dir is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    # Cookie is the web path; the Bearer token is the native-shell path (#1310). Same signer/secret,
    # same payload shape — try the cookie first, then fall back to the header.
    payload = app_sessions.verify(request.cookies.get(app_sessions.SESSION_COOKIE), secret)
    if not payload:
        payload = app_sessions.verify(_bearer_token(request), secret)
    user_id = payload.get("user_id") if payload else None
    user = get_user(data_dir, str(user_id)) if user_id else None
    if user is None or user.disabled:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


def get_optional_user(request: Request) -> User | None:
    """Resolve the session to a ``User``, or ``None`` when unauthenticated (no 401).

    For read surfaces that personalize *when* signed in but stay open otherwise (e.g. the
    discovery feed): an anonymous request simply gets the un-personalized response.
    """
    try:
        return get_current_user(request)
    except HTTPException:
        return None


def get_admin_user(request: Request) -> User:
    """Like :func:`get_current_user` but requires the ``admin`` role (403 otherwise)."""
    user = get_current_user(request)
    if not app_roles.is_admin(user.role):
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


def require_viewer_access(request: Request) -> User:
    """Require a signed-in user with **at least ``creator``** (RFC-108 operator surfaces).

    Mounted as a router-level dependency on the operator-read routers **only** in the
    public operator serve mode (``PODCAST_SERVE_OPERATOR_PUBLIC``); the tailnet-only
    operator serve leaves them ungated (tailnet privacy is the gate). A signed-in
    ``listener`` gets 403 — the operator surface is creator/admin only.
    """
    user = get_current_user(request)
    if not app_roles.can_use_viewer(user.role):
        raise HTTPException(status_code=403, detail="Creator or admin role required.")
    return user


def _safe_return_to(value: str | None) -> str | None:
    """Open-redirect guard for the post-login ``return_to``.

    Allow ONLY a same-origin *relative* path (single leading ``/``). Rejects protocol-relative
    (``//host``), absolute URLs, backslashes, and CRLF so a poisoned ``return_to`` can't bounce
    the post-login redirect off-site. Used by the MCP ``/authorize`` bounce (RFC-112): an
    unauthenticated remote-client authorize is sent through Google sign-in and back here.
    """
    if not value or not isinstance(value, str):
        return None
    if not value.startswith("/") or value.startswith("//"):
        return None
    if "://" in value or "\\" in value or "\n" in value or "\r" in value:
        return None
    return value


@router.get("/auth/login")
async def app_auth_login(
    request: Request,
    as_: str | None = Query(default=None, alias="as", description="Mock identity hint (dev/e2e)."),
    grant: str | None = Query(
        default=None, description="Role hint for new users; only 'creator' is honoured."
    ),
    platform: str | None = Query(
        default=None, description="'native' → callback returns a deep-link token, not a cookie."
    ),
    return_to: str | None = Query(
        default=None,
        description="Same-origin path to return to after login (open-redirect-guarded).",
    ),
) -> RedirectResponse:
    """Begin the OAuth flow: redirect to the provider with a CSRF state cookie.

    ``?as=<name>`` is an optional identity hint honoured **only by the mock provider** (dev/e2e) so
    parallel e2e specs can sign in as isolated users; real providers ignore it.

    ``?grant=creator`` is the viewer's login hint: a *new* (or ``listener``) user is promoted to
    ``creator`` on callback. Only ``creator`` is ever granted this way — never ``admin``.
    """
    provider = _provider(request)
    secret = _secret(request)
    if provider is None or not secret:
        raise HTTPException(status_code=503, detail="Auth is not configured.")
    state = secrets.token_urlsafe(24)
    url = provider.authorization_url(
        state=state, redirect_uri=_callback_uri(request), login_hint=as_
    )
    resp = RedirectResponse(url, status_code=307)
    resp.set_cookie(
        app_sessions.STATE_COOKIE,
        app_sessions.sign(
            {
                "state": state,
                "iat": int(time.time()),
                "grant": grant or "",
                "platform": "native" if platform == "native" else "",
                "return_to": _safe_return_to(return_to) or "",
            },
            secret,
        ),
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
    policy = getattr(request.app.state, "access_policy", None)
    if policy is not None and not policy.is_allowed(identity.email):
        raise HTTPException(status_code=403, detail="This account is not allowed to sign in.")
    user = get_or_create_user(
        data_dir,
        provider=identity.provider,
        subject=identity.subject,
        email=identity.email,
        name=identity.name,
    )
    # Apply the role policy: admin allowlist > creator grant > existing role (never downgraded).
    admin_emails: frozenset[str] = getattr(request.app.state, "admin_emails", frozenset())
    effective = app_roles.resolve_login_role(
        user.role, email=user.email, grant=saved.get("grant"), admin_emails=admin_emails
    )
    if effective != user.role:
        set_role(data_dir, user.user_id, effective)
        user = replace(user, role=effective)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, secret)
    # Native shell (#1310): the OAuth completed in an external browser, so a cookie can't reach the
    # WebView. Hand the SAME signed token back via the app's custom-scheme deep link; the app stores
    # it and sends it as a Bearer header. The token rides the URL fragment (never logged/cached by
    # proxies the way a query string is), and the state cookie is cleared either way.
    if saved.get("platform") == "native":
        deep_link = f"{_native_scheme(request)}://auth#token={token}"
        resp = RedirectResponse(deep_link, status_code=307)
        resp.delete_cookie(app_sessions.STATE_COOKIE)
        return resp
    # Return to where login was initiated (e.g. the MCP /authorize consent, RFC-112) when a
    # guarded same-origin return_to rode the state cookie; otherwise the player home.
    dest = _safe_return_to(saved.get("return_to")) or "/"
    resp = RedirectResponse(dest, status_code=307)
    resp.set_cookie(
        app_sessions.SESSION_COOKIE,
        token,
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


def _user_dict(user: User) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "disabled": user.disabled,
        "mcp_access": user.mcp_access,  # RFC-112: gates the MCP connection UI
    }


@router.get("/me")
def app_me(user: User = Depends(get_current_user)) -> dict[str, object]:
    """Return the signed-in user's basic profile + role (401 when not authenticated)."""
    return _user_dict(user)


@router.get("/auth/dev-users")
def app_auth_dev_users(request: Request) -> dict[str, object]:
    """Predefined dev identities for the sign-in picker — only when the MOCK provider is active.

    With the fake (mock) OAuth provider on, the sign-in UI lets you pick a seeded user (or type a
    custom name) and signs in as ``?as=<hint>``. With a real provider (Google), ``enabled`` is
    ``False`` and the UI shows the normal provider button instead.
    """
    provider = _provider(request)
    is_mock = getattr(provider, "name", "") == "mock"
    users: list[dict[str, str]] = []
    if is_mock:
        from podcast_scraper.server.app_oauth import _safe_hint
        from podcast_scraper.server.app_user_seed import seeds_from_env

        for seed in seeds_from_env():
            hint = _safe_hint(str(seed.get("hint", "")))
            if not hint:
                continue
            users.append(
                {
                    "hint": hint,
                    "name": str(seed.get("name") or hint),
                    "role": app_roles.normalize_role(seed.get("role")),
                }
            )
    return {"enabled": is_mock, "users": users}


@router.get("/auth/status")
def app_auth_status(request: Request) -> dict[str, object]:
    """Whether platform auth is *configured*, plus the signed-in user (if any) — never 401s.

    The viewer gates its UI on this: when auth is not configured (no session secret / provider /
    data dir — e.g. a bare deployment or a backend-less e2e), the app renders **open**, preserving
    the pre-auth behaviour. Only when auth is enabled does an anonymous request get the login gate.
    """
    enabled = bool(_secret(request) and _provider(request) is not None and _data_dir(request))
    user = get_optional_user(request) if enabled else None
    return {"enabled": enabled, "user": _user_dict(user) if user is not None else None}
