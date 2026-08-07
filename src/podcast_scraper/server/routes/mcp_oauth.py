"""MCP OAuth 2.1 authorization-server endpoints (RFC-112 slice 3, #1471).

We are the authorization server for the MCP resource. ``wellknown_router`` (root-mounted) serves the
RFC 8414 metadata so a client discovers the endpoints; ``router`` (under ``/api/app``)
carries Dynamic Client Registration, the session-gated consent + authorize, and the token endpoint.

Public clients + PKCE only. ``/authorize`` reuses the platform session (the human must be logged in
and hold ``mcp_access``); a **GET** renders a prefetch-safe consent page, a **POST** approves and
redirects back with a single-use code. Issuer/base URL from ``APP_MCP_ISSUER_URL``; unset →
the endpoints 503 (OAuth disabled) — same posture as the other configured features.
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from podcast_scraper.server import app_oauth_server, app_rate_limit
from podcast_scraper.server.app_audit import audit_event
from podcast_scraper.server.app_user_store import get_user
from podcast_scraper.server.routes.app_auth import get_current_user

router = APIRouter(tags=["app"])
wellknown_router = APIRouter(tags=["app"])

# Per-principal rate limits (app-level; the edge already limits per-IP). Beyond these → 429.
_REGISTER_LIMIT, _REGISTER_WINDOW_S = 5, 60.0  # DCR per client IP
_TOKEN_LIMIT, _TOKEN_WINDOW_S = 10, 60.0  # token exchanges per OAuth client


def _client_ip(request: Request) -> str:
    """Best-effort client IP for the DCR rate-limit key — the left-most X-Forwarded-For hop.

    Trust model: the app is loopback-bound behind the edge (Cloudflare → Caddy → nginx). The true
    outer edge (Cloudflare, ADR-118) overwrites XFF, so the left-most hop is edge-attested there —
    the same posture the existing nginx `limit_req` relies on. Without that edge XFF is client-
    spoofable, so this per-IP DCR limit is **best-effort**; the unbypassable backstop against DCR
    abuse is the hard ``_MAX_CLIENTS`` registration cap, and the per-*client* token limit below keys
    on the server-minted ``client_id`` (not spoofable).
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def _issuer() -> str:
    return os.environ.get("APP_MCP_ISSUER_URL", "").strip().rstrip("/")


def _require_issuer() -> str:
    issuer = _issuer()
    if not issuer:
        raise HTTPException(status_code=503, detail="mcp oauth not configured")
    return issuer


@wellknown_router.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata() -> JSONResponse:
    """RFC 8414 metadata — how a client discovers the authorize / token / register endpoints."""
    issuer = _require_issuer()
    base = f"{issuer}/api/app/mcp/oauth"
    return JSONResponse(
        {
            "issuer": issuer,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["mcp:read"],
        }
    )


@router.post("/mcp/oauth/register", status_code=201)
async def register(request: Request) -> JSONResponse:
    """Dynamic Client Registration — the client self-registers its redirect URIs (public client)."""
    _require_issuer()
    if not app_rate_limit.allow(
        f"mcp_register:{_client_ip(request)}", limit=_REGISTER_LIMIT, window_s=_REGISTER_WINDOW_S
    ):
        raise HTTPException(status_code=429, detail="too many registrations, slow down")
    body = await request.json()
    redirect_uris = body.get("redirect_uris") if isinstance(body, dict) else None
    if not isinstance(redirect_uris, list) or not redirect_uris:
        raise HTTPException(status_code=400, detail="redirect_uris (array) required")
    try:
        client = app_oauth_server.register_client(
            _data_dir(request),
            redirect_uris=[str(u) for u in redirect_uris],
            client_name=str(body.get("client_name", "")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(client, status_code=201)


def _validate_authorize(
    request: Request, client_id: str, redirect_uri: str, method: str, scope: str
) -> None:
    client = app_oauth_server.get_client(_data_dir(request), client_id)
    if client is None:
        raise HTTPException(status_code=400, detail="unknown client_id")
    if redirect_uri not in client["redirect_uris"]:
        raise HTTPException(
            status_code=400, detail="redirect_uri not registered"
        )  # anti open-redirect
    if method != "S256":
        raise HTTPException(status_code=400, detail="code_challenge_method must be S256")
    # Only mint scopes we actually support — an unknown scope must not be silently granted.
    if not app_oauth_server.is_scope_supported(scope):
        raise HTTPException(status_code=400, detail=f"unsupported scope: {scope}")


def _mint_and_redirect(
    request: Request,
    user_id: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    scope: str,
    state: str,
) -> RedirectResponse:
    """Mint a single-use code + 302 back to the client's redirect_uri (preserving state).

    We redirect ONLY to a URI taken from the client's **registered allow-list** — not the raw
    request value. Re-resolving the target from stored client data (rather than reflecting the
    request param) is the anti-open-redirect guard made explicit: an unregistered value can never
    reach the ``Location`` header (defense-in-depth on top of ``_validate_authorize``).
    """
    client = app_oauth_server.get_client(_data_dir(request), client_id)
    allowed = list((client or {}).get("redirect_uris", []))
    if redirect_uri not in allowed:  # unreachable post-validation; fail closed regardless
        raise HTTPException(status_code=400, detail="redirect_uri not registered")
    safe_redirect = allowed[allowed.index(redirect_uri)]  # from stored data, not the request
    code = app_oauth_server.create_authorization_code(
        _data_dir(request),
        user_id=user_id,
        client_id=client_id,
        redirect_uri=safe_redirect,
        code_challenge=code_challenge,
        scope=scope,
    )
    params = {"code": code}
    if state:
        params["state"] = state
    sep = "&" if "?" in safe_redirect else "?"
    return RedirectResponse(url=f"{safe_redirect}{sep}{urlencode(params)}", status_code=302)


def hidden_fields(**fields: str) -> str:
    """Render the authorize params as hidden inputs (HTML-escaped) for the consent form POST."""
    return "".join(
        f'<input type=hidden name="{html.escape(k)}" value="{html.escape(v)}">'
        for k, v in fields.items()
    )


def _origin_of(url: str) -> str:
    """The scheme://host[:port] of a redirect_uri, for the consent-screen disclosure."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else url


def _consent_page(client: dict, redirect_uri: str, scope: str, hidden: str) -> str:
    """The consent screen. Discloses WHO (client name + registration) and WHERE the code goes
    (redirect origin) so a user can't be tricked by a look-alike ``client_name`` — DCR is open, so
    an attacker can register "Claude" pointing at their own redirect (review H3). Offers Deny too.
    """
    name = html.escape(str(client.get("client_name") or "an application"))
    origin = html.escape(_origin_of(redirect_uri))
    scope_txt = html.escape(scope)
    created = client.get("created_at")
    when = ""
    if isinstance(created, (int, float)):
        day = time.strftime("%Y-%m-%d", time.gmtime(int(created)))
        when = f" · registered {day}"
    return (
        "<!doctype html><html lang=en><meta charset=utf-8><meta name=robots content=noindex>"
        "<title>Authorize</title><body style='font-family:system-ui;max-width:32rem;margin:4rem "
        "auto;padding:0 1rem'>"
        f"<h1>Allow <b>{name}</b> to access your closelistening corpus?</h1>"
        f"<p>It will be able to search and read your podcast knowledge base as you (scope "
        f"<code>{scope_txt}</code>).</p>"
        f"<p style='color:#555;font-size:.9rem'>You will be redirected to "
        f"<b>{origin}</b>{when}. Only approve if you recognise this destination.</p>"
        "<div style='display:flex;gap:.75rem;margin-top:1.25rem'>"
        f"<form method=post action='/api/app/mcp/oauth/authorize'>{hidden}"
        "<button type=submit style='padding:.6rem 1.2rem;font-size:1rem'>Allow</button></form>"
        "<a href='/' style='padding:.6rem 1.2rem;font-size:1rem;border:1px solid #ccc;"
        "border-radius:.4rem;text-decoration:none;color:#333'>Deny</a>"
        "</div></body></html>"
    )


@router.get("/mcp/oauth/authorize", response_class=HTMLResponse)
async def authorize_page(
    request: Request,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str = "",
    scope: str = "mcp:read",
    code_challenge_method: str = "S256",
    response_type: str = "code",
) -> Response:
    """Authorize: silent redirect if consent is remembered, else render the consent screen.

    Session-gated + `mcp_access`. A GET issues a code ONLY when the user has already approved this
    client + scope (a top-level client navigation, not a prefetch); otherwise it renders the form.
    """
    _require_issuer()
    try:
        user = get_current_user(request)
    except HTTPException as exc:
        # Not signed into the player yet. A remote MCP client (e.g. claude.ai) opens /authorize
        # as a top-level navigation, so a hard 401 dead-ends the connector with no login prompt
        # (INCIDENT 2026-08-08). Bounce through Google sign-in and return here — the consent
        # screen needs a known user, and the Lax session cookie set on callback is carried back
        # on the return redirect. Only the unauthenticated (401) case redirects; 403/others raise.
        if exc.status_code == 401:
            # /auth/login lives under the SAME app prefix as this route (…/api/app). Derive it
            # from the current path so the redirect stays same-origin regardless of mount prefix
            # or proxy host (url_for would emit the internal uvicorn host behind the edge).
            suffix = "/mcp/oauth/authorize"
            prefix = request.url.path[: -len(suffix)] if request.url.path.endswith(suffix) else ""
            nxt = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(
                f"{prefix}/auth/login?return_to={quote(nxt, safe='')}", status_code=302
            )
        raise
    if not user.mcp_access:
        raise HTTPException(status_code=403, detail="mcp access not granted")
    if response_type != "code":
        raise HTTPException(status_code=400, detail="response_type must be code")
    _validate_authorize(request, client_id, redirect_uri, code_challenge_method, scope)
    # Remembered consent → skip the prompt, mint a code, redirect (silent re-auth).
    if app_oauth_server.has_consent(
        _data_dir(request), user_id=user.user_id, client_id=client_id, scope=scope
    ):
        return _mint_and_redirect(
            request, user.user_id, client_id, redirect_uri, code_challenge, scope, state
        )
    client = app_oauth_server.get_client(_data_dir(request), client_id)
    assert client is not None  # _validate_authorize raised otherwise
    hidden = hidden_fields(
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        scope=scope,
        state=state,
    )
    # Clickjacking defense-in-depth on a one-button approval page (review L4).
    headers = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'"}
    return HTMLResponse(_consent_page(client, redirect_uri, scope, hidden), headers=headers)


@router.post("/mcp/oauth/authorize")
async def authorize_approve(
    request: Request,
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    code_challenge: str = Form(...),
    code_challenge_method: str = Form("S256"),
    scope: str = Form("mcp:read"),
    state: str = Form(""),
) -> RedirectResponse:
    """Consent approved → remember it, mint a code, redirect back (RFC-8252/OAuth 2.1)."""
    _require_issuer()
    user = get_current_user(request)
    if not user.mcp_access:
        raise HTTPException(status_code=403, detail="mcp access not granted")
    _validate_authorize(request, client_id, redirect_uri, code_challenge_method, scope)
    app_oauth_server.remember_consent(
        _data_dir(request), user_id=user.user_id, client_id=client_id, scope=scope
    )
    audit_event(
        request, "mcp.consent.granted", user_id=user.user_id, client_id=client_id, scope=scope
    )
    return _mint_and_redirect(
        request, user.user_id, client_id, redirect_uri, code_challenge, scope, state
    )


@router.post("/mcp/oauth/token")
async def token(
    request: Request,
    grant_type: str = Form(...),
    client_id: str = Form(...),
    code: str = Form(""),
    code_verifier: str = Form(""),
    redirect_uri: str = Form(""),
    refresh_token: str = Form(""),
) -> JSONResponse:
    """Exchange an authorization code (with PKCE) or a refresh token for tokens."""
    _require_issuer()
    if not app_rate_limit.allow(
        f"mcp_token:{client_id}", limit=_TOKEN_LIMIT, window_s=_TOKEN_WINDOW_S
    ):
        raise HTTPException(status_code=429, detail="too many token requests, slow down")
    data_dir = _data_dir(request)

    def _entitled(uid: str) -> bool:
        u = get_user(data_dir, uid)
        return u is not None and u.mcp_access

    if grant_type == "authorization_code":
        result = app_oauth_server.exchange_authorization_code(
            data_dir,
            code=code,
            code_verifier=code_verifier,
            client_id=client_id,
            redirect_uri=redirect_uri,
            is_entitled=_entitled,
        )
    elif grant_type == "refresh_token":
        result = app_oauth_server.refresh_access_token(
            data_dir, refresh_token=refresh_token, client_id=client_id, is_entitled=_entitled
        )
    else:
        raise HTTPException(status_code=400, detail="unsupported grant_type")
    if result is None:
        audit_event(request, "mcp.oauth.token_denied", grant_type=grant_type, client_id=client_id)
        return JSONResponse({"error": "invalid_grant"}, status_code=400)
    audit_event(request, "mcp.oauth.token_issued", grant_type=grant_type, client_id=client_id)
    return JSONResponse(result)
