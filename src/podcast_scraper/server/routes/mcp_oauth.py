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
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from podcast_scraper.server import app_oauth_server
from podcast_scraper.server.routes.app_auth import get_current_user

router = APIRouter(tags=["app"])
wellknown_router = APIRouter(tags=["app"])


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


def _validate_authorize(request: Request, client_id: str, redirect_uri: str, method: str) -> None:
    client = app_oauth_server.get_client(_data_dir(request), client_id)
    if client is None:
        raise HTTPException(status_code=400, detail="unknown client_id")
    if redirect_uri not in client["redirect_uris"]:
        raise HTTPException(
            status_code=400, detail="redirect_uri not registered"
        )  # anti open-redirect
    if method != "S256":
        raise HTTPException(status_code=400, detail="code_challenge_method must be S256")


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
) -> HTMLResponse:
    """The consent screen (session + mcp_access gated); a GET never issues a code."""
    _require_issuer()
    user = get_current_user(request)  # raises 401 when not logged in
    if not user.mcp_access:
        raise HTTPException(status_code=403, detail="mcp access not granted")
    if response_type != "code":
        raise HTTPException(status_code=400, detail="response_type must be code")
    _validate_authorize(request, client_id, redirect_uri, code_challenge_method)
    client = app_oauth_server.get_client(_data_dir(request), client_id)
    assert client is not None  # _validate_authorize raised otherwise
    fields = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "scope": scope,
        "state": state,
    }
    hidden = "".join(
        f'<input type=hidden name="{html.escape(k)}" value="{html.escape(v)}">'
        for k, v in fields.items()
    )
    name = html.escape(str(client["client_name"]))
    page = (
        "<!doctype html><html lang=en><meta charset=utf-8><meta name=robots content=noindex>"
        "<title>Authorize</title><body style='font-family:system-ui;max-width:32rem;margin:4rem "
        "auto;padding:0 1rem'>"
        f"<h1>Allow {name} to access your closelistening corpus?</h1>"
        "<p>It will be able to search and read your podcast knowledge base as you.</p>"
        f"<form method=post action='/api/app/mcp/oauth/authorize'>{hidden}"
        "<button type=submit style='padding:.6rem 1.2rem;font-size:1rem'>Allow</button></form>"
        "</body></html>"
    )
    return HTMLResponse(page)


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
    """Consent approved → mint a code + redirect back to the client (RFC-8252/OAuth 2.1)."""
    _require_issuer()
    user = get_current_user(request)
    if not user.mcp_access:
        raise HTTPException(status_code=403, detail="mcp access not granted")
    _validate_authorize(request, client_id, redirect_uri, code_challenge_method)
    code = app_oauth_server.create_authorization_code(
        _data_dir(request),
        user_id=user.user_id,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        scope=scope,
    )
    params = {"code": code}
    if state:
        params["state"] = state
    sep = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(url=f"{redirect_uri}{sep}{urlencode(params)}", status_code=302)


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
    data_dir = _data_dir(request)
    if grant_type == "authorization_code":
        result = app_oauth_server.exchange_authorization_code(
            data_dir,
            code=code,
            code_verifier=code_verifier,
            client_id=client_id,
            redirect_uri=redirect_uri,
        )
    elif grant_type == "refresh_token":
        result = app_oauth_server.refresh_access_token(
            data_dir, refresh_token=refresh_token, client_id=client_id
        )
    else:
        raise HTTPException(status_code=400, detail="unsupported grant_type")
    if result is None:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)
    return JSONResponse(result)
