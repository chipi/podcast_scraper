"""MCP personal-access token management — RFC-112 slice 1 (#1471).

Auth-gated by the logged-in session AND the `mcp_access` entitlement: a user without the grant sees
403 (and, in the UI, no MCP section). The human manages their tokens from a session here; the token
itself is what their agent presents to the MCP server. The plaintext is returned exactly once on
create.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from podcast_scraper.server import app_mcp_tokens, app_oauth_server
from podcast_scraper.server.app_audit import audit_event
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    McpConnection,
    McpConnectionConfig,
    McpConnectionsResponse,
    McpTokenCreate,
    McpTokenCreated,
    McpTokenMeta,
    McpTokensResponse,
)

router = APIRouter(tags=["app"])

# The FastMCP Streamable-HTTP mount path (mcp.server.fastmcp default). The connector URL a client
# pastes is the resource origin + this path.
_MCP_ENDPOINT_PATH = "/mcp"


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def require_mcp_access(user: User = Depends(get_current_user)) -> User:
    """Gate on the MCP entitlement — 403 for a logged-in user who lacks it."""
    if not user.mcp_access:
        raise HTTPException(status_code=403, detail="mcp access not granted")
    return user


@router.get("/mcp/config", response_model=McpConnectionConfig)
async def connection_config(user: User = Depends(require_mcp_access)) -> McpConnectionConfig:
    """The connector wiring the 'Connected agents' UI shows — connector URL + OAuth status.

    Values come from deploy-time env: ``APP_MCP_RESOURCE_URL`` (the public MCP server ORIGIN) and
    ``APP_MCP_ISSUER_URL`` (the OAuth authorization server). Either unset → null + the UI hides that
    affordance. The connector URL a client pastes is the origin + the MCP
    endpoint path (``/mcp``, the FastMCP Streamable-HTTP mount) — the RFC 9728 discovery doc + the
    ``resource`` identifier stay on the origin.
    """
    origin = os.environ.get("APP_MCP_RESOURCE_URL", "").strip().rstrip("/") or None
    issuer = os.environ.get("APP_MCP_ISSUER_URL", "").strip().rstrip("/") or None
    return McpConnectionConfig(
        connector_url=f"{origin}{_MCP_ENDPOINT_PATH}" if origin else None,
        authorization_server=issuer,
        oauth_enabled=bool(issuer),
    )


@router.get("/mcp/tokens", response_model=McpTokensResponse)
async def list_tokens(
    request: Request, user: User = Depends(require_mcp_access)
) -> McpTokensResponse:
    """The user's MCP tokens (metadata only — the secret is never returned after creation)."""
    rows = app_mcp_tokens.list_tokens(_data_dir(request), user.user_id)
    return McpTokensResponse(items=[McpTokenMeta(**r) for r in rows])


@router.post("/mcp/tokens", response_model=McpTokenCreated, status_code=201)
async def create_token(
    request: Request, body: McpTokenCreate, user: User = Depends(require_mcp_access)
) -> McpTokenCreated:
    """Mint a token; the plaintext is returned ONCE here and never again."""
    plaintext, meta = app_mcp_tokens.create_token(_data_dir(request), user.user_id, body.label)
    audit_event(
        request, "mcp.pat.created", user_id=user.user_id, token_id=meta["id"], label=meta["label"]
    )
    return McpTokenCreated(token=plaintext, meta=McpTokenMeta(**meta))


@router.delete("/mcp/tokens/{token_id}", response_model=McpTokensResponse)
async def revoke_token(
    request: Request, token_id: str, user: User = Depends(require_mcp_access)
) -> McpTokensResponse:
    """Revoke a token by id; returns the remaining tokens."""
    app_mcp_tokens.revoke_token(_data_dir(request), user.user_id, token_id)
    audit_event(request, "mcp.pat.revoked", user_id=user.user_id, token_id=token_id)
    rows = app_mcp_tokens.list_tokens(_data_dir(request), user.user_id)
    return McpTokensResponse(items=[McpTokenMeta(**r) for r in rows])


@router.get("/mcp/connections", response_model=McpConnectionsResponse)
async def list_connections(
    request: Request, user: User = Depends(require_mcp_access)
) -> McpConnectionsResponse:
    """The OAuth agents (claude.ai etc.) the user has connected — for the 'Connected agents' UI."""
    rows = app_oauth_server.list_consents(_data_dir(request), user.user_id)
    return McpConnectionsResponse(items=[McpConnection(**r) for r in rows])


@router.delete("/mcp/connections/{client_id}", response_model=McpConnectionsResponse)
async def revoke_connection(
    request: Request, client_id: str, user: User = Depends(require_mcp_access)
) -> McpConnectionsResponse:
    """Disconnect an OAuth agent: forget the consent AND drop its live access/refresh tokens.

    A full disconnect (not just re-consent-on-next-authorize) — the agent's current session dies at
    its next tool call (the dropped access token fails the verify seam).
    """
    data_dir = _data_dir(request)
    app_oauth_server.revoke_consent(data_dir, user_id=user.user_id, client_id=client_id)
    dropped = app_oauth_server.revoke_client_grants(
        data_dir, user_id=user.user_id, client_id=client_id
    )
    audit_event(
        request, "mcp.consent.revoked", user_id=user.user_id, client_id=client_id, grants=dropped
    )
    rows = app_oauth_server.list_consents(data_dir, user.user_id)
    return McpConnectionsResponse(items=[McpConnection(**r) for r in rows])
