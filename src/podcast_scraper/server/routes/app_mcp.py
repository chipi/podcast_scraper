"""MCP personal-access token management — RFC-112 slice 1 (#1471).

Auth-gated by the logged-in session AND the `mcp_access` entitlement: a user without the grant sees
403 (and, in the UI, no MCP section). The human manages their tokens from a session here; the token
itself is what their agent presents to the MCP server. The plaintext is returned exactly once on
create.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from podcast_scraper.server import app_mcp_tokens
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    McpTokenCreate,
    McpTokenCreated,
    McpTokenMeta,
    McpTokensResponse,
)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def require_mcp_access(user: User = Depends(get_current_user)) -> User:
    """Gate on the MCP entitlement — 403 for a logged-in user who lacks it."""
    if not user.mcp_access:
        raise HTTPException(status_code=403, detail="mcp access not granted")
    return user


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
    return McpTokenCreated(token=plaintext, meta=McpTokenMeta(**meta))


@router.delete("/mcp/tokens/{token_id}", response_model=McpTokensResponse)
async def revoke_token(
    request: Request, token_id: str, user: User = Depends(require_mcp_access)
) -> McpTokensResponse:
    """Revoke a token by id; returns the remaining tokens."""
    app_mcp_tokens.revoke_token(_data_dir(request), user.user_id, token_id)
    rows = app_mcp_tokens.list_tokens(_data_dir(request), user.user_id)
    return McpTokensResponse(items=[McpTokenMeta(**r) for r in rows])
