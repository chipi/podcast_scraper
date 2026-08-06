"""Internal MCP verify seam — RFC-112 §4 (#1471).

The MCP server process (a separate process, the transport half of slice 2) presents a bearer token
here to resolve it to a platform user + entitlement, rather than mounting the app's per-user store
directly. Mounted under ``/internal`` (service-to-service, tailnet-only), gated by the shared
``INTERNAL_MCP_TOKEN`` (``X-Internal-Token`` header) — same pattern as the outbox seam. Unconfigured
→ 503; wrong token → 401.

Verification is O(1) via the token index (``app_mcp_tokens``). A token that resolves to a user who
no longer holds ``mcp_access`` returns ``authenticated: false`` — the entitlement is checked at
connect time, not only at creation, so a revoked grant takes effect immediately.
"""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from podcast_scraper.server import app_mcp_tokens, app_oauth_server
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_user_store import get_user
from podcast_scraper.server.schemas import McpVerifyBody, McpVerifyResponse

router = APIRouter(tags=["internal"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def require_internal_mcp_token(
    request: Request, x_internal_token: str | None = Header(default=None)
) -> None:
    """Gate on the shared ``INTERNAL_MCP_TOKEN``. 503 when unconfigured, 401 on mismatch."""
    configured = getattr(request.app.state, "internal_mcp_token", "") or ""
    if not configured:
        raise HTTPException(status_code=503, detail="internal mcp verify not configured")
    if not hmac.compare_digest(x_internal_token or "", configured):
        raise HTTPException(status_code=401, detail="invalid internal token")


@router.post(
    "/mcp/verify",
    response_model=McpVerifyResponse,
    dependencies=[Depends(require_internal_mcp_token)],
)
async def verify(request: Request, body: McpVerifyBody) -> McpVerifyResponse:
    """Resolve a presented MCP bearer token to its user + live entitlement.

    Accepts **both** auth paths: a PAT (slice 1) or an OAuth 2.1 access token (slice 3). Either
    resolves to a ``user_id``; the ``mcp_access`` entitlement is then re-checked live.
    """
    data_dir = _data_dir(request)
    # A PAT carries the full read scope; an OAuth access token carries its granted scope.
    scope = "mcp:read"
    user_id = app_mcp_tokens.verify_token(data_dir, body.token)
    if user_id is None:
        oauth = app_oauth_server.verify_access_token(data_dir, body.token)
        if oauth is not None:
            user_id = oauth["user_id"]
            scope = str(oauth.get("scope", "mcp:read"))
    audit_path = getattr(request.app.state, "audit_path", None)
    if user_id is None:
        # A presented token that resolved to nothing (unknown / expired / malformed). Audit the
        # DENIAL (the security signal) — never the token itself. Successes are NOT audited here
        # (every tool call verifies → too noisy); credential issuance/revocation is audited instead.
        append_audit(audit_path, {"event": "mcp.auth.denied", "reason": "unresolved_token"})
        return McpVerifyResponse(authenticated=False)
    user = get_user(data_dir, user_id)
    if user is None or not user.mcp_access:
        # Token valid but the entitlement was revoked (or the user is gone) → deny at connect time.
        append_audit(
            audit_path,
            {"event": "mcp.auth.denied", "reason": "entitlement_revoked", "user_id": user_id},
        )
        return McpVerifyResponse(authenticated=False, user_id=user_id, mcp_access=False)
    return McpVerifyResponse(authenticated=True, user_id=user_id, mcp_access=True, scope=scope)
