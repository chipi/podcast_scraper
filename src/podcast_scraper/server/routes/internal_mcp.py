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

from podcast_scraper.server import app_mcp_tokens, app_oauth_server, app_rate_limit
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_user_store import get_user
from podcast_scraper.server.schemas import McpVerifyBody, McpVerifyResponse

# Cap denial-audit writes: hammering public /mcp with junk bearers triggers a verify denial per
# request; uncapped that is unbounded disk write-amplification (M4). Past this rate the denials are
# dropped from the audit (the edge fail2ban jail handles the flood itself).
_DENIAL_AUDIT_LIMIT, _DENIAL_AUDIT_WINDOW_S = 60, 60.0


def _audit_denial(audit_path: Path | None, **fields: object) -> None:
    if app_rate_limit.allow(
        "mcp_verify_denied", limit=_DENIAL_AUDIT_LIMIT, window_s=_DENIAL_AUDIT_WINDOW_S
    ):
        append_audit(audit_path, {"event": "mcp.auth.denied", **fields})


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
    # A PAT carries the full read scope + no audience; an OAuth access token carries its granted
    # scope + the `aud` it was issued for (the resource server checks aud at its boundary).
    scope = "mcp:read"
    aud = ""
    user_id = app_mcp_tokens.verify_token(data_dir, body.token)
    if user_id is None:
        oauth = app_oauth_server.verify_access_token(data_dir, body.token)
        if oauth is not None:
            user_id = oauth["user_id"]
            scope = str(oauth.get("scope", "mcp:read"))
            aud = str(oauth.get("aud", ""))
    audit_path = getattr(request.app.state, "audit_path", None)
    if user_id is None:
        # A presented token that resolved to nothing (unknown / expired / malformed). Audit the
        # DENIAL (the security signal, throttled) — never the token. Successes are NOT audited here
        # (every call verifies → noisy); issuance/revocation is audited elsewhere.
        _audit_denial(audit_path, reason="unresolved_token")
        return McpVerifyResponse(authenticated=False)
    user = get_user(data_dir, user_id)
    if user is None or not user.mcp_access:
        # Token valid but the entitlement was revoked (or the user is gone) → deny at connect time.
        _audit_denial(audit_path, reason="entitlement_revoked", user_id=user_id)
        return McpVerifyResponse(authenticated=False, user_id=user_id, mcp_access=False)
    # Surface the role so a rank-scoped MCP server (e.g. the observability MCP, admin-only #56)
    # can gate on it. The content MCP ignores it — mcp_access alone is its gate.
    return McpVerifyResponse(
        authenticated=True, user_id=user_id, mcp_access=True, scope=scope, aud=aud, role=user.role
    )
