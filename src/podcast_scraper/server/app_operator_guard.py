"""Authz + audit for the operator API (#1071 epic #911; admin gating #1128).

The operator API — ``/api/feeds``, ``/api/operator-config``, ``/api/ops``, ``/api/jobs*``,
``/api/scheduled-jobs``, ``/api/enrichment/config``, ``/api/index/rebuild`` — backs the viewer's
**admin-only** surfaces (Dashboard, Ops, Configuration). These are network-gated (Tailscale,
RFC-082) and this middleware adds an application-layer gate on top:

- **Access rule (read *and* write):** allow when the request carries a valid **admin session**
  (the shared ``lp_session`` cookie → ``role == admin``) **OR** a valid operator **key**
  (``X-Operator-Key`` matching ``APP_OPERATOR_API_KEY``). Otherwise 403. Either credential grants
  access, so browser admins use their session and headless automation can use the key.
- **Enforced only when it can be:** the gate activates when platform auth is configured (a session
  secret + per-user data dir) **or** a key is set. On a bare deployment with neither (the legacy
  Tailscale-only posture), the API keeps its prior network-only behavior — no lockout on upgrade.
- Every *mutating* operator request is appended to the audit log (best-effort).

Consumer routes (``/api/app/*``) have their own auth and are never gated here.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from podcast_scraper.server import app_roles, app_sessions
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_user_store import get_user

_WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

#: Operator API base paths — gated for *all* methods (reads included). A request matches when its
#: path equals a base or is a sub-path of it (``/api/jobs`` → ``/api/jobs/enrichment`` too).
_OPERATOR_BASES = (
    "/api/feeds",
    "/api/operator-config",
    "/api/index/rebuild",
    "/api/ops",
    "/api/jobs",
    "/api/scheduled-jobs",
    # Cover the WHOLE /api/enrichment namespace, not just /config — the
    # status/health/metrics/events GETs + the health re-enable POST are operator
    # surface too, and were unguarded (gated only on the internal
    # jobs_api_enabled flag). Whole-codebase review 2026-07-17 (H5).
    "/api/enrichment",
)


def is_operator_path(path: str) -> bool:
    """True for any request (read or write) to an operator API endpoint."""
    return any(path == base or path.startswith(base + "/") for base in _OPERATOR_BASES)


def is_operator_write(method: str, path: str) -> bool:
    """True for mutating requests to operator endpoints (drives audit + back-compat callers)."""
    return method.upper() in _WRITE_METHODS and is_operator_path(path)


def _valid_key(request: Request, key: str) -> bool:
    """True when a configured operator key matches the request's ``X-Operator-Key`` header."""
    return bool(key) and hmac.compare_digest(request.headers.get("x-operator-key", ""), key)


def _is_admin_session(request: Request, secret: str, data_dir: object) -> bool:
    """True when the request's session cookie resolves to an enabled ``admin`` user."""
    if not secret or data_dir is None:
        return False
    payload = app_sessions.verify(request.cookies.get(app_sessions.SESSION_COOKIE), secret)
    user_id = payload.get("user_id") if payload else None
    if not user_id:
        return False
    user = get_user(Path(str(data_dir)), str(user_id))
    return user is not None and not user.disabled and app_roles.is_admin(user.role)


class OperatorWriteGuard(BaseHTTPMiddleware):
    """Admin-session-or-key gate + audit for the operator API (RFC-082; #1128)."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Gate operator reads+writes on admin-session-or-key (when enforceable); audit writes."""
        path = request.url.path
        if not is_operator_path(path):
            return await call_next(request)

        state = request.app.state
        key = getattr(state, "operator_api_key", "") or ""
        secret = getattr(state, "session_secret", "") or ""
        data_dir = getattr(state, "app_data_dir", None)
        audit_path = getattr(state, "audit_path", None)
        is_write = request.method.upper() in _WRITE_METHODS
        base = {"method": request.method, "path": path, "actor": "operator"}

        # Enforce only when a credential could exist: platform auth configured, or a key set.
        enforce = bool(secret and data_dir is not None) or bool(key)
        if enforce and not (
            _valid_key(request, key) or _is_admin_session(request, secret, data_dir)
        ):
            if is_write:
                append_audit(audit_path, {**base, "outcome": "denied"})
            return JSONResponse(status_code=403, content={"detail": "Admin access required."})

        if is_write:
            append_audit(audit_path, {**base, "outcome": "allowed", "enforced": enforce})
        return await call_next(request)
