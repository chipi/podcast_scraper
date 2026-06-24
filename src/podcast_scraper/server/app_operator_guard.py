"""Authz + audit for operator write endpoints (#1071, epic #911).

The operator API (``/api/feeds``, ``/api/operator-config``, ``/api/jobs*``,
``/api/index/rebuild``) is network-gated (Tailscale, RFC-082) and otherwise unauthenticated.
This middleware adds **optional** API-key auth on its *mutating* routes plus an audit trail:

- When ``APP_OPERATOR_API_KEY`` is set, mutating operator requests must carry a matching
  ``X-Operator-Key`` header (else 401). When unset, the key check is skipped
  (backward-compatible with the Tailscale-only model).
- Every mutating operator request is appended to the audit log (best-effort).

Consumer routes (``/api/app/*``) have their own auth and are never gated here.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from podcast_scraper.server.app_audit import append_audit

_WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
_WRITE_EXACT = {"/api/feeds", "/api/operator-config", "/api/index/rebuild"}
_WRITE_PREFIXES = ("/api/jobs",)


def is_operator_write(method: str, path: str) -> bool:
    """True for mutating requests to operator write endpoints (not consumer ``/api/app``)."""
    if method.upper() not in _WRITE_METHODS:
        return False
    if path in _WRITE_EXACT:
        return True
    return any(path == prefix or path.startswith(prefix + "/") for prefix in _WRITE_PREFIXES)


class OperatorWriteGuard(BaseHTTPMiddleware):
    """API-key gate + audit for mutating operator routes."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Gate (if a key is set) + audit mutating operator requests; pass the rest through."""
        if not is_operator_write(request.method, request.url.path):
            return await call_next(request)

        state = request.app.state
        key = getattr(state, "operator_api_key", "") or ""
        audit_path = getattr(state, "audit_path", None)
        base = {"method": request.method, "path": request.url.path, "actor": "operator"}

        if key and not hmac.compare_digest(request.headers.get("x-operator-key", ""), key):
            append_audit(audit_path, {**base, "outcome": "denied"})
            return JSONResponse(status_code=401, content={"detail": "Operator API key required."})

        append_audit(audit_path, {**base, "outcome": "allowed", "authenticated": bool(key)})
        return await call_next(request)
