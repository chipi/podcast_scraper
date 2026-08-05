"""Remote-transport auth for the corpus MCP server (RFC-112 slice 2, #1471).

The HTTP transport authenticates every connection as a platform user. The MCP server does **not**
mount the app's user store — it presents the bearer token to the app's internal verify endpoint
(``POST /internal/mcp/verify``, RFC-112 §4) over the tailnet, keeping the store app-owned. stdio has
no auth (local trust); this only guards the HTTP transport.

Two pieces:
- :func:`verify_bearer` — resolve a token → ``user_id`` (or None) via the app endpoint.
- :class:`McpAuthMiddleware` — a pure-ASGI wrapper that gates the HTTP app: it extracts the bearer,
  verifies it, sets the authenticated ``user_id`` in a contextvar (for attribution — v1 serves the
  shared corpus, so this gates + attributes, it does not yet scope), and returns 401 otherwise.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: The authenticated platform user for the current MCP request (None outside a request / stdio).
current_mcp_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_mcp_user", default=None
)

_VERIFY_URL_ENV = "APP_MCP_VERIFY_URL"  # e.g. http://app.internal:8000/internal/mcp/verify
_INTERNAL_TOKEN_ENV = "INTERNAL_MCP_TOKEN"


def _verify_config() -> tuple[str, str]:
    return (
        os.environ.get(_VERIFY_URL_ENV, "").strip(),
        os.environ.get(_INTERNAL_TOKEN_ENV, "").strip(),
    )


def verify_bearer(token: str, *, timeout: float = 5.0) -> str | None:
    """Resolve ``token`` to a ``user_id`` via the app's internal verify endpoint, or None.

    None on: no token, unconfigured verify URL/secret, a network/HTTP error, or an
    ``authenticated: false`` result. Fails **closed** — any uncertainty denies.
    """
    if not token:
        return None
    verify_url, internal_token = _verify_config()
    if not verify_url or not internal_token:
        logger.warning(
            "MCP HTTP auth not configured (%s / %s)", _VERIFY_URL_ENV, _INTERNAL_TOKEN_ENV
        )
        return None
    body = json.dumps({"token": token}).encode("utf-8")
    req = urllib.request.Request(
        verify_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-Internal-Token": internal_token},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        logger.warning("MCP token verify failed: %s", exc)
        return None
    if isinstance(data, dict) and data.get("authenticated") and data.get("mcp_access"):
        uid = data.get("user_id")
        return str(uid) if uid else None
    return None


def _bearer_from_scope(scope: dict[str, Any]) -> str | None:
    for name, value in scope.get("headers") or []:
        if name == b"authorization":
            raw = value.decode("latin-1")
            if raw.lower().startswith("bearer "):
                return str(raw[7:].strip())
    return None


async def _send_401(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="mcp"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})


class McpAuthMiddleware:
    """Pure-ASGI gate: verify the bearer, set ``current_mcp_user``, else 401.

    ``verifier`` is injected (defaults to :func:`verify_bearer`) so tests substitute it without a
    live app. Non-HTTP scopes (lifespan) pass through untouched.
    """

    def __init__(self, app: Any, *, verifier: Callable[[str], str | None] = verify_bearer) -> None:
        self._app = app
        self._verifier = verifier

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        token = _bearer_from_scope(scope)
        user_id = self._verifier(token) if token else None
        if user_id is None:
            await _send_401(send)
            return
        tok = current_mcp_user.set(user_id)
        try:
            await self._app(scope, receive, send)
        finally:
            current_mcp_user.reset(tok)
