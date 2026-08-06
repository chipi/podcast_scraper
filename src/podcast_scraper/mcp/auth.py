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

from anyio import to_thread

logger = logging.getLogger(__name__)

#: The authenticated platform user for the current MCP request (None outside a request / stdio).
current_mcp_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_mcp_user", default=None
)

_VERIFY_URL_ENV = "APP_MCP_VERIFY_URL"  # e.g. http://app.internal:8000/internal/mcp/verify
_INTERNAL_TOKEN_ENV = "INTERNAL_MCP_TOKEN"
_RESOURCE_URL_ENV = "APP_MCP_RESOURCE_URL"  # this MCP server's public URL (the OAuth 'resource')
_ISSUER_URL_ENV = "APP_MCP_ISSUER_URL"  # the authorization server (the app)
_ALLOWED_ORIGINS_ENV = (
    "APP_MCP_ALLOWED_ORIGINS"  # comma-sep browser Origins allowed (DNS-rebind guard)
)

_PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"


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


def _header(scope: dict[str, Any], want: bytes) -> str | None:
    for name, value in scope.get("headers") or []:
        if name == want:
            return str(value.decode("latin-1"))
    return None


def _origin_allowed(scope: dict[str, Any]) -> bool:
    """DNS-rebinding guard (MCP spec): reject a browser `Origin` outside the allowlist.

    Server-to-server clients (claude.ai's connector) send no `Origin`, so a missing header always
    passes. When ``APP_MCP_ALLOWED_ORIGINS`` is unset we do not gate on Origin at all (the public
    TLS deployment relies on the bearer + TLS); set it to lock the browser surface down.
    """
    origin = _header(scope, b"origin")
    if origin is None:
        return True
    allowed = os.environ.get(_ALLOWED_ORIGINS_ENV, "").strip()
    if not allowed:
        return True
    return origin in {o.strip() for o in allowed.split(",") if o.strip()}


def _resource_metadata() -> dict[str, Any] | None:
    """RFC 9728 protected-resource metadata pointing a client at the authorization server."""
    resource = os.environ.get(_RESOURCE_URL_ENV, "").strip().rstrip("/")
    issuer = os.environ.get(_ISSUER_URL_ENV, "").strip().rstrip("/")
    if not resource or not issuer:
        return None
    return {"resource": resource, "authorization_servers": [issuer]}


async def _send_json(
    send: Callable[[dict[str, Any]], Awaitable[None]], status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_401(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
    # RFC 9728: point the client at the protected-resource metadata so it can discover the AS.
    resource = os.environ.get(_RESOURCE_URL_ENV, "").strip().rstrip("/")
    challenge = b'Bearer realm="mcp"'
    if resource:
        meta_url = f"{resource}{_PROTECTED_RESOURCE_PATH}"
        challenge = f'Bearer resource_metadata="{meta_url}"'.encode("latin-1")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", challenge),
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
        # RFC 9728 protected-resource metadata is PUBLIC — a client fetches it (unauthenticated)
        # to discover the authorization server before it has any token. Serve it before the gate.
        if scope.get("path") == _PROTECTED_RESOURCE_PATH and scope.get("method") == "GET":
            meta = _resource_metadata()
            if meta is None:
                await _send_json(send, 503, {"error": "mcp oauth not configured"})
            else:
                await _send_json(send, 200, meta)
            return
        if not _origin_allowed(scope):
            await _send_json(send, 403, {"error": "origin not allowed"})
            return
        token = _bearer_from_scope(scope)
        # verify_bearer does a blocking HTTP round-trip — run it OFF the event loop so one slow/hung
        # verify (or the app being down) can't stall every other MCP connection on this server.
        user_id = await to_thread.run_sync(self._verifier, token) if token else None
        if user_id is None:
            await _send_401(send)
            return
        tok = current_mcp_user.set(user_id)
        try:
            await self._app(scope, receive, send)
        finally:
            current_mcp_user.reset(tok)
