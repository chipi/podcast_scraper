"""Remote-transport auth for the corpus MCP server (RFC-112 slice 2, #1471).

The HTTP transport authenticates every connection as a platform user. The MCP server does **not**
mount the app's user store — it presents the bearer token to the app's internal verify endpoint
(``POST /internal/mcp/verify``, RFC-112 §4) over the tailnet, keeping the store app-owned. stdio has
no auth (local trust); this only guards the HTTP transport.

Two pieces:
- :func:`verify_bearer` — resolve a token → ``VerifiedToken`` (or None) via the app endpoint.
- :class:`McpAuthMiddleware` — a pure-ASGI wrapper that gates the HTTP app: it extracts the bearer,
  verifies it, sets the authenticated ``user_id`` AND its granted scopes in contextvars, and
  returns 401 otherwise.
- :func:`require_scope` — what a tool calls to enforce the scope it needs (#1916).

Scope was plumbed end to end and then thrown away: the app's verify endpoint has always returned
the token's granted ``scope``, and this module never read it. Every tool therefore ran on nothing
but the ``mcp_access`` entitlement — including ``reenrich`` and ``reindex``, which are corpus
WRITES. Any entitled user's agent could trigger a reindex with a read-only token.
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

#: Scopes granted to the current request's token.
#:
#: ``None`` means "no HTTP auth context at all" — i.e. stdio, which is local-trust by design and
#: has no token to carry a scope. An empty frozenset means "an HTTP request whose token granted
#: nothing", which must be REFUSED. The distinction is the whole reason this is not just a set:
#: collapsing them would either break local use or silently open the remote surface.
current_mcp_scopes: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "current_mcp_scopes", default=None
)

#: The only scope the authorization server mints today (``app_oauth_server._SUPPORTED_SCOPES``).
SCOPE_READ = "mcp:read"
#: Required by corpus-mutating tools. NOT mintable yet, and that is deliberate: until there is a
#: recorded decision to grant it, a remote agent should not be able to mutate the corpus at all.
SCOPE_WRITE = "mcp:write"


class McpScopeError(PermissionError):
    """A tool was called without the scope it requires."""


def require_scope(scope: str) -> None:
    """Refuse unless the current request's token granted ``scope``.

    stdio (``current_mcp_scopes`` unset) passes: it has no token, no transport auth, and is
    local-trust by design — the same reasoning that lets it run unauthenticated at all.
    """
    granted = current_mcp_scopes.get()
    if granted is None:
        return
    if scope not in granted:
        raise McpScopeError(
            f"this tool requires the '{scope}' scope; the token presented granted "
            f"{sorted(granted) or 'none'}"
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


def verify_bearer(token: str, *, timeout: float = 5.0) -> tuple[str, frozenset[str]] | None:
    """Resolve ``token`` to ``(user_id, granted_scopes)`` via the app's verify endpoint, or None.

    None on: no token, unconfigured verify URL/secret, a network/HTTP error, or an
    ``authenticated: false`` result. Fails **closed** — any uncertainty denies.

    The scopes come from the app, which has always returned them; this server simply never looked
    (#1916). A token that grants nothing authenticates but authorises nothing.
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
        # Audience check (RFC 8707). A PAT (empty aud) is user-scoped and skips this. An aud-BOUND
        # token requires this server to know its own resource: if it's bound but we have no
        # APP_MCP_RESOURCE_URL to compare, FAIL CLOSED (a resource server that can't identify itself
        # must not accept a token minted for some resource; M1) — and of course reject a mismatch.
        aud = str(data.get("aud") or "")
        resource = os.environ.get(_RESOURCE_URL_ENV, "").strip().rstrip("/")
        if aud and aud != resource:
            logger.warning("MCP token audience mismatch (aud=%s, resource=%s)", aud, resource)
            return None
        uid = data.get("user_id")
        if not uid:
            return None
        return (str(uid), _parse_scopes(data.get("scope")))
    return None


def _parse_scopes(raw: Any) -> frozenset[str]:
    """OAuth scope strings are space-delimited (RFC 6749 §3.3); tolerate commas and junk."""
    if not isinstance(raw, str):
        return frozenset()
    return frozenset(part for part in raw.replace(",", " ").split() if part)


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

    def __init__(self, app: Any, *, verifier: Callable[[str], Any] = verify_bearer) -> None:
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
        verified = await to_thread.run_sync(self._verifier, token) if token else None
        if verified is None:
            await _send_401(send)
            return
        # A verifier may still return a bare user_id (older injected doubles, and any caller that
        # only needs identity). Treat that as "authenticated, no scopes" rather than crashing —
        # and note that no-scopes is the CLOSED state: every scoped tool then refuses.
        if isinstance(verified, tuple):
            user_id, scopes = verified
        else:
            user_id, scopes = str(verified), frozenset()
        tok = current_mcp_user.set(user_id)
        scope_tok = current_mcp_scopes.set(scopes)
        try:
            await self._app(scope, receive, send)
        finally:
            current_mcp_user.reset(tok)
            current_mcp_scopes.reset(scope_tok)
