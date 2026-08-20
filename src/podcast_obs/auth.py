"""Admin-gated remote-transport auth for the observability MCP (#56).

The obs control plane is a SECOND MCP server on the prod edge (`ops.<domain>`), reached by an
external agent. Like the content MCP it does not own the user store — it presents the bearer to
the app's internal verify endpoint (``POST /internal/mcp/verify``) over the tailnet. Unlike the
content MCP, it is **admin-only**: observability exposes prod health, so a listener/creator token
is rejected even if it is otherwise MCP-entitled.

This mirrors ``podcast_scraper.mcp.auth`` deliberately rather than importing it: the obs image
ships ONLY ``src/podcast_obs`` (a tiny, zero-coupling control plane), so it carries its own copy.
Kept small on purpose — the shared contract is the verify endpoint's JSON, not the code.

stdio has no auth (local trust); this only guards the HTTP transport.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_VERIFY_URL_ENV = "APP_MCP_VERIFY_URL"  # e.g. http://api:8000/internal/mcp/verify
_INTERNAL_TOKEN_ENV = "INTERNAL_MCP_TOKEN"
_RESOURCE_URL_ENV = "APP_MCP_RESOURCE_URL"  # this server's public URL (the OAuth 'resource')
_ISSUER_URL_ENV = "APP_MCP_ISSUER_URL"  # the authorization server (the app)
_ALLOWED_ORIGINS_ENV = "APP_MCP_ALLOWED_ORIGINS"  # comma-sep browser Origins (DNS-rebind guard)
_REQUIRE_ADMIN_ENV = "APP_MCP_REQUIRE_ADMIN"  # "true" → admit only role==admin

_PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"
_ADMIN_ROLE = "admin"


def _require_admin() -> bool:
    """Admin gate — SAFE DEFAULT is REQUIRED. Observability is admin-only by design, so only an
    explicit ``false``/``0``/``no`` disables the gate; an unset, empty, or garbled value keeps it
    ON (a fail-open default is how one typo silently exposes prod health to any entitled listener).
    """
    val = os.environ.get(_REQUIRE_ADMIN_ENV, "").strip().lower()
    if val in {"0", "false", "no"}:
        return False
    if val and val not in {"1", "true", "yes"}:
        logger.warning(
            "obs MCP: unrecognized %s=%r — defaulting to admin-required", _REQUIRE_ADMIN_ENV, val
        )
    return True


def _verify_config() -> tuple[str, str]:
    return (
        os.environ.get(_VERIFY_URL_ENV, "").strip(),
        os.environ.get(_INTERNAL_TOKEN_ENV, "").strip(),
    )


def verify_principal(token: str, *, timeout: float = 5.0) -> Optional[dict[str, Any]]:
    """Resolve ``token`` to its principal (``{user_id, role, mcp_access}``) or None.

    Fails **closed** — no token, unconfigured verify URL/secret, a network/HTTP error, an
    ``authenticated: false`` result, a missing entitlement, or (when required) a non-admin role
    all return None. The audience check (RFC 8707) mirrors the content MCP: an aud-bound token
    that this server cannot match to its own ``resource`` is rejected.
    """
    if not token:
        return None
    verify_url, internal_token = _verify_config()
    if not verify_url or not internal_token:
        logger.warning(
            "obs MCP auth not configured (%s / %s)", _VERIFY_URL_ENV, _INTERNAL_TOKEN_ENV
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
        with urllib.request.urlopen(
            req, timeout=timeout
        ) as resp:  # noqa: S310 - fixed internal URL
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        logger.warning("obs MCP token verify failed: %s", exc)
        return None
    if not (isinstance(data, dict) and data.get("authenticated") and data.get("mcp_access")):
        return None
    aud = str(data.get("aud") or "")
    resource = os.environ.get(_RESOURCE_URL_ENV, "").strip().rstrip("/")
    if aud and aud != resource:
        logger.warning("obs MCP token audience mismatch (aud=%s, resource=%s)", aud, resource)
        return None
    role = str(data.get("role") or "")
    if _require_admin() and role != _ADMIN_ROLE:
        logger.warning("obs MCP: non-admin token rejected (role=%s)", role or "<none>")
        return None
    uid = data.get("user_id")
    if not uid:
        return None
    return {"user_id": str(uid), "role": role, "mcp_access": True}


def _header(scope: dict[str, Any], want: bytes) -> Optional[str]:
    for name, value in scope.get("headers") or []:
        if name.lower() == want:
            return str(value.decode("latin-1"))
    return None


def _bearer_from_scope(scope: dict[str, Any]) -> Optional[str]:
    auth = _header(scope, b"authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _origin_allowed(scope: dict[str, Any]) -> bool:
    allowed = os.environ.get(_ALLOWED_ORIGINS_ENV, "").strip()
    if not allowed:
        return True
    origin = _header(scope, b"origin")
    if not origin:
        return True  # server-side clients (claude.ai) send no Origin
    return origin in {o.strip() for o in allowed.split(",") if o.strip()}


def _resource_metadata() -> Optional[dict[str, Any]]:
    """RFC 9728 protected-resource metadata pointing a client at the authorization server."""
    resource = os.environ.get(_RESOURCE_URL_ENV, "").strip().rstrip("/")
    issuer = os.environ.get(_ISSUER_URL_ENV, "").strip().rstrip("/")
    if not resource or not issuer:
        return None
    return {"resource": resource, "authorization_servers": [issuer]}


async def _send_json(
    send: Callable[[dict[str, Any]], Awaitable[None]], status: int, payload: dict[str, Any]
) -> None:
    data = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": data})


async def _send_401(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
    # RFC 9728: point the client at the protected-resource metadata so it can find the AS.
    resource = os.environ.get(_RESOURCE_URL_ENV, "").strip().rstrip("/")
    challenge = b'Bearer realm="mcp"'
    if resource:
        challenge = f'Bearer resource_metadata="{resource}{_PROTECTED_RESOURCE_PATH}"'.encode(
            "latin-1"
        )
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


class ObsAuthMiddleware:
    """Pure-ASGI gate: verify the bearer (admin-only when required), else 401.

    ``verifier`` is injected (defaults to :func:`verify_principal`) so tests substitute it without
    a live app. The RFC 9728 discovery doc is served un-authenticated (a cold client fetches it
    before it has any token). Non-HTTP scopes (lifespan) pass through untouched.
    """

    def __init__(
        self,
        app: Any,
        *,
        verifier: Callable[[str], Optional[dict[str, Any]]] = verify_principal,
    ) -> None:
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
        # verify_principal does a blocking HTTP round-trip — run it OFF the event loop so one
        # slow/hung verify can't stall every other connection on this server.
        principal = None
        if token:
            try:
                from anyio import to_thread

                principal = await to_thread.run_sync(self._verifier, token)
            except Exception:  # noqa: BLE001 - a verifier crash must fail closed, never 500-leak
                logger.warning("obs MCP verify raised; denying", exc_info=True)
                principal = None
        if principal is None:
            await _send_401(send)
            return
        # Attribution: an obs admin can trigger real reads (and, if PODCAST_OBS_ALLOW_WRITES is
        # ever set, writes) against prod backends — log who was admitted (never the token).
        logger.info(
            "obs MCP: admitted user_id=%s role=%s", principal.get("user_id"), principal.get("role")
        )
        await self._app(scope, receive, send)
