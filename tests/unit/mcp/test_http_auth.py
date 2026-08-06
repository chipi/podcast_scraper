"""Unit tests for the remote MCP auth (RFC-112 slice 2, #1471)."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp import auth

pytestmark = pytest.mark.unit


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_urlopen(monkeypatch, payload: bytes | None, *, raise_exc: Exception | None = None):
    def _open(req, timeout=5.0):
        if raise_exc is not None:
            raise raise_exc
        return _FakeResp(payload or b"")

    monkeypatch.setattr(auth.urllib.request, "urlopen", _open)


def _config(monkeypatch) -> None:
    monkeypatch.setenv("APP_MCP_VERIFY_URL", "http://app/internal/mcp/verify")
    monkeypatch.setenv("INTERNAL_MCP_TOKEN", "tok")


def test_verify_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    assert auth.verify_bearer("") is None


def test_verify_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_VERIFY_URL", raising=False)
    monkeypatch.delenv("INTERNAL_MCP_TOKEN", raising=False)
    assert auth.verify_bearer("clp_mcp_x") is None  # fails closed


def test_verify_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": true, "user_id": "u_1", "mcp_access": true}')
    assert auth.verify_bearer("clp_mcp_x") == "u_1"


def test_verify_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": false}')
    assert auth.verify_bearer("clp_mcp_x") is None


def test_verify_no_entitlement(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": true, "user_id": "u_1", "mcp_access": false}')
    assert auth.verify_bearer("clp_mcp_x") is None


def test_verify_network_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, None, raise_exc=OSError("boom"))
    assert auth.verify_bearer("clp_mcp_x") is None


# --- ASGI middleware ---


def _run_asgi(app, headers: list[tuple[bytes, bytes]]):
    """Drive an ASGI app once for an http scope; return (status, captured_user)."""
    import asyncio

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}
    sent: list[dict] = []
    captured: dict[str, str | None] = {"user": "sentinel"}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    async def inner(s, r, sd):
        captured["user"] = auth.current_mcp_user.get()
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    mw = auth.McpAuthMiddleware(inner, verifier=lambda t: "u_1" if t == "good" else None)
    asyncio.run(mw(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, captured["user"]


def test_middleware_401_without_token() -> None:
    status, user = _run_asgi(_app_unused(), [])
    assert status == 401 and user == "sentinel"  # inner never ran


def test_middleware_401_bad_token() -> None:
    status, _ = _run_asgi(_app_unused(), [(b"authorization", b"Bearer bad")])
    assert status == 401


def test_middleware_passes_and_sets_user() -> None:
    status, user = _run_asgi(_app_unused(), [(b"authorization", b"Bearer good")])
    assert status == 200 and user == "u_1"


def _app_unused():
    return None  # the middleware's inner is provided inside _run_asgi


# --- RFC 9728 protected-resource discovery ---


def _drive(app, scope: dict) -> tuple[int, dict[bytes, bytes], bytes]:
    """Drive an ASGI app once; return (status, headers-dict, body)."""
    import asyncio

    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = {k: v for k, v in start.get("headers", [])}
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], headers, body


def _mw(verifier=lambda t: None):
    async def inner(s, r, sd):
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    return auth.McpAuthMiddleware(inner, verifier=verifier)


def test_protected_resource_metadata_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.setenv("APP_MCP_ISSUER_URL", "https://app.example.com")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/.well-known/oauth-protected-resource",
        "headers": [],
    }
    status, _headers, body = _drive(_mw(), scope)  # no bearer → still served
    assert status == 200
    import json

    meta = json.loads(body)
    assert meta["resource"] == "https://mcp.example.com"
    assert meta["authorization_servers"] == ["https://app.example.com"]


def test_protected_resource_metadata_503_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_RESOURCE_URL", raising=False)
    monkeypatch.delenv("APP_MCP_ISSUER_URL", raising=False)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/.well-known/oauth-protected-resource",
        "headers": [],
    }
    status, _headers, _body = _drive(_mw(), scope)
    assert status == 503


def test_401_carries_resource_metadata_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com")
    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": []}
    status, headers, _body = _drive(_mw(), scope)
    assert status == 401
    www = headers[b"www-authenticate"].decode()
    assert 'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"' in www


def test_origin_guard_blocks_disallowed_browser_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_ALLOWED_ORIGINS", "https://claude.ai")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"origin", b"https://evil.example"), (b"authorization", b"Bearer good")],
    }
    status, _h, _b = _drive(_mw(lambda t: "u_1"), scope)  # even a valid bearer is rejected
    assert status == 403


def test_origin_guard_allows_listed_origin_and_serverside_no_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_MCP_ALLOWED_ORIGINS", "https://claude.ai")
    listed = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"origin", b"https://claude.ai"), (b"authorization", b"Bearer good")],
    }
    no_origin = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", b"Bearer good")],  # server-to-server sends no Origin
    }
    assert _drive(_mw(lambda t: "u_1"), listed)[0] == 200
    assert _drive(_mw(lambda t: "u_1"), no_origin)[0] == 200
