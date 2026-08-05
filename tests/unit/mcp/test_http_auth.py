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
