"""Admin-gated auth for the observability MCP (#56).

The obs control plane is admin-only: a valid, MCP-entitled listener/creator token is still
rejected when ``APP_MCP_REQUIRE_ADMIN`` is set. The verify round-trip is stubbed; the middleware
is driven directly as an ASGI app with an injected verifier (no network).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import pytest

from podcast_obs import auth

pytestmark = pytest.mark.unit


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def _stub_urlopen(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(auth.urllib.request, "urlopen", lambda req, timeout=5.0: _FakeResp(payload))


def _cfg(monkeypatch: pytest.MonkeyPatch, *, require_admin: bool = False) -> None:
    monkeypatch.setenv("APP_MCP_VERIFY_URL", "http://api:8000/internal/mcp/verify")
    monkeypatch.setenv("INTERNAL_MCP_TOKEN", "tok")
    monkeypatch.setenv("APP_MCP_REQUIRE_ADMIN", "true" if require_admin else "false")


class TestVerifyPrincipal:
    def test_unconfigured_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APP_MCP_VERIFY_URL", raising=False)
        monkeypatch.delenv("INTERNAL_MCP_TOKEN", raising=False)
        assert auth.verify_principal("clp_mcp_x") is None

    def test_empty_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _cfg(monkeypatch)
        assert auth.verify_principal("") is None

    def test_admin_required_admits_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _cfg(monkeypatch, require_admin=True)
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "admin"}',
        )
        assert auth.verify_principal("clp_mcp_x") == {
            "user_id": "u_1",
            "role": "admin",
            "mcp_access": True,
        }

    def test_admin_required_rejects_non_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: an entitled listener/creator must NOT reach observability."""
        _cfg(monkeypatch, require_admin=True)
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "creator"}',
        )
        assert auth.verify_principal("clp_mcp_x") is None

    def test_no_admin_requirement_admits_any_entitled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cfg(monkeypatch, require_admin=False)
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "listener"}',
        )
        assert auth.verify_principal("clp_mcp_x") is not None

    def test_missing_entitlement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _cfg(monkeypatch, require_admin=True)
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": false, "role": "admin"}',
        )
        assert auth.verify_principal("clp_mcp_x") is None

    def test_audience_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _cfg(monkeypatch, require_admin=True)
        monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://ops.example.com")
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "admin",'
            b' "aud": "https://someone-else.example.com"}',
        )
        assert auth.verify_principal("clp_mcp_x") is None

    def test_aud_bound_token_rejected_when_resource_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail closed: an aud-BOUND token but no APP_MCP_RESOURCE_URL to match → deny."""
        _cfg(monkeypatch, require_admin=True)
        monkeypatch.delenv("APP_MCP_RESOURCE_URL", raising=False)
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "admin",'
            b' "aud": "https://ops.example.com"}',
        )
        assert auth.verify_principal("clp_mcp_x") is None

    def test_admin_required_by_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SAFE DEFAULT: an UNSET require-admin env must still gate to admin (no fail-open)."""
        monkeypatch.setenv("APP_MCP_VERIFY_URL", "http://api:8000/internal/mcp/verify")
        monkeypatch.setenv("INTERNAL_MCP_TOKEN", "tok")
        monkeypatch.delenv("APP_MCP_REQUIRE_ADMIN", raising=False)
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "creator"}',
        )
        assert auth.verify_principal("clp_mcp_x") is None

    def test_admin_required_on_garbage_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo (e.g. "ture") must NOT silently disable the admin gate."""
        monkeypatch.setenv("APP_MCP_VERIFY_URL", "http://api:8000/internal/mcp/verify")
        monkeypatch.setenv("INTERNAL_MCP_TOKEN", "tok")
        monkeypatch.setenv("APP_MCP_REQUIRE_ADMIN", "ture")
        _stub_urlopen(
            monkeypatch,
            b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "role": "listener"}',
        )
        assert auth.verify_principal("clp_mcp_x") is None


def _drive(mw: auth.ObsAuthMiddleware, *, path: str, method: str, headers: Optional[list] = None):
    """Run the ASGI middleware once and return the captured send messages."""
    scope = {"type": "http", "path": path, "method": method, "headers": headers or []}
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    asyncio.run(mw(scope, receive, send))
    return sent


def _status(sent: list) -> int:
    return int(next(m["status"] for m in sent if m["type"] == "http.response.start"))


class TestObsAuthMiddleware:
    def test_discovery_doc_is_public(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://ops.example.com")
        monkeypatch.setenv("APP_MCP_ISSUER_URL", "https://app.example.com")
        mw = auth.ObsAuthMiddleware(_unreachable_app, verifier=lambda _t: None)
        sent = _drive(mw, path="/.well-known/oauth-protected-resource", method="GET")
        assert _status(sent) == 200
        body = json.loads(next(m for m in sent if m["type"] == "http.response.body")["body"])
        assert body["resource"] == "https://ops.example.com"
        assert body["authorization_servers"] == ["https://app.example.com"]

    def test_no_token_is_401(self) -> None:
        mw = auth.ObsAuthMiddleware(_unreachable_app, verifier=lambda _t: None)
        sent = _drive(mw, path="/mcp", method="POST")
        assert _status(sent) == 401

    def test_admin_principal_reaches_inner_app(self) -> None:
        called = {"hit": False}

        async def _inner(scope, receive, send):  # noqa: ANN001
            called["hit"] = True
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = auth.ObsAuthMiddleware(_inner, verifier=lambda _t: {"user_id": "u_1", "role": "admin"})
        headers = [(b"authorization", b"Bearer clp_mcp_x")]
        sent = _drive(mw, path="/mcp", method="POST", headers=headers)
        assert called["hit"] is True
        assert _status(sent) == 200

    def test_rejected_principal_never_reaches_inner_app(self) -> None:
        mw = auth.ObsAuthMiddleware(_unreachable_app, verifier=lambda _t: None)
        headers = [(b"authorization", b"Bearer clp_mcp_x")]
        sent = _drive(mw, path="/mcp", method="POST", headers=headers)
        assert _status(sent) == 401


async def _unreachable_app(scope, receive, send):  # noqa: ANN001
    raise AssertionError("inner app must not be reached when auth denies")
