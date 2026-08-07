"""The wired-up HTTP transport actually serves the public discovery doc + gates unauth (RFC-112).

`build_http_app` = the auth middleware wrapping FastMCP's Streamable-HTTP app. Both paths under test
are short-circuited by the middleware BEFORE it delegates to the inner MCP app, so they need neither
a bearer nor the session-manager lifespan — which is exactly why they're safe to assert here. This
closes the gap where only a manual local smoke covered the deployed serving path.
"""

from __future__ import annotations

import asyncio

import pytest

# The MCP SDK is a [dev] extra, guaranteed present in the unit env (like test_server.py).
pytestmark = pytest.mark.unit

_CORPUS = "tests/fixtures/app-validation-corpus"


def _drive(app, scope: dict) -> tuple[int, dict[bytes, bytes], bytes]:
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


def test_http_app_serves_discovery_and_gates_unauth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.setenv("APP_MCP_ISSUER_URL", "https://app.example.com")
    from podcast_scraper.mcp.server import build_http_app

    app = build_http_app(_CORPUS)

    # Public RFC 9728 discovery — served WITHOUT a bearer.
    status, _h, body = _drive(
        app,
        {
            "type": "http",
            "method": "GET",
            "path": "/.well-known/oauth-protected-resource",
            "headers": [],
        },
    )
    assert status == 200
    import json

    meta = json.loads(body)
    assert meta["resource"] == "https://mcp.example.com"
    assert meta["authorization_servers"] == ["https://app.example.com"]

    # A token-less MCP request is refused at the gate with the resource_metadata pointer.
    status, headers, _b = _drive(
        app, {"type": "http", "method": "POST", "path": "/mcp", "headers": []}
    )
    assert status == 401
    assert b"resource_metadata=" in headers[b"www-authenticate"]
