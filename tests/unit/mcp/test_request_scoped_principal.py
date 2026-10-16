"""Tool calls are authorised and attributed as the request that CARRIES them.

FastMCP runs a tool in the task of the request that opened the MCP session, so the ContextVars
:class:`McpAuthMiddleware` sets reached the tool only from that first request: every later call in
the session was scope-checked, logged and traced as the opener (player-mcp 2026-10-15: three
``resolve_entity`` calls two minutes apart shared one trace id). Driven through the real
streamable-HTTP app in-process; the SDK binds a session to one user, so the case is the same user
presenting a token with different scopes on a later request.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from podcast_scraper.mcp import auth, telemetry

pytestmark = pytest.mark.unit

_TOKENS = {
    "read-only": ("u_1", frozenset({auth.SCOPE_READ})),
    "read-write": ("u_1", frozenset({auth.SCOPE_READ, auth.SCOPE_WRITE})),
}


def _build() -> tuple[Any, Any]:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("t")

    @server.tool()
    def write_probe() -> dict:
        auth.require_scope(auth.SCOPE_WRITE)
        with telemetry.observe_tool_call("write_probe") as call:
            result = {"ok": True}
            call.set_result(result)
            return result

    app = auth.McpAuthMiddleware(server.streamable_http_app(), verifier=_TOKENS.get)
    return server, app


async def _call(client: httpx.AsyncClient, body: dict, token: str, session: str | None) -> Any:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session:
        headers["mcp-session-id"] = session
    r = await client.post("/mcp", content=json.dumps(body), headers=headers)
    payload = None
    for line in r.text.splitlines():
        if line.startswith("data:"):
            payload = json.loads(line[5:])
    return r, payload


async def _session_then_call(open_token: str, call_token: str) -> dict:
    server, app = _build()
    async with server.session_manager.run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8009") as c:
            init = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"},
                },
            }
            r, _ = await _call(c, init, open_token, None)
            sid = r.headers.get("mcp-session-id")
            assert sid, r.text
            await _call(
                c, {"jsonrpc": "2.0", "method": "notifications/initialized"}, open_token, sid
            )
            body = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "write_probe", "arguments": {}},
            }
            _, payload = await _call(c, body, call_token, sid)
            assert payload is not None
            result: dict = payload["result"]
            return result


def test_scope_check_uses_the_calling_request() -> None:
    result = asyncio.run(_session_then_call("read-only", "read-write"))
    assert result["isError"] is False, result


def test_a_read_only_call_in_a_read_write_session_is_refused() -> None:
    result = asyncio.run(_session_then_call("read-write", "read-only"))
    assert result["isError"] is True
    assert "mcp:write" in json.dumps(result)


def test_tool_line_is_attributed_to_the_calling_request(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.mcp.tool"):
        asyncio.run(_session_then_call("read-only", "read-write"))
    lines = [r.getMessage() for r in caplog.records if r.name == "podcast_scraper.mcp.tool"]
    assert lines and "tool=write_probe user=u_1" in lines[0]
