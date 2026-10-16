"""The two MCP servers (player-mcp, player-obs) share their transport, auth and telemetry.

They serve different tools, but the transport, auth and telemetry mechanisms are the same, and
they drifted: obs ran without session binding and without DNS-rebinding protection, and
player-mcp logged and traced its own healthcheck. Each check here runs against BOTH servers'
real pieces (auth middleware, transport security, HTTP stack, tool wrapper) through the
streamable-HTTP app, in-process.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import pytest

pytestmark = pytest.mark.unit

RESOURCE = "https://srv.example.com"
PUBLIC_CLIENT = ("172.20.0.1", 41000)  # through Caddy + the docker bridge
LOOPBACK_CLIENT = ("127.0.0.1", 41001)  # the compose healthcheck


@dataclass
class Server:
    name: str
    build: Callable[[], tuple[Any, Any]]  # -> (FastMCP server, full ASGI stack)
    tool_logger: str
    access_logger: str


def _content() -> tuple[Any, Any]:
    from mcp.server.fastmcp import FastMCP

    from podcast_scraper.mcp import auth, server

    tokens = {
        "alice": ("u_alice", frozenset({auth.SCOPE_READ})),
        "alice-2": ("u_alice", frozenset({auth.SCOPE_READ})),
        "bob": ("u_bob", frozenset({auth.SCOPE_READ})),
    }
    fast = FastMCP("t", transport_security=server._transport_security())

    def probe() -> dict:
        return {"ok": True}

    fast.tool()(server._enveloped(probe))
    return fast, server._http_stack(
        auth.McpAuthMiddleware(fast.streamable_http_app(), verifier=tokens.get)
    )


def _obs() -> tuple[Any, Any]:
    from mcp.server.fastmcp import FastMCP

    from podcast_obs import auth, mcp_server, telemetry

    tokens = {
        "alice": {"user_id": "u_alice", "role": "admin", "analytics_id": "anl_a1"},
        "alice-2": {"user_id": "u_alice", "role": "admin", "analytics_id": "anl_a2"},
        "bob": {"user_id": "u_bob", "role": "admin", "analytics_id": "anl_b"},
    }
    fast = FastMCP("t", transport_security=mcp_server._transport_security())

    def probe() -> dict:
        return {"ok": True}

    fast.tool()(telemetry.observed(probe))
    return fast, telemetry.http_stack(
        auth.ObsAuthMiddleware(fast.streamable_http_app(), verifier=tokens.get)
    )


SERVERS = [
    Server("player-mcp", _content, "podcast_scraper.mcp.tool", "podcast.access"),
    Server("player-obs", _obs, "podcast_obs.tool", "podcast_obs.access"),
]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.setenv("APP_MCP_ISSUER_URL", "https://issuer.example.com")
    for var in ("APP_MCP_ALLOWED_HOSTS", "APP_MCP_ALLOWED_ORIGINS"):
        monkeypatch.delenv(var, raising=False)


def _headers(token: str | None, session: str | None = None) -> dict[str, str]:
    h = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if session:
        h["mcp-session-id"] = session
    return h


def _data(r: httpx.Response) -> Any:
    out = None
    for line in r.text.splitlines():
        if line.startswith("data:"):
            out = json.loads(line[5:])
    return out


_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"},
    },
}
_CALL = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "probe", "arguments": {}},
}


async def _run(srv: Server, steps: Callable[[httpx.AsyncClient], Any], client=PUBLIC_CLIENT) -> Any:
    fast, app = srv.build()
    async with fast.session_manager.run():
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url=RESOURCE) as c:
            return await steps(c)


async def _open(c: httpx.AsyncClient, token: str) -> str:
    r = await c.post("/mcp", content=json.dumps(_INIT), headers=_headers(token))
    sid = r.headers.get("mcp-session-id")
    assert sid, (r.status_code, r.text)
    await c.post(
        "/mcp",
        content=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        headers=_headers(token, sid),
    )
    return sid


@pytest.mark.parametrize("srv", SERVERS, ids=lambda s: s.name)
def test_unauthenticated_mcp_is_401_with_the_resource_metadata_pointer(srv: Server) -> None:
    async def steps(c: httpx.AsyncClient) -> httpx.Response:
        return await c.post("/mcp", content=json.dumps(_INIT), headers=_headers(None))

    r = asyncio.run(_run(srv, steps))
    assert r.status_code == 401
    assert f"{RESOURCE}/.well-known/oauth-protected-resource" in r.headers["www-authenticate"]


@pytest.mark.parametrize("srv", SERVERS, ids=lambda s: s.name)
def test_another_user_cannot_drive_an_open_session(srv: Server) -> None:
    async def steps(c: httpx.AsyncClient) -> tuple[int, int]:
        sid = await _open(c, "alice")
        own = await c.post("/mcp", content=json.dumps(_CALL), headers=_headers("alice", sid))
        other = await c.post("/mcp", content=json.dumps(_CALL), headers=_headers("bob", sid))
        return own.status_code, other.status_code

    own, other = asyncio.run(_run(srv, steps))
    assert own == 200
    assert other in (403, 404), other


@pytest.mark.parametrize("srv", SERVERS, ids=lambda s: s.name)
def test_a_foreign_host_header_is_refused(srv: Server) -> None:
    async def steps(c: httpx.AsyncClient) -> tuple[int, int]:
        good = await c.post("/mcp", content=json.dumps(_INIT), headers=_headers("alice"))
        bad = await c.post(
            "/mcp",
            content=json.dumps(_INIT),
            headers={**_headers("alice"), "Host": "rebind.attacker.example"},
        )
        return good.status_code, bad.status_code

    good, bad = asyncio.run(_run(srv, steps))
    assert good == 200
    assert bad == 421


@pytest.mark.parametrize("srv", SERVERS, ids=lambda s: s.name)
def test_the_tool_line_names_the_calling_request(
    srv: Server, caplog: pytest.LogCaptureFixture
) -> None:
    async def steps(c: httpx.AsyncClient) -> Any:
        sid = await _open(c, "alice")
        r = await c.post("/mcp", content=json.dumps(_CALL), headers=_headers("alice-2", sid))
        return _data(r)

    with caplog.at_level(logging.INFO, logger=srv.tool_logger):
        payload = asyncio.run(_run(srv, steps))
    assert payload["result"]["isError"] is False
    lines = [r.getMessage() for r in caplog.records if r.name == srv.tool_logger]
    assert len(lines) == 1 and "tool=probe user=u_alice" in lines[0]
    if srv.name == "player-obs":
        # Same user, second token: the participant comes from the CALLING request, not the opener.
        assert "participant=anl_a2" in lines[0]


@pytest.mark.parametrize("srv", SERVERS, ids=lambda s: s.name)
def test_the_container_healthcheck_is_not_access_logged(
    srv: Server, caplog: pytest.LogCaptureFixture
) -> None:
    async def get(c: httpx.AsyncClient) -> int:
        return (await c.get("/.well-known/oauth-protected-resource")).status_code

    with caplog.at_level(logging.INFO, logger=srv.access_logger):
        assert asyncio.run(_run(srv, get, client=LOOPBACK_CLIENT)) == 200
        quiet = [r for r in caplog.records if r.name == srv.access_logger]
        assert asyncio.run(_run(srv, get, client=PUBLIC_CLIENT)) == 200
        public = [r for r in caplog.records if r.name == srv.access_logger]
    assert quiet == []
    assert len(public) == 1


@pytest.mark.parametrize("srv", SERVERS, ids=lambda s: s.name)
def test_metrics_are_served_without_a_bearer(srv: Server) -> None:
    async def steps(c: httpx.AsyncClient) -> httpx.Response:
        return await c.get("/metrics")

    r = asyncio.run(_run(srv, steps))
    assert r.status_code == 200
    assert "mcp_tool_calls_total" in r.text
