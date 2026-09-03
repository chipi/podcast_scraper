"""Scope enforcement over the REAL Streamable-HTTP transport (#1916, advisor 4.1).

The unit tests set `current_mcp_scopes` in their own context and call a tool in the same task.
That proves the middleware sets the contextvar, and separately that the contextvar gates the tool
— but NOT that the value survives the transport, which is the seam the whole feature lives on.

It matters because the SDK spawns a session task from inside the *initialize* request, so the
tools dispatch from a context captured then, not from each later request's context. These drive
the actual ASGI app end to end: initialize, then `tools/call`, over HTTP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from podcast_scraper.mcp import auth
from podcast_scraper.mcp.server import build_http_app

pytestmark = [pytest.mark.integration]

_ACCEPT = "application/json, text/event-stream"


def _client(tmp_path: Path, verifier) -> TestClient:
    app = build_http_app(tmp_path)
    app._verifier = verifier  # type: ignore[attr-defined]
    # The SDK ships its own DNS-rebind guard allowing only 127.0.0.1/localhost hosts, so the
    # TestClient's default `testserver` Host is refused with 421 before anything under test runs.
    return TestClient(app, base_url="http://127.0.0.1:8009")


def _rpc(client: TestClient, method: str, params: dict, *, token: str, session: str | None = None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": _ACCEPT,
        "Content-Type": "application/json",
    }
    if session:
        headers["mcp-session-id"] = session
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
    )


def _initialize(client: TestClient, token: str) -> str | None:
    resp = _rpc(
        client,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
        token=token,
    )
    assert resp.status_code == 200, resp.text
    session = resp.headers.get("mcp-session-id")
    return str(session) if session is not None else None


def _tool_result(resp) -> dict:
    """Pull the tool's JSON envelope out of a JSON or SSE response."""
    body = resp.text
    for line in body.splitlines():
        if line.startswith("data: "):
            body = line[len("data: ") :]
            break
    payload = json.loads(body)
    content = payload["result"]["content"][0]["text"]
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    return parsed


def test_a_read_token_cannot_reach_a_corpus_write_over_http(tmp_path: Path) -> None:
    """The whole feature, end to end: the scope reaches the tool across the transport."""
    auth.__reset_transport_trust()
    client = _client(tmp_path, lambda t: ("u_read", frozenset({"mcp:read"})))
    with client:
        session = _initialize(client, "read-token")
        resp = _rpc(
            client,
            "tools/call",
            {"name": "reindex", "arguments": {}},
            token="read-token",
            session=session,
        )
        assert resp.status_code == 200, resp.text
        out = _tool_result(resp)
        assert out["ok"] is False
        assert out["note"] == "McpScopeError"


def test_a_read_tool_still_works_over_http(tmp_path: Path) -> None:
    """The gate must not have closed on readers — the thing Phase 0 promised not to do."""
    auth.__reset_transport_trust()
    client = _client(tmp_path, lambda t: ("u_read", frozenset({"mcp:read"})))
    with client:
        session = _initialize(client, "read-token")
        resp = _rpc(
            client,
            "tools/call",
            {"name": "corpus_status", "arguments": {}},
            token="read-token",
            session=session,
        )
        assert resp.status_code == 200, resp.text
        assert _tool_result(resp)["ok"] is not False


def test_a_token_granting_nothing_is_refused_over_http(tmp_path: Path) -> None:
    """Authenticated but unauthorised: the CLOSED state, asserted across the transport."""
    auth.__reset_transport_trust()
    client = _client(tmp_path, lambda t: ("u_none", frozenset()))
    with client:
        session = _initialize(client, "empty-token")
        resp = _rpc(
            client,
            "tools/call",
            {"name": "reenrich", "arguments": {}},
            token="empty-token",
            session=session,
        )
        assert _tool_result(resp)["note"] == "McpScopeError"


def test_an_unauthenticated_request_never_reaches_a_tool(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda t: None)
    with client:
        resp = client.post(
            "/mcp",
            headers={"Accept": _ACCEPT, "Content-Type": "application/json"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        assert resp.status_code == 401


def test_another_users_token_cannot_drive_an_established_session(tmp_path: Path) -> None:
    """Session hijacking (advisor 2.3).

    The SDK binds a session to the credential that created it and refuses a mismatch — but only
    when `scope["user"]` carries a principal. We never set one, so its check compared None against
    None and ANY holder of ANY valid token who learned a session id could drive that session,
    with the victim's identity attached.
    """
    auth.__reset_transport_trust()
    tokens = {
        "alice": ("u_alice", frozenset({"mcp:read"})),
        "mallory": ("u_mallory", frozenset({"mcp:read"})),
    }
    client = _client(tmp_path, lambda t: tokens.get(t))
    with client:
        session = _initialize(client, "alice")
        assert session, "the transport did not issue a session id"
        stolen = _rpc(
            client,
            "tools/call",
            {"name": "corpus_status", "arguments": {}},
            token="mallory",
            session=session,
        )
        # Answered "as if the session did not exist" — never with Alice's session.
        assert stolen.status_code in (400, 404), stolen.text
