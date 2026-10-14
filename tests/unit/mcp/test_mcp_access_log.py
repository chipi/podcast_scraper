"""The MCP's HTTP stack logs one trace-correlated access line per request (like the api).

player-mcp's only per-request log was uvicorn's access line, which runs after the span closes and
has no trace id, so no player-mcp log line could be found from its trace (checked on two GlitchTip
errors, 2026-10-14). The api's pure-ASGI access log captures the id at ``http.response.start``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from podcast_scraper.mcp import server
from podcast_scraper.utils import access_log

pytestmark = pytest.mark.unit

_TRACE = "0123456789abcdef0123456789abcdef"


async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"{}"})


def _call(app, path: str) -> None:
    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(_message):
        return None

    scope = {"type": "http", "method": "POST", "path": path, "headers": []}
    asyncio.run(app(scope, _receive, _send))


def test_mcp_request_logs_its_trace_id(monkeypatch, caplog):
    monkeypatch.setattr(access_log, "current_trace_id", lambda: _TRACE)
    caplog.set_level(logging.INFO, logger="podcast.access")
    _call(server._http_stack(_ok_app), "/mcp")
    lines = [r.getMessage() for r in caplog.records if r.name == "podcast.access"]
    assert len(lines) == 1
    assert lines[0].startswith("POST /mcp -> 200 in ")
    assert lines[0].endswith(f"trace={_TRACE}")


def test_metrics_scrapes_are_not_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="podcast.access")
    _call(server._http_stack(_ok_app), "/metrics")
    assert [r for r in caplog.records if r.name == "podcast.access"] == []
