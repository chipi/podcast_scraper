"""The GET-only operator key: obs read probes get in, nothing it carries can write (2026-10-03).

Prod shape: player-api has no write key and an admin-session gate, so the obs container's
``GET /api/jobs`` was 403. It now carries ``APP_OPERATOR_READ_KEY``, which this gate accepts for
reads of operator endpoints only.
"""

from __future__ import annotations

from typing import Optional

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from podcast_scraper.server.app_operator_guard import OperatorWriteGuard

pytestmark = [pytest.mark.unit]

WRITE = "write-key"
READ = "read-key"


async def _ok(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _client(write_key: str = WRITE, read_key: str = READ) -> TestClient:
    app = Starlette(
        routes=[
            Route("/api/jobs", _ok, methods=["GET", "POST"]),
            Route("/api/jobs/{job_id}/cancel", _ok, methods=["POST"]),
        ]
    )
    app.add_middleware(OperatorWriteGuard)
    app.state.operator_api_key = write_key
    app.state.operator_read_key = read_key
    app.state.session_secret = ""
    app.state.app_data_dir = None
    app.state.audit_path = None
    return TestClient(app)


def _status(client: TestClient, method: str, path: str, key: Optional[str]) -> int:
    headers = {"X-Operator-Key": key} if key else {}
    return int(client.request(method, path, headers=headers).status_code)


def test_the_read_key_lists_jobs() -> None:
    assert _status(_client(), "GET", "/api/jobs", READ) == 200


def test_the_read_key_can_never_write() -> None:
    c = _client()
    assert _status(c, "POST", "/api/jobs", READ) == 403
    assert _status(c, "POST", "/api/jobs/abc/cancel", READ) == 403


def test_the_write_key_still_reads_and_writes() -> None:
    c = _client()
    assert _status(c, "GET", "/api/jobs", WRITE) == 200
    assert _status(c, "POST", "/api/jobs", WRITE) == 200


def test_no_key_and_a_wrong_key_are_refused() -> None:
    c = _client()
    assert _status(c, "GET", "/api/jobs", None) == 403
    assert _status(c, "GET", "/api/jobs", "nope") == 403


def test_a_read_key_alone_enforces_the_gate() -> None:
    """player-api has no write key: setting only the read key must not open the plane to all."""
    c = _client(write_key="")
    assert _status(c, "GET", "/api/jobs", None) == 403
    assert _status(c, "GET", "/api/jobs", READ) == 200
    assert _status(c, "POST", "/api/jobs", READ) == 403
