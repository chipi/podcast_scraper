"""Unit tests for the trace-correlated request access-log middleware (ADR-119 G1).

The middleware logs one `podcast.access` line per real request with the active OTEL
trace id stamped inline (`trace=<hex>`, or `-` with no span) so a VictoriaLogs line pivots
to its VictoriaTraces span — and it skips `/health` + `/metrics` so probe/scrape traffic
does not flood the log.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from podcast_scraper.server.app import _install_access_logging


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/app/ping")
    def _ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/api/app/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def _metrics() -> str:
        return "# metrics"

    _install_access_logging(app)
    return app


def test_access_log_emits_line_with_trace_token(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="podcast.access")
    resp = TestClient(_app()).get("/api/app/ping")
    assert resp.status_code == 200
    records = [r for r in caplog.records if r.name == "podcast.access"]
    assert len(records) == 1, "exactly one access line per request"
    msg = records[0].getMessage()
    assert "GET /api/app/ping -> 200" in msg
    # trace= is always present: a 32-hex id when a span is active, "-" otherwise (no OTEL
    # in the unit env). The literal token is what the VictoriaLogs derivedField keys on.
    assert "trace=" in msg


@pytest.mark.parametrize("path", ["/api/app/health", "/metrics"])
def test_access_log_skips_probe_and_scrape_paths(
    path: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="podcast.access")
    TestClient(_app()).get(path)
    assert [r for r in caplog.records if r.name == "podcast.access"] == []
