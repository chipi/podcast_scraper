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


def test_run_server_http_wires_otel_sentry_and_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The http path inits OTel + GlitchTip, then serves the metrics-wrapped instrumented app."""
    import podcast_scraper.mcp.server as srv
    import podcast_scraper.utils.otel_init as otel_init
    import podcast_scraper.utils.sentry_init as sentry_init

    seen: dict = {"otel": 0, "sentry": None, "served": None}

    monkeypatch.setattr(srv, "build_http_app", lambda corpus: {"app": corpus})
    monkeypatch.setattr(otel_init, "init_otel", lambda: seen.__setitem__("otel", seen["otel"] + 1))
    monkeypatch.setattr(sentry_init, "init_sentry", lambda comp: seen.__setitem__("sentry", comp))

    import uvicorn

    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, host, port: seen.__setitem__("served", (callable(app), host, port)),
    )

    srv.run_server(_CORPUS, transport="http", host="127.0.0.1", port=8009)

    assert seen["otel"] == 1
    assert seen["sentry"] == "mcp"  # GlitchTip component tag
    assert seen["served"] == (True, "127.0.0.1", 8009)  # a callable ASGI app was served


def test_run_server_http_survives_observability_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving never blocks on a failing OTel/GlitchTip init — both except branches (ADR-120)."""
    import podcast_scraper.mcp.server as srv
    import podcast_scraper.utils.otel_init as otel_init
    import podcast_scraper.utils.sentry_init as sentry_init

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("observability backend down")

    monkeypatch.setattr(srv, "build_http_app", lambda corpus: {"app": corpus})
    monkeypatch.setattr(otel_init, "init_otel", _boom)
    monkeypatch.setattr(sentry_init, "init_sentry", _boom)

    import uvicorn

    served: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: served.update(ok=callable(app)))
    srv.run_server(_CORPUS, transport="http", host="127.0.0.1", port=8009)
    assert served == {"ok": True}  # served despite both inits raising


def test_with_metrics_noop_when_prometheus_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """No prometheus_client → /metrics not served, the app passes through unchanged (714-715)."""
    import sys
    import types

    from podcast_scraper.mcp.server import _with_metrics

    broken = types.ModuleType("prometheus_client")  # lacks generate_latest / CONTENT_TYPE_LATEST
    monkeypatch.setitem(sys.modules, "prometheus_client", broken)

    async def _app(scope: dict, receive: object, send: object) -> None:  # pragma: no cover - marker
        return None

    assert _with_metrics(_app) is _app


def test_instrument_asgi_returns_app_when_instrumentor_absent() -> None:
    """No OTel ASGI instrumentation installed → the app is served un-wrapped (the except path)."""
    from podcast_scraper.mcp.server import _instrument_asgi

    async def _app(scope: dict, receive: object, send: object) -> None:  # pragma: no cover - marker
        return None

    assert _instrument_asgi(_app) is _app


def test_instrument_asgi_wraps_when_instrumentor_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the instrumentor IS importable, the app is wrapped in OpenTelemetryMiddleware."""
    import sys
    import types

    from podcast_scraper.mcp import server as srv

    pkg = types.ModuleType("opentelemetry.instrumentation")
    asgi = types.ModuleType("opentelemetry.instrumentation.asgi")

    class _FakeMW:
        def __init__(self, app: object) -> None:
            self.app = app

    asgi.OpenTelemetryMiddleware = _FakeMW  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation", pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.asgi", asgi)

    async def _app(scope: dict, receive: object, send: object) -> None:  # pragma: no cover - marker
        return None

    wrapped = srv._instrument_asgi(_app)
    assert isinstance(wrapped, _FakeMW)
    assert wrapped.app is _app
