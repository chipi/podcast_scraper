"""Self-observability of the obs MCP server (player-obs): tool telemetry, auth context, ASGI stack.

No network: Umami's urlopen is stubbed, the verify round-trip is stubbed, and the ASGI stack is
driven directly.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Dict, Optional

import pytest

from podcast_obs import auth, telemetry

pytestmark = pytest.mark.unit


def _sample(tool: str, ok: str) -> Optional[float]:
    assert telemetry.REGISTRY is not None
    value = telemetry.REGISTRY.get_sample_value("mcp_tool_calls_total", {"tool": tool, "ok": ok})
    return None if value is None else float(value)


class _InlinePool:
    def submit(self, fn: Any) -> None:
        fn()


@pytest.fixture
def umami_sent(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    sent: list[Any] = []
    monkeypatch.setattr(telemetry, "_UMAMI_POOL", _InlinePool())
    monkeypatch.setattr(
        telemetry.urllib.request, "urlopen", lambda req, timeout=4: sent.append(req)
    )
    return sent


def _with_context(user: Optional[str], participant: Optional[str]) -> Any:
    tokens = (auth.current_obs_user.set(user), auth.current_obs_participant.set(participant))
    return tokens


def _reset(tokens: Any) -> None:
    auth.current_obs_user.reset(tokens[0])
    auth.current_obs_participant.reset(tokens[1])


class TestObserved:
    def test_keeps_the_schema_fastmcp_reads(self) -> None:
        def prod_health(target: Optional[str] = None) -> dict:
            """Full /api/health."""
            return {"ok": True}

        wrapped = telemetry.observed(prod_health)
        assert wrapped.__name__ == "prod_health"
        assert wrapped.__doc__ == "Full /api/health."
        assert inspect.signature(wrapped) == inspect.signature(prod_health)

    def test_counts_ok_and_failed_envelopes(self) -> None:
        before_ok = _sample("t_env", "true") or 0.0
        before_bad = _sample("t_env", "false") or 0.0

        def t_env(ok: bool) -> dict:
            return {"ok": ok}

        wrapped = telemetry.observed(t_env)
        wrapped(True)
        wrapped(False)
        wrapped(False)
        assert _sample("t_env", "true") == before_ok + 1
        assert _sample("t_env", "false") == before_bad + 2

    def test_exception_counts_as_failed_and_propagates(self) -> None:
        def t_boom() -> dict:
            raise RuntimeError("backend down")

        before = _sample("t_boom", "false") or 0.0
        with pytest.raises(RuntimeError, match="backend down"):
            telemetry.observed(t_boom)()
        assert _sample("t_boom", "false") == before + 1

    def test_log_line_names_user_participant_and_trace(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def t_log() -> dict:
            return {"ok": True}

        tokens = _with_context("u_1", "anl_9")
        try:
            with caplog.at_level(logging.INFO, logger="podcast_obs.tool"):
                telemetry.observed(t_log)()
        finally:
            _reset(tokens)
        line = next(r.getMessage() for r in caplog.records if r.name == "podcast_obs.tool")
        assert "tool=t_log user=u_1 participant=anl_9 ok=True" in line
        assert "trace_id=" in line

    def test_preregister_creates_zero_series(self) -> None:
        telemetry.preregister_tools(["t_unused"])
        assert _sample("t_unused", "true") == 0.0
        assert _sample("t_unused", "false") == 0.0


class TestUmami:
    def test_noop_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, umami_sent: list[Any]
    ) -> None:
        monkeypatch.delenv("PODCAST_MCP_UMAMI_WEBSITE_ID", raising=False)
        monkeypatch.delenv("PODCAST_MCP_UMAMI_URL", raising=False)
        telemetry._emit_umami("prod_health", True, "anl_9")
        assert umami_sent == []

    def test_event_is_filed_under_the_participant_with_a_browser_ua(
        self, monkeypatch: pytest.MonkeyPatch, umami_sent: list[Any]
    ) -> None:
        monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "site-1")
        monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://analytics.example/api/send")
        monkeypatch.setenv("PODCAST_MCP_UMAMI_HOSTNAME", "https://obs.example.com/mcp")
        telemetry._emit_umami("prod_health", False, "anl_9")
        (req,) = umami_sent
        payload = json.loads(req.data)["payload"]
        assert payload == {
            "website": "site-1",
            "hostname": "obs.example.com",
            "name": "obs_tool:prod_health",
            "data": {"tool": "prod_health", "ok": False},
            "id": "anl_9",
        }
        ua = req.get_header("User-agent")
        # Umami's bot filter drops non-browser UAs without storing anything.
        assert ua.startswith("Mozilla/5.0") and "Chrome/" in ua and ua.endswith("podcast-obs/1.0")

    def test_no_id_without_participant(
        self, monkeypatch: pytest.MonkeyPatch, umami_sent: list[Any]
    ) -> None:
        monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "site-1")
        monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://analytics.example/api/send")
        telemetry._emit_umami("prod_health", True, None)
        (req,) = umami_sent
        assert "id" not in json.loads(req.data)["payload"]


class TestAuthContext:
    def test_verify_returns_analytics_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_MCP_VERIFY_URL", "http://api:8000/internal/mcp/verify")
        monkeypatch.setenv("INTERNAL_MCP_TOKEN", "tok")
        monkeypatch.setenv("APP_MCP_REQUIRE_ADMIN", "true")

        class _Resp:
            def __init__(self, body: bytes) -> None:
                self._b = body

            def read(self) -> bytes:
                return self._b

            def __enter__(self) -> "_Resp":
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        body = {"authenticated": True, "user_id": "u_1", "mcp_access": True, "role": "admin"}
        monkeypatch.setattr(
            auth.urllib.request,
            "urlopen",
            lambda req, timeout=5.0: _Resp(json.dumps({**body, "analytics_id": "anl_9"}).encode()),
        )
        principal = auth.verify_principal("t")
        assert principal is not None and principal["analytics_id"] == "anl_9"
        monkeypatch.setattr(
            auth.urllib.request,
            "urlopen",
            lambda req, timeout=5.0: _Resp(json.dumps(body).encode()),
        )
        principal = auth.verify_principal("t")
        assert principal is not None and principal["analytics_id"] is None

    def test_middleware_sets_and_resets_the_caller(self) -> None:
        seen: dict[str, Any] = {}

        async def app(scope: Any, receive: Any, send: Any) -> None:
            seen["user"] = auth.current_obs_user.get()
            seen["participant"] = auth.current_obs_participant.get()

        mw = auth.ObsAuthMiddleware(
            app, verifier=lambda tok: {"user_id": "u_1", "role": "admin", "analytics_id": "anl_9"}
        )
        scope = {
            "type": "http",
            "path": "/mcp",
            "method": "POST",
            "headers": [(b"authorization", b"Bearer t")],
        }

        async def _noop_send(m: Any) -> None:
            return None

        async def _no_receive() -> Dict[str, Any]:
            return {}

        asyncio.run(mw(scope, _no_receive, _noop_send))
        assert seen == {"user": "u_1", "participant": "anl_9"}
        assert auth.current_obs_user.get() is None
        assert auth.current_obs_participant.get() is None


class _Recorder:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.calls += 1
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def _drive(app: Any, path: str, client: str = "172.20.0.1") -> list[dict]:
    sent: list[dict] = []

    async def _send(m: dict) -> None:
        sent.append(m)

    scope = {"type": "http", "path": path, "method": "GET", "headers": [], "client": (client, 5)}
    asyncio.run(app(scope, None, _send))
    return sent


class TestHttpStack:
    def test_metrics_need_no_bearer(self) -> None:
        inner = _Recorder()
        sent = _drive(telemetry.http_stack(inner), "/metrics")
        assert inner.calls == 0
        assert sent[0]["status"] == 200
        assert b"mcp_tool_calls_total" in sent[1]["body"]
        assert b"process_" in sent[1]["body"] or b"python_gc" in sent[1]["body"]

    def test_request_is_access_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="podcast_obs.access"):
            _drive(telemetry.http_stack(_Recorder()), "/mcp")
        lines = [r.getMessage() for r in caplog.records if r.name == "podcast_obs.access"]
        assert len(lines) == 1 and lines[0].startswith("GET /mcp -> 401 in ")
        assert "trace_id=" in lines[0]

    def test_container_healthcheck_is_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        inner = _Recorder()
        with caplog.at_level(logging.INFO, logger="podcast_obs.access"):
            _drive(
                telemetry.http_stack(inner), "/.well-known/oauth-protected-resource", "127.0.0.1"
            )
        assert inner.calls == 1
        assert not [r for r in caplog.records if r.name == "podcast_obs.access"]

    def test_external_discovery_fetch_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="podcast_obs.access"):
            _drive(telemetry.http_stack(_Recorder()), "/.well-known/oauth-protected-resource")
        assert [r for r in caplog.records if r.name == "podcast_obs.access"]


class TestInit:
    def test_sentry_off_without_dsn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PODCAST_SENTRY_DSN_OBS", raising=False)
        assert telemetry.init_sentry() is False

    def test_otel_needs_the_exporter_opt_in_not_just_the_read_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # obs reads VictoriaTraces through the same endpoint var; that alone must not export.
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://vtraces:10428/v1/traces")
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
        assert telemetry.init_otel() is False
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
        assert telemetry.init_otel() is False


class TestRequestScopedAttribution:
    def test_tool_uses_the_calling_request_not_the_session_opener(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # FastMCP runs tools in the task of the request that opened the session, so the
        # ContextVars hold the opener; the SDK's request_ctx holds the CallTool request.
        from types import SimpleNamespace

        from mcp.server.lowlevel.server import request_ctx

        def t_attr() -> dict:
            return {"ok": True}

        request = SimpleNamespace(
            scope={"state": {"obs_principal": {"user_id": "u_call", "analytics_id": "anl_call"}}}
        )
        ctx_token = request_ctx.set(SimpleNamespace(request=request))  # type: ignore[arg-type]
        tokens = _with_context("u_opener", "anl_opener")
        try:
            with caplog.at_level(logging.INFO, logger="podcast_obs.tool"):
                telemetry.observed(t_attr)()
        finally:
            _reset(tokens)
            request_ctx.reset(ctx_token)
        line = next(r.getMessage() for r in caplog.records if r.name == "podcast_obs.tool")
        assert "user=u_call participant=anl_call" in line

    def test_auth_middleware_puts_the_principal_on_the_request_scope(self) -> None:
        async def app(scope: Any, receive: Any, send: Any) -> None:
            return None

        mw = auth.ObsAuthMiddleware(
            app, verifier=lambda tok: {"user_id": "u_1", "role": "admin", "analytics_id": "anl_9"}
        )
        scope: dict[str, Any] = {
            "type": "http",
            "path": "/mcp",
            "method": "POST",
            "headers": [(b"authorization", b"Bearer t")],
        }

        async def _noop_send(m: Any) -> None:
            return None

        async def _no_receive() -> Dict[str, Any]:
            return {}

        asyncio.run(mw(scope, _no_receive, _noop_send))
        assert scope["state"]["obs_principal"] == {"user_id": "u_1", "analytics_id": "anl_9"}
