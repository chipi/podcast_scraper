"""Per-tool MCP observability (#1505): span/log/metric emission is best-effort + never breaks."""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.mcp.telemetry import observe_tool_call


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def test_observe_logs_tool_and_ok_true(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.mcp.tool"):
        with observe_tool_call("search_corpus") as call:
            call.set_result({"ok": True, "data": {"results": [1]}})
    assert any("tool=search_corpus" in m and "ok=True" in m for m in _messages(caplog))


def test_observe_logs_ok_false_from_envelope(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.mcp.tool"):
        with observe_tool_call("bridge") as call:
            call.set_result({"ok": False, "data": {}, "note": "unsupported"})
    assert any("tool=bridge" in m and "ok=False" in m for m in _messages(caplog))


def test_observe_records_ok_false_and_reraises_on_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.mcp.tool"):
        with pytest.raises(ValueError):
            with observe_tool_call("resolve_entity"):
                raise ValueError("boom")
    # A raised body → recorded as not-ok, and the failure still propagates (telemetry is passive).
    assert any("tool=resolve_entity" in m and "ok=False" in m for m in _messages(caplog))


def test_observe_carries_user_id_when_set(caplog: pytest.LogCaptureFixture) -> None:
    from podcast_scraper.mcp.auth import current_mcp_user

    token = current_mcp_user.set("u_test123")
    try:
        with caplog.at_level(logging.INFO, logger="podcast_scraper.mcp.tool"):
            with observe_tool_call("corpus_trending") as call:
                call.set_result({"ok": True, "data": {}})
    finally:
        current_mcp_user.reset(token)
    assert any("user=u_test123" in m for m in _messages(caplog))


def test_umami_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    import podcast_scraper.mcp.telemetry as tele

    monkeypatch.delenv("PODCAST_MCP_UMAMI_WEBSITE_ID", raising=False)
    monkeypatch.delenv("PODCAST_MCP_UMAMI_URL", raising=False)

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("must not submit a sender when Umami is unconfigured")

    monkeypatch.setattr(tele._UMAMI_POOL, "submit", _boom)
    tele._emit_umami("search_corpus", True)  # returns early — no submit, no raise


def test_umami_posts_event_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    import podcast_scraper.mcp.telemetry as tele

    monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "site-abc")
    monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://analytics.example/api/send")
    captured: dict = {}

    def fake_urlopen(req: object, timeout: object = None) -> None:
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        captured["body"] = _json.loads(req.data)  # type: ignore[attr-defined]
        captured["ua"] = req.headers.get("User-agent")  # type: ignore[attr-defined]

    # Run the sender inline (no real thread) so the assertion is deterministic.
    monkeypatch.setattr(tele._UMAMI_POOL, "submit", lambda fn: fn())
    monkeypatch.setattr(tele.urllib.request, "urlopen", fake_urlopen)

    tele._emit_umami("who_said_about_topic", False)

    assert captured["url"] == "https://analytics.example/api/send"
    assert captured["body"]["payload"]["name"] == "mcp_tool:who_said_about_topic"
    assert captured["body"]["payload"]["website"] == "site-abc"
    assert captured["body"]["payload"]["data"] == {"tool": "who_said_about_topic", "ok": False}
    assert captured["ua"]  # Umami drops UA-less events


def test_umami_hostname_is_bare_host_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Umami wants a bare host; the env often carries the full resource URL (advisor M2)."""
    import podcast_scraper.mcp.telemetry as tele

    monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "site-abc")
    monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://analytics.example/api/send")
    monkeypatch.setenv("PODCAST_MCP_UMAMI_HOSTNAME", "https://mcp.closelistening.app")
    captured: dict = {}

    def fake_urlopen(req: object, timeout: object = None) -> None:
        import json as _json

        payload = _json.loads(req.data)["payload"]  # type: ignore[attr-defined]
        captured["hostname"] = payload["hostname"]

    monkeypatch.setattr(tele._UMAMI_POOL, "submit", lambda fn: fn())
    monkeypatch.setattr(tele.urllib.request, "urlopen", fake_urlopen)
    tele._emit_umami("corpus_trending", True)
    assert captured["hostname"] == "mcp.closelistening.app"  # scheme stripped


def test_set_result_ignores_non_envelope() -> None:
    """A bare (non-envelope) return value leaves ok=True — the set_result early exit."""
    with observe_tool_call("bridge") as call:
        call.set_result("plain-string")
        call.set_result({"data": 1})  # dict without an "ok" key
    assert call.ok is True


def test_emit_sets_span_attributes_and_metric() -> None:
    """_emit stamps the span + increments the counter when a span + prometheus are present."""
    import podcast_scraper.mcp.telemetry as tele

    class _FakeSpan:
        def __init__(self) -> None:
            self.attrs: dict = {}

        def set_attribute(self, k: str, v: object) -> None:
            self.attrs[k] = v

    span = _FakeSpan()
    tele._emit("search_corpus", True, 0.01, "u_1", span)
    assert span.attrs["mcp.tool"] == "search_corpus"
    assert span.attrs["mcp.ok"] is True
    assert span.attrs["mcp.user_id"] == "u_1"


def test_maybe_span_yields_none_when_tracer_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising tracer degrades to an un-spanned call rather than breaking it (89-93)."""
    from opentelemetry import trace

    import podcast_scraper.mcp.telemetry as tele

    def _boom(_name: str) -> object:
        raise RuntimeError("no tracer provider")

    monkeypatch.setattr(trace, "get_tracer", _boom)
    with tele._maybe_span("resolve_entity") as span:
        assert span is None


def test_umami_backpressure_drops_when_inflight_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the in-flight cap the event is dropped at the semaphore guard — never queued (178)."""
    import podcast_scraper.mcp.telemetry as tele

    monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "s")
    monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://a/api/send")

    def _must_not_submit(_fn: object) -> None:
        raise AssertionError("must not submit a sender under backpressure")

    monkeypatch.setattr(tele._UMAMI_POOL, "submit", _must_not_submit)
    held = [tele._UMAMI_INFLIGHT.acquire(blocking=False) for _ in range(8)]
    try:
        assert all(held)
        tele._emit_umami("corpus_trending", True)  # returns at the guard, no submit, no raise
    finally:
        for _ in held:
            tele._UMAMI_INFLIGHT.release()


def test_umami_send_swallows_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable Umami host is swallowed in _send and the semaphore released (193-196)."""
    import podcast_scraper.mcp.telemetry as tele

    monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "s")
    monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://a/api/send")
    monkeypatch.setattr(tele._UMAMI_POOL, "submit", lambda fn: fn())  # run inline

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("unreachable analytics host")

    monkeypatch.setattr(tele.urllib.request, "urlopen", _boom)
    tele._emit_umami("who_said_about_topic", True)  # no raise
    # released by _send's finally, so a fresh acquire succeeds
    assert tele._UMAMI_INFLIGHT.acquire(blocking=False)
    tele._UMAMI_INFLIGHT.release()


def test_umami_pool_submit_failure_releases_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shut-down pool drops the event AND releases the semaphore — no leak (200-201)."""
    import podcast_scraper.mcp.telemetry as tele

    monkeypatch.setenv("PODCAST_MCP_UMAMI_WEBSITE_ID", "s")
    monkeypatch.setenv("PODCAST_MCP_UMAMI_URL", "https://a/api/send")

    def _boom(_fn: object) -> None:
        raise RuntimeError("pool is shut down")

    monkeypatch.setattr(tele._UMAMI_POOL, "submit", _boom)
    tele._emit_umami("top_people", False)  # no raise
    assert tele._UMAMI_INFLIGHT.acquire(blocking=False)  # not leaked
    tele._UMAMI_INFLIGHT.release()


def test_observe_swallows_user_context_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken user contextvar degrades to user=None rather than breaking the call (113-114)."""
    import podcast_scraper.mcp.telemetry as tele

    class _BrokenVar:
        def get(self) -> str:
            raise RuntimeError("contextvar broken")

    monkeypatch.setattr(tele, "current_mcp_user", _BrokenVar())
    with tele.observe_tool_call("episode_digest") as call:
        call.set_result({"ok": True})  # no raise


def test_emit_swallows_span_metric_and_log_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every emit sink (span attr, metric, log) is independently best-effort (125-132, 141-142)."""
    import podcast_scraper.mcp.telemetry as tele

    class _BadSpan:
        def set_attribute(self, _k: str, _v: object) -> None:
            raise RuntimeError("span dead")

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("sink dead")

    assert tele._CALLS is not None  # prometheus installed in the unit env
    monkeypatch.setattr(tele._CALLS, "labels", _boom)
    monkeypatch.setattr(tele._LOGGER, "info", _boom)
    tele._emit("t", True, 0.0, "u", _BadSpan())  # all three failures swallowed → no raise


def test_metrics_endpoint_serves_prometheus_and_delegates_else() -> None:
    import asyncio

    from podcast_scraper.mcp.server import _with_metrics

    delegated = {"hit": False}

    async def _downstream(scope: dict, receive: object, send: object) -> None:
        delegated["hit"] = True

    app = _with_metrics(_downstream)
    sent: list = []

    async def _send(m: dict) -> None:
        sent.append(m)

    async def _receive() -> dict:
        return {"type": "http.request"}

    asyncio.run(app({"type": "http", "path": "/metrics"}, _receive, _send))
    assert sent[0]["status"] == 200
    assert any(h == (b"content-type",) or h[0] == b"content-type" for h in sent[0]["headers"])
    assert not delegated["hit"]  # /metrics handled here, not delegated

    delegated["hit"] = False
    asyncio.run(app({"type": "http", "path": "/mcp"}, _receive, _send))
    assert delegated["hit"]  # everything else passes through to the auth-gated app
