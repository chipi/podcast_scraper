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
