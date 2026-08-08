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
