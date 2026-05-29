"""Unit tests for DGX fallback telemetry (ADR-096)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from podcast_scraper.providers.tailnet_dgx import telemetry


@patch("sentry_sdk.add_breadcrumb")
def test_emit_dgx_fallback_breadcrumb_records_sentry(mock_breadcrumb: MagicMock) -> None:
    telemetry.emit_dgx_fallback_breadcrumb(
        stage="transcription",
        model="whisper-large-v3",
        failure_reason="timeout",
        episode_id="ep-1",
        extra={"retry": 2},
    )
    mock_breadcrumb.assert_called_once()
    kwargs = mock_breadcrumb.call_args.kwargs
    assert kwargs["category"] == "dgx.fallback"
    assert kwargs["data"]["dgx_fallback_active"] is True
    assert kwargs["data"]["episode_id"] == "ep-1"
    assert kwargs["data"]["retry"] == 2


@patch("sentry_sdk.add_breadcrumb", side_effect=RuntimeError("no sentry"))
def test_emit_dgx_fallback_breadcrumb_logs_when_sentry_unavailable(
    _mock_breadcrumb: MagicMock,
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING):
        telemetry.emit_dgx_fallback_breadcrumb(
            stage="transcription",
            model="whisper-large-v3",
            failure_reason="health_check_failed",
        )
    assert any("DGX fallback for stage=transcription" in rec.message for rec in caplog.records)
