"""Unit tests for ``enrichment.observability`` — Sentry / Langfuse helpers."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

from podcast_scraper.enrichment.observability import (
    breadcrumb_circuit_opened,
    langfuse_kwargs_for,
    message_auto_disabled,
    message_stall_escalation,
    stamp_sentry_correlation,
)
from podcast_scraper.enrichment.protocol import RunContext


def _ctx(*, parent_run_id: str | None = "parent-1") -> RunContext:
    return RunContext(
        run_id="run-1",
        parent_run_id=parent_run_id,
        enricher_id="nli_contradiction",
        enricher_version="1.0.0",
        tier="ml",
        attempt=2,
        job_id="job-1",
        cancel_event=asyncio.Event(),
    )


# ---------------------------------------------------------------------------
# stamp_sentry_correlation
# ---------------------------------------------------------------------------


def test_stamp_sentry_correlation_sets_envelope_tags() -> None:
    fake = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": fake}):
        stamp_sentry_correlation(_ctx())
    fake.set_tag.assert_any_call("run_id", "run-1")
    fake.set_tag.assert_any_call("enricher_id", "nli_contradiction")
    fake.set_tag.assert_any_call("tier", "ml")
    fake.set_tag.assert_any_call("attempt", "2")


def test_stamp_sentry_correlation_handles_standalone_run() -> None:
    """parent_run_id=None becomes "(standalone)" tag string."""
    fake = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": fake}):
        stamp_sentry_correlation(_ctx(parent_run_id=None))
    fake.set_tag.assert_any_call("parent_run_id", "(standalone)")


def test_stamp_sentry_correlation_noop_without_sdk() -> None:
    with patch.dict(sys.modules, {"sentry_sdk": None}):
        stamp_sentry_correlation(_ctx())  # must not raise


# ---------------------------------------------------------------------------
# breadcrumb_circuit_opened
# ---------------------------------------------------------------------------


def test_breadcrumb_circuit_opened_fires_breadcrumb_with_category_and_data() -> None:
    fake = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": fake}):
        breadcrumb_circuit_opened(
            _ctx(),
            consecutive_failures=5,
            cooldown_until="2026-06-26T16:01:42Z",
        )
    fake.add_breadcrumb.assert_called_once()
    kwargs = fake.add_breadcrumb.call_args.kwargs
    assert kwargs["category"] == "enrichment.circuit_opened"
    assert kwargs["level"] == "warning"
    assert "nli_contradiction circuit opened" in kwargs["message"]
    assert kwargs["data"]["enricher_id"] == "nli_contradiction"
    assert kwargs["data"]["consecutive_failures"] == 5
    assert kwargs["data"]["cooldown_until"] == "2026-06-26T16:01:42Z"


def test_breadcrumb_circuit_opened_noop_without_sdk() -> None:
    with patch.dict(sys.modules, {"sentry_sdk": None}):
        breadcrumb_circuit_opened(
            _ctx(), consecutive_failures=3, cooldown_until=None
        )  # must not raise


# ---------------------------------------------------------------------------
# message_auto_disabled
# ---------------------------------------------------------------------------


def test_message_auto_disabled_captures_warning_message() -> None:
    fake = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": fake}):
        message_auto_disabled(_ctx(), consecutive_failed_runs=2, reason="circuit opened twice")
    fake.capture_message.assert_called_once()
    args = fake.capture_message.call_args.args
    assert "nli_contradiction auto-disabled" in args[0]
    assert "circuit opened twice" in args[0]


def test_message_auto_disabled_noop_without_sdk() -> None:
    with patch.dict(sys.modules, {"sentry_sdk": None}):
        message_auto_disabled(_ctx(), consecutive_failed_runs=3, reason="x")  # must not raise


# ---------------------------------------------------------------------------
# message_stall_escalation
# ---------------------------------------------------------------------------


def test_message_stall_escalation_captures_error_message() -> None:
    fake = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": fake}):
        message_stall_escalation(
            _ctx(),
            last_heartbeat_at="2026-06-26T15:03:14Z",
            escalated_to="cancel",
        )
    fake.capture_message.assert_called_once()
    args = fake.capture_message.call_args.args
    assert "nli_contradiction stall escalated to cancel" in args[0]


def test_message_stall_escalation_noop_without_sdk() -> None:
    with patch.dict(sys.modules, {"sentry_sdk": None}):
        message_stall_escalation(
            _ctx(), last_heartbeat_at="t", escalated_to="cancel"
        )  # must not raise


# ---------------------------------------------------------------------------
# langfuse_kwargs_for
# ---------------------------------------------------------------------------


def test_langfuse_kwargs_for_returns_emit_langfuse_span_extras() -> None:
    kwargs = langfuse_kwargs_for(_ctx())
    assert kwargs["enricher_id"] == "nli_contradiction"
    assert kwargs["enricher_tier"] == "ml"
    assert kwargs["run_seed"] == "run-1"


def test_langfuse_kwargs_for_consistent_with_correlation_helper() -> None:
    """The kwargs derive from ``langfuse_metadata_for_context`` — same surface."""
    from podcast_scraper.enrichment.correlation import langfuse_metadata_for_context

    ctx = _ctx()
    md = langfuse_metadata_for_context(ctx)
    kwargs = langfuse_kwargs_for(ctx)
    # The trace_seed is the run_id (the join key #1053 used).
    assert kwargs["run_seed"] == md["run_id"]
    assert kwargs["enricher_id"] == md["enricher_id"]
    assert kwargs["enricher_tier"] == md["tier"]
