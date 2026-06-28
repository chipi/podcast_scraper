"""Unit tests for ``enrichment.correlation`` helpers."""

from __future__ import annotations

import asyncio

from podcast_scraper.enrichment.correlation import (
    correlation_extras_for_logging,
    jsonl_event_extras,
    langfuse_metadata_for_context,
    sentry_tags_for_context,
)
from podcast_scraper.enrichment.protocol import RunContext


def _make_ctx(*, run_id: str = "run-1", parent_run_id: str | None = "parent-1") -> RunContext:
    return RunContext(
        run_id=run_id,
        parent_run_id=parent_run_id,
        enricher_id="topic_cooccurrence",
        enricher_version="1.0.0",
        tier="deterministic",
        attempt=2,
        job_id="job-1",
        cancel_event=asyncio.Event(),
    )


# ---------------------------------------------------------------------------
# correlation_extras_for_logging
# ---------------------------------------------------------------------------


def test_correlation_extras_for_logging_carries_full_envelope() -> None:
    extras = correlation_extras_for_logging(_make_ctx())
    assert extras["run_id"] == "run-1"
    assert extras["parent_run_id"] == "parent-1"
    assert extras["enricher_id"] == "topic_cooccurrence"
    assert extras["enricher_version"] == "1.0.0"
    assert extras["tier"] == "deterministic"
    assert extras["attempt"] == 2
    assert extras["job_id"] == "job-1"


def test_correlation_extras_preserves_none_parent_run_id() -> None:
    """Standalone enrichment runs carry ``parent_run_id == None``."""
    extras = correlation_extras_for_logging(_make_ctx(parent_run_id=None))
    assert extras["parent_run_id"] is None


# ---------------------------------------------------------------------------
# sentry_tags_for_context
# ---------------------------------------------------------------------------


def test_sentry_tags_stringifies_attempt() -> None:
    """Sentry tags MUST be strings — ``attempt`` is coerced."""
    tags = sentry_tags_for_context(_make_ctx())
    assert tags["attempt"] == "2"
    assert isinstance(tags["attempt"], str)


def test_sentry_tags_standalone_run_has_sentinel_parent() -> None:
    """Sentry tags must be strings; ``None`` becomes ``"(standalone)"``."""
    tags = sentry_tags_for_context(_make_ctx(parent_run_id=None))
    assert tags["parent_run_id"] == "(standalone)"


def test_sentry_tags_carries_correlation_envelope() -> None:
    tags = sentry_tags_for_context(_make_ctx())
    assert tags["run_id"] == "run-1"
    assert tags["enricher_id"] == "topic_cooccurrence"
    assert tags["enricher_version"] == "1.0.0"
    assert tags["tier"] == "deterministic"


# ---------------------------------------------------------------------------
# langfuse_metadata_for_context
# ---------------------------------------------------------------------------


def test_langfuse_metadata_includes_run_correlation() -> None:
    md = langfuse_metadata_for_context(_make_ctx())
    assert md["run_id"] == "run-1"
    assert md["parent_run_id"] == "parent-1"
    assert md["enricher_id"] == "topic_cooccurrence"
    assert md["tier"] == "deterministic"


def test_langfuse_metadata_does_not_include_attempt() -> None:
    """Langfuse metadata stays minimal — attempt is per-trace internal detail."""
    md = langfuse_metadata_for_context(_make_ctx())
    assert "attempt" not in md
    assert "job_id" not in md


# ---------------------------------------------------------------------------
# jsonl_event_extras
# ---------------------------------------------------------------------------


def test_jsonl_event_extras_carries_per_event_fields() -> None:
    extras = jsonl_event_extras(_make_ctx())
    assert extras["run_id"] == "run-1"
    assert extras["enricher_id"] == "topic_cooccurrence"
    assert extras["enricher_version"] == "1.0.0"
    assert extras["tier"] == "deterministic"
    assert extras["attempt"] == 2
    # ``parent_run_id`` is NOT in per-event extras — it only appears on
    # ``enrichment.run.{started,completed}`` events written elsewhere.
    assert "parent_run_id" not in extras


# ---------------------------------------------------------------------------
# Cross-surface consistency
# ---------------------------------------------------------------------------


def test_correlation_envelope_consistent_across_surfaces() -> None:
    """Same ctx → all surfaces agree on run_id/enricher_id/tier."""
    ctx = _make_ctx()
    log_ex = correlation_extras_for_logging(ctx)
    sentry_t = sentry_tags_for_context(ctx)
    lf_md = langfuse_metadata_for_context(ctx)
    jsonl_ex = jsonl_event_extras(ctx)

    for surface in (log_ex, sentry_t, lf_md, jsonl_ex):
        assert surface["run_id"] == "run-1"
        assert surface["enricher_id"] == "topic_cooccurrence"
        assert surface["tier"] == "deterministic"
