"""Correlation ID helpers for the enrichment layer.

``RunContext`` is the canonical correlation envelope (defined in
``protocol.py``). This module provides helpers that emit correlation
extras into the various o11y surfaces so ``prod_correlate(run_id)``
returns one consistent story across pipeline + enrichment + LLM calls.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.protocol import RunContext


def correlation_extras_for_logging(ctx: RunContext) -> dict[str, Any]:
    """Build the ``extra={...}`` dict for structured logger calls.

    Loki picks these up as structured fields so ``prod_recent_logs``
    can filter by ``run_id`` / ``enricher_id`` / ``tier``. Usage:

        logger.info(
            "enricher completed",
            extra=correlation_extras_for_logging(ctx),
        )
    """
    return {
        "run_id": ctx.run_id,
        "parent_run_id": ctx.parent_run_id,
        "enricher_id": ctx.enricher_id,
        "enricher_version": ctx.enricher_version,
        "tier": ctx.tier,
        "attempt": ctx.attempt,
        "job_id": ctx.job_id,
    }


def sentry_tags_for_context(ctx: RunContext) -> dict[str, str]:
    """Build the Sentry scope tag set for an enrichment-emitted event.

    Sentry tags are searchable; an agent can filter to all events for
    a particular ``run_id`` or ``enricher_id`` via the Sentry MCP
    surface. ``parent_run_id == None`` becomes ``"(standalone)"`` for
    the tag (Sentry tags must be strings).
    """
    return {
        "run_id": ctx.run_id,
        "parent_run_id": ctx.parent_run_id or "(standalone)",
        "enricher_id": ctx.enricher_id,
        "enricher_version": ctx.enricher_version,
        "tier": ctx.tier,
        "attempt": str(ctx.attempt),
    }


def langfuse_metadata_for_context(ctx: RunContext) -> dict[str, Any]:
    """Build the Langfuse trace metadata dict for LLM-tier enrichers.

    Future LLM-tier query enrichers' provider calls carry this metadata
    so Langfuse traces are filterable by ``enricher_id`` / ``run_id``
    via the existing ``prod_recent_traces`` MCP tool.
    """
    return {
        "run_id": ctx.run_id,
        "parent_run_id": ctx.parent_run_id,
        "enricher_id": ctx.enricher_id,
        "enricher_version": ctx.enricher_version,
        "tier": ctx.tier,
    }


def jsonl_event_extras(ctx: RunContext) -> dict[str, Any]:
    """Correlation fields included on every ``run.jsonl`` line.

    Same set the structured logger gets, minus ``parent_run_id`` which
    only appears on ``enrichment.run.{started,completed}`` events.
    """
    return {
        "run_id": ctx.run_id,
        "enricher_id": ctx.enricher_id,
        "enricher_version": ctx.enricher_version,
        "tier": ctx.tier,
        "attempt": ctx.attempt,
    }
