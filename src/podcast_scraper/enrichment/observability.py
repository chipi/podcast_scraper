"""High-level observability helpers tying RunContext + state changes
to Sentry breadcrumbs + Langfuse trace metadata.

Thin wrappers over ``utils.sentry_init`` + ``utils.langfuse_tracing``
extensions that already exist; this module composes the correlation
envelope into the helper-call shapes so executor + resilience code
stays clean.

All helpers no-op when the underlying o11y SDK isn't installed
(Sentry / Langfuse are optional extensions).
"""

from __future__ import annotations

from podcast_scraper.enrichment.correlation import (
    langfuse_metadata_for_context,
    sentry_tags_for_context,
)
from podcast_scraper.enrichment.protocol import RunContext
from podcast_scraper.utils.sentry_init import (
    capture_enrichment_message,
    emit_enrichment_breadcrumb,
    set_correlation_tags,
)


def stamp_sentry_correlation(ctx: RunContext) -> None:
    """Tag the Sentry scope with the run context.

    Any exception captured (e.g. via the executor's safety net) will
    carry the same ``run_id`` / ``enricher_id`` tags the rest of the
    o11y surface uses; the MCP ``prod_correlate(run_id)`` join works
    out of the box.
    """
    set_correlation_tags(sentry_tags_for_context(ctx))


def breadcrumb_circuit_opened(
    ctx: RunContext, *, consecutive_failures: int, cooldown_until: str | None
) -> None:
    """Fire ``enrichment.circuit_opened`` Sentry breadcrumb.

    Not a Sentry issue — just an entry on the breadcrumb timeline.
    Operators alert on aggregate count via Grafana / Sentry alert
    rules (e.g. "more than 5 enrichment.circuit_opened breadcrumbs
    per hour → page").
    """
    emit_enrichment_breadcrumb(
        "enrichment.circuit_opened",
        f"{ctx.enricher_id} circuit opened (consecutive_failures={consecutive_failures})",
        level="warning",
        data={
            "enricher_id": ctx.enricher_id,
            "tier": ctx.tier,
            "run_id": ctx.run_id,
            "consecutive_failures": int(consecutive_failures),
            "cooldown_until": cooldown_until,
        },
    )


def message_auto_disabled(ctx: RunContext, *, consecutive_failed_runs: int, reason: str) -> None:
    """Fire ``enrichment.auto_disabled`` Sentry message (warning level).

    Notable enough to warrant its own issue — operators can alert on
    the message string in Sentry to detect a lost enricher.
    """
    capture_enrichment_message(
        f"{ctx.enricher_id} auto-disabled after {consecutive_failed_runs} "
        f"failed runs: {reason}",
        level="warning",
        tags=sentry_tags_for_context(ctx),
    )


def message_stall_escalation(ctx: RunContext, *, last_heartbeat_at: str, escalated_to: str) -> None:
    """Fire ``enrichment.stall_escalation`` Sentry message (error level).

    Indicates the heartbeat watchdog had to cancel an enricher.
    Likely a real bug or a corrupt input — worth investigating.
    """
    capture_enrichment_message(
        f"{ctx.enricher_id} stall escalated to {escalated_to} "
        f"(last_heartbeat_at={last_heartbeat_at})",
        level="error",
        tags=sentry_tags_for_context(ctx),
    )


def langfuse_kwargs_for(ctx: RunContext) -> dict:
    """Return the kwargs to pass into ``emit_langfuse_span(...)`` for an
    enrichment-driven LLM call.

    Composes the ``enricher_id`` / ``enricher_tier`` keys the
    ``langfuse_tracing`` extension expects so the trace is filterable
    by enricher via ``prod_recent_traces``. Caller still passes
    ``provider`` / ``capability`` / ``model`` / ``cost`` directly.
    """
    md = langfuse_metadata_for_context(ctx)
    return {
        "enricher_id": md["enricher_id"],
        "enricher_tier": md["tier"],
        "run_seed": md["run_id"],
    }
