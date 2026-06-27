"""QueryEnricher protocol — per-request enrichment of search results (RFC-088 Phase 4).

A QueryEnricher decorates a search response in-flight. Examples:

* ``query_topic_relatedness`` (chunk 5) — adds precomputed
  ``topic_similarity`` ranks to each hit's topic metadata.
* future LLM-tier query enrichers (separate RFC) — rerank, summarise,
  add follow-up questions.

Cross-request isolation: every invocation gets a fresh per-request
``RunContext`` derived from the search route's ``request_id`` UUID.
Cost-cap enforcement plumbing from chunk 1 is reused — LLM query
enrichers only populate the manifest's cost fields; the executor's
cap logic doesn't change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from podcast_scraper.enrichment.protocol import EnricherTier, RunContext


@dataclass(frozen=True)
class QueryEnricherManifest:
    """Declares a query enricher's identity, tier, and cost-cap fields."""

    id: str
    version: str
    tier: EnricherTier
    description: str
    # Reuses the same cost-cap fields as the artifact-enricher manifest
    # so LLM query enrichers plug into chunk-1 enforcement without
    # new code.
    max_cost_usd_per_request: float | None = None
    expected_duration_s: int | None = None


@dataclass
class QueryResultEnvelope:
    """The in-flight payload passed through the query-enricher chain.

    ``hits`` is the list the search route returns; query enrichers may
    annotate (in place or via the returned envelope) but never remove
    hits — removing breaks the route's contract.
    """

    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class QueryEnricher(Protocol):
    """Async protocol for in-flight search-result enrichers."""

    @property
    def manifest(self) -> QueryEnricherManifest:
        """Declare identity, tier, and cost cap."""
        ...

    async def enrich_query_result(
        self,
        *,
        envelope: QueryResultEnvelope,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> QueryResultEnvelope:
        """Annotate the envelope and return it. Never raise; convert
        backend exceptions to the cooperative-cancel result via
        ``ctx.cancel_event``, or just return the envelope unmodified."""
        ...


# Public helpers ---------------------------------------------------------


def make_request_ctx(*, request_id: str, enricher_id: str, tier: str) -> RunContext:
    """Per-request RunContext factory for the search route.

    The search route generates a fresh ``request_id`` UUID per call;
    every query enricher in the chain shares it.
    """
    return RunContext(
        run_id=request_id,
        parent_run_id=None,
        enricher_id=enricher_id,
        enricher_version="query",
        tier=tier,
        attempt=1,
        job_id=request_id,
        cancel_event=asyncio.Event(),
    )


__all__ = [
    "QueryEnricher",
    "QueryEnricherManifest",
    "QueryResultEnvelope",
    "make_request_ctx",
]
