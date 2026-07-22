"""Shared plumbing for the RFC-088 QueryEnricher chain, used by both the operator
viewer's ``/api/search`` and the consumer app's ``/api/app/search``.

The chain runs Python-side after retrieval and only *decorates* hits with
``metadata.query_enrichments.related_topics`` — it never mutates ordering or the
error field. Failures inside the chain are swallowed (logged) so a broken
enricher never breaks the search endpoint.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from podcast_scraper.enrichment.query_enrichers import (
    register_deterministic_query_enrichers,
)
from podcast_scraper.enrichment.query_protocol import QueryResultEnvelope
from podcast_scraper.enrichment.query_registry import QueryEnricherRegistry

if TYPE_CHECKING:
    from fastapi import Request

    from podcast_scraper.server.schemas import SearchHitModel

logger = logging.getLogger(__name__)


def _registry_for(request: Request, corpus_root: Path) -> QueryEnricherRegistry:
    """Cache the registry on app.state — it's stateless across requests; the
    corpus root flows in via ``corpus_root_provider`` at ``run_chain()`` time."""
    registry = getattr(request.app.state, "query_enricher_registry", None)
    if registry is None:
        registry = QueryEnricherRegistry()
        request.app.state.query_enricher_registry = registry
        request.app.state.query_enricher_corpus_root = corpus_root
        register_deterministic_query_enrichers(
            registry,
            corpus_root_provider=lambda: (
                request.app.state.query_enricher_corpus_root or corpus_root
            ),
        )
    request.app.state.query_enricher_corpus_root = corpus_root
    return registry


async def apply_query_enrichers(
    request: Request,
    corpus_root: Path,
    query: str,
    hits: list[SearchHitModel],
) -> None:
    """Run the QueryEnricher chain over ``hits`` and merge decorations back.

    Only field currently produced: ``metadata.query_enrichments.related_topics``.
    Mutates ``hits`` in place; failures inside the chain are logged and swallowed
    (the endpoint returns the plain top-k page in that case).
    """
    try:
        registry = _registry_for(request, corpus_root)
        envelope = QueryResultEnvelope(
            query=query,
            hits=[{"doc_id": h.doc_id, "metadata": dict(h.metadata)} for h in hits],
        )
        decorated = await registry.run_chain(envelope=envelope, request_id=str(uuid.uuid4()))
        for hit, decorated_hit in zip(hits, decorated.hits):
            related = decorated_hit.get("related_topics")
            if related is not None:
                hit.metadata.setdefault("query_enrichments", {})["related_topics"] = related
    except Exception as exc:  # noqa: BLE001 — enrichment never breaks search
        logger.warning("query enricher chain failed: %s", exc)
