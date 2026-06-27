"""QueryEnricherRegistry — flat registry of in-flight query enrichers.

Mirrors :class:`podcast_scraper.enrichment.registry.EnricherRegistry`
in shape and contract, but operates on
:class:`~podcast_scraper.enrichment.query_protocol.QueryEnricher`.

The search route holds one instance; chunk 7 wires per-profile sets.
"""

from __future__ import annotations

import logging
from typing import Iterable

from podcast_scraper.enrichment.query_protocol import (
    make_request_ctx,
    QueryEnricher,
    QueryResultEnvelope,
)

logger = logging.getLogger(__name__)


class QueryEnricherRegistry:
    """Flat registry of ``QueryEnricher`` instances keyed by ``manifest.id``."""

    def __init__(self) -> None:
        self._enrichers: dict[str, QueryEnricher] = {}

    def register(self, enricher: QueryEnricher) -> None:
        """Register one QueryEnricher; raises on duplicate id."""
        mid = enricher.manifest.id
        if mid in self._enrichers:
            raise ValueError(f"query enricher already registered: {mid!r}")
        self._enrichers[mid] = enricher

    def get(self, enricher_id: str) -> QueryEnricher:
        """Look up a registered QueryEnricher by id."""
        return self._enrichers[enricher_id]

    def all_ids(self) -> list[str]:
        """Return every registered QueryEnricher id, insertion order."""
        return list(self._enrichers.keys())

    def clear(self) -> None:
        """Drop every registered QueryEnricher (test fixture cleanup)."""
        self._enrichers.clear()

    async def run_chain(
        self,
        *,
        envelope: QueryResultEnvelope,
        request_id: str,
        enricher_ids: Iterable[str] | None = None,
        config_by_id: dict[str, dict] | None = None,
    ) -> QueryResultEnvelope:
        """Run the named query enrichers (or all of them) sequentially.

        Each enricher runs with its own per-request RunContext derived
        from ``request_id``. Failures from one enricher don't break
        the chain — they're logged and the original envelope is passed
        through to the next.
        """
        ids = list(enricher_ids) if enricher_ids is not None else self.all_ids()
        configs = config_by_id or {}
        current = envelope
        for eid in ids:
            enricher = self._enrichers.get(eid)
            if enricher is None:
                logger.warning("query enricher %r not registered; skipping", eid)
                continue
            ctx = make_request_ctx(
                request_id=request_id, enricher_id=eid, tier=enricher.manifest.tier.value
            )
            try:
                current = await enricher.enrich_query_result(
                    envelope=current, config=configs.get(eid, {}), ctx=ctx
                )
            except Exception as exc:  # noqa: BLE001 — never break the chain
                logger.warning(
                    "query enricher %r raised %s; passing envelope through unmodified",
                    eid,
                    exc,
                )
        return current


__all__ = ["QueryEnricherRegistry"]
