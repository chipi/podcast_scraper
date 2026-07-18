"""Hybrid retrieval layer: signals → RRF → compound dedup (RFC-090 §3.5).

Ties the backend (#855), fusion (RRF), router (intent → weights), and compound
dedup together. The KG-proximity signal (RFC-091 / #859) plugs into the reserved
slot below without touching this layer's contract (RFC-090 KD-1).
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .backend import SearchBackend, SearchQuery, Tier
from .dedup import deduplicate, Result
from .fusion import rrf_fuse
from .router import classify_query, signal_weights_for, tier_weights_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..identity.resolver import EntityResolver
    from .kg_proximity import KGProximitySearch
    from .query_router import QueryRouter


class RetrievalLayer:
    """Runs the configured signals over a ``SearchBackend`` and fuses them."""

    def __init__(
        self,
        backend: SearchBackend,
        *,
        kg_proximity: "Optional[KGProximitySearch]" = None,
        entity_resolver: "Optional[EntityResolver]" = None,
        router: "Optional[QueryRouter]" = None,
    ):
        self.backend = backend
        self.kg_proximity = kg_proximity
        self.entity_resolver = entity_resolver
        self.router = router

    def _classify(self, text: str) -> str:
        """Intent for *text* via the injected router, else the rules default."""
        return self.router.classify(text) if self.router is not None else classify_query(text)

    @staticmethod
    def classify(text: str) -> str:
        """Detected query intent for *text* (delegates to the rules router)."""
        return classify_query(text)

    def retrieve(
        self,
        text: str,
        embedding: List[float],
        *,
        filters: Optional[dict] = None,
        k: int = 20,
        intent: Optional[str] = None,
        signals: str = "hybrid",
        tier: Tier = "all",
    ) -> List[Result]:
        """Retrieve fused, deduplicated results.

        ``intent`` selects per-query signal/tier weights (auto-classified when
        None). ``signals`` chooses which signals to run (``hybrid`` | ``bm25`` |
        ``vector``). Returns a score-ordered list of ``ScoredResult`` /
        ``CompoundResult``.
        """
        query = SearchQuery(text=text, embedding=embedding, filters=filters or {}, k=k, tier=tier)

        # #1205: LanceDB's native in-engine hybrid (``backend.search_hybrid``, ADR-099 Stage 2)
        # hard-crashes the api — its score-normalize step calls native ``pyarrow.compute`` and
        # SIGSEGVs the worker under the digest route's search fan-out (confirmed in the stack-test
        # api faulthandler: pyarrow.compute <- lancedb ``_normalize_scores`` <- ``_combine_hybrid_
        # results``). Route the default ``hybrid`` signal through the Python-side vector+BM25+RRF
        # fan-out below (the pre-Stage-2 path) instead: two single-modality queries fused by
        # ``rrf_fuse`` never touch the crashing native combine, and this restores the router's
        # per-intent tier/signal weighting the in-engine reranker had dropped.
        intent = intent or self._classify(text)

        ranked_lists = []
        if signals in ("hybrid", "bm25"):
            ranked_lists.append(self.backend.search_bm25(query))
        if signals in ("hybrid", "vector"):
            ranked_lists.append(self.backend.search_vector(query))

        # KG-proximity signal (RFC-091): resolve an entity from the query, traverse
        # the cross-layer graph, append a third ranked list. Graceful skip when the
        # KG components are absent or no entity resolves.
        if self.kg_proximity is not None and self.entity_resolver is not None:
            entity_id = self.entity_resolver.resolve(text)
            if entity_id:
                kg_results = self.kg_proximity.search(entity_id, k=k, filters=filters or {})
                if kg_results:
                    ranked_lists.append(kg_results)

        if len(ranked_lists) == 1:
            return deduplicate(ranked_lists[0])
        fused = rrf_fuse(
            ranked_lists,
            signal_weights=signal_weights_for(intent),
            tier_weights=tier_weights_for(intent),
        )
        return deduplicate(fused)
