"""Hybrid retrieval layer: signals → RRF → compound dedup (RFC-090 §3.5).

Ties the backend (#855), fusion (RRF), router (intent → weights), and compound
dedup together. The KG-proximity signal (RFC-091 / #859) plugs into the reserved
slot below without touching this layer's contract (RFC-090 KD-1).
"""

from __future__ import annotations

from typing import List, Optional

from .backend import SearchBackend, SearchQuery, Tier
from .dedup import deduplicate, Result
from .fusion import rrf_fuse
from .router import classify_query, signal_weights_for, tier_weights_for


class RetrievalLayer:
    """Runs the configured signals over a ``SearchBackend`` and fuses them."""

    def __init__(self, backend: SearchBackend):
        self.backend = backend

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
        intent = intent or classify_query(text)
        query = SearchQuery(text=text, embedding=embedding, filters=filters or {}, k=k, tier=tier)

        ranked_lists = []
        if signals in ("hybrid", "bm25"):
            ranked_lists.append(self.backend.search_bm25(query))
        if signals in ("hybrid", "vector"):
            ranked_lists.append(self.backend.search_vector(query))

        # KG-proximity slot — RFC-091 / #859 resolves an entity from the query and
        # appends a third ranked list here. No fusion/backend change needed.

        if len(ranked_lists) == 1:
            return deduplicate(ranked_lists[0])
        fused = rrf_fuse(
            ranked_lists,
            signal_weights=signal_weights_for(intent),
            tier_weights=tier_weights_for(intent),
        )
        return deduplicate(fused)
