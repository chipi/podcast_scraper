"""KG-proximity retrieval signal (RFC-091 / #859).

The third RRF signal: traverse the cross-layer ``CorpusGraph`` (#849) from a
query's resolved entity and return reachable insight/segment nodes scored by
inverse hop distance (``1/(hop+1)``). This encodes relational context — that two
insights connect because they're about the same person's position on the same
topic across shows — which embedding similarity cannot capture. It plugs into the
reserved slot in ``RetrievalLayer`` without any backend/fusion change.

Prerequisites (merged, #849): ``CorpusGraph`` (``bfs``/``get_node``) and
``EntityResolver`` (seeds the traversal from query text).
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from .backend import ScoredResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .corpus_graph import CorpusGraph, Node

# Nodes worth returning as retrieval hits (insights are ranked; segments arrive
# with two-tier indexing). Other node types (person/topic/episode) are traversal
# waypoints, not results.
_RESULT_TYPES = ("insight", "segment")


def _passes_filters(node: "Node", filters: Optional[Dict]) -> bool:
    if not filters:
        return True
    return all(node.payload.get(key) == value for key, value in filters.items())


class KGProximitySearch:
    """Scores insight/segment nodes by graph hop-distance from a seed entity."""

    def __init__(self, graph: "CorpusGraph", *, max_hops: int = 3):
        self.graph = graph
        self.max_hops = max_hops

    def search(
        self,
        entity_id: str,
        *,
        k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[ScoredResult]:
        """BFS from *entity_id*; return top-*k* insight/segment hits (signal ``kg``).

        Score decays as ``1/(hop+1)`` (1.0 at the seed, 0.5 one hop out, …). Nodes
        that are pure traversal waypoints (person/topic/episode) are skipped.
        """
        distances = self.graph.bfs(entity_id, max_hops=self.max_hops)
        results: List[ScoredResult] = []
        for node_id, hops in distances.items():
            node = self.graph.get_node(node_id)
            if node is None or node.type not in _RESULT_TYPES:
                continue
            if not _passes_filters(node, filters):
                continue
            results.append(
                ScoredResult(
                    doc_id=node_id,
                    score=1.0 / (hops + 1),
                    rank=0,
                    payload=node.payload,
                    signal="kg",
                    source_tier=node.type,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        return results[:k]
