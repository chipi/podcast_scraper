"""In-memory cross-layer corpus graph — unified GIL + KG traversal (#849 Slice B).

RFC-091's KG-proximity signal needs to traverse from a query entity to the
*insight* nodes that retrieval ranks. Insights live only in the GIL layer, while
entities/episodes/topics live in the KG layer; the two are joined on **exact
shared canonical IDs** (`person:{slug}`, `org:{slug}`, `topic:{slug}`,
`episode:{episode_id}` are identical across `*.gi.json` and `*.kg.json`,
RFC-072). This module loads both layers and unifies them into one undirected
in-memory graph with `neighbors()` / `get_node()` / `bfs()`.

Design notes:

- **Undirected adjacency.** Proximity is reachability, so an edge connects its
  endpoints in both directions (a 2-hop person→episode→insight path is what the
  signal exploits).
- **Type from id prefix.** Node `type` is derived from the canonical id prefix so
  the GIL `Person` node and the KG `Entity(kind=person)` node unify into one
  `person` node, and RFC-091's `type in ("insight", "segment")` check works.
- **Hand-rolled adjacency, live BFS.** At corpus scale (~700 episodes, ~15k
  insights, ~20–30k nodes) a dict adjacency with live ≤3-hop BFS is sufficient;
  no graph library or precomputed adjacency cache is needed (RFC-091 OQ-1).

Consumer: RFC-091 `KGProximitySearch` (separate module, builds on `bfs()`).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..builders.bridge_builder import strip_layer_prefixes

logger = logging.getLogger(__name__)

# Canonical id prefix → normalized node type. Keeps GIL `Person` and KG
# `Entity(kind=person)` as one `person` node, etc.
_TYPE_BY_PREFIX = {
    "person": "person",
    "org": "org",
    "topic": "topic",
    "episode": "episode",
    "insight": "insight",
    "quote": "quote",
    "podcast": "podcast",
    "segment": "segment",
}


def _normalize_type(node_id: str, raw_type: Any) -> str:
    """Derive node type from the canonical id prefix; fall back to artifact type."""
    prefix = node_id.split(":", 1)[0] if ":" in node_id else ""
    mapped = _TYPE_BY_PREFIX.get(prefix)
    if mapped:
        return mapped
    return str(raw_type or "").lower()


@dataclass
class Node:
    """A unified corpus-graph node."""

    id: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    layers: Set[str] = field(default_factory=set)  # {"gi", "kg"}


class CorpusGraph:
    """Undirected, in-memory union of the GIL and KG artifact graphs."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._adj: Dict[str, Set[str]] = {}

    # --- construction ----------------------------------------------------------

    def _upsert_node(self, node_id: str, ntype: str, props: Dict[str, Any], source: str) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            self._nodes[node_id] = Node(
                id=node_id, type=ntype, payload=dict(props), layers={source}
            )
            return
        # Later source (GI is ingested after KG) wins on overlapping keys — GIL
        # payloads are richer for insights/quotes/persons.
        node.payload.update(props)
        node.layers.add(source)
        if not node.type and ntype:
            node.type = ntype

    def _add_edge(self, frm: str, to: str) -> None:
        self._adj.setdefault(frm, set()).add(to)
        self._adj.setdefault(to, set()).add(frm)  # undirected

    def _ingest(self, artifact: Dict[str, Any], source: str) -> None:
        for node in artifact.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            raw_id = node.get("id")
            if raw_id is None:
                continue
            nid = strip_layer_prefixes(str(raw_id))
            props_any = node.get("properties")
            props = props_any if isinstance(props_any, dict) else {}
            self._upsert_node(nid, _normalize_type(nid, node.get("type")), props, source)
        for edge in artifact.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            frm = edge.get("from")
            to = edge.get("to")
            if frm is None or to is None:
                continue
            self._add_edge(strip_layer_prefixes(str(frm)), strip_layer_prefixes(str(to)))

    @classmethod
    def build(cls, corpus_dir: Path | str, *, validate: bool = False) -> "CorpusGraph":
        """Build the unified graph from all GI + KG artifacts under *corpus_dir*."""
        from ..gi.corpus import load_gi_artifacts
        from ..gi.explore import scan_artifact_paths as scan_gi_paths
        from ..kg.corpus import load_kg_artifacts, scan_kg_artifact_paths

        corpus_dir = Path(corpus_dir)
        graph = cls()
        # KG first, then GI: GI payloads win on overlap (see _upsert_node).
        for _path, data in load_kg_artifacts(scan_kg_artifact_paths(corpus_dir), validate=validate):
            graph._ingest(data, "kg")
        for _path, data in load_gi_artifacts(scan_gi_paths(corpus_dir), validate=validate):
            graph._ingest(data, "gi")
        logger.debug(
            "Built corpus graph: %d nodes, %d adjacency entries",
            len(graph._nodes),
            len(graph._adj),
        )
        return graph

    # --- access ----------------------------------------------------------------

    def get_node(self, node_id: Optional[str]) -> Optional[Node]:
        if node_id is None:
            return None
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> List[str]:
        """Undirected neighbors of *node_id* (sorted for determinism)."""
        return sorted(self._adj.get(node_id, set()))

    def degree(self, node_id: str) -> int:
        return len(self._adj.get(node_id, ()))

    def nodes_by_type(self, node_type: str) -> List[str]:
        return sorted(nid for nid, n in self._nodes.items() if n.type == node_type)

    def bfs(self, start_id: str, max_hops: int = 3) -> Dict[str, int]:
        """BFS hop-distances from *start_id* (inclusive), bounded by *max_hops*.

        Returns ``{node_id: hop_distance}`` for every node reachable within
        ``max_hops`` (start at hop 0). The primitive RFC-091 scores as
        ``1 / (hop + 1)``.
        """
        if start_id not in self._adj and start_id not in self._nodes:
            return {}
        dist: Dict[str, int] = {start_id: 0}
        queue: deque[str] = deque([start_id])
        while queue:
            cur = queue.popleft()
            hop = dist[cur]
            if hop >= max_hops:
                continue
            for nbr in self._adj.get(cur, ()):  # noqa: SIM118 - set membership iter
                if nbr not in dist:
                    dist[nbr] = hop + 1
                    queue.append(nbr)
        return dist

    def __len__(self) -> int:
        return len(self._nodes)


# Process-level cache (mirrors providers/ml/embedding_loader.py): graphs are
# expensive to build and reused across searches. Keyed by resolved corpus path.
_corpus_graphs: Dict[str, CorpusGraph] = {}
_corpus_graphs_lock = threading.Lock()


def get_corpus_graph(corpus_dir: Path | str, *, validate: bool = False) -> CorpusGraph:
    """Return the cached cross-layer graph for *corpus_dir*, building it once."""
    key = str(Path(corpus_dir).resolve())
    with _corpus_graphs_lock:
        if key not in _corpus_graphs:
            _corpus_graphs[key] = CorpusGraph.build(corpus_dir, validate=validate)
        return _corpus_graphs[key]


def clear_corpus_graph_cache() -> None:
    """Clear the process cache (tests / after corpus re-index)."""
    with _corpus_graphs_lock:
        _corpus_graphs.clear()
