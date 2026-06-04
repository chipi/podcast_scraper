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
from typing import Any, Dict, List, Optional, Set, Tuple

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

    def __init__(self, identity_map: Optional[Dict[str, str]] = None) -> None:
        self._nodes: Dict[str, Node] = {}
        self._adj: Dict[str, Set[str]] = {}
        # Typed adjacency (RFC-094 / #882): node -> [(neighbor, edge_type)], undirected.
        # Lets the relational-query layer distinguish meaning-bearing edges on the same
        # node pair (e.g. Person→Insight "STATES" vs Insight→MENTIONS→Entity). The
        # untyped `_adj` still backs `neighbors`/`bfs` (proximity is type-agnostic).
        self._typed_adj: Dict[str, List[Tuple[str, str]]] = {}
        # variant_id -> canonical_id (#852). Collapses cross-episode entity-spelling
        # variants at read-time; empty = faithful artifact union.
        self._id_map: Dict[str, str] = dict(identity_map or {})

    # --- construction ----------------------------------------------------------

    def _canon(self, node_id: str) -> str:
        return self._id_map.get(node_id, node_id)

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

    def _add_edge(self, frm: str, to: str, edge_type: str = "") -> None:
        self._adj.setdefault(frm, set()).add(to)
        self._adj.setdefault(to, set()).add(frm)  # undirected
        self._typed_adj.setdefault(frm, []).append((to, edge_type))
        self._typed_adj.setdefault(to, []).append((frm, edge_type))

    def _ingest(self, artifact: Dict[str, Any], source: str) -> None:
        for node in artifact.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            raw_id = node.get("id")
            if raw_id is None:
                continue
            nid = self._canon(strip_layer_prefixes(str(raw_id)))
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
            self._add_edge(
                self._canon(strip_layer_prefixes(str(frm))),
                self._canon(strip_layer_prefixes(str(to))),
                str(edge.get("type") or ""),
            )

    def _derive_speaker_links(self) -> None:
        """Add direct ``person ↔ insight`` edges via a shared quote (#849 Slice C).

        The KG/GIL artifacts have no `SPEAKER_OF` edge — a person reaches an
        insight only in 2 hops (person—`SPOKEN_BY`—quote—`SUPPORTED_BY`—insight).
        For the RFC-091 proximity signal the person→insight path is the most
        valuable, so we synthesize a 1-hop shortcut. This is purely structural
        (person→quote→insight via node types), so it needs no new artifact edge
        types — which is why Slice C does **not** touch the KG schema.
        """
        for nid, node in list(self._nodes.items()):
            if node.type != "person":
                continue
            # Snapshot neighbor sets — _add_edge mutates self._adj[nid] below.
            for quote_id in list(self._adj.get(nid, set())):
                q = self._nodes.get(quote_id)
                if q is None or q.type != "quote":
                    continue
                for insight_id in list(self._adj.get(quote_id, set())):
                    ins = self._nodes.get(insight_id)
                    if ins is not None and ins.type == "insight":
                        self._add_edge(nid, insight_id, "STATES")

    @classmethod
    def build(
        cls,
        corpus_dir: Path | str,
        *,
        validate: bool = False,
        derive_speaker_links: bool = False,
        identity_map: Optional[Dict[str, str]] = None,
    ) -> "CorpusGraph":
        """Build the unified graph from all GI + KG artifacts under *corpus_dir*.

        When ``derive_speaker_links`` is True, add 1-hop ``person ↔ insight``
        shortcuts (see ``_derive_speaker_links``). Off by default so the graph is
        a faithful union of the artifacts; RFC-091's proximity layer opts in.

        ``identity_map`` (#852) is an optional ``variant_id → canonical_id`` map
        (e.g. from entity canonicalization) applied at ingest so cross-episode
        spelling variants collapse to one node.
        """
        from ..gi.corpus import load_gi_artifacts
        from ..gi.explore import scan_artifact_paths as scan_gi_paths
        from ..kg.corpus import load_kg_artifacts, scan_kg_artifact_paths

        corpus_dir = Path(corpus_dir)
        graph = cls(identity_map=identity_map)
        # KG first, then GI: GI payloads win on overlap (see _upsert_node).
        for _path, data in load_kg_artifacts(scan_kg_artifact_paths(corpus_dir), validate=validate):
            graph._ingest(data, "kg")
        for _path, data in load_gi_artifacts(scan_gi_paths(corpus_dir), validate=validate):
            graph._ingest(data, "gi")
        if derive_speaker_links:
            graph._derive_speaker_links()
        logger.debug(
            "Built corpus graph: %d nodes, %d adjacency entries",
            len(graph._nodes),
            len(graph._adj),
        )
        return graph

    # --- access ----------------------------------------------------------------

    def get_node(self, node_id: Optional[str]) -> Optional[Node]:
        """Return the node for *node_id*, or ``None`` if absent/``None``."""
        if node_id is None:
            return None
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> List[str]:
        """Undirected neighbors of *node_id* (sorted for determinism)."""
        return sorted(self._adj.get(node_id, set()))

    def typed_neighbors(self, node_id: str, edge_type: str) -> List[str]:
        """Neighbors reached from *node_id* by an edge of *edge_type* (RFC-094 / #882).

        Sorted, de-duplicated. Edge type plus the neighbor's node type disambiguate the
        meaning-bearing edges — e.g. ``typed_neighbors(person, "STATES")`` yields the
        insights a person stated, while ``typed_neighbors(entity, "MENTIONS")`` yields
        insights that mention the entity, even though both are person/org↔insight pairs.
        """
        return sorted(
            {nbr for nbr, etype in self._typed_adj.get(node_id, ()) if etype == edge_type}
        )

    def degree(self, node_id: str) -> int:
        """Number of (undirected) neighbors of *node_id*."""
        return len(self._adj.get(node_id, ()))

    def nodes_by_type(self, node_type: str) -> List[str]:
        """Sorted ids of all nodes whose normalized type is *node_type*."""
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
_corpus_graphs: Dict[tuple[str, bool, bool], CorpusGraph] = {}
_corpus_graphs_lock = threading.Lock()


def get_corpus_graph(
    corpus_dir: Path | str,
    *,
    validate: bool = False,
    derive_speaker_links: bool = False,
    canonicalize_entities: bool = True,
) -> CorpusGraph:
    """Return the cached cross-layer graph for *corpus_dir*, building it once.

    ``canonicalize_entities`` (default True — the production path) builds the
    cross-episode entity canonical map (#852) from the corpus and applies it so
    spelling variants (`Tracy`/`Tracey Alloway`) collapse to one node. The map is
    derived deterministically from ``corpus_dir``, so it stays out of the cache key.
    Pass False for a faithful artifact union.
    """
    key = (str(Path(corpus_dir).resolve()), derive_speaker_links, canonicalize_entities)
    with _corpus_graphs_lock:
        if key not in _corpus_graphs:
            identity_map: Optional[Dict[str, str]] = None
            if canonicalize_entities:
                from ..kg.entity_clusters import build_entity_id_map

                identity_map = build_entity_id_map(corpus_dir)
            _corpus_graphs[key] = CorpusGraph.build(
                corpus_dir,
                validate=validate,
                derive_speaker_links=derive_speaker_links,
                identity_map=identity_map,
            )
        return _corpus_graphs[key]


def clear_corpus_graph_cache() -> None:
    """Clear the process cache (tests / after corpus re-index)."""
    with _corpus_graphs_lock:
        _corpus_graphs.clear()
