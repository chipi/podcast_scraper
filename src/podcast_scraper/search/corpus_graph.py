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
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..builders.bridge_builder import strip_layer_prefixes

logger = logging.getLogger(__name__)

# A raw diarization label that never got a real name (#1056): "SPEAKER_03",
# "Speaker 3", "speaker-12". Used to tell an unnamed host *voice* from a real
# person when reconciling network-feed hosts across a show's episodes.
_SPEAKER_LABEL_RE = re.compile(r"^\s*speaker[\s_\-]*\d+\s*$", re.IGNORECASE)


def _looks_like_speaker_label(name: Optional[str]) -> bool:
    """True when *name* is a bare diarization id (``SPEAKER_03``), not a real name."""
    return bool(name and _SPEAKER_LABEL_RE.match(str(name)))


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

    # --- feed-anchored host reconciliation (#1056) -----------------------------

    def _episode_podcast(self, episode_id: str) -> Optional[str]:
        """The podcast a given episode belongs to (via ``HAS_EPISODE``), or None."""
        for nbr in self.typed_neighbors(episode_id, "HAS_EPISODE"):
            node = self._nodes.get(nbr)
            if node is not None and node.type == "podcast":
                return nbr
        return None

    def _person_episodes(self, person_id: str) -> List[str]:
        """Episodes a person is attached to (Person→``MENTIONS``→Episode, #1056)."""
        return [
            nbr
            for nbr in self.typed_neighbors(person_id, "MENTIONS")
            if (n := self._nodes.get(nbr)) is not None and n.type == "episode"
        ]

    def _person_has_speech(self, person_id: str) -> bool:
        """True when the person is attributed any speech in the corpus — a ``SPOKEN_BY``
        quote or a derived ``STATES`` insight.

        The voice-match signal (#1169): a ``role == "host"`` person who never speaks anywhere
        in the corpus is a **metadata-only name** — feed-description NER noise ("Twitter") or a
        non-speaking mention — not a real host *voice*. This is the deterministic filter that
        recurrence alone cannot do: a feed description is constant across episodes, so a bogus
        name recurs just like the real host; only "does this name actually speak" separates them.
        """
        for etype in ("SPOKEN_BY", "STATES"):
            for nbr in self.typed_neighbors(person_id, etype):
                n = self._nodes.get(nbr)
                if n is not None and n.type in ("quote", "insight"):
                    return True
        return False

    def _demote_speechless_hosts(self) -> None:
        """Demote a named ``role == "host"`` person with no attributed speech to ``mentioned``.

        A host name that never speaks in the corpus is a metadata-only artefact (feed-description
        NER noise, a non-speaking mention), not a real host *voice* (#1169). Unnamed ``SPEAKER_NN``
        host voices are left untouched — the back-fill merge in :meth:`_reconcile_feed_hosts`
        handles those, not this demotion.

        No-op when the corpus carries no speech attribution at all (no ``SPOKEN_BY`` / ``STATES``
        anywhere): without it there is no basis to tell a real host voice from a name, so every
        host is left as-is (a KG-only corpus with no GI quotes stays a faithful union).
        """
        persons = [(nid, node) for nid, node in self._nodes.items() if node.type == "person"]
        if not any(self._person_has_speech(nid) for nid, _ in persons):
            return
        for nid, node in persons:
            if str(node.payload.get("role") or "").lower() != "host":
                continue
            if _looks_like_speaker_label(node.payload.get("name") or node.payload.get("label")):
                continue  # unnamed SPEAKER_NN host voice — back-fill handles it, not demotion
            if not self._person_has_speech(nid):
                node.payload["role"] = "mentioned"
                node.payload["host_demoted_reason"] = "no attributed speech (metadata-only, #1169)"

    def _reconcile_feed_hosts(self) -> None:
        """Name recurring hosts of network-authored feeds across a show (#1056).

        Each episode is diarized independently, so a recurring host whose name the
        per-episode roster never resolves (network feed: author is the org, no
        self-intro) surfaces as a bare ``SPEAKER_03`` Person — even when a *sibling*
        episode of the same show *did* name that host. We anchor cross-episode
        identity on **(feed, host role)** rather than voice fingerprints (no ML):
        when a show has exactly one *recurring* named host (host in ≥2 episodes), its
        unnamed host voices are merged into that person. Ambiguous shows (co-hosts,
        no recurring named host) are left unnamed but **tagged** so the surface can
        say "recurring host — not auto-named" instead of a bare ``SPEAKER_03``.

        Guards (deliberately conservative — a wrong name is worse than none):
        - only ``role == "host"`` voices are ever touched (never guests);
        - an unnamed voice is merged only if it is *feed-exclusive* (its episodes sit
          under a single podcast — a shared ``person:speaker-00`` node spanning shows
          is ambiguous and skipped);
        - the target must recur as host in ≥2 episodes of that feed;
        - exactly one such recurring named host per feed (else it's a co-host show).

        Runs a **voice-match confirmation** first (#1169): a named ``role == "host"`` person
        with no attributed speech anywhere is a metadata-only name (feed-description NER noise,
        a non-speaking mention) and is demoted to ``mentioned`` before the merge below — so the
        show-description host path can supply high-recall candidates without leaking false hosts.
        """
        self._demote_speechless_hosts()
        named, unnamed_pods, unnamed_eps = self._collect_feed_host_voices()

        # Aggregate feed-exclusive unnamed host voices per podcast. Episode-scoped ids (#1b) turn
        # a recurring unnamed host into many singleton nodes under one show rather than a single
        # shared node; deciding merge/tag on the *union* of their episodes makes the outcome
        # identical whether the corpus carries the legacy shared id or the new per-episode one.
        pod_voices: Dict[str, Set[str]] = defaultdict(set)  # pod -> its feed-exclusive voice ids
        pod_unnamed_eps: Dict[str, Set[str]] = defaultdict(set)  # pod -> union of their episodes
        for voice_id, pods in unnamed_pods.items():
            if len(pods) != 1:
                continue  # ambiguous: a shared SPEAKER_00 node spanning shows (legacy) — skip
            pod = next(iter(pods))
            pod_voices[pod].add(voice_id)
            pod_unnamed_eps[pod] |= unnamed_eps[voice_id]

        merge_map: Dict[str, str] = {}
        for pod, voices in pod_voices.items():
            recurring = {h for h, eps in named.get(pod, {}).items() if len(eps) >= 2}
            if len(recurring) == 1:
                merge_map.update({v: next(iter(recurring)) for v in voices})
            elif len(pod_unnamed_eps[pod]) >= 2:
                # Opt1: can't name it, but the show has a recurring unnamed host — say so honestly.
                note = self._recurring_host_note(pod)
                for voice_id in voices:
                    self._nodes[voice_id].payload["recurring_host_note"] = note

        if merge_map:
            self._apply_identity_map(merge_map)

    def _collect_feed_host_voices(
        self,
    ) -> "tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Set[str]], Dict[str, Set[str]]]":
        """Group host Person nodes per podcast into named vs. unnamed voices (#1056 helper).

        Returns ``(named, unnamed_pods, unnamed_eps)``: ``named[pod][host_id] -> episodes``;
        ``unnamed_pods[voice_id] -> podcasts it spans``; ``unnamed_eps[voice_id] -> its episodes``.
        """
        named: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        unnamed_pods: Dict[str, Set[str]] = defaultdict(set)
        unnamed_eps: Dict[str, Set[str]] = defaultdict(set)
        for pid, node in self._nodes.items():
            episodes = self._host_person_episodes(pid, node)
            if not episodes:
                continue
            is_unnamed = _looks_like_speaker_label(
                node.payload.get("name") or node.payload.get("label")
            )
            for ep in episodes:
                pod = self._episode_podcast(ep)
                if pod is None:
                    continue
                if is_unnamed:
                    unnamed_pods[pid].add(pod)
                    unnamed_eps[pid].add(ep)
                else:
                    named[pod][pid].add(ep)
        return named, unnamed_pods, unnamed_eps

    def _host_person_episodes(self, pid: str, node: Any) -> Optional[List[str]]:
        """Episodes a ``role == "host"`` Person node owns, or None when it isn't a host voice."""
        if node.type != "person":
            return None
        if str(node.payload.get("role") or "").lower() != "host":
            return None
        return self._person_episodes(pid) or None

    def _recurring_host_note(self, pod: str) -> str:
        """The honest "recurring host — not auto-named" tag for a show's unnamed host voice."""
        show = self._nodes.get(pod)
        show_title = (show.payload.get("title") if show else None) or "this show"
        return f"recurring host of {show_title} — not auto-named"

    def _apply_identity_map(self, extra_map: Dict[str, str]) -> None:
        """Collapse nodes per ``extra_map`` (variant→canonical), rebuilding adjacency.

        Generalises the ingest-time ``identity_map`` to a post-build merge: a remapped
        node's edges move onto its canonical target and the node disappears. A merged-away
        node never overrides the target's ``name``/``label`` (the target is the real
        person; the voice id is not), so a ``SPEAKER_03`` merge can't un-name a host.
        """
        self._id_map.update(extra_map)
        old_nodes, old_typed = self._nodes, self._typed_adj
        self._nodes, self._adj, self._typed_adj = {}, {}, {}
        for nid, node in old_nodes.items():
            cid = self._canon(nid)
            payload = node.payload
            if cid != nid:  # merged away — drop identity keys so it can't rename the target
                payload = {k: v for k, v in node.payload.items() if k not in ("name", "label")}
            self._upsert_node(cid, node.type, payload, next(iter(node.layers), "kg"))
        seen_edges: Set[Tuple[str, str, str]] = set()
        for frm, neighbors in old_typed.items():
            for to, etype in neighbors:
                cf, ct = self._canon(frm), self._canon(to)
                if cf == ct:
                    continue  # self-loop created by the merge
                key = (cf, ct, etype) if cf <= ct else (ct, cf, etype)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                self._add_edge(cf, ct, etype)

    @classmethod
    def build(
        cls,
        corpus_dir: Path | str,
        *,
        validate: bool = False,
        derive_speaker_links: bool = False,
        reconcile_hosts: bool = False,
        identity_map: Optional[Dict[str, str]] = None,
    ) -> "CorpusGraph":
        """Build the unified graph from all GI + KG artifacts under *corpus_dir*.

        When ``derive_speaker_links`` is True, add 1-hop ``person ↔ insight``
        shortcuts (see ``_derive_speaker_links``). Off by default so the graph is
        a faithful union of the artifacts; RFC-091's proximity layer opts in.

        When ``reconcile_hosts`` is True, name recurring network-feed hosts across a
        show's episodes (see ``_reconcile_feed_hosts``, #1056). Off by default; the
        exploration/relational surfaces opt in.

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
        if reconcile_hosts:
            graph._reconcile_feed_hosts()
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
_corpus_graphs: Dict[tuple[str, bool, bool, bool], CorpusGraph] = {}
_corpus_graphs_lock = threading.Lock()


def get_corpus_graph(
    corpus_dir: Path | str,
    *,
    validate: bool = False,
    derive_speaker_links: bool = False,
    reconcile_hosts: bool = False,
    canonicalize_entities: bool = True,
) -> CorpusGraph:
    """Return the cached cross-layer graph for *corpus_dir*, building it once.

    ``canonicalize_entities`` (default True — the production path) builds the
    cross-episode entity canonical map (#852) from the corpus and applies it so
    spelling variants (`Tracy`/`Tracey Alloway`) collapse to one node. The map is
    derived deterministically from ``corpus_dir``, so it stays out of the cache key.
    Pass False for a faithful artifact union.

    ``reconcile_hosts`` (#1056) names recurring network-feed hosts across a show's
    episodes; the exploration/relational surfaces opt in.
    """
    key = (
        str(Path(corpus_dir).resolve()),
        derive_speaker_links,
        reconcile_hosts,
        canonicalize_entities,
    )
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
                reconcile_hosts=reconcile_hosts,
                identity_map=identity_map,
            )
        return _corpus_graphs[key]


def clear_corpus_graph_cache() -> None:
    """Clear the process cache (tests / after corpus re-index)."""
    with _corpus_graphs_lock:
        _corpus_graphs.clear()
