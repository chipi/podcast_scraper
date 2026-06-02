# RFC-091: KG Proximity Signal

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-032-hybrid-corpus-search.md` — hybrid corpus search
  - `docs/prd/PRD-031-search.md` — Search product
- **Related ADRs**:
  - _(none yet)_
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — provides the RRF fusion slot (hard dependency)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — entity resolution
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` — contradiction signals (future edges)
- **Related Documents**:
  - `docs/architecture/kg/kg.schema.json` — KG edge schema (today: `MENTIONS`/`RELATED_TO`; GI also emits `ABOUT`)

> **Stabilization note (2026-05-30):** Split out of an earlier combined draft (RFC-079) that bundled
> KG proximity, the ML router, and LITM context packs under one number. This RFC is the KG-proximity
> third signal only; the router is RFC-092 and the packs are RFC-093. **This RFC is the most
> prerequisite-heavy of the three** — see Constraints. Its core dependencies (an in-memory KG access
> layer, typed traversal edges, and a freeform-text → canonical-ID entity resolver) are the genuine
> survivors of issue #466 and **do not exist in the codebase yet**.

---

## Abstract

Add the third retrieval signal to RFC-090: KG graph proximity. For a query, resolve the most
relevant canonical entity, traverse the knowledge graph from it over typed edges, and return
reachable insight/segment nodes scored by inverse hop distance. This list becomes the third input
to RRF in `RetrievalLayer.retrieve()`. It is additive — nothing in RFC-090 changes; the reserved
slot is filled.

**Architecture Alignment:** The signal plugs into the existing `RetrievalLayer` via the
already-reserved third-list slot (RFC-090 KD-1), so no backend or fusion changes are needed.
**However**, it introduces two new architectural dependencies that must be built first: a graph
access layer over the KG artifacts and an entity resolver over the identity layer.

## Problem Statement

BM25 and dense vector are commoditized signals. Graph hop distance — computed over this project's
specific ontology (person → topic → insight → episode → show) — is structurally unique to this
corpus. It encodes relational context that embedding similarity cannot capture: that two insights
are connected because they are about the same person's position on the same topic across different
shows, not because they share words.

Today the KG contributes nothing to ranking. The KG also is not currently queryable as a graph:
it is stored as per-episode `*.kg.json` artifacts (`src/podcast_scraper/kg/`) with no in-memory
graph object exposing `neighbors()` / `get_node()`, and the edge schema is limited to `MENTIONS` /
`RELATED_TO` (plus `ABOUT` emitted by the GI pipeline). So this RFC is gated on building that
access layer and enriching the edge set.

**Use Cases:**

1. **Relational recall**: A query about a person surfaces insights connected to them via typed
   edges even when wording differs from the query.
2. **Centrality boost**: Centrally-positioned, well-connected content ranks above isolated nodes
   for a relevant entity query.
3. **Graceful skip**: A query that references no known entity simply runs on two signals.

## Goals

1. **KG proximity as a third RRF signal**: scored by `1 / (hop + 1)`, max 3 hops.
2. **Entity resolution before traversal**: resolve query text → canonical ID; skip cleanly on miss.
3. **Zero RFC-090 disruption**: fill the reserved slot; no change to fusion or backend.
4. **Bounded cost**: pre-computable adjacency; live BFS acceptable at current corpus scale.

## Constraints & Assumptions

**Constraints — Prerequisites (tracked in #849):**

> **Cross-layer reality (2026-05-30 audit).** This RFC's traversal (`person → insight → episode →
> show`) assumes a single graph, but GIL and KG are **deliberately separate layers**: insights and
> quotes live in the GIL artifacts; entities, topics, and episodes live in the KG artifacts
> (`docs/architecture/kg/ontology.md` enforces the separation). They share canonical IDs
> (`person:…` / `topic:…`) and are joined per episode by `bridge.json`. So the prerequisite is **not**
> "add edges to the KG" — it is a **cross-layer graph** that unifies both via those shared IDs.

1. **In-memory cross-layer corpus graph.** — **SHIPPED (Slice B of #849):**
   `src/podcast_scraper/search/corpus_graph.py`. `CorpusGraph` unifies GIL + KG nodes/edges
   (id-keyed union on shared canonical IDs) and exposes `get_node()`, `neighbors()` (undirected),
   `bfs()`, `degree()`, `nodes_by_type()`, plus a process-cached `get_corpus_graph()`.
2. **Traversal edges — satisfied natively; no KG schema change (Slice C of #849, resolved):**
   - `ABOUT` (insight ↔ topic): exists in GIL (`gi/about_edges.py`).
   - `SPEAKER_OF` (person → insight): synthesized as an **opt-in 1-hop derived shortcut**
     (`CorpusGraph.build(derive_speaker_links=True)`, composing `SPOKEN_BY` + `SUPPORTED_BY`); also
     reachable in 2 hops without it.
   - `IN_EPISODE` (insight ↔ episode): GIL `HAS_INSIGHT`.
   - `COVERS` (topic ↔ episode), `MENTIONED_IN` (person ↔ episode): the existing KG `MENTIONS` edge
     (undirected, so no relabel needed for traversal).
   - `FROM_SHOW` (episode ↔ show): GIL `Podcast` nodes + `HAS_EPISODE` edges — already in the
     unified graph; **no new Show node / KG edge type added.**
3. **Entity resolver.** `EntityResolver.resolve(text) → canonical_id | None` —
   **SHIPPED (Slice A of #849):** `src/podcast_scraper/identity/resolver.py`, a corpus-wide registry
   over GIL + KG canonical entities with exact + fuzzy matching (reuses `bridge_builder`).

**Assumptions:**

- At current scale (~14 shows, ~700 episodes, ~15k insights) live BFS is fast enough; revisit if
  p99 latency is unacceptable (OQ-1).
- Conservative entity resolution (return `None` on low confidence) is preferred over aggressive
  matching (OQ-2).

## Design & Implementation

### 1. KG Proximity Search

```python
# src/podcast_scraper/search/kg_proximity.py
#
# PREREQUISITE (now shipped, #849): the in-memory ``CorpusGraph`` provides
# bfs()/get_node()/neighbors(). As built, traversal delegates to ``CorpusGraph.bfs``
# rather than the hand-rolled queue below, and the param is named ``graph`` (keyword-
# only). This sketch is illustrative; see ``search/kg_proximity.py`` for the real code.

class KGProximitySearch:
    def __init__(self, graph, *, max_hops: int = 3):
        self.kg = kg
        self.max_hops = max_hops

    def search(self, entity_id: str, k: int = 20, filters: dict = None):
        """BFS from entity_id over typed edges. Returns insight/segment nodes
        with score = 1 / (hop + 1)."""
        visited = {}                    # node_id -> min hop distance
        queue = [(entity_id, 0)]
        while queue:
            node_id, hops = queue.pop(0)
            if node_id in visited or hops > self.max_hops:
                continue
            visited[node_id] = hops
            for neighbor_id in self.kg.neighbors(node_id):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, hops + 1))

        results = []
        for node_id, hops in visited.items():
            node = self.kg.get_node(node_id)
            if node is None or node.type not in ("insight", "segment"):
                continue
            if not self._passes_filters(node, filters):
                continue
            results.append(ScoredResult(
                doc_id=node_id,
                score=1.0 / (hops + 1),
                rank=0,
                payload=node.payload,
                signal="kg",
                source_tier=node.type,
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results[:k]
```

### 2. Edge Types Used

| Edge | Direction | Layer / status |
| --- | --- | --- |
| `ABOUT` | insight → topic | **GIL — exists** (`gi/about_edges.py`) |
| `SPEAKER_OF` | person → insight | GIL — from `SPOKEN_BY` (insight nodes are GIL-only) |
| `IN_EPISODE` | insight → episode | GIL — from `HAS_INSIGHT` (insight nodes are GIL-only) |
| `COVERS` | episode → topic | KG — relabel of `MENTIONS` (Slice C) |
| `MENTIONED_IN` | person → episode | KG — relabel of `MENTIONS` (Slice C) |
| `FROM_SHOW` | episode → show | KG — needs a Show node type (Slice C) |

The insight-reaching edges (`SPEAKER_OF`, `IN_EPISODE`, `ABOUT`) live in the **GIL** layer and the
rest in the **KG** layer; proximity traverses the **unified cross-layer graph** (Slice B), joined
on shared canonical IDs. Max traversal: 3 hops (score decays `1/(hop+1)`; beyond 3 hops proximity
is noise).

### 3. Entity Resolution for Queries (integration into RetrievalLayer)

```python
# src/podcast_scraper/search/retrieval.py  (RFC-090 slot filled)

class RetrievalLayer:
    def __init__(self, backend, kg, entity_resolver):
        self.backend = backend
        self.kg_proximity = KGProximitySearch(kg)
        self.entity_resolver = entity_resolver      # PREREQUISITE — see Constraints

    def retrieve(self, text, embedding, filters=None, k=20, query_type="hybrid",
                 tier="all", signal_weights=None, tier_weights=None):
        query = SearchQuery(text=text, embedding=embedding,
                            filters=filters or {}, k=k, tier=tier)
        ranked_lists = []
        if query_type in ("hybrid", "bm25"):
            ranked_lists.append(self.backend.search_bm25(query))
        if query_type in ("hybrid", "vector"):
            ranked_lists.append(self.backend.search_vector(query))

        entity_id = self.entity_resolver.resolve(text)   # None on miss
        if entity_id:
            ranked_lists.append(self.kg_proximity.search(entity_id, k=k, filters=filters))

        if len(ranked_lists) == 1:
            return deduplicate(ranked_lists[0])
        fused = rrf_fuse(ranked_lists, signal_weights=signal_weights or {},
                         tier_weights=tier_weights or TIER_WEIGHTS)
        return deduplicate(fused)
```

### 4. Graceful Degradation

If `entity_resolver.resolve()` returns `None` (no known entity in the query), KG proximity is
skipped and RRF runs over two signals. No error, no degradation of the BM25 + vector result.

## Key Decisions

1. **BFS over the KG, max 3 hops.**
   - **Decision**: Limit traversal to 3 hops.
   - **Rationale**: Score decays as `1/(hop+1)` (a 3-hop node contributes 0.25 vs 1.0 for a direct
     connection); also bounds compute on large graphs.
2. **Entity resolution before traversal; graceful skip on failure.**
   - **Decision**: Resolve the query to a canonical entity first; skip KG entirely if unresolved.
   - **Rationale**: KG proximity is only meaningful with a canonical entity to traverse from.

## Alternatives Considered

1. **Embed the KG structure into vectors (node2vec-style) instead of live traversal.**
   - **Pros**: One vector signal; no traversal at query time.
   - **Cons**: Requires retraining on graph change; loses exact hop semantics; opaque.
   - **Why rejected**: Live typed-edge traversal is interpretable and cheap at this scale.
2. **Use `MENTIONS` edges only (ship without new edge types).**
   - **Pros**: No KG schema work.
   - **Cons**: `MENTIONS` (topic/entity → episode) is too coarse for person-position proximity.
   - **Why rejected**: The signal's value comes from the richer typed ontology; ship after edges.

## Testing Strategy

**Test Coverage:**

- **Unit**: BFS traversal correctness, score decay by hop, `k`-truncation, filter application,
  graceful skip on unresolved entity.
- **Integration**: KG signal changes ranking for entity queries on a fixture KG; two-signal
  fallback identical to RFC-090 output when entity unresolved.

**Test Organization:** `tests/unit/podcast_scraper/search/test_kg_proximity.py`; fixture KG built
from a small committed corpus slice with the new typed edges.

**Test Execution:** Unit in `ci-fast`; eval contribution measured in the RFC-090 eval harness
(hybrid+KG vs hybrid).

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 0 — Prerequisites (blocking)**: KG access layer, typed edges in KG extraction, entity
  resolver. Tracked in #849.
- **Phase 1 — KG proximity**: `KGProximitySearch`, wire into `RetrievalLayer`, signal weight for
  `kg` in `SIGNAL_WEIGHTS`, tests.

**Monitoring:** adjacency is built live from the corpus graph (no pre-build target; see the
"live BFS, no pre-computation" decision below); track p99 traversal latency and
KG-signal contribution to nDCG.

**Success Criteria:** KG signal improves nDCG@10 on entity/relational queries over hybrid-only,
with no regression on unresolved-entity queries.

## Relationship to Other RFCs

This RFC is signal 3 of 3 in the Search initiative.

**Key Distinction:**

- **RFC-090**: BM25 + vector + RRF foundation (the slot).
- **RFC-091 (this)**: KG proximity signal (fills the slot) — gated on KG/resolver prerequisites.
- **RFC-092 / RFC-093**: router and context packs (independent of this RFC's prerequisites).

| RFC | Relationship |
| --- | --- |
| RFC-090 | Provides the reserved third-list slot and RRF fusion (hard dependency) |
| RFC-072 | Entity resolver uses the canonical identity layer (Partial — resolver pending) |
| RFC-088 | Contradiction signals; when modelled as typed KG edges, they surface via proximity |

## Benefits

1. **A signal no generic search tool has**: domain-specific graph proximity.
2. **Interpretable**: exact hop semantics, not opaque graph embeddings.
3. **Additive and safe**: fills a reserved slot; degrades gracefully to two signals.

## Migration Path

1. **Phase 0**: Build prerequisites (graph access, typed edges, resolver) behind feature flags.
2. **Phase 1**: Enable KG signal in `RetrievalLayer`; A/B against hybrid-only in eval.
3. **Phase 2**: Promote adjacency pre-computation if live BFS latency is unacceptable (OQ-1).

## Open Questions

1. **OQ-1 KG adjacency pre-computation.** Live BFS vs cached adjacency list (node → [(neighbor,
   hop)]) at index-build time. Trade-off: stale cache vs live cost. Likely fine live at current
   scale; revisit on p99 latency.
2. **OQ-2 Entity resolver precision.** A false positive (wrong entity ID) produces a misleading
   signal. Conservative resolution (return `None` when unsure) is safer than aggressive matching.
3. **OQ-3 Typed-edge backfill.** Adding `SPEAKER_OF`/`IN_EPISODE`/`FROM_SHOW`/`COVERS`/
   `MENTIONED_IN` requires re-running KG extraction over the corpus. Sequence vs the RFC-090
   migration.

## References

- **Related PRD**: `docs/prd/PRD-032-hybrid-corpus-search.md`
- **Related RFC**: `docs/rfc/RFC-090-hybrid-retrieval.md`,
  `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- **Source Code**: `src/podcast_scraper/kg/` (artifact loaders — no traversal API yet),
  `src/podcast_scraper/identity/slugify.py` (slug only — no resolver),
  `src/podcast_scraper/gi/about_edges.py` (existing `ABOUT` edges)
- **Prerequisites**: issue #849 (KG depth survivors of #466) · **Parent epic**: #466 (superseded)
