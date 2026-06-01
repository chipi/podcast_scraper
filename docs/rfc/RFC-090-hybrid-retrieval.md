# RFC-090: Hybrid Retrieval Pipeline

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-032-hybrid-corpus-search.md` — hybrid corpus search (parent)
  - `docs/prd/PRD-031-search.md` — Search product surface
  - `docs/prd/PRD-021-semantic-corpus-search.md` — predecessor (FAISS)
- **Related ADRs**:
  - _(none yet)_
- **Related RFCs**:
  - `docs/rfc/RFC-091-kg-proximity-signal.md` — third RRF signal (additive)
  - `docs/rfc/RFC-092-ml-query-router.md` — ML query router (additive)
  - `docs/rfc/RFC-093-litm-context-packs.md` — LITM context packs (additive)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — canonical identity
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` — enrichment layer
- **Related UX specs**:
  - _(viewer Search surface — see PRD-031)_
- **Related Documents**:
  - `docs/architecture/kg/kg.schema.json` — KG edge schema (today: `MENTIONS`/`RELATED_TO`)

> **Stabilization note (2026-05-30):** Rebased from an earlier draft (RFC-078) authored against a
> different numbering universe. Module paths corrected to `src/podcast_scraper/…`; references to an
> "enrichment" RFC and an "MCP integration" RFC corrected to RFC-088 and **[TBD — no MCP RFC
> exists]**. The third signal, ML router, and context packs are split into RFC-091/092/093.
> **Prerequisite reality check** (see Constraints): the entity resolver, in-memory KG access layer,
> and most typed KG edges this pipeline's _additive_ signals assume **do not exist yet** — they are
> the genuine survivors of issue #466 and are tracked in #849. RFC-090 itself
> (BM25 + vector + RRF over two tiers) does **not** depend on them.

---

## Abstract

Replace single-signal FAISS vector search with a two-tier, three-signal hybrid retrieval pipeline.
Tier 1 indexes transcript segments (raw evidence); Tier 2 indexes GIL insight nodes (synthesized
intelligence). Signals — BM25, dense vector, and (additively, in RFC-091) KG proximity — are fused
via Reciprocal Rank Fusion (RRF) into a single mixed result list. A `SearchBackend` protocol
decouples signal generation from fusion, enabling backend substitution without touching retrieval
or MCP layers. LanceDB embedded is the initial backend.

**Architecture Alignment:** Extends the existing search package
(`src/podcast_scraper/search/`) rather than replacing it; reuses the GIL insight contracts
(`src/podcast_scraper/gi/contracts.py`) and the existing chunker
(`src/podcast_scraper/search/chunker.py`). The `SearchBackend` protocol follows the RFC-016
modularization principle (swap implementations behind a stable interface).

## Problem Statement

Single-signal FAISS over insights only produces three failure classes:

1. **Missed raw evidence.** Transcript segments not captured as GIL insights are invisible to
   search — exact phrase matches, specific terminology, minority-view content below the insight
   extraction threshold are all unreachable.
2. **Named-entity degradation.** Embedding models under-weight proper nouns; BM25 fixes this.
3. **No relational context.** KG edge structure contributes nothing to ranking. RFC-091 adds this
   as the third signal; this RFC builds the fusion layer it plugs into.

**Use Cases:**

1. **Raw evidence**: A user searches for an exact phrase and finds the transcript segment, even if
   no insight was extracted from it.
2. **Named entity**: A user searches "Sam Altman" and gets results containing the name directly.
3. **Mixed-tier synthesis**: A cross-show question returns both distilled insights and supporting
   raw segments, deduplicated into compound results.

## Goals

1. **Two-tier retrieval**: Segments and insights both indexed and retrievable; results mixed by
   score.
2. **Named-entity recall**: BM25 signal restores proper-noun matching.
3. **Clean fusion layer**: RRF in the retrieval layer, ready to accept a third (KG) signal with no
   backend change.
4. **Compound dedup**: Segment+insight pairs referring to the same content merge into one result.
5. **Swappable backend**: `SearchBackend` protocol; LanceDB first.

## Constraints & Assumptions

**Constraints:**

- Embedded, local-first backend (LanceDB) on the operator's machine; no cloud dependency.
- `lancedb` is **not currently a dependency** (`pyproject.toml` ships `faiss-cpu`); adding it is
  part of this RFC.
- No changes to GIL grounding invariants or the enrichment pipeline.

**Assumptions:**

- GIL insight grounding quotes carry `timestamp_start_ms` / `timestamp_end_ms`
  (`src/podcast_scraper/gi/contracts.py`), enabling timestamp-based segment↔insight linking.
- Whisper output segments (`{text, start, end, speaker_id?}`) are available at index time. Today
  they are passed into GI/KG builders as optional sidecar data
  (`src/podcast_scraper/gi/pipeline.py`), not retained as a standalone artifact (OQ-5).
- **Additive-signal prerequisites are NOT assumed by this RFC.** KG proximity (RFC-091) needs an
  in-memory KG access layer and an entity resolver that do not exist yet; RFC-090 ships BM25 +
  vector + RRF without them.

## Design & Implementation

### 1. Document Schemas

**Segment document (Tier 1)**

```python
# src/podcast_scraper/search/backend.py

@dataclass
class SegmentDocument:
    id: str                      # "{episode_id}_chunk_{n}"
    text: str                    # 200-300 word chunk, 50-word overlap
    embedding: list[float]
    show_id: str
    episode_id: str
    speaker_id: str | None       # if diarized
    start_time: float            # seconds from episode start
    end_time: float
    linked_insight_ids: list[str]  # GIL insight IDs whose grounding quote
                                   # falls within this chunk (timestamp overlap)
    source_tier: str = "segment"
```

**Insight document (Tier 2)**

```python
@dataclass
class InsightDocument:
    id: str                      # GIL insight node ID
    text: str
    embedding: list[float]
    show_id: str
    episode_id: str
    speaker_id: str | None
    entity_type: str
    confidence: float
    derived: bool
    source_segment_id: str | None  # back-ref to segment with grounding quote
    source_tier: str = "insight"
```

### 2. SearchBackend Protocol

```python
# src/podcast_scraper/search/backend.py

from typing import Protocol, runtime_checkable, Literal
from dataclasses import dataclass, field

Tier = Literal["segment", "insight", "all"]

@dataclass
class SearchQuery:
    text: str
    embedding: list[float]
    filters: dict = field(default_factory=dict)
    k: int = 20
    tier: Tier = "all"

@dataclass
class ScoredResult:
    doc_id: str
    score: float
    rank: int
    payload: dict
    signal: str                  # "bm25" | "vector" | "kg" | "rrf"
    source_tier: str             # "segment" | "insight" | "compound"

@dataclass
class CompoundResult:
    """Merged result when a segment and insight refer to the same content."""
    doc_id: str                  # segment id (primary key)
    score: float                 # max(segment_score, insight_score)
    rank: int
    segment: ScoredResult
    insight: ScoredResult
    signal: str = "rrf"
    source_tier: str = "compound"

@runtime_checkable
class SearchBackend(Protocol):
    def search_bm25(self, query: SearchQuery) -> list[ScoredResult]: ...
    def search_vector(self, query: SearchQuery) -> list[ScoredResult]: ...
    def upsert_segment(self, doc: SegmentDocument) -> None: ...
    def upsert_insight(self, doc: InsightDocument) -> None: ...
    def delete(self, doc_id: str, tier: Tier) -> None: ...
    def create_indices(self) -> None: ...
    def health(self) -> dict: ...
```

### 3. RRF Fusion with Tier Weights

```python
# src/podcast_scraper/search/fusion.py

TIER_WEIGHTS = {"insight": 1.2, "segment": 1.0}

def rrf_fuse(ranked_lists, k=60, signal_weights=None, tier_weights=None):
    """RRF score(d) = sum( (signal_weight * tier_weight) / (k + rank_i(d)) )"""
    signal_weights = signal_weights or {}
    tier_weights = tier_weights or TIER_WEIGHTS
    scores, payloads, tiers = {}, {}, {}
    for result_list in ranked_lists:
        if not result_list:
            continue
        signal = result_list[0].signal
        for result in result_list:
            sw = signal_weights.get(signal, 1.0)
            tw = tier_weights.get(result.source_tier, 1.0)
            scores.setdefault(result.doc_id, 0.0)
            payloads.setdefault(result.doc_id, result.payload)
            tiers.setdefault(result.doc_id, result.source_tier)
            scores[result.doc_id] += (sw * tw) / (k + result.rank)
    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [
        ScoredResult(d, scores[d], i + 1, payloads[d], "rrf", tiers[d])
        for i, d in enumerate(sorted_ids)
    ]
```

### 4. Deduplication — Compound Results

After RRF, deduplicate segment+insight pairs referring to the same content into a `CompoundResult`
taking the higher score. Full implementation in `src/podcast_scraper/search/dedup.py`: for each
insight result, if a segment result exists whose `linked_insight_ids` (or the insight's
`source_segment_id`) matches, merge the two; carry both, take `max` score and `min` rank;
re-sort by score.

### 5. Retrieval Layer

```python
# src/podcast_scraper/search/retrieval.py

class RetrievalLayer:
    def __init__(self, backend: SearchBackend):
        self.backend = backend

    def retrieve(self, text, embedding, filters=None, k=20,
                 query_type="hybrid", tier="all",
                 signal_weights=None, tier_weights=None):
        query = SearchQuery(text=text, embedding=embedding,
                            filters=filters or {}, k=k, tier=tier)
        ranked_lists = []
        if query_type in ("hybrid", "bm25"):
            ranked_lists.append(self.backend.search_bm25(query))
        if query_type in ("hybrid", "vector"):
            ranked_lists.append(self.backend.search_vector(query))

        # KG proximity slot — filled by RFC-091 (requires KG access + entity resolver):
        # entity_id = self.entity_resolver.resolve(text)
        # if entity_id:
        #     ranked_lists.append(self.kg_proximity.search(entity_id, k=k))

        if len(ranked_lists) == 1:
            return deduplicate(ranked_lists[0])
        fused = rrf_fuse(ranked_lists, signal_weights=signal_weights or {},
                         tier_weights=tier_weights or TIER_WEIGHTS)
        return deduplicate(fused)
```

### 6. Query-Type Router (Rules-Based v1)

Full ML router is RFC-092. Minimal rules-based version ships here
(`src/podcast_scraper/search/router.py`): keyword/regex routing to one of `raw_evidence`,
`temporal_tracking`, `cross_show_synthesis`, `entity_lookup`, `semantic`, plus per-type
`SIGNAL_WEIGHTS` and `TIER_WEIGHTS_BY_QUERY`. `raw_evidence` flips tier weights to favour segments
(segment 1.3, insight 0.9).

### 7. LanceDB Backend

Two tables (`segments`, `insights`); `search_bm25` / `search_vector` route by `query.tier`.
Implementation in `src/podcast_scraper/search/backends/lancedb_backend.py`: FTS index on `text`,
vector index on `embedding`, `merge_insert` upserts, `where(...)` SQL filters (see OQ-3 on
parameterisation), and a `health()` returning per-table row counts.

### 8. Segment Chunking

**Reuse the existing chunker** at `src/podcast_scraper/search/chunker.py` (already used by
`indexer.py`) rather than creating a new pipeline package. Extend it to emit `SegmentDocument`s and
to link insights by timestamp overlap:

```python
# src/podcast_scraper/search/chunker.py  (extend existing)

CHUNK_WORDS = 250
OVERLAP_WORDS = 50

def link_insights_to_segments(chunks, insights, tolerance_seconds=2.0):
    """Populate linked_insight_ids on segments and source_segment_id on insights
    by timestamp overlap with a tolerance window. Mutates in place."""
    for insight in insights:
        quote_start = insight.payload.get("quote_start_time")
        quote_end = insight.payload.get("quote_end_time")
        if quote_start is None:
            continue
        for chunk in chunks:
            if (chunk.start_time - tolerance_seconds <= quote_start
                    and quote_end <= chunk.end_time + tolerance_seconds):
                chunk.linked_insight_ids.append(insight.id)
                insight.source_segment_id = chunk.id
                break
```

### 9. Migration & Configuration

`search/migration.py::migrate_faiss_to_lance` re-projects the existing FAISS store into the
`segments` (Tier 1, from `transcript` chunks) and `insights` (Tier 2) LanceDB tables with FTS +
vector indices. It **reuses the FAISS embeddings verbatim** rather than re-embedding — deliberately,
so the Stage-4 eval holds the dense signal constant and isolates the BM25 + RRF contribution. The
migration is idempotent (merge-insert on `id`). Backend selected via `config/search.yaml`
(`backend: lancedb`); the router mode (`rules` | `ml`) is configured in the same file.

This migration is the first step of the 2.6 → 2.7 corpus upgrade; the managed upgrade-path runner
(ordering, version stamp, dry-run, rollback) that registers it is **#862**. A from-corpus indexer
(chunk via #857 + embed via `indexer` + link insights) is future work.

#### Stage-4 eval result (#858)

`scripts/eval_two_tier_retrieval.py` on the real corpus (149 known-item queries, k=10):

| system | recall@10 | MRR@10 | nDCG@10 |
| ------ | --------- | ------ | ------- |
| FAISS  | 1.000     | 0.993  | 0.995   |
| hybrid | 1.000     | 0.997  | 0.998   |

**Verdict:** hybrid does not regress and marginally improves ranking (MRR +0.003). The known-item
proxy **saturates** (recall 1.0 on both), so it confirms parity but cannot justify FAISS removal on
its own. **Decision: deprecate FAISS by notice, do not remove** (Phase 3 below is re-gated). Actual
removal waits on a discriminating, human-judged query set — which is also the labeled-query source
RFC-092 (#860) needs, so the two unblock together.

The discriminating eval is built: `scripts/eval_hybrid_judged.py` (RFC-057) runs a query set
through both backends, emits a graded-judgment template, and scores mean nDCG@k / recall@k per
backend once a human fills relevance. Its verdict requires a clear margin over ≥30 judged queries —
that result is what flips FAISS removal (#858) and ML-router promotion (#860) from gated to ready.

## Key Decisions

1. **Separate ranked lists, client-side RRF.**
   - **Decision**: Backend exposes `search_bm25()` and `search_vector()` separately; RRF lives in
     the retrieval layer.
   - **Rationale**: Enables KG proximity as a clean additive third list (RFC-091) without touching
     the backend.
2. **Two LanceDB tables, not one.**
   - **Decision**: Separate `segments` and `insights` tables.
   - **Rationale**: Different schemas, payloads, and FTS field requirements; one table would need
     nullable columns and complicate FTS config.
3. **Dedup at the retrieval layer, not the consumer.**
   - **Decision**: `CompoundResult` produced once in the retrieval layer.
   - **Rationale**: Consumers (viewer, MCP, autoresearch) receive a clean list.
4. **Tier weights in RRF, not pre-filter.**
   - **Decision**: Always search both tiers; weights adjust rank contribution.
   - **Rationale**: Pre-filtering by tier would miss cross-tier evidence; weights are cheaper and
     more robust.
5. **200–300 words, 50-word overlap.**
   - **Decision**: Standard RAG chunking over Whisper sentence segments.
   - **Rationale**: Balances context coherence with retrieval precision; revisit if eval shows
     boundary misalignment with grounding quotes.

## Alternatives Considered

1. **Single weighted index (one table, `source_tier` discriminator).**
   - **Pros**: Simpler schema; one FTS index.
   - **Cons**: Nullable columns, awkward FTS config, harder per-tier weighting.
   - **Why rejected**: KD-2 — two tables are cleaner for distinct schemas.
2. **Keep FAISS, add a separate BM25 store.**
   - **Pros**: No new vector dependency.
   - **Cons**: Two storage systems to keep in sync; no unified filter/upsert path; FAISS has no
     native FTS.
   - **Why rejected**: LanceDB unifies FTS + vector + filters behind one backend.
3. **Cross-encoder rerank instead of RRF.**
   - **Pros**: Potentially higher precision.
   - **Cons**: Heavier compute; needs a reranker model; harder to add a graph signal.
   - **Why rejected**: Deferred to post-RRF baseline (Non-Goal).

## Testing Strategy

**Test Coverage:**

- **Unit**: RRF math (score accumulation, weighting), compound dedup logic, chunker
  windowing/overlap, `SearchBackend` contract conformance.
- **Integration**: `raw_evidence` queries return segment-heavy results; named-entity queries return
  exact-match hits; compound results materialize when timestamps overlap.
- **Eval**: Two-tier hybrid vs FAISS baseline on a held-out query set — nDCG@10, named-entity
  recall, compound-result rate, segment-only result rate.

**Test Organization:** Under `tests/unit/podcast_scraper/search/` and the existing eval harness;
fixtures from a small committed corpus slice.

**Test Execution:** Unit/integration in `ci-fast`; eval as an operator/CI job against a fixture
index (not per-PR).

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1 — Core abstractions + LanceDB**: schemas, `LanceDBBackend`, `rrf_fuse()`,
  `deduplicate()`, `RetrievalLayer` (BM25 + vector, KG slot commented), chunker extension, migration
  script, unit tests.
- **Phase 2 — Query router + tier weights**: rules-based `classify_query()`, per-type weights,
  wire into `RetrievalLayer`, integration tests.
- **Phase 3 — Eval + FAISS deprecation**: baseline vs hybrid eval (done — see §9 Stage-4 result).
  FAISS is **deprecated by notice** (`faiss_store.py` docstring); **removal is re-gated** on a
  discriminating human-judged eval, because the known-item proxy saturated. Cutover orchestration:
  #862.

**Monitoring:** `make search-health` (segment/insight counts); eval metrics tracked over time;
log unlinked insights post-migration to inspect linking miss rate.

**Success Criteria:** nDCG@10 improvement over FAISS; named-entity recall ≥ 90% @10; >20% of top-10
segment-tier on raw-evidence queries.

## Relationship to Other RFCs

This RFC is the foundation of the Search initiative:

1. **RFC-091 KG Proximity Signal** — fills the reserved third-signal slot; adds the KG-access and
   entity-resolver dependencies (which do not exist yet).
2. **RFC-092 ML Query Router** — replaces the rules-based router once eval data exists.
3. **RFC-093 LITM Context Packs** — consumes `RetrievalLayer` output for agent-facing MCP packs.

**Key Distinction:**

- **RFC-090 (this)**: Two-tier index, BM25 + vector, RRF fusion, compound dedup — shippable today.
- **RFC-091/092/093**: Additive signals/routing/packaging — gated on prerequisites.

Other relationships:

| RFC | Relationship |
| --- | --- |
| RFC-072 | Canonical IDs used as entity filters in `SearchQuery` (slug IDs today; resolver pending) |
| RFC-088 | Enriched insight nodes are Tier 2 documents; `derived: true` is a payload field |
| RFC-080 | Viewer Graph surface consumes `RetrievalLayer` signals (node size / edge weight) |

## Benefits

1. **Named-entity recall**: BM25 restores proper-noun matching that embeddings lose.
2. **Raw evidence reachable**: Transcript segments become first-class search targets.
3. **Extensible fusion**: RRF accepts a third signal with zero backend change.
4. **Swappable backend**: `SearchBackend` protocol isolates storage choice.

## Migration Path

1. **Phase 1**: Add `lancedb`; build two-tier index via the migration script alongside the existing
   FAISS index (no removal yet).
2. **Phase 2**: Route viewer/MCP search through `RetrievalLayer`; keep FAISS as fallback.
3. **Phase 3**: FAISS deprecated by notice (eval confirmed parity, not a clear win — §9). `faiss-cpu`
   removal deferred to a later PR, gated on a human-judged eval and the #862 upgrade runner.

## Open Questions

1. **OQ-1 FTS rebuild on delta upserts.** Does LanceDB FTS require full rebuild per upsert batch or
   support incremental updates? Schedule indexing post-ingestion regardless.
2. **OQ-2 Embedding alignment on migration.** Validate dimensionality before write; re-embed from
   raw text if mismatch.
3. **OQ-3 Filter SQL injection.** `_to_sql()` is naive string interpolation — parameterise before
   any user-facing filter input reaches it.
4. **OQ-4 Chunk boundary alignment.** 2-second tolerance in `link_insights_to_segments()` may need
   tuning; log unlinked insights post-migration.
5. **OQ-5 Speaker diarization integration.** `speaker_id` on segments is nullable; when diarization
   is available, segments inherit the dominant speaker within the chunk. Separate PR.

## References

- **Related PRD**: `docs/prd/PRD-032-hybrid-corpus-search.md`, `docs/prd/PRD-031-search.md`
- **Related RFC**: `docs/rfc/RFC-091-kg-proximity-signal.md`, `docs/rfc/RFC-092-ml-query-router.md`,
  `docs/rfc/RFC-093-litm-context-packs.md`
- **Source Code**: `src/podcast_scraper/search/` (existing FAISS pipeline, chunker, indexer),
  `src/podcast_scraper/gi/contracts.py` (insight grounding quotes)
- **Prerequisites**: issue #849 · **Parent epic**: issue #466 (GI/KG depth roadmap — superseded)
