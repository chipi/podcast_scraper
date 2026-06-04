# PRD-032: Hybrid Corpus Search

- **Status**: Draft
- **Author**: Marko
- **Created**: 2026-05-24
- **Target**: v2.7
- **Related PRDs**:
  - `docs/prd/PRD-031-search.md` — product surface that consumes this backend
  - `docs/prd/PRD-033-search-powered-surfaces.md` — cross-surface propagation
  - `docs/prd/PRD-021-semantic-corpus-search.md` — predecessor (FAISS semantic search)
  - `docs/prd/PRD-028-position-tracker.md` — position evolution data (temporal queries)
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — two-tier index + BM25 + vector + RRF (implements core)
  - `docs/rfc/RFC-091-kg-proximity-signal.md` — KG proximity signal
  - `docs/rfc/RFC-092-ml-query-router.md` — ML query router
  - `docs/rfc/RFC-093-litm-context-packs.md` — LITM-aware context packs
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — canonical identity
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` — enrichment layer
- **Related issues**: #849 (Search retrieval prerequisites), #466 (GI/KG depth roadmap — CLOSED, superseded), #484 (Semantic Corpus Search — CLOSED, predecessor), #485 (Topic nodes + ABOUT edges — CLOSED)

> **Stabilization note (2026-05-30):** Rebased to this repo. The original draft pointed at an
> MCP-integration RFC and "Position Tracker"/"Guest Intelligence Brief" PRDs that do not exist
> under those numbers here. Corrected: enrichment → RFC-088; Position Tracker → PRD-028; the
> MCP tool layer and a guest-briefing PRD are **[TBD — not yet specified]**. Module paths are
> corrected to `src/podcast_scraper/…`.

---

## Summary

Podcast Scraper's search today is single-signal FAISS vector retrieval over GIL insight nodes only
(`src/podcast_scraper/search/`, shipped via #484). This misses raw transcript evidence, fails on
named-entity queries, ignores KG relational structure, and gives agents unstructured result dumps.
This PRD covers the evolution to **hybrid corpus search**: a two-tier index (transcript segments +
GIL insights) with two retrieval signals fused via RRF (BM25 + dense vector), intent-aware
query routing, mixed-tier result ranking, and LITM-aware agent context packs. It is implemented by
RFC-090 (core); RFC-091's KG-proximity signal was **evaluated and rejected** (Decision Record) —
relational structure comes from typed edges instead (#874) — with RFC-092/093 additive.

## Background & Context

- **What problem this solves.** Single-signal vector search over insights answers similarity but
  not exact-match, relational, or raw-evidence questions. The corpus has structure (typed entities,
  topics, shows, time) that retrieval ignores.
- **Why now.** Semantic corpus search (#484, PRD-021/RFC-061) shipped and exposed its own ceiling:
  named-entity queries degrade, transcript-level evidence is unreachable, and the KG contributes
  nothing to ranking. The GI/KG depth roadmap (#466) names exactly these gaps.
- **How it relates to existing features.** Builds on the existing FAISS pipeline, the GIL insight
  contracts (`src/podcast_scraper/gi/contracts.py`), the KG artifacts (`*.kg.json`), and the
  canonical identity layer (RFC-072). It is the backend for the Search product (PRD-031).

## Goals

- **Two-tier retrieval.** Transcript segments and GIL insights are both indexed and retrievable;
  results are mixed by score, not grouped by tier.
- **Named-entity recall.** Searches for person names, show names, and specific terminology return
  directly relevant results, not just semantic neighbours.
- **Graph-aware structure.** KG-proximity was evaluated as a third retrieval signal and **rejected**
  (RFC-091 Decision Record); relational structure comes from typed edges (`Person→Insight`,
  `Insight→Entity`, #874) that surfaces traverse, not from proximity ranking.
- **Intent-aware routing.** Query intent is classified and retrieval strategy adjusted: person
  lookup, synthesis, raw evidence, and temporal queries each get the right signal/tier mix.
- **Clean backend abstraction.** A `SearchBackend` protocol makes the backend swappable; LanceDB
  embedded is the initial implementation. Cloud backends require no retrieval-logic changes.
- **Agent-ready context packs.** MCP search tools return LITM-aware, compressed, grounded context,
  not raw lists (RFC-093).

## Non-Goals

- Real-time / sub-second indexing of new episodes.
- Cross-language search (corpus is English-only).
- User-facing faceted search UI (filter chips are existing UX work; this PRD provides the backend
  filter mechanism).
- Search personalisation (no auth layer yet).
- Cross-encoder reranking (deferred, post-RRF baseline).
- Cloud search backends (turbopuffer, Qdrant Cloud) — supported by the abstraction, not implemented.

## Personas

- **Beta researcher**: Wants exact quotes, named-entity hits, and cross-show breadth — not just
  similar snippets.
- **Autoresearch / agent consumer**: Issues typed queries and needs shaped, tier-weighted results
  without specifying signal weights manually. _(Autoresearch is today an eval/prompt-tuning harness
  — `autoresearch/` — and is not yet a live search consumer; this is the integration target.)_
- **Operator (you)**: Tunes retrieval quality via the eval loop and owns the backend choice.

## User Stories

- _As a beta researcher, I can search a specific phrase or quote and find the transcript segment
  containing it, so that raw evidence is reachable even when no insight was extracted from it._
- _As a beta researcher, I can search a person's name and get results containing their name
  directly, so that named-entity queries don't degrade to topical neighbours._
- _As a beta researcher, I can ask a cross-show question and see results from multiple shows
  weighted by breadth, so that synthesis queries surface diversity._
- _As an agent via MCP, I can receive a shaped context pack — grounded insights first, supporting
  segments middle, caveats last, within a token budget — so that I extract more value per token._
- _As the autoresearch loop, I can issue typed queries and receive tier-weighted results
  appropriate to my intent without specifying signal weights manually._

## Functional Requirements

### FR1: Two-Tier Index

- **FR1.1 Segment tier (Tier 1)**: Transcript chunks, 200–300 words, 50-word overlap over Whisper
  output segments. Indexed fields: chunk text (BM25 + vector), `show_id`, `episode_id`,
  `speaker_id` (if diarized), `start_time`, `end_time`, `linked_insight_ids`. Catches exact
  phrases, terminology, raw evidence, content that did not surface as an insight.
- **FR1.2 Insight tier (Tier 2)**: GIL insight nodes. Indexed fields: insight text (BM25 + vector),
  `show_id`, `episode_id`, `speaker_id`, `entity_type`, `confidence`, `derived`,
  `source_segment_id`. Catches distilled claims and positions.
- **FR1.3**: A chunker already exists at `src/podcast_scraper/search/chunker.py` (used by
  `indexer.py`); RFC-090 extends rather than recreates it.

### FR2: Mixed Ranking (RRF)

- **FR2.1**: Both tiers searched in parallel; results merged via RRF into a single ranked list.
- **FR2.2**: `source_tier: "segment" | "insight"` is a payload field on every result — the score
  decides order, not the tier.
- **FR2.3**: Default RRF tier weights — insights 1.2, segments 1.0. Query router overrides per
  intent (e.g. raw-evidence → segments 1.3, insights 0.9).

### FR3: Deduplication (Compound Results)

- **FR3.1**: When a segment result and an insight result refer to the same underlying content
  (segment contains the insight's grounding quote), merge into a single **compound result** at the
  retrieval layer, carrying both the raw segment and the synthesized insight.
- **FR3.2**: Compound result takes the higher of the two scores; the consumer receives one object.

### FR4: Intent-Aware Routing

- **FR4.1**: Classify query intent (rules-based in RFC-090; ML in RFC-092) and adjust signal/tier
  weights accordingly.
- **FR4.2**: Misclassification degrades to sub-optimal weights, not wrong results (RRF is robust).

### FR5: Backend Abstraction

- **FR5.1**: A `SearchBackend` protocol decouples signal generation from fusion (RFC-090 §3.2).
- **FR5.2**: LanceDB embedded is the initial backend; swapping backends costs zero changes outside
  `config/search.yaml`.

### FR6: Agent Context Packs

- **FR6.1**: MCP search tools return LITM-aware, token-budgeted context packs (RFC-093).
  _Requires an MCP tool layer, which **does not exist yet** — **[TBD]**._

## Success Metrics

| Metric | Target |
| --- | --- |
| Named-entity recall@10 | ≥ 90% top-10 contains exact-match result for queried person/show |
| nDCG@10 vs FAISS baseline | Statistically significant improvement on held-out query set |
| Segment-only result rate | > 20% of top-10 are segment-tier (validates two-tier value) |
| Cross-show diversity (synthesis queries) | ≥ 3 distinct shows in top-10 |
| Compound-result dedup rate | Measurable — confirms segment↔insight linking works |
| Backend swap cost | Zero changes outside `config/search.yaml` |

## Dependencies

- **RFC-090** Hybrid Retrieval Pipeline — **Hard** — Draft.
- **RFC-091/092/093** — **Soft/additive** — Draft; the KG and MCP features depend on prerequisites
  below.
- **RFC-072** Canonical Identity — **Hard for KG proximity** — **Partial**: only
  `identity/slugify.py` exists; no entity resolver yet.
- **GIL contracts** — **Hard** — shipped; grounding quotes carry `timestamp_start_ms` /
  `timestamp_end_ms`, enabling segment↔insight linking.
- **Prerequisites not yet built** (tracked in #849): file-local entity resolver; in-memory
  KnowledgeGraph access layer (`neighbors()`/`get_node()`); typed KG edges beyond the current
  `MENTIONS`/`RELATED_TO` (note: `ABOUT` edges + Topic nodes already exist via #485); MCP tool layer.

## Constraints & Assumptions

**Constraints:**

- Embedded, local-first backend (LanceDB) on the operator's machine; no cloud dependency.
- English-only corpus.

**Assumptions:**

- Whisper output segments (`{text, start, end, speaker_id?}`) are available to the chunker. Today
  segments are passed into GI/KG builders as optional sidecar data
  (`src/podcast_scraper/gi/pipeline.py`), not retained as a first-class artifact — RFC-090 OQ-5
  covers diarization integration.
- Embedding model `all-MiniLM-L6-v2` (384-dim); validate dimensionality on migration.

## Design Considerations

Decisions carried from draft review (now resolved):

- **Dedup strategy**: Compound result at the retrieval layer; consumer receives one object with
  both segment and insight.
- **Result presentation**: Uniform ranking by score; `source_tier` as a payload field; no forced
  grouping.
- **Default tier weights**: Insight 1.2, segment 1.0; query router overrides per intent.
- **Segment granularity**: 200–300 words, 50-word overlap over Whisper output.
- **Segment–insight linking**: `linked_insight_ids` on the segment doc; `source_segment_id` on the
  insight doc.

## Open Questions

Risks and their mitigations (tracked as open until validated):

1. **Segment index size.** ~700 episodes × ~200 chunks ≈ ~140k segment docs vs ~15k insights.
   LanceDB handles this at embedded scale — validate before assuming.
2. **LanceDB FTS rebuild on upserts.** Schedule as a post-ingestion step, not per-upsert
   (RFC-090 OQ-1).
3. **Compound-result linking quality.** Depends on timestamp overlap between Whisper segments and
   GIL grounding quotes. Mitigation: tolerance window; log unlinked insights.
4. **Tier-weight calibration.** Defaults are heuristic; weights are config-driven and the eval loop
   produces tuning signal.
5. **Rules-based router misclassification.** Degrades to wrong weights, not wrong results; RFC-092
   ML router fixes systematic misclassification once eval data exists.

## Related Work

- `docs/rfc/RFC-090-hybrid-retrieval.md`
- `docs/rfc/RFC-091-kg-proximity-signal.md`
- `docs/rfc/RFC-092-ml-query-router.md`
- `docs/rfc/RFC-093-litm-context-packs.md`
- `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- `docs/rfc/RFC-088-enrichment-layer-architecture.md`
- `docs/prd/PRD-031-search.md`
- `docs/prd/PRD-028-position-tracker.md`
- Issue #849 (Search retrieval prerequisites), #466 (GI/KG depth roadmap — superseded), #484, #485.

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-090 (core) approved; prerequisite issues for entity resolver / graph access / typed edges
      filed in #849
- [ ] Two-tier index + RRF fusion + compound dedup implemented
- [ ] Segment chunking wired post-transcription (extend existing chunker)
- [ ] Eval: two-tier hybrid vs FAISS baseline on held-out query set
- [ ] `config/search.yaml` backend abstraction in place
- [ ] FAISS deprecated on confirmed improvement

## Future

- Turbopuffer or Qdrant Cloud backend (abstraction ready).
- Cross-encoder reranking post-RRF baseline.
- Query expansion / HyDE for sparse queries.
- Personalisation (requires auth layer).
