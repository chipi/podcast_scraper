# Hybrid Search Validation — Hypotheses and Eval Spec

*Diagnostic and measurement framework for PRD-032 / RFC-078 / RFC-079*
*Written after initial implementation showed marginal gains over FAISS baseline*

> **Outcome note (2026-06-04).** These hypotheses were investigated; this records what the data said,
> so the spec reads as history + open work, not pending questions:
>
> - **H2 (linking)** — confirmed: linking was 0% (a real bug, `summary.timestamps` unpopulated); fixed
>   via text-containment linking (now 87–93%); compound results fire.
> - **H6 (KG proximity)** — **rejected** (RFC-091 Decision Record): proximity hurt or was neutral on
>   every corpus/axis; co-occurrence is redundant with dense embeddings. Value moved to relational
>   edges (#874). Treat the KG-proximity rows here as historical hypotheses, not active signals.
> - **H1/H7** — partly confirmed (per-intent stratification; only semantic/entity intents classified).
> - **H3 (k-grid), H4 (MRR/reranking), H5 (chunking)** — still open / not yet run (hybrid tuning upside).

---

## Context

After implementing the two-tier hybrid retrieval pipeline (RFC-078), measured gains over FAISS were present but somewhat marginal — better, but not the qualitative shift expected. This document captures the hypotheses for why, what might be missing, and the full measurement framework to validate that the new approach is working as intended.

**Key diagnostic framing:** aggregate nDCG over a generic query set is the wrong primary metric for this architecture. The gains are concentrated in specific query types and failure modes. Marginal aggregate improvement + significant stratified improvement = the architecture is working but the eval was measuring the wrong thing. Confirm or rule this out first.

---

## Hypotheses

### H1 — Eval query set doesn't target the failure modes
*Likelihood: high*

If the held-out query set was drawn generically, it's dominated by semantic/conceptual queries — exactly what FAISS already handles well. BM25 and KG proximity gains only appear on specific query types: named entity lookups, exact phrase searches, cross-show synthesis. If those represent 20% of the eval set, an 80% improvement on them shows as ~16% overall — looks marginal in aggregate, is significant for the affected class.

**Implication:** stratified eval per query type is the fix. Aggregate nDCG is a secondary metric here, not the primary one.

---

### H2 — Segment-insight linking has low hit rate
*Likelihood: high*

The two-tier architecture's qualitative gain — compound results, raw evidence accessible alongside synthesized insights — only materialises if `linked_insight_ids` is correctly populated on segment documents. If timestamp alignment between Whisper segments and GIL grounding quotes is off (tolerance window too tight, quote timestamps not stored on insights, or not populated at migration time), you have two separate indexes that coexist rather than a genuinely integrated two-tier system. The compound result dedup fires rarely or not at all. The segment tier adds coverage but not the intelligence linking that makes it meaningful.

**Implication:** check linking coverage rate before any other diagnostic. If < 50%, fix the linker first — everything else is secondary.

---

### H3 — RRF k=60 is over-smoothing for this corpus size
*Likelihood: medium*

k=60 is the empirical default for large corpora (millions of documents). At ~15k insights + ~140k segments, it may be too conservative — it flattens rank differences between signals that shouldn't be equal. At lower k (20–30), rank position matters more and the BM25 signal is amplified on named entity queries where it should dominate. The default was right to ship with; wrong to leave unchecked.

**Implication:** grid search k before concluding the architecture is working or not. Cheap to run, potentially non-trivial effect on entity_lookup queries.

---

### H4 — No post-RRF reranking
*Likelihood: medium*

RRF improves recall. It does not improve precision at rank 1. After fusion there is no signal that identifies the *defining* document for a query — the first-introduction boost and position-statement boost described in RFC-079 are not implemented. For a person lookup, the most relevant result is where they most clearly state their core position — not just the document that happens to rank highest by RRF score. Without post-RRF reranking, you get better recall but not better ranking quality at the top of results.

**Implication:** measure MRR on entity_lookup queries specifically. Low MRR despite good recall@10 = the right document is in the result set but not at position 1. That is a reranking problem, not a retrieval problem. Different fix.

---

### H5 — Chunk boundaries misalign with meaning units in podcast transcripts
*Likelihood: medium*

200–300 word sliding window chunks are standard RAG practice designed for written text. Podcast transcripts are structurally different: speaker turns, interruptions, topic shifts mid-sentence, filler words. A 250-word chunk in podcast audio probably spans 2–3 speaker turns and multiple topic shifts. BM25 over a multi-speaker, multi-topic chunk has diluted signal — the chunk is not coherently about any one thing. Speaker-turn-aware chunking (chunk by speaker turn, merge short turns below a minimum word count) would produce sharper, more precise BM25 signal per chunk.

**Implication:** measure average distinct speakers per chunk. If > 1.5 on average, chunks are crossing speaker boundaries too often. Run a speaker-turn-aware chunking variant and compare on raw_evidence queries.

---

### H6 — KG proximity not yet contributing
*Likelihood: depends on implementation state*

The RFC-079 KG proximity signal is the reserved slot in `RetrievalLayer.retrieve()` — if it is still commented out, the system has two signals (BM25 + vector), not three. The qualitative shift described in the vision document — relational intelligence that embedding similarity cannot capture, cross-show synthesis that reflects graph structure — was always primarily about graph proximity. BM25 over vector is an incremental improvement. Graph proximity over BM25+vector is the new dimension. If it is not wired in, the "new dimension" has not been tested yet.

**Implication:** before concluding gains are marginal, confirm whether KG proximity is active. If not, the architecture is incomplete. Once wired, measure cross-show diversity specifically — that is the metric graph proximity moves.

---

### H7 — Query router misclassifying most queries as semantic
*Likelihood: medium*

The rules-based router uses regex patterns and keyword lists. If most queries don't match the patterns (short queries, novel phrasing, queries that don't use "compare" or "vs" explicitly), the router defaults to "semantic" — and signal weight differentiation does nothing. All queries get equal BM25:vector weights. The router is a no-op in practice.

**Implication:** log query type distribution in production. If > 70% of queries classify as "semantic", the router is not discriminating. Fix the rules first; ML router (RFC-079 Phase 3) is the longer-term answer.

---

## Eval Spec

### Principles

- **Stratified by query type, not aggregate only.** Aggregate nDCG is a secondary health metric. Primary metrics are per query type.
- **Known-answer queries wherever possible.** For each query type, the correct result should be known in advance so precision can be measured, not just ranked.
- **Always-on pipeline health metrics.** Linking coverage, compound result rate, query type distribution — run continuously, not just at eval time.
- **Baselines:** FAISS-only (original), BM25+vector (RFC-078 without KG), BM25+vector+KG (RFC-078+079 full). Three baselines lets you isolate each signal's contribution.

---

### Query Sets

Build and maintain labeled query sets per type. Minimum sizes below are for statistical significance at p < 0.05 on nDCG comparisons.

**Entity lookup queries** (min n=50)
- Form: "[person name]", "what did [person] say about [topic]", "[show name] episodes on [topic]"
- Labeled correct result: the insight or segment where the person most directly addresses the topic, or the canonical introduction of the person to the corpus
- Examples: "Sam Altman AI safety", "Planet Money inflation", "Lex Fridman consciousness"

**Raw evidence queries** (min n=30)
- Form: exact or near-exact phrases, specific quotes, "what exactly did X say about Y"
- Labeled correct result: the segment containing the verbatim phrase
- Examples: "race to the bottom", "we're in a bubble", exact quotes from high-profile episodes

**Cross-show synthesis queries** (min n=30)
- Form: "how do different shows approach X", "compare X vs Y on Z", "who disagrees about X"
- Labeled correct result: a result set containing insights from ≥ 3 distinct shows
- Examples: "compare tech podcasts on AI regulation", "how has startup advice changed"

**Temporal tracking queries** (min n=20)
- Form: "how has X's position on Y evolved", "when did Z change their view on W"
- Labeled correct result: a set of results spanning multiple dates showing position change
- Examples: "how has [person]'s view on remote work changed"

**Semantic queries** (min n=50)
- Form: conceptual questions with no specific entity
- Labeled correct result: top-5 most semantically relevant insights
- Examples: "what makes a great founder", "lessons from failed startups"
- Note: this is the FAISS-friendly query type; hybrid should be flat or marginally better here

---

### Metrics Per Query Type

**Entity lookup**

| Metric | Description | Target |
|---|---|---|
| Named entity recall@10 | Top-10 contains exact-match result for queried entity | ≥ 90% |
| MRR | Mean Reciprocal Rank — is the defining document at position 1? | ≥ 0.7 |
| nDCG@10 vs FAISS | Improvement over baseline | Significant (p < 0.05) |
| BM25 contribution | % of top-10 results where BM25 rank < vector rank | > 40% |

**Raw evidence**

| Metric | Description | Target |
|---|---|---|
| Segment-tier rate in top-10 | % of top-10 results that are segment-tier | > 50% |
| Exact match recall | Does verbatim phrase appear in top-10 | ≥ 80% |
| Compound result rate | % of top-10 that are compound (segment + insight linked) | > 20% |
| MRR | Is the exact-match segment at position 1? | ≥ 0.6 |

**Cross-show synthesis**

| Metric | Description | Target |
|---|---|---|
| Distinct show count in top-10 | Breadth across shows | ≥ 3 |
| KG proximity contribution | Rerank delta vs BM25+vector only (once KG is active) | Measurable positive |
| Insight-tier rate | % of top-10 that are insight-tier (synthesis should prefer insights) | > 60% |

**Temporal tracking**

| Metric | Description | Target |
|---|---|---|
| Temporal span of top-10 | Date range covered by top-10 results | > 6 months for tracked topics |
| Multi-episode rate | % of queries where top-10 spans > 3 episodes | > 70% |

**Semantic**

| Metric | Description | Target |
|---|---|---|
| nDCG@10 vs FAISS | Should not regress | ≥ FAISS baseline |

---

### Always-On Pipeline Health Metrics

Run after every ingestion cycle and on demand. These are diagnostic — they tell you whether the architecture is functioning as designed, independent of retrieval quality.

| Metric | How to compute | Target | Alert if |
|---|---|---|---|
| Linking coverage rate | `% of insights with source_segment_id != null` | > 70% | < 50% |
| Compound result rate | `% of top-10 results that are CompoundResult` across last N queries | > 15% | < 5% |
| Query type distribution | `% of queries per type` from router logs | entity+evidence ≥ 30% combined | semantic > 80% |
| BM25 index freshness | Time since last FTS index rebuild vs last ingestion | < 24h lag | > 48h lag |
| Avg speakers per chunk | Mean distinct speaker_ids per SegmentDocument | < 1.5 | > 2.0 |
| k sensitivity delta | nDCG@10 at k=30 vs k=60 on entity queries | < 5% | > 15% (means k needs tuning) |

---

### RRF k Grid Search

Run once after initial implementation, then whenever corpus size doubles.

```
k values to test: [20, 30, 45, 60]
query types: entity_lookup, raw_evidence (k matters most here)
metric: nDCG@10, MRR
baseline: k=60 (current default)
```

If k=30 or k=20 shows > 5% improvement on entity_lookup MRR, update default in `rrf_fuse()`. Document result in this file.

*k grid search result (fill after running):*
- k=20: —
- k=30: —
- k=45: —
- k=60: baseline

---

### Chunking Strategy Comparison

Run if avg speakers per chunk > 1.5 or compound result rate < 5%.

| Strategy | Description |
|---|---|
| Word-count sliding window | Current default — 250 words, 50-word overlap |
| Speaker-turn-aware | Chunk by speaker turn, merge turns < 50 words with adjacent turn |
| Sentence-boundary-aware | Chunk at sentence boundaries nearest to 250-word target |

Compare on raw_evidence queries: exact match recall, segment-tier rate in top-10.

---

### Baseline Comparison Table

Fill as each stage is implemented and eval is run.

| Config | Entity recall@10 | Raw evidence exact match | Synthesis show diversity | Semantic nDCG@10 |
|---|---|---|---|---|
| FAISS only (original baseline) | — | — | — | — |
| BM25 + vector, k=60 (RFC-078) | — | — | — | — |
| BM25 + vector, k=best (tuned) | — | — | — | — |
| BM25 + vector + KG proximity (RFC-079) | — | — | — | — |
| + post-RRF first-intro reranking | — | — | — | — |

---

## Diagnostic Decision Tree

Use this when gains look marginal after implementation.

```
Gains look marginal vs FAISS
│
├── Is eval stratified by query type?
│   └── No → stratify first, re-measure before any other action
│
├── What is linking coverage rate?
│   └── < 50% → fix linker before anything else
│
├── What is query type distribution?
│   └── semantic > 80% → router not discriminating, fix rules or lower thresholds
│
├── Is KG proximity wired in?
│   └── No → implement RFC-079 Phase 1 before concluding architecture is limited
│
├── What does k grid search show?
│   └── k=30 >> k=60 on entity MRR → update k default
│
├── What is MRR on entity_lookup?
│   └── Low MRR + good recall@10 → reranking problem, not retrieval problem
│       → implement first-introduction boost
│
└── What is avg speakers per chunk?
    └── > 1.5 → try speaker-turn-aware chunking on raw_evidence queries
```

---

## Notes on Results

*Fill as eval is run. Running log of findings.*

| Date | Finding | Action taken |
|---|---|---|
| — | — | — |

