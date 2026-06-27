# ADR-104: Enrichment-layer boundary vs KG-direct connectivity

- **Status**: Accepted (promoted 2026-06-27 with RFC-088 chunks 0-8)
- **Date**: 2026-06-26
- **Authors**: Marko Dragoljevic, Claude (Opus 4.7)
- **Related ADRs**:
  - [ADR-101](ADR-101-drop-legacy-kg-gi-shape.md) — strict KG v2.0 /
    GI v3.0 contract this ADR partitions.
  - [ADR-103](ADR-103-deterministic-connectivity-under-llm-free-profiles.md)
    — RFC-097 chunk 9 KG-direct path that motivates the partition.
  - [ADR-051](ADR-051-per-episode-json-artifacts-with-logical-union.md) —
    per-episode JSON archive convention enrichment outputs extend.
  - [ADR-052](ADR-052-separate-gil-and-kg-artifact-layers.md) —
    GIL / KG separation enrichment layer sits on top of.
- **Related RFCs**:
  - [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) — defines
    the enrichment-layer protocol + registry + executor. This ADR amends
    its Key Decision #1.
  - [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) — chunk 9
    cross-show Topic clustering writes `concept:topic-{slug}` Topic
    nodes + `RELATED_TO` edges directly into KG artifacts. This ADR
    legitimises that as a core pipeline output, not an enrichment-layer
    violation.

## Context

RFC-088 §Key Decisions #1 reads "Enrichers never modify core artifacts.
`*.gi.json`, `*.kg.json`, `*.bridge.json` are immutable after the core
pipeline writes them." That phrasing was written before RFC-097 chunk 9
shipped (PR #1094, 2026-06-25).

Chunk 9 ships `src/podcast_scraper/kg/topic_clustering.py`, which runs
cross-show Topic clustering and **adds synthetic
`concept:topic-{slug}` Topic nodes + `RELATED_TO` edges directly into
the per-episode `*.kg.json` files**. The function fires from
`workflow/orchestration.py` Step 16 (after `_finalize_pipeline`) on
every corpus run where `cfg.kg_topic_corpus_clustering` is true —
default-true for the `airgapped*` profiles. ADR-103 ratified this as
how we close the airgapped-CI connectivity gap without introducing stub
mode.

Read literally against RFC-088 Decision #1, chunk 9 violates the
enrichment-layer contract: it writes derived signals into core
artifacts. Read against the actual operator goal, chunk 9 is the right
shape — typed connectivity belongs in the KG so the relational query
layer + viewer + future LLM grounding all see the same graph.

The contradiction is a vocabulary mismatch, not a design disagreement.
"Core" and "enrichment" weren't precise enough to cover the case where
a deterministic post-pass produces typed edges the schema treats as
first-class.

This ADR fixes the vocabulary. It does not retract chunk 9 or rewrite
RFC-088 from scratch.

## Decision

**The boundary is by ontology contract + audience, not by file
identity.**

- **Core artifacts** (`*.gi.json`, `*.kg.json`, `*.bridge.json`) are
  whatever the **core pipeline** produces. The pipeline can include
  deterministic post-passes that synthesize typed nodes + edges *if and
  only if* those outputs conform to the schema (KG v2.0 / GI v3.0) and
  participate in the relational query layer the same as any other typed
  data. RFC-097 chunk 9's `concept:topic-{slug}` + `RELATED_TO` qualify.
- **Enrichment outputs** (`metadata/enrichments/{stem}.{id}.json`,
  `{corpus_root}/enrichments/{id}.json`) are **derived signals that do
  not have a place in the typed ontology** — scores, ranks,
  calibrations, candidate pairs, per-Person aggregates, trend windows.
  They live in the enrichment layer because they are consumed for UI
  rendering, ranking, and autoresearch tuning, not for typed graph
  traversal or LLM grounding.

**Reconciliation rule when a signal could land in either:**

- If a downstream consumer would reasonably **traverse** it (CIL
  resolver, relational query layer, KG export, viewer graph edges,
  airgapped LLM grounding context) → it belongs in **KG / GIL / bridge**
  (core).
- If a downstream consumer would reasonably **rank, threshold, or
  visualize** it (search result decoration, "related topics" UI chip
  with confidence, Topic-Entity-View trend sparkline, autoresearch param
  sweep) → it belongs in **enrichments/** (derived).

The same underlying signal can have two outputs, one in each layer,
when both audiences need it. `topic_similarity` (RFC-088 chunk 3) is
that case: KG keeps `RELATED_TO` edges (chunk 9 — typed connectivity,
LLM-grounding-friendly); enrichments keep the cosine scores +
top-K-per-topic ranking (RFC-088 chunk 3 — UI consumption + autoresearch
tuning). Both are correct; neither is redundant.

**Amendment to RFC-088 Key Decision #1:**

> "Enrichers never modify core artifacts produced by core pipeline
> stages. The RFC-097 chunk 9 cross-show Topic clustering is part of
> the core pipeline (it runs in `workflow/orchestration.py` and
> conforms to the KG v2.0 ontology), not an enrichment-layer
> violation. Enrichment outputs are derived signals that live under
> `enrichments/` and are read-only from the perspective of core
> artifacts."

## Rationale

Three observations made this the cleanest framing:

1. **The ontology is the contract, not the directory name.** KG v2.0
   defines `Topic`, `RELATED_TO`, `is_concept` as first-class. Anything
   conforming to that contract is on-graph — regardless of which
   pipeline stage wrote it. ADR-101 ratified the strict-v2.0 flip
   precisely so consumers could trust the schema without origin
   tracking; that trust survives chunk 9.
2. **The consumer pattern decides the layer.** Traversal (graph
   queries, LLM grounding) requires typed edges in core artifacts.
   Ranking + thresholding (UI decoration, autoresearch sweeps) require
   scored signals in derived artifacts. Same data, two consumer
   patterns, two physical homes.
3. **Coexistence beats migration.** Retiring chunk 9 to move
   `RELATED_TO` into enrichments would break every consumer that
   already reads it (the relational query layer, the viewer graph
   surface, ADR-103's airgapped CI story) without buying back the
   audience that wanted scores (which is best served by a separate
   enricher anyway). Coexistence preserves both audiences with zero
   migration.

## Boundary in operational terms

| Concern | Core (KG / GIL / bridge) | Enrichment (`enrichments/`) |
| --- | --- | --- |
| Wrote by | Core pipeline (`workflow/orchestration.py`) | Enrichment pass (`enrichment/executor.py`) |
| Schema | KG v2.0 / GI v3.0 strict | Per-enricher `schema_version` |
| Mutation envelope | Allowed when conforming to schema (e.g. chunk 9) | Never mutates core; only writes new files |
| Audience | CIL, relational queries, KG export, LLM grounding | UI ranking, autoresearch sweeps, candidate-pair surfacing |
| Required by | Default-on profiles (deterministic typed connectivity) | Opt-in per profile preset (RFC-088 chunk 7 matrix) |
| Idempotency | Idempotent additive writes (chunk 9 pattern) | Full recompute on re-run (RFC-088 Decision #8) |

## Alternatives considered

- **Retract RFC-097 chunk 9; move `concept:topic-{slug}` +
  `RELATED_TO` to enrichments/.** Rejected. Cleaner architecturally
  on paper, but breaks every consumer that already reads `RELATED_TO`
  from KG (viewer graph, ABOUT∩MENTIONS_PERSON joins, relational
  query layer, airgapped CI per ADR-103), invalidates every corpus
  generated since PR #1094, and forces a v3 KG ontology change.
  Cost / benefit not justified.
- **Mark RFC-088 Decision #1 as "aspirational only"; ignore the
  chunk-9 contradiction.** Rejected. Future enrichment authors need
  a clear rule. "Aspirational" is not a rule.
- **Introduce a new "pipeline-derived" tier in the enrichment-layer
  taxonomy.** Rejected as additional ceremony without paying back the
  precision. The KG / enrichment boundary covers it — chunk 9 is a
  pipeline post-pass that writes core artifacts, period.

## Consequences

**Positive:**

- Future enrichment authors have a precise rule for where their output
  belongs (traversal → core; ranking → enrichment).
- RFC-097 chunk 9's airgapped-CI story (ADR-103) is preserved without
  awkward exception-carving.
- `topic_similarity` (RFC-088 chunk 3) ships alongside chunk 9
  `RELATED_TO` with no architectural friction — they serve different
  consumers.
- Sets precedent: future deterministic post-passes that synthesize
  typed nodes / edges can land in core, provided they conform to the
  schema and have a traversal-oriented consumer.

**Negative:**

- "Same signal, two physical homes" can confuse a reader who only sees
  one layer's output. The ENRICHMENT_LAYER_GUIDE (chunk 8) needs to
  explicitly call out the coexistence rule and point at this ADR.
- Enricher authors have to think about which side they're on before
  starting. The rubric ("traversal vs ranking") is the test.

**Neutral:**

- No code change required by this ADR. The amendment to RFC-088
  Decision #1 is a clarification, not a rewrite. RFC-097 carries a
  cross-ref but its body is unchanged.

## Validation

- RFC-088 + RFC-097 cross-refs in place (this chunk).
- `make docs` strict green.
- Implementation chunks (RFC-088 #1103 onward) honour the rubric:
  - chunk 2 (deterministic) — six enrichers, none touch core artifacts;
  - chunk 3 (`topic_similarity`) — writes ranks to enrichments, does
    not mutate KG `RELATED_TO`;
  - chunk 4 (`nli_contradiction`) — writes candidate pairs to
    enrichments, does not introduce a `CONTRADICTS` edge type in KG.

## Implementation references

- `src/podcast_scraper/kg/topic_clustering.py` — chunk 9 KG-direct
  cross-show Topic clustering (the core path).
- `src/podcast_scraper/workflow/orchestration.py` Step 16 — where the
  KG-direct post-pass fires.
- `src/podcast_scraper/enrichment/` (incoming, RFC-088 chunk 1) — the
  enrichment-layer module the rubric protects.
- Profile-preset enricher matrix in
  [#1101](https://github.com/chipi/podcast_scraper/issues/1101) — which
  enrichments (vs core) run per environment.
