# RFC-097: Unified KG + GI Ontology v2

- **Status**: Completed (v2.6.0–v2.7) — Anchor #1036 closed; chunks 1–8 shipped in PR #1039 (foundation + edge styling); chunk 9 (#1073) shipped on `feat/rfc097-followups` (PR #1089) — strict KG v2.0 / GI v3.0 schemas + [ADR-101](../adr/ADR-101-drop-legacy-kg-gi-shape.md). [ADR-102](../adr/ADR-102-retro-audit-marker-for-in-place-artifact-mutation.md) + [ADR-103](../adr/ADR-103-deterministic-connectivity-under-llm-free-profiles.md) Accepted. Person Profile + Position Tracker viewer surfaces live via #1048/#1049/#1050. Follow-ups: #1075 / #1076 / #1058 all closed. NER post-pass and corpus-level Topic clustering ship deterministic typed connectivity under airgapped profiles (PR #1094).
- **Authors**: Marko Dragoljevic (chipi), Claude (Opus 4.7)
- **Stakeholders**: Operator (sign-off), KG/GI maintainers, viewer maintainers
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md` (entity v1.1 deferral retired)
  - `docs/prd/PRD-019-knowledge-graph-layer.md` (Entity → Person/Org/Podcast first-class)
  - `docs/prd/PRD-026-topic-entity-view.md` (CIL enablement)
  - `docs/prd/PRD-027-enriched-search.md` (chunk-to-Insight lift foundation)
  - `docs/prd/PRD-028-position-tracker.md` (data foundation delivered; viewer surface delivered by #1048 shared shell + #1049 Position Tracker panel — per-(Person, Topic) timeline now live)
  - `docs/prd/PRD-029-person-profile.md` (data foundation delivered; viewer surface delivered by #1048 shared shell + #1049 Position Tracker + #1050 Person Profile aggregate — UXS-010 sections complete)
  - `docs/prd/PRD-031-search.md` (entity-aware retrieval foundation)
- **Related ADRs**:
  - `docs/adr/ADR-095-viewer-test-pyramid.md` (real-bug-to-matrix-row discipline)
  - `docs/adr/ADR-101-drop-legacy-kg-gi-shape.md` (reserved, lands with chunk 9)
- **Related RFCs**:
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` (GI core; v1.1 entity deferral retired)
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md` (cross-layer use cases moved to RFC-072)
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` (KG core; Entity → typed nodes)
  - `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md` (cross-layer use cases moved to RFC-072)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` (CIL + bridge; v1.1 additive fields land)
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` (enricher inputs unblocked)
  - `docs/rfc/RFC-090-hybrid-retrieval.md` (orthogonal; LanceDB-first stands)
  - `docs/rfc/RFC-091-kg-proximity-signal.md` (REJECTED 2026-06-03; v2 confirms)
  - `docs/rfc/RFC-092-ml-query-router.md` (intent taxonomy may extend post-v2)
  - `docs/rfc/RFC-093-litm-context-packs.md` (richer packs via insight_type/position_hint)
  - `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md` (relational queries stabilized via CIL)
  - `docs/rfc/RFC-095-generic-mcp-server.md` (CIL prerequisite for stable tool schemas)
- **Related UX specs**:
  - PRD-028 / PRD-029 cover the viewer Position Tracker + Person Profile surfaces
- **Related Documents**:
  - `docs/architecture/corpus/ontology.md` (NEW, lands in chunk 1)
  - `docs/architecture/kg/ontology.md` (superseded by chunk 1)
  - `docs/architecture/gi/ontology.md` (superseded by chunk 1)
  - `docs/wip/SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md` (round-1 spec, archaeology)
  - `docs/wip/SPEC_KG_GI_ONTOLOGY_V2_2026-06-20.md` (round-2 spec, archaeology)
  - `docs/wip/SPEC_KG_GI_ONTOLOGY_V2_ROUND3_2026-06-20.md` (round-3 spec, the live design)
  - `docs/wip/SPEC_KG_GI_ONTOLOGY_V3_WISHLIST_2026-06-20.md` (deferred items + preservation checklist)

## Abstract

RFC-097 unifies the Knowledge Graph (KG) and Grounded Insight (GI)
ontologies into a single per-corpus shape that the viewer can render
directly, formally retires the entity-extraction deferral, and lights
up the cross-layer descriptive edges that prior schemas named but
never emitted. It is **not** a new graph database, **not** a merger
of `kg.json` and `gi.json` files at the filesystem layer, **not** a
change to the grounding contract, and **not** a retrieval-side change
(RFC-091's 2026-06-03 rejection of KG proximity stands).

v2 is a **formalization layer** on top of what RFC-072 (Canonical
Identity Layer + cross-layer bridge) already shipped + what #1035
(NER pre-pass) made possible. The infrastructure that round-2 of the
WIP spec framed as forward-looking is already behind us; the real
remaining work is turning the cross-layer queries the CIL makes
possible into shipped per-artifact data.

**Architecture Alignment:** Builds on RFC-072 (CIL + bridge),
preserves RFC-091's relational-not-proximity retrieval decision,
unblocks RFC-088 enrichers and RFC-094 relational queries, and gates
two PRD-029 / PRD-028 viewer surfaces.

## Problem Statement

After three rounds of WIP spec doc iteration and a full re-read of
PRD-017, PRD-019, RFC-049/050, RFC-055/056, RFC-072, RFC-091, the
state of the corpus ontology has three connected problems:

**1. Entity extraction is shipped but the schema still calls it
deferred.** PRD-017 (line 26) and RFC-049 (Resolved Q2) deferred
`Entity` extraction to v1.1; v1.1 never landed as a formal release.
But the substance arrived anyway: #1035's NER pre-pass shipped
entity extraction at 97–100% recall, and RFC-072 (Canonical Identity
Layer) made `person:` / `org:` / `topic:` IDs canonical. The
documentation still describes an `Entity(kind=...)` discriminator
that no longer matches the operational state.

**2. Cross-layer edges are defined but never emitted.** The GI
schema defines `ABOUT (Insight → Topic)` and `MENTIONS (Insight →
Entity)` edges, but the default GI pipeline never emits them. The
CorpusGraph (RFC-072 Slice B) composes them in-memory at query
time, but the per-artifact JSON does not contain them — so the
viewer cannot render the relationships without rebuilding the graph
at request time.

**3. Two flagship use cases the CIL was built to enable never
shipped.** RFC-072 named **Position Tracker** (Person × Topic over
time) and **Person Profile** (everything about a Person) as the
products the bridge unlocks. Both depend on two additive Insight
fields RFC-072 §2 specified — `insight_type` and `position_hint` —
that never made it into a release.

**Use Cases:**

1. **Position Tracker** — trace how a Person's stated positions on
   a Topic evolved across episodes and across time. Requires:
   typed Person + Topic nodes (✓ shipped, this RFC formalizes),
   `ABOUT` + `MENTIONS_PERSON` edges (this RFC), `insight_type`
   classification (this RFC), `position_hint` arithmetic (this RFC),
   `Episode.publish_date` (✓ shipped).
2. **Person Profile** — single page per Person showing insights
   voiced, quotes attributable, topics discussed, episodes appeared
   in, organizations associated with. Same data foundations as
   Position Tracker.
3. **Relational queries via RFC-094** — `who_said_what_about(topic)`,
   `insights_about(person)`, `mentions_of(org)` — all newly viable
   once descriptive edges materialize per-artifact.

## Goals

1. **Formalize the entity taxonomy.** Replace `Entity(kind=person)` /
   `Entity(kind=organization)` with first-class `Person` and
   `Organization` node types in both KG and GI ontologies. Add
   `Podcast` as a first-class node (RFC-091 wanted `FROM_SHOW`;
   #849 descoped it; v2 picks it up).
2. **Land the v1.1 additive Insight fields.** Ship `insight_type`
   (LLM-classified enum: claim | recommendation | observation |
   question | unknown) and `position_hint` (0.0–1.0, derived from
   Quote timestamps via 4-step waterfall) as Optional fields on
   `Insight`.
3. **Materialize cross-layer descriptive edges.** GI pipeline emits
   `ABOUT (Insight → Topic)`, `MENTIONS_PERSON (Insight → Person)`,
   `MENTIONS_ORG (Insight → Organization)` as part of the per-artifact
   shape. KG pipeline emits `HAS_EPISODE (Podcast → Episode)`.
4. **Declare a two-tier edge contract.** Each edge carries an
   `edge_class` metadata field: `evidentiary` (grounding contract,
   load-bearing), `descriptive` (classification), `discovery`
   (KG co-occurrence), `structural` (parent/child), `attribution`
   (speaker). Future contributors can see at a glance which edges
   are grounding-load-bearing and which are not.
5. **Ship the two flagship viewer surfaces.** Position Tracker and
   Person Profile. The data foundation is delivered by v2 (chunk 7).
   **Scope-cut 2026-06-21**: the viewer UI for both surfaces is split
   into follow-up tickets (A/B/C in
   `docs/wip/RFC097_CHUNK8_FOLLOWUP_TICKETS.md`); revised chunk 8
   ships only the two-tier edge contract visual styling so the v2
   foundation PR closes without a multi-week UI build.
6. **Re-baseline measurement.** Full silver rebuild (`silver_opus47_*`
   + `silver_sonnet46_*_benchmark_v2`) on the new shape, with the
   silver/judge vendor-bias rule honored (no candidate vendor matches
   silver/judge vendor — #939 lesson).
7. **Preserve the grounding contract bit-for-bit.** `Insight.grounded
   ⇔ ≥1 SUPPORTED_BY → Quote` invariant is non-negotiable. Viewer
   dashed-border styling at `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts:407`
   (selector `node[type = "Insight"].insight-ungrounded`) is the visible
   manifestation; `hideUngroundedInsights` is the user-facing filter.

## Constraints & Assumptions

**Constraints:**

- **Grounding contract preserved bit-for-bit.** Only `SUPPORTED_BY`
  edges count toward `Insight.grounded = true`. `ABOUT` /
  `MENTIONS_PERSON` / `MENTIONS_ORG` are descriptive and do **not**
  promote ungrounded Insights.
- **No retrieval-side changes.** RFC-091's 2026-06-03 rejection of
  KG proximity as a ranking signal stands. v2 enables relational
  queries (RFC-094), not proximity ranking.
- **No artifact file merger.** `kg.json` and `gi.json` remain
  separately emitted, separately toggleable per PRD-019 line 45.
  v2 unifies the **ontology**, not the operational file layout.
- **Permissive transition.** Schemas v2.0 (KG) + v3.0 (GI) accept
  both legacy and new shape during the bake window. Hard rejection
  of legacy lands in chunk 9, gated on 2–4 weeks of production
  operation under v2.
- **Corpora-first migration order.** Migration scripts rewrite
  existing corpora before the viewer ships against the new shape
  (matches RFC-072 migration pattern; minimizes dual-shape
  complexity in viewer code).
- **DGX research mode for eval runs.** The silver rebuild + candidate
  sweep uses the project's `gpu-mode-swap.sh research` slot — never
  `code` (which is coder-next, off-limits).
- **Silver/judge vendor disjoint from candidates.** Re-baseline must
  not pick a silver+judge vendor that matches any single candidate
  (#939 Sonnet-mimicry lesson; documented in
  `autoresearch/JUDGING.md`).

**Assumptions:**

- `Episode.duration_ms` is recoverable to ≥99.9% via the 4-step
  position_hint waterfall (RSS → segments → max Quote ts → skip).
  Empirically validated on 477 GI artifacts in
  `.test_outputs/manual/**/*.gi.json` (86% RSS coverage today;
  segments fill the gap).
- LLM classification of `insight_type` is tractable in one extraction
  prompt change. Cost validated by a small accuracy sweep before
  full silver regen (chunk 5).
- Semantic-match threshold for ABOUT / MENTIONS_PERSON / MENTIONS_ORG
  emission can reuse `search/topic_clusters.py`'s Pareto-0.75
  pattern from #1035 NER pre-pass. Precision over recall.
- Production prod-v2 corpus (99 episodes at
  `.test_outputs/manual/prod-v2/corpus`) is intact and migration-ready.

## Design & Implementation

### 1. Two-Tier Edge Contract

The grounding-asymmetry paradox round-2 missed:

> GIL requires verbatim quote evidence (grounding contract).
> KG only requires co-occurrence (MENTIONS).
> Merging them naively would force all entities into the grounding
> contract (bloat) — or allow ungrounded entities (contradiction).

v2 resolves this by declaring `edge_class` per edge:

| Edge | `edge_class` | Grounding-load-bearing? |
|---|---|---|
| `SUPPORTED_BY` (Insight → Quote) | **evidentiary** | ✓ — the grounding contract |
| `HAS_EPISODE` (Podcast → Episode) | structural | ✗ |
| `SPOKE_IN` (Person → Episode) | structural | ✗ |
| `HAS_INSIGHT` (Episode → Insight) | structural | ✗ |
| `SPOKEN_BY` (Quote → Person) | attribution | ✗ |
| `ABOUT` (Insight → Topic) | descriptive | ✗ |
| `MENTIONS_PERSON` (Insight → Person) | descriptive | ✗ |
| `MENTIONS_ORG` (Insight → Organization) | descriptive | ✗ |
| `MENTIONS` (Topic/Person/Org → Episode) | discovery | ✗ — co-occurrence only |

Only `SUPPORTED_BY` decides `Insight.grounded`. Descriptive edges
classify; they do not prove.

`edge_class` is **schema metadata**, not a runtime field — it lives
in the schema definition so the viewer's stylesheet, the relational
query layer, and future contributors all read the same source of
truth about which edges are grounding-load-bearing.

### 2. Node Taxonomy Upgrade

**KG ontology v2.0:**

```diff
  nodes:
- - Episode | Topic | Entity(kind: person|org)
+ - Episode | Topic | Person | Organization | Podcast

  edges:
- - MENTIONS (Topic|Entity → Episode)
+ - MENTIONS (Topic|Person|Organization → Episode)
+ - HAS_EPISODE (Podcast → Episode)
  - RELATED_TO (still reserved, still not emitted)
```

**GI ontology v3.0:**

```diff
  nodes:
- - Podcast | Episode | Person | Topic | Insight | Quote (Entity deferred)
+ - Podcast | Episode | Person | Organization | Topic | Insight | Quote

  Insight properties (additive, Optional):
+ - insight_type: enum [claim, recommendation, observation, question, unknown]
+ - position_hint: number 0.0-1.0

  edges:
  - HAS_EPISODE         (Podcast → Episode)
  - SPOKE_IN            (Person → Episode)
  - HAS_INSIGHT         (Episode → Insight)
  - SUPPORTED_BY        (Insight → Quote)              [evidentiary]
  - SPOKEN_BY           (Quote → Person)               [attribution]
+ - ABOUT               (Insight → Topic)              [descriptive]
+ - MENTIONS_PERSON     (Insight → Person)             [descriptive]
+ - MENTIONS_ORG        (Insight → Organization)       [descriptive]
```

Both schemas keep accepting the legacy shape during the bake window
(chunk 2 is **permissive**). Hard rejection lands in chunk 9 (gated
on 2–4 weeks of production operation under v2).

### 3. `insight_type` and `position_hint`

Both fields originate in RFC-072 §2 as v1.1 additive specifications
that never shipped. v2 lands them.

**`insight_type`** — LLM-classified enum on each Insight. Options:

- `claim` — a factual assertion ("X causes Y")
- `recommendation` — prescriptive ("you should do X")
- `observation` — descriptive without assertion ("I noticed X")
- `question` — interrogative ("but is X really true?")
- `unknown` — classifier couldn't decide

**Implementation status (post-chunk-5 retroactive sweep, 2026-06-20)**:
chunk 5 shipped insight_type via the **megabundle / extraction-bundled
prompt path** (`prompting/megabundle.py`) — provider responses there
include `insight_type` and the parser threads it through to the GI
pipeline. The legacy per-provider `prompts/{anthropic,gemini,openai,
deepseek,grok,mistral,ollama}/insight_extraction/v1.j2` prompts emit
plain-text insights only; their outputs degrade to `insight_type:
"unknown"` via `_normalize_insight_type`'s default. Updating each
per-provider prompt to emit structured `{text, insight_type}` JSON is a
**parked v2.1 follow-up** — additive (default stays valid), no schema
change required, but each provider needs re-tuning and an accuracy
sweep before flipping over.

Implemented as a single prompt extension in the GI extraction prompt
(across all provider shards in `prompts/{anthropic,gemini,openai,
deepseek,grok,mistral,ollama,vllm}/gi/*`). Classification accuracy
measured against an N=100 hand-labelled sample on the Tier-3 fixture
before silver regen (chunk 5 gate).

**`position_hint`** — pure arithmetic, no LLM. 4-step waterfall:

1. `Episode.duration_ms` from RSS `<itunes:duration>` (~86% corpus
   coverage today)
2. Last segment's `end × 1000` from `*.segments.json` (~99.9%
   coverage — every transcribed episode has segments)
3. `max(Quote.timestamp_end_ms)` across the episode's Quotes
   (lower-bound; preserves ordering for Position Tracker)
4. Skip emission — field is Optional (< 0.1% edge case)

Computed as `mean(Quote.timestamp_start_ms) / duration_ms`, range
[0.0, 1.0]. Implementation in NEW `src/podcast_scraper/gi/position_hint.py`
as a pure helper (testable in isolation).

### 4. Cross-Layer Descriptive Edges (per-artifact materialization)

GI pipeline gains a post-pass after Insight extraction that:

1. Reads the episode's `bridge.json` (or sibling `*.kg.json`) for
   canonical Topic / Person / Organization IDs
2. Semantic-matches each Insight against the episode's
   Topics / Persons / Orgs (sentence-transformer similarity or
   LLM judge; reuses `search/topic_clusters.py` Pareto-0.75
   threshold pattern from #1035 NER pre-pass)
3. Emits `ABOUT` / `MENTIONS_PERSON` / `MENTIONS_ORG` edges with
   confidence scores

Same architectural pattern as #1035 — deterministic helper around
an LLM call (or pure similarity). Lives in extensions to
`src/podcast_scraper/gi/about_edges.py`, `gi/relational_edges.py`,
`gi/edge_normalization.py`.

**Threshold calibration** is a chunk-4 gate: small precision/recall
sweep on the Tier-3 fixture before merge. Descriptive edges should
err on precision (descriptive ≠ noisy).

### 5. Unified Ontology Document

NEW: `docs/architecture/corpus/ontology.md` — single textual
ontology covering:

- All node types: Podcast, Episode, Person, Organization, Topic,
  Insight, Quote
- All edges with `edge_class` metadata column
- Canonical ID rules (re-stated from RFC-072 + slugify)
- Grounding contract invariant with explicit citation to viewer
  dashed-border styling
- Explicit RFC-091 citation: "relational, not proximity-based"
- `insight_type` enum + `position_hint` 4-step waterfall
- Two-tier edge contract

`docs/architecture/kg/ontology.md` and `docs/architecture/gi/ontology.md`
gain "**Superseded by `corpus/ontology.md` for v2.0+**" banners
but stay readable for archaeology.

### 6. Migration Story

Three migration scripts (chunk 6, RFC-072 migration pattern):

- `scripts/migrate_kg_entity_to_person_org.py` — legacy
  `Entity(kind=person)` → `Person` node, `Entity(kind=organization)`
  → `Organization` node; rewrite edges
- `scripts/backfill_gi_insight_type.py` — optional LLM-classify
  existing Insights without re-extracting them (cheap model;
  amortizes a re-extraction we don't need)
- `scripts/compute_gi_position_hints.py` — pure arithmetic, works on
  every existing artifact with Quote timestamps

Registered in `src/podcast_scraper/migrations/gil_kg_identity_migrations.py`
+ `src/podcast_scraper/upgrade/migrations/`. `corpus_version.py`
constant bumped.

Pre-flight: dry-run all three against `.test_outputs/manual/prod-v2/corpus`
(99 eps) in copy mode; diff before/after.

## Key Decisions

1. **`insight_type` lands in v2, not v2.1.**
   - **Decision**: Ship insight_type classification in chunk 5
     alongside position_hint.
   - **Rationale**: Person Profile is materially weaker without it;
     the prompt change is small once you're already touching the GI
     extraction prompt; classification accuracy measurable against
     hand-labelled sample before silver regen gates the change.

2. **Podcast / Show as first-class node in v2.**
   - **Decision**: KG ontology v2.0 promotes Podcast to a node type;
     emits `HAS_EPISODE (Podcast → Episode)` edge from existing
     `Episode.podcast_id` / `Episode.feed_id`.
   - **Rationale**: RFC-091 wanted `FROM_SHOW`; #849 descoped it.
     Data is already on every Episode; cost is near-zero; viewer
     clicks ("Singletrack Sessions → list all episodes") become
     trivial.

3. **Unified ontology doc explicitly cites RFC-091.**
   - **Decision**: The new `docs/architecture/corpus/ontology.md`
     opens with "v2 is relational, not proximity-based" and cites
     RFC-091 as the decision record.
   - **Rationale**: Prevents a future contributor from proposing
     proximity-based retrieval again. RFC-091's empirical −0.018
     to −0.170 nDCG hits are not theoretical.

4. **Corpora-first migration order.**
   - **Decision**: Chunk 6 (migration scripts) lands and runs against
     existing corpora **before** chunk 8 (viewer) ships against the
     new shape.
   - **Rationale**: Cleaner state; no dual-shape complexity in viewer
     code. Matches RFC-072 migration pattern which shipped cleanly.

5. **Two-tier edge contract is schema metadata, not runtime.**
   - **Decision**: `edge_class` lives in the schema definition; not
     a per-edge runtime field.
   - **Rationale**: Avoids per-edge bloat; viewer styling and
     relational queries both consume the schema metadata; future
     contributors see the contract in one place.

6. **Full silver rebuild, accept loss of direct comparability.**
   - **Decision**: Regenerate `silver_opus47_*` and
     `silver_sonnet46_*_benchmark_v2` on the new shape; previous
     scoreboards become directly incomparable to post-v2.
   - **Rationale**: Clean measurement beats false comparability.
     Partial re-baseline (compare-on-overlap) would force two-shape
     scoring logic and obscure the metric story.

7. **Single long-lived branch with one PR at the end.**
   - **Decision**: `feat/corpus-ontology-v2` accumulates one commit
     per chunk; one PR opens after chunk 9 (or chunk 8 if chunk 9
     bake-gate hasn't elapsed).
   - **Rationale**: Operator preference. Bisectable per chunk.
     Avoids churn of nine separate PRs reviewing the same context.

## Alternatives Considered

1. **Direction B — merge `kg.json` and `gi.json` into one corpus
   artifact (round-2 sketch).**
   - **Description**: Single per-episode JSON containing both KG
     and GI nodes/edges.
   - **Pros**: One file to read; viewer state simpler.
   - **Cons**: Reverses PRD-019 line 26 Non-Goal explicitly;
     forces a grounding-contract bloat (every KG entity would need
     a grounding edge, or we'd silently allow ungrounded entities
     in GI). Permanently couples two pipelines that have separate
     enable/disable knobs (PRD-019 line 45).
   - **Why Rejected**: Two-tier edge contract solves the
     classification problem without violating prior decisions.

2. **Defer `insight_type` to v2.1.**
   - **Description**: Ship only `position_hint` (arithmetic) in v2;
     `insight_type` later once we measure classification accuracy.
   - **Pros**: Smaller prompt change footprint; one fewer LLM
     accuracy bet.
   - **Cons**: Person Profile materially degraded; v2's viewer
     surfaces ship without their canonical filter dimension.
   - **Why Rejected**: Operator-locked decision (Q-3.1) after
     weighing Person Profile impact.

3. **Topic semantic dedup in v2.**
   - **Description**: Merge "AI Regulation" + "AI Policy" into one
     canonical Topic.
   - **Pros**: Cleaner viewer browse.
   - **Cons**: Corpus-level work; needs measurement against the
     v2-emitted Topic set (chicken-and-egg).
   - **Why Rejected**: Deferred to v3. Topic dedup needs the
     v2-emitted corpus as input; sequencing prevents it from
     fitting in v2.

4. **CONTRADICTS edges between Insights in v2.**
   - **Description**: NLI-driven contradiction detection between
     Insights.
   - **Pros**: Powers Search "contradiction" badges, LITM pack
     `top_contradiction` field.
   - **Cons**: Requires a new NLI pipeline stage that does not
     exist today; adds extraction cost and a calibration problem
     out of proportion to the v2 scope.
   - **Why Rejected**: Deferred to v3 (RFC-088 analysis layer
     territory).

5. **Re-baseline on overlap (partial silver regen).**
   - **Description**: Keep prior silvers; score the v2 shape only
     against fields present in both old + new.
   - **Pros**: Direct comparability with prior scoreboards.
   - **Cons**: Two-shape scoring logic; metrics that depend on new
     fields (ABOUT coverage, insight_type accuracy) excluded;
     obscures the metric story.
   - **Why Rejected**: Clean measurement beats false comparability.

## Testing Strategy

**Test Coverage:**

- **Schema unit tests** (`tests/unit/{kg,gi}/test_schema.py`,
  `test_schema_validator.py`): both legacy + new shape validate
  during permissive window.
- **Pipeline unit tests** (`tests/unit/kg/test_pipeline.py`,
  `tests/unit/gi/test_pipeline.py`, plus new
  `tests/unit/gi/test_position_hint.py`): typed-node emission,
  descriptive-edge post-pass, position_hint waterfall (exhaustive
  coverage of all four steps).
- **Pipeline integration tests** (`tests/integration/{kg,gi}/test_pipeline_*.py`):
  golden artifacts regenerate.
- **Tier-2 matrix rows** (`tests/integration/test_multi_run_corpus_fixture.py`):
  one row each for chunks 3, 4, 5 per ADR-095's real-bug-to-matrix-row
  discipline.
- **Migration tests** (`tests/integration/upgrade/test_migration_*.py`):
  idempotency, round-trip, malformed-legacy rejection.
- **Scorer tests** (`tests/eval/test_{kg,gi,ner}_scorer*.py`): new
  shape, new coverage metrics (ABOUT coverage, MENTIONS_PERSON
  coverage, insight_type accuracy, position_hint sanity).
- **Viewer tests** (`web/gi-kg-viewer/src/**/__tests__/*.test.ts` +
  Playwright): Position Tracker component, Person Profile component,
  edge-class stylesheet, grounded-Insight dashed border preserved
  bit-for-bit.
- **Stack-test** (`tests/stack-test/`): full pipeline + viewer
  integration via `make ci-ui-full` (chip/testid refactor cadence
  per project memory).

**Test Organization:**

- Unit tests next to the module under test
- Integration tests in `tests/integration/{kg,gi,upgrade}/`
- Eval tests in `tests/eval/`
- Viewer tests under `web/gi-kg-viewer/src/` (cd into the directory
  before running vitest per project memory)
- Stack-test under `tests/stack-test/`

**Test Execution:**

- Subtarget reverify after each chunk: `make docs`, `make lint`,
  module-specific pytest selections (per project memory: don't
  re-run `ci-fast` between chunks)
- `make ci-fast` / `ci-ui-fast` only as the final pre-push step
- `make ci-ui-full` before push for chunk 8 (viewer chip/testid
  surfaces touched)
- `make docs` before push for any chunk that edits *.md
- `npm run build` locally in `web/gi-kg-viewer/` before push for
  chunk 8

## Rollout & Monitoring

**Rollout Plan (chunked, one commit per chunk on `feat/corpus-ontology-v2`,
one PR at the end):**

- **Chunk 0 (this RFC)** — RFC-097 + cross-doc closure sweep.
  Risk: zero. Days: 1.
- **Chunk 1** — `docs/architecture/corpus/ontology.md` + supersession
  banners. Risk: zero. Days: 1.
- **Chunk 2** — KG schema v2.0 + GI schema v3.0 (permissive).
  Risk: low. Days: 1.
- **Chunk 3** — KG pipeline emits Person + Organization + Podcast +
  HAS_EPISODE. Risk: medium. Days: 2–3.
- **Chunk 4** — GI pipeline emits ABOUT / MENTIONS_PERSON /
  MENTIONS_ORG via post-pass. Risk: medium. Days: 2–3 (incl.
  threshold calibration sweep).
- **Chunk 5** — Insight.insight_type (LLM) + position_hint
  (waterfall). Risk: medium. Days: 3–4 (incl. accuracy sweep).
- **Chunk 6** — Migration scripts + corpus_version bump.
  Risk: medium. Days: 1–2.
- **Chunk 7** — Full silver rebuild + scoreboard re-baseline.
  Risk: medium-high. Days: 2–4 (half-day LLM time + candidate
  sweep + vendor-bias check).
- **Chunk 8 (scope-cut 2026-06-21)** — Two-tier edge contract visual
  styling only. The three flagship-view items originally bundled
  (Person Landing shared component, Position Tracker view, Person
  Profile view) are split into follow-up tickets so the v2 foundation
  PR closes cleanly without a multi-week UI build.
  Risk: low. Days: 0.5–1.
  Follow-up tickets (drafted in
  `docs/wip/RFC097_CHUNK8_FOLLOWUP_TICKETS.md`):
  - **Ticket A** — Person Landing shared component (PRD-029 spec)
  - **Ticket B** — Position Tracker view (PRD-028 spec, UXS-009)
  - **Ticket C** — Person Profile view (PRD-029 spec, UXS-010)
  Both PRD-028 and PRD-029 v2-closure stanzas updated to reflect the
  split (data foundation = chunk 7; viewer surface = follow-ups).
- **Chunk 9 (deferred to follow-up PR, bake-gated)** — Drop legacy
  schema support; ADR-101 records the decision. Gate: 2–4 weeks of
  production operation under v2. Risk: low (code-wise). Days: 1.
  **Not in this PR**: with the flagship-view items split out of
  chunk 8, the v2 foundation PR closes after revised chunk 8.
  Chunk 9 lands in a follow-up PR after the bake window.

**Monitoring:**

- After chunk 5 lands in prod: track `insight_type` distribution
  per provider; flag classification collapse (one class > 80%) as a
  prompt regression.
- After chunk 5 lands in prod: track `position_hint` waterfall step
  distribution (step 1 / step 2 / step 3 / step 4 frequencies); flag
  step-4 (skip) rate creep above 1% as a corpus-quality regression.
- After chunk 4 lands in prod: track ABOUT / MENTIONS_PERSON /
  MENTIONS_ORG edges-per-Insight; flag mean drop below 1.0 (per
  Insight) or above 10.0 (noisy threshold).
- After chunk 8 ships: track viewer Position Tracker + Person
  Profile click-through; if both stay near zero for 2 weeks, the
  surfaces don't justify their maintenance cost.

**Success Criteria:**

1. Full silver rebuild scoreboards (Opus 4.7 + Sonnet 4.6 +
   DGX Qwen3-30B + others) published with non-regressive
   KG/GI coverage on the new shape.
2. Position Tracker + Person Profile usable end-to-end on the
   prod-v2 corpus (99 eps).
3. Migration scripts dry-run clean on prod-v2 corpus.
4. Grounding contract preserved bit-for-bit: every Insight with
   `grounded=true` has ≥1 SUPPORTED_BY edge; no descriptive edge
   promotes ungrounded Insights.
5. Chunk 9 (legacy drop) lands without reading any legacy artifact
   in CI.

## Relationship to Other RFCs

This RFC is part of the autoresearch programme (epic #907) +
corpus-ontology consolidation effort that includes:

1. **RFC-049 (GI core)** — entity-extraction v1.1 deferral retired
   by v2; ABOUT + MENTIONS edges (defined but unemitted in v1) now
   shipped per-artifact. CONTRADICTS edges remain deferred to v3.
2. **RFC-050 / RFC-056 (GI / KG use cases)** — Position Tracker and
   Person Profile use cases (named here, moved to RFC-072) get
   delivered surfaces in chunk 8. NL query layer remains deferred.
3. **RFC-055 (KG core)** — Entity discriminator replaced by typed
   Person + Organization + Podcast nodes; KG ontology v2.0.
4. **RFC-072 (CIL + bridge)** — v2 materializes the cross-layer
   edges the CorpusGraph composes in-memory; v1.1 additive fields
   (`insight_type`, `position_hint`) ship; Known Limitations 1, 2,
   3, 5 remain open as designed.
5. **RFC-088 (enrichment layer)** — v2 unblocks enricher inputs
   (`bridge.json` is the foundation); enricher protocol unchanged.
6. **RFC-090 (hybrid retrieval)** — orthogonal; LanceDB-first stands;
   segment document `insight_document.entity_type` may gain
   precision from v2 typed nodes (informational, not blocking).
7. **RFC-091 (KG proximity signal)** — REJECTED 2026-06-03; v2
   **confirms** the rejection. Relational queries (RFC-094) are
   the value path, not proximity ranking.
8. **RFC-092 (ML query router)** — orthogonal; intent taxonomy may
   gain classes if new node types enable new intent classes
   post-v2.
9. **RFC-093 (LITM context packs)** — `CorpusBriefingPack` builder
   gets richer inputs (`insight_type`, `position_hint`); MCP tool
   layer (RFC-095) remains the publish gate.
10. **RFC-094 (relational query layer)** — v2 CIL ensures stable
    `person:` / `topic:` / `org:` IDs; `who_said_what_about`,
    `insights_about`, `mentions_of_person` queries newly viable
    against per-artifact edges.
11. **RFC-095 (generic MCP server)** — v2 CIL is prerequisite for
    stable tool schemas; MCP layer itself unchanged.

**Key Distinction:**

- **RFC-072** — defined the cross-layer infrastructure (CIL, bridge,
  CorpusGraph) and named v1.1 additive fields.
- **RFC-097 (this RFC)** — formalizes the ontology textually,
  materializes the cross-layer edges per-artifact, lands the
  additive fields, and ships the two viewer surfaces.

Together they provide:

- Stable canonical IDs across artifacts (RFC-072)
- Per-artifact materialization of cross-layer edges (RFC-097)
- v1.1 additive Insight fields (RFC-097)
- Two flagship viewer surfaces (RFC-097)

## Benefits

1. **Documentation reflects operational state.** The "v1.1 deferral"
   language survives in PRD-017 / RFC-049 despite #1035 + RFC-072
   having shipped the substance; RFC-097 retires it.
2. **Cross-layer edges queryable without rebuilding CorpusGraph.**
   Per-artifact `ABOUT` / `MENTIONS_PERSON` / `MENTIONS_ORG` lets
   the viewer + relational query layer skip in-memory composition.
3. **Two flagship surfaces ship.** Position Tracker + Person Profile
   are the products RFC-072 was built to enable.
4. **Cleaner schemas.** Typed Person + Organization + Podcast nodes
   eliminate the discriminator hack and reduce repair-pass surface
   area (`kg/filters.py::repair_entity_kind`).
5. **Edge-class metadata.** Future contributors don't have to
   re-derive which edges are grounding-load-bearing — it's in the
   schema.
6. **Re-baseline measurement.** Full silver rebuild gives a clean
   per-shape scoreboard the autoresearch programme can build on
   without legacy-shape noise.

## Migration Path

1. **Chunk 1** — docs only; no migration.
2. **Chunk 2** — schemas permissive; legacy artifacts still validate.
3. **Chunks 3–5** — pipelines emit new shape; older artifacts still
   readable.
4. **Chunk 6** — `scripts/migrate_kg_entity_to_person_org.py`,
   `scripts/backfill_gi_insight_type.py`,
   `scripts/compute_gi_position_hints.py` rewrite existing corpora.
   Dry-run first against prod-v2 corpus (99 eps) in copy mode.
5. **Chunk 7** — silver rebuild on new shape; full candidate sweep
   re-baseline.
6. **Chunk 8** — viewer ships against new shape only (post-migration).
7. **Chunk 9** — schemas reject legacy; gated on 2–4 weeks of
   production operation under v2 (ADR-101 records the decision).

## Open Questions

**Resolved by operator (2026-06-20):**

- **Q-3.1** — `insight_type` lands in v2 (not v2.1). ✓
- **Q-3.2** — `position_hint` 4-step waterfall (RSS → segments →
  max Quote ts → skip). ✓
- **Q-3.3** — Podcast / Show as first-class node in v2. ✓
- **Q-3.4** — Unified ontology doc explicitly cites RFC-091. ✓
- **Q-3.5** — Corpora-first migration order. ✓

**Remaining (implementation-time decisions):**

1. ABOUT / MENTIONS_PERSON / MENTIONS_ORG similarity threshold —
   resolved by the chunk-4 calibration sweep.
2. `insight_type` provider sweep — does the cheap model classify
   well enough that we don't pay Opus-tier extraction for it?
   Resolved by the chunk-5 accuracy sweep.
3. Migration script semantic-match confidence floor for
   `backfill_gi_insight_type.py` — what classification confidence
   counts as "good enough to backfill"? Resolved at chunk-6
   implementation.

## References

- **Related PRDs**: PRD-017, PRD-019, PRD-026 through PRD-029, PRD-031
- **Related RFCs**: RFC-049, RFC-050, RFC-055, RFC-056, RFC-072,
  RFC-088, RFC-090, RFC-091, RFC-092, RFC-093, RFC-094, RFC-095
- **Round-1 WIP spec**: `docs/wip/SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md`
- **Round-2 WIP spec**: `docs/wip/SPEC_KG_GI_ONTOLOGY_V2_2026-06-20.md`
- **Round-3 WIP spec (live design)**:
  `docs/wip/SPEC_KG_GI_ONTOLOGY_V2_ROUND3_2026-06-20.md`
- **v3 wishlist**: `docs/wip/SPEC_KG_GI_ONTOLOGY_V3_WISHLIST_2026-06-20.md`
- **GH anchor**: #1036 (issue body updated to reflect round-3 framing)
- **Programme epic**: #907 (autoresearch)
- **Source code touched**: `src/podcast_scraper/{kg,gi}/`,
  `src/podcast_scraper/identity/`, `src/podcast_scraper/search/`,
  `src/podcast_scraper/evaluation/`, `web/gi-kg-viewer/src/`,
  `scripts/migrate_*`, `scripts/backfill_*`, `scripts/compute_*`

---

## Appendix A — Cross-reference closures

For each related document, this appendix records what RFC-097 v2
closes, what remains open, and what closure annotation the
companion doc receives (applied in the same commit as this RFC).
Operators can audit at a glance which threads survive v2 and where
to find them.

### `docs/rfc/RFC-049-grounded-insight-layer-core.md` — Completed

- **Closed by v2**: entity-extraction v1.1 deferral retired
  (Person + Organization first-class); `ABOUT (Insight → Topic)`
  edge emitted; `MENTIONS` split into `MENTIONS_PERSON` +
  `MENTIONS_ORG` and emitted; `Insight.insight_type` (additive v1.1
  field) ships; `Insight.position_hint` (additive v1.1 field) ships.
- **Remains open**: Topic semantic deduplication (Resolved Q1 →
  v3); `CONTRADICTS` edges (Resolved Q1 → v3, needs NLI pipeline);
  Phase 5 chunk-to-Insight lift (KL1, char-offset alignment blocker).
- **Companion-doc annotation**: closure note added pointing to
  RFC-097 §2 + §3 + §4.

### `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md` — Completed

- **Closed by v2**: cross-layer use cases (originally named in
  RFC-050, moved to RFC-072) get shipped surfaces via Position
  Tracker + Person Profile (chunk 8 + PRDs-028/029).
- **Remains open**: NL query layer (deferred indefinitely; RFC-092
  intent routing is the partial substitute); grounding-rate
  empirical thresholds (validate post-v2).
- **Companion-doc annotation**: cross-ref to RFC-072 §Vision +
  RFC-097 §Use Cases.

### `docs/rfc/RFC-055-knowledge-graph-layer-core.md` — Completed

- **Closed by v2**: `Entity(kind=...)` discriminator replaced by
  typed Person + Organization + Podcast nodes; KG ontology v2.0.
- **Remains open**: Topic semantic dedup (v3, RFC-075 territory);
  `MENTIONS` edge semantics — KG co-occurrence stays as `discovery`
  edge_class, retrieval-relevance settled by RFC-091.
- **Companion-doc annotation**: ontology migration note pointing
  to RFC-097 §2 + `scripts/migrate_kg_entity_to_person_org.py`.

### `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md` — Completed

- **Closed by v2**: same cross-layer use cases as RFC-050 (moved
  to RFC-072) get shipped surfaces.
- **Remains open**: NL query translation (post-v2); perfect entity
  resolution (PRD-019 non-goal); KG semantic deduplication (v3).
- **Companion-doc annotation**: mirror RFC-050 cross-ref.

### `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — Draft

- **Closed by v2**: v1.1 additive fields (`insight_type`,
  `position_hint`) ship per-artifact; cross-layer descriptive
  edges materialize in per-artifact JSON (not just CorpusGraph
  composition).
- **Remains open**: Phase 5 chunk-to-Insight lift (KL1, char-offset
  alignment); cross-episode Person identity merging / alias registry
  (KL2, v3); Topic semantic dedup (KL3, v3); `CONTRADICTS` edges
  (KL5, v3); position-change detection (analysis layer territory).
- **Companion-doc annotation**: v2 closure note for §2a / §2b
  fields + §5 cross-layer query patterns.

### `docs/rfc/RFC-088-enrichment-layer-architecture.md` — Draft

- **Closed by v2**: enricher input data (bridge.json + typed
  descriptive edges) shipped per-artifact.
- **Remains open**: QueryEnricher protocol (Phase 4); typed
  contradiction enricher output (needs CONTRADICTS edges); LLM
  tier enrichers (gated on opt-in approval).
- **Companion-doc annotation**: prerequisite note pointing to
  RFC-097 §4 (bridge materialization).

### `docs/rfc/RFC-090-hybrid-retrieval.md` — Draft

- **Closed by v2**: nothing directly (orthogonal); typed node
  precision may improve segment document `entity_type` field
  post-v2.
- **Remains open**: FAISS removal (Phase 3, gated on discriminating
  eval #858); KG proximity signal slot (REJECTED 2026-06-03); ML
  query router (RFC-092).
- **Companion-doc annotation**: informational cross-ref noting v2
  is orthogonal to ranking pipeline.

### `docs/rfc/RFC-091-kg-proximity-signal.md` — Rejected

- **Closed by v2**: nothing directly (v2 **confirms** the rejection;
  does not reverse).
- **Remains open**: relational structure (RFC-094) survives as the
  value path; co-occurrence ranking does not.
- **Companion-doc annotation**: confirmation note + cross-ref to
  RFC-097 §Goals "no retrieval-side changes" + §Key Decision 3.

### `docs/rfc/RFC-092-ml-query-router.md` — Draft

- **Closed by v2**: nothing directly (orthogonal).
- **Remains open**: ML router deployment gate (≥30 human-judged
  queries with margin ≥ rules baseline); template-defined training
  labels.
- **Companion-doc annotation**: informational cross-ref noting
  intent taxonomy may extend post-v2.

### `docs/rfc/RFC-093-litm-context-packs.md` — Draft

- **Closed by v2**: `CorpusBriefingPack` builder gains richer
  inputs (`insight_type` for filtering, `position_hint` for
  temporal sorting).
- **Remains open**: MCP tool wrapper (RFC-095 prerequisite);
  `top_contradiction` content (needs CONTRADICTS edges);
  `coverage_gaps` surface (corpus-impact surface TBD).
- **Companion-doc annotation**: v2 enablement note for builder
  inputs.

### `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md` — Draft

- **Closed by v2**: stable `person:` / `topic:` / `org:` IDs for
  relational queries (RFC-072 CIL formalized in RFC-097); ABOUT
  + MENTIONS_PERSON + MENTIONS_ORG edges per-artifact enable
  `who_said_what_about`, `insights_about`, `mentions_of_person`.
- **Remains open**: MCP exposure of relational queries (RFC-095);
  panel caching (OQ-1); coverage-aware query filtering.
- **Companion-doc annotation**: v2 prerequisite note for query
  layer.

### `docs/rfc/RFC-095-generic-mcp-server.md` — Draft

- **Closed by v2**: nothing directly (prerequisite: stable CIL IDs
  for tool inputs).
- **Remains open**: HTTP/SSE transport (OQ-1); MCP resource layer
  (OQ-2); QueryEnricher tool (RFC-088 Phase 4).
- **Companion-doc annotation**: prerequisite note pointing to
  RFC-097 §3 (CIL formalization).

### `docs/prd/PRD-017-grounded-insight-layer.md` — Partially implemented

- **Closed by v2**: FR1.3 (Insight → Topic via ABOUT); FR1.4
  (`insight_type` marking); entity v1.1 deferral retired;
  Quote/Insight entity attribution via MENTIONS_PERSON /
  MENTIONS_ORG.
- **Remains open**: RFC-050 Insight Explorer surface; RFC-051 DB
  projection; cross-episode Insight dedup (RFC-072 KL).
- **Companion-doc annotation**: closure note pointing to
  RFC-097 §2 + §3 + GI schema v3.0.

### `docs/prd/PRD-019-knowledge-graph-layer.md` — Partially implemented

- **Closed by v2**: FR3.1 (documented entity node types);
  entity identity fragmentation (`entity:person:` → `person:`);
  Non-Goal "merging KG into gi.json" clarified (we unify the
  ontology, not the file layout).
- **Remains open**: RFC-056 use-case spec; RFC-051 DB projection;
  perfect entity resolution (Non-Goal).
- **Companion-doc annotation**: closure note pointing to
  RFC-097 §2 + KG schema v2.0 + migration script.

### `docs/prd/PRD-026-topic-entity-view.md` — Draft

- **Closed by v2**: Topic-centric querying foundation (stable
  `topic:` IDs); cross-episode topic aggregation (bridge connects
  episodes via CIL).
- **Remains open**: UI/UX surface (PRD-026 scope); contradiction
  detection across persons (analysis layer).
- **Companion-doc annotation**: v2 enablement note (data layer
  ready; UI is separate work).

### `docs/prd/PRD-027-enriched-search.md` — Draft

- **Closed by v2**: chunk-to-Insight lift foundation (RFC-072
  Phase 5 prerequisite); synthesis quality (Insight.insight_type
  + position_hint improve enriched synthesis).
- **Remains open**: QueryEnricher protocol (RFC-088 Phase 4);
  Enriched Search UI; char-offset alignment verification (RFC-072
  KL1, hard blocker for full chunk-to-Insight lift).
- **Companion-doc annotation**: v2 prerequisite note.

### `docs/prd/PRD-028-position-tracker.md` — Draft

- **Closed by v2**: Position Tracker data foundation + viewer
  surface (chunk 8); `position_arc` query pattern (RFC-072 §5A);
  `insight_type` filtering; `position_hint` temporal ordering.
- **Remains open**: position-change detection (analysis layer);
  multi-corpus cross-show view.
- **Companion-doc annotation**: v2 completeness note for data +
  UI.

### `docs/prd/PRD-029-person-profile.md` — Draft

- **Closed by v2**: Person Profile data foundation + viewer
  surface (chunk 8); `person_profile` query pattern (RFC-072 §5B).
- **Remains open**: analytical ranking / curation (analysis
  layer); multi-corpus aggregation.
- **Companion-doc annotation**: v2 completeness note for data +
  UI.

### `docs/prd/PRD-031-search.md` — Draft

- **Closed by v2**: named-entity recall foundation (stable
  `person:` / `org:` IDs); relational context (bridge + RFC-094).
- **Remains open**: full Search UI (FR1–FR6); discriminating eval
  for FAISS removal (RFC-090 Phase 3 gate).
- **Companion-doc annotation**: v2 enablement note for
  entity-aware retrieval.

---

## Appendix B — What v2 explicitly does NOT do (parked for v3)

Pinned so the chunked execution doesn't accidentally do them:

1. Topic semantic deduplication
2. Cross-episode Person identity merging / alias registry
3. `CONTRADICTS` edges between Insights
4. Phase 5 chunk-to-Insight lift (RFC-072 KL1)
5. Natural-language query layer
6. Enrichment workers (RFC-088 territory)
7. MCP tool layer (RFC-095 territory)
8. ML query intent router (RFC-092 territory)
9. KG proximity-based retrieval (RFC-091 stays REJECTED)

v3 wishlist with rationale lives at
`docs/wip/SPEC_KG_GI_ONTOLOGY_V3_WISHLIST_2026-06-20.md` (preserved
through v2 implementation — when v2 ships in production, this
wishlist + 2–4 weeks of observations become the v3 spec input).
