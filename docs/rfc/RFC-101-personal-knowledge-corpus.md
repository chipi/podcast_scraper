# RFC-101: Personal Knowledge Corpus

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Server API, Search/Retrieval, Consumer App
- **Related PRDs**:
  - `docs/prd/PRD-035-learning-platform.md` (Principle 2 — the consolidation moat)
  - `docs/prd/PRD-041-consolidation.md`
  - `docs/prd/PRD-040-capture.md` (raw material)
- **Related RFCs**:
  - `docs/rfc/RFC-098-learning-platform-foundation.md` (identity + per-user store + retrieval wiring)
  - `docs/rfc/RFC-090-hybrid-retrieval.md` (retrieval engine, scoped here)
  - `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md` (relational traversal, scoped here)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` (canonical identity)
  - `docs/rfc/RFC-075-corpus-topic-clustering.md` (clustered topic threads + interest profile)
  - `docs/rfc/RFC-088-enrichment-layer-architecture.md` (enrichment signals — **built in parallel; consume + stay in sync**)
  - `docs/rfc/RFC-095-generic-mcp-server.md` (bring-your-own-agent north-star over the personal corpus)
  - RFC-068 / RFC-023 (digest engine → "Your Week"); RFC-069 (optional consumer graph browser, via RFC-099)

## Abstract

This RFC defines the platform's differentiator: a **personal knowledge corpus** — a per-user *projection*
over the existing GIL/KG ontology, scoped to the episodes a user has actually heard or captured from. It
answers "what have I learned about X / from this guest" by **retrieval, not generation** (PRD-035 D6),
draws cross-episode connections, and resurfaces past highlights for reflection — so a user's knowledge
compounds over time. It adds a per-user layer over infrastructure we already have, not new ML.

**Architecture Alignment:** Reuses RFC-090 retrieval and RFC-094 relational traversal, scoped by the user's
episode set (from playback history + captures). No new index type, no request-time LLM.

## Problem Statement

Listening alone doesn't make knowledge stick (PRD-035 thesis). Capture (PRD-040) records what mattered;
consolidation must turn those captures + listening history into a durable, connected, grounded corpus the
user can interrogate and revisit. Nothing per-user exists today (gap G13–G15). The challenge is to do this
**without** a per-user index rebuild or an LLM — by projecting the shared ontology through the lens of one
user's experience.

**Use Cases:**

1. **Recall**: "what have I learned about transformers" → grounded items drawn only from episodes I've heard.
2. **Connect**: everything `<guest>` has said across the episodes I've heard, assembled.
3. **Reflect**: periodic resurfacing of my past highlights with a prompt and jump-to-moment.

## Goals

1. **Per-user corpus projection**: a view over highlights, saved insights, notes, and the entities/topics of
   the user's heard/captured episodes, grounded via RFC-072.
2. **Grounded recall (no LLM)**: retrieval (RFC-090) + relational traversal (RFC-094) scoped to the user's
   episode set, returning an assembled grounded set with jump-to-moment.
3. **Cross-episode connections** within the user's set (person/topic threads).
4. **Spaced resurfacing** of past highlights with pacing controls.
5. **Interest profile** to (opt-in) personalise Catalog ordering.

## Constraints & Assumptions

**Constraints:**

- **No request-time LLM** (D6): recall is extractive; results are verbatim grounded items.
- Scope is strictly the user's **heard or captured** episodes — recall must not draw from the global corpus.
- Reuse the existing retrieval + relational layers; no new index type for v2.7.

**Assumptions:**

- Playback history + captures (RFC-098 store, PRD-040) define the user's episode set.
- The shared hybrid index (RFC-090) supports an episode-set filter (verified in the gap analysis).

## Design & Implementation

### 1. The user's episode set

A derived set = episodes the user has **heard** (**≥30% played**, default/tunable) ∪ episodes the user has
**captured** from (any highlight/note). Maintained by reading the user's per-user files (RFC-098); no
duplication of artifacts.

### 2. Corpus projection (read-time, not a rebuilt graph)

The personal corpus is assembled at read time: take the user's highlights/saved insights/notes (per-user store) plus the
insights/entities of the episode set (shared artifacts via the relational layer, RFC-094), unified by
canonical identity (RFC-072) so a person/topic is one node across episodes. No per-user graph is persisted in
v2.7 — it's a filtered view. (If read-time cost is too high, a per-user materialised summary is a later
optimisation — see Open Questions.)

### 3. Grounded recall (retrieval, no LLM)

```text
POST /api/app/corpus/recall { "q": "transformers" }
-> { "groups": [ { "by": "episode|guest", "items": [ grounded insight|quote|highlight + jump ] } ],
     "coverage": "n episodes" }
```

- Runs hybrid search (RFC-090) **filtered to the user's episode set** + relational traversal for the queried
  entity/topic; merges with the user's own highlights; groups and ranks. The "answer" **is** the assembled
  grounded set — no prose generation.
- **Zero-coverage honesty**: if the user's set has nothing on `q`, say so ("nothing in your corpus yet") —
  never fall back to the global corpus.

### 4. Cross-episode connections

- `GET /api/app/corpus/person/{id}` and `/corpus/topic/{id}`: the existing RFC-094 traversals (positions_of /
  who_said / topic threads) **scoped to the user's set**. Surface as "you also heard `<guest>` discuss this
  in …".

### 5. Spaced resurfacing

- Computed **on read**, not via a background job: `GET /api/app/resurfacing` selects highlights whose
  `created_at` + interval ladder (e.g. 2d/1w/1mo/3mo) is due and `last_surfaced` is older than the step.
  Returns items + a reflection prompt + jump-to-moment. Pacing controls (frequency, pause, dismiss) are
  per-user settings. No scheduler needed for v2.7.

### 6. Interest profile

- Aggregate topic/person frequencies from captures + history into `interest_topics`. Opt-in personalised
  ordering of Catalog/Home re-ranks shared-corpus results by personal relevance (off by default until
  validated).

### FR6: Harvested surfaces (build-on, 2026-06-23)

- **FR6.1 Enrichment-powered reflection (consumes RFC-088 — built in parallel, stay in sync):**
  `temporal_velocity` → "trending in your corpus"; contradictions → "you've heard opposing takes on X";
  `grounding_rate` → flag/filter low-confidence items. The platform consumes RFC-088 outputs; it does not
  re-implement them. If RFC-088's field/artifact contract shifts, update these consumption points.
- **FR6.2 "Your Week" personal digest:** reuse the digest engine (RFC-068 / RFC-023) scoped to the user's
  library + interests — new episodes in followed shows, topics trending in *their* corpus, and a recap of
  what they captured. Computed on read; no new pipeline.
- **FR6.3 Clustered topic threads (RFC-075):** cross-episode connections and the interest profile use topic
  clusters so variants collapse to one canonical thread.
- **FR6.4 Bring-your-own-agent north-star (RFC-095):** expose the user's *personal* corpus through the MCP
  server so the user's **own** agent/LLM performs generative synthesis ("ask Claude about your learning").
  This is the D6-safe path to generative answers — the model is the user's, never our server. North-star,
  not v2.7.

## Key Decisions

1. **Projection, not a rebuilt per-user graph**
   - **Decision**: assemble the personal corpus at read time by filtering the shared ontology to the user's
     episode set.
   - **Rationale**: reuses RFC-090/094/072; no new index; respects shared-artifacts / personal-overlay.
2. **Recall is retrieval, not generation**
   - **Decision**: return a grounded set, no LLM.
   - **Rationale**: PRD-035 D6 — simpler, cheaper, nothing to hallucinate, no CI-LLM concern.
3. **Resurfacing computed on read**
   - **Decision**: due-item selection at request time, not a background scheduler.
   - **Rationale**: minimal infra for v2.7; deterministic and testable.

## Alternatives Considered

1. **Per-user materialised knowledge graph** — Deferred: heavier to build/maintain; the read-time projection
   is enough until proven otherwise (optimisation path noted).
2. **LLM-synthesised recall answers** — Rejected for v2.7 (D6); parked as a future layer on top of retrieval.
3. **Global-corpus recall (ignore the user's set)** — Rejected: violates the "your own experience" scope and
   the trust story.

## Testing Strategy

**Test Coverage:**

- **Unit**: episode-set derivation; due-item selection ladder; grouping/ranking; coverage/zero-coverage.
- **Integration**: multi-user fixtures (seeded history + highlights); recall returns only the user's set;
  user A's corpus excludes user B's captures; cross-episode person/topic scoping; **no LLM** invoked.
- **E2E**: recall + resurfacing surfaces in the consumer app (RFC-099) with working jump-to-moment.

**Test Organization:** `tests/integration/app_api/test_personal_corpus.py`; deterministic fixtures; assert
no LLM/network calls.

## Rollout & Monitoring

- **P3** (after P0–P2). Ships recall + connections + resurfacing; interest-driven ordering opt-in.
- **Monitoring**: recall coverage distribution, zero-coverage rate, resurfacing engagement, read-time
  latency of the projection.
- **Success**: recall cites only the user's heard/captured episodes with working jumps; cross-episode
  threads assemble correctly via canonical identity; resurfacing respects pacing.

## Open Questions

1. Read-time projection latency at large personal corpora — when to materialise a per-user summary.
2. ~~Episode-set threshold~~ **Resolved (2026-06-23):** "heard" = ≥30% played or any capture (default, tunable).
3. Whether interest-driven ordering can be validated without a multi-user feedback loop yet.

## References

- **Related PRDs**: `docs/prd/PRD-041-consolidation.md`, `docs/prd/PRD-040-capture.md`
- **Related RFCs**: `docs/rfc/RFC-090-hybrid-retrieval.md`, `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md`, `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
- **Analysis**: `docs/wip/player/SERVER-SIDE-GAP-ANALYSIS.md` (G13–G15)
