# Remember half — scope sketch (Capture + Consolidation)

- **Status**: WIP planning sketch (not yet a PRD/RFC). For `feat/consumer-capture`.
- **Covers**: PRD-040 Capture (P2) + PRD-041 Consolidation (P3) / RFC-101.
- **Hard dependency**: the **Enrichment Layer (RFC-088 / ADR-104)** — currently in the
  `podcast_scraper-FUTURE` worktree, landing on `main` soon. **P3 is designed to consume it; we
  wait for it before building P3.** P2 (Capture) has no enrichment dependency and can start first.

## Thesis

Listening is the input; a growing, **grounded, personal knowledge corpus** is the output. P2
**captures** the raw material; P3 **consolidates** it into recall + connections + resurfacing. Both
are **per-user projections over the same GIL/KG ontology + the new enrichment layer** the pipeline
already produces — we reuse, we don't rebuild.

## Settled decisions (this sketch)

1. **Persistence** — per-user **files** for P2 (mirrors favorites/queue/interests/listen-log). P3
   recall is a **scoped query** over the existing shared index (a `user_heard_set` filter), not a
   second store.
2. **API convention** — all new routes under `/api/app/*` (not PRD-040's older `/api/user`,
   `/api/episodes`).
3. **Resurfacing channel** — **one** channel: a digest, delivered as an **in-app push/inbox**
   (no email/SMS infra in v1).
4. **Export** — **Markdown only** for now (the single export option).
5. **Test pyramid** — same shape we just hardened: unit (store/recall logic) → integration
   (`/api/app/*` route tests) → e2e on the committed `tests/fixtures/app-validation-corpus` (extend
   it with a heard/captured + enrichment fixture).

## P2 — Capture (PRD-040) — the prerequisite (no enrichment dependency)

One-tap "mark this moment" during playback; transcript span-select; "save this insight"; plain-text
notes; review per-episode + a global "My highlights"; jump back within 0.5s.

- **Data (per-user overlay rows / files):**
  - `highlight {id, episode_slug, kind(span|moment|insight), start_ms/end_ms, char_start/end,
    segment_ids[], quote_text, speaker?, source_insight_id?, color?, created_at}` — **timestamp is
    the stable anchor; re-anchors on re-scrape** (a highlight is never silently dropped).
  - `note {id, target(highlight|insight|episode), target_id, text, created/updated_at}`.
- **API (`/api/app/*`):** per-episode + global highlights (filters: podcast/topic/color), highlight
  create/edit/delete, notes CRUD. Markdown export endpoint.
- **UX surfaces:** capture control in the player hero; transcript span-select; the insight card's
  "save to highlights"; a Library **Highlights** view (per-episode + global); inline notes.
- **Sub-epics:** (a) capture store + routes + MD export; (b) player/transcript/insight capture UX;
  (c) Library highlights/notes review.

## P3 — Consolidation (PRD-041 / RFC-101) — rebased on the Enrichment Layer

Turn captures + listening history into a **per-user knowledge graph projection**, answer **grounded
recall**, surface **cross-episode connections**, and **resurface** highlights on a spaced schedule.
Scoped strictly to episodes the user has **heard or captured** (recall cites their own experience,
never the global corpus). No request-time LLM (D6) — extractive/verbatim retrieval.

### What the Enrichment Layer (RFC-088) gives us — and how P3 consumes it

RFC-088 adds a 4th artifact tier: per-episode + corpus **envelopes** under `enrichments/`, with these
enricher signals (consumed read-only; ADR-104 boundary — we never recompute):

| Enricher signal | P3 use |
| --- | --- |
| `topic_cooccurrence` | Cross-episode **topic threads** + "related topics in your corpus" |
| `topic_similarity` | "Similar threads you've heard"; recall expansion |
| `temporal_velocity` | "This topic is trending across what you've heard" resurfacing cue |
| `nli_contradiction` | "You've heard **opposing** views on X" — the standout recall signal |
| person/topic landing + `RELATED_TO` edges | Person/topic **connections** (reuse our entity-card machinery, scoped per-user) |

**New consumer read surface required (a P3 dependency/gap):** the shipped enrichment HTTP routes
(`/api/enrichment/*`) are **operator/ops-facing** (run/status/health/metrics). P3 needs a
**consumer projection** over the enrichment **envelope data** per episode — e.g.
`GET /api/app/episodes/{slug}/enrichment` (+ a corpus-scope read) — scoped to the user's heard set.

### P3 functional shape

- **Personal corpus (FR1):** per-user projection over GIL/KG **+ enrichment envelopes**, nodes =
  highlights / saved insights / notes / heard episodes; every node keeps grounding (slug + ts +
  quote).
- **Grounded recall (FR2):** hybrid retrieval (RFC-090) + relational (RFC-072) over the user's set,
  **enriched** with co-occurrence / similarity / velocity / contradiction. Zero-coverage → honest
  "nothing in your corpus yet."
- **Connections (FR3):** person view (a guest across the user's heard episodes) + topic view +
  "you also heard <guest> discuss this in …" — powered by enrichment edges/co-occurrence.
- **Spaced resurfacing (FR4):** in-app digest/inbox, reflection prompt + one-tap re-listen; pacing
  controls (frequency/pause/dismiss); velocity signal informs what to resurface.
- **Interest profile (FR5):** derived from captures + history, cross-referenced with enrichment
  topic signals — extends the interest-token model we shipped (cluster/`topic:`/`person:`).
- **Sub-epics:** (a) consumer enrichment read surface; (b) per-user corpus projection + scoping;
  (c) grounded recall over hybrid/relational + enrichment; (d) connections UI; (e) spaced
  resurfacing + reflection inbox; (f) interest-profile evolution.

## Sequencing

```text
[Enrichment Layer RFC-088 lands on main]  ← we wait for this for P3
            │
P2 Capture ─┼─────────────► P3 Consolidation (recall · connections · resurfacing)
(can start now)             (consumes enrichment envelopes, scoped per-user)
```

- **P2 Capture** can start immediately (no enrichment dependency).
- **P3 Consolidation** begins once the enrichment layer is on `main`; captures (P2) are its raw
  material, and enrichment envelopes are its connection/recall substrate.
- **Optional follow-ons:** consumer KG browser (RFC-099 §9, P2+); per-app-activity e2e + a consumer
  `stack-test` (deferred this session); MCP exposure of the personal corpus (RFC-095, north-star).

## Open questions

- Consumer enrichment read surface — fold into RFC-101, or its own thin RFC?
- Resurfacing cadence defaults + the spaced schedule (simple intervals in v1; full SRS later).
- Highlight tag taxonomy — fixed colours/labels only (v1) vs free tags (later).
- Whether a thin "recall" slice belongs in P2, or P3 stays the clean home for all recall.
