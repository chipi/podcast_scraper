# RFC landscape vs. the Learning Platform — harvest, expand, prune

- **Status**: Analysis (for operator decision; no RFC statuses changed)
- **Date**: 2026-06-23
- **Inputs**: full inventory of `docs/rfc/` (102 RFCs) + deep read of RFC-098–101 and the gap analysis.
- **Purpose**: fold existing ideas into the platform, surface adjacent ideas worth reviving, recommend
  prunes, and list gaps in our own RFC-098–101. **Nothing here changes any RFC's status** — recommendations
  only; the operator decides what to retire.

---

## TL;DR

The four platform RFCs **conflict with nothing** — they're additive on the intelligence layer. The
interesting work is (a) a handful of **shipped-but-under-used** capabilities the platform should plug in
nearly for free, (b) a few **adjacent ideas** worth reviving as new surfaces, (c) one clear **prune**
(RFC-051 DB projection, now contradicted by the no-DB decision), and (d) **gaps in our own RFC-098–101**
worth closing while we're still at requirements stage.

---

## 1. Fold-in — prerequisites to name explicitly (already depended on)

These are shipped/foundational and our RFCs already lean on them; make the dependency explicit so nobody
re-invents them: **RFC-090** (hybrid retrieval — powers ask/recall, the no-LLM answer path), **RFC-094**
(relational query layer — `positions_of` / `who_said` / `cross_show_synthesis` for the Knowledge Panel and
recall), **RFC-072** (canonical identity — load-bearing for the per-user corpus projection), **RFC-097**
(ontology v2 — typed entities, `insight_type`, `position_hint`). Action: cite the specific RFC-094 queries
in RFC-099/101 rather than hand-waving "relational traversal".

## 2. Adjacent-expand — older ideas worth reviving as platform scope

The genuinely interesting harvest. Each is mostly-built and becomes "free" surface once there's a consumer app.

| Idea (RFC) | What it gives the platform | Where it plugs in | Phase |
| --- | --- | --- | --- |
| **Enrichment layer (RFC-088)** — `topic_cooccurrence`, `temporal_velocity`, `grounding_rate`, contradictions | "Trending in your week", related-topic chips, a credibility signal ("80% grounded"), contradiction surfacing for reflection | RFC-099 Knowledge Panel + RFC-101 Consolidation | P3 (signals already deterministic) |
| **Topic clustering (RFC-075)** — semantic topic groups + canonical aliases | Cross-episode topic browse; "explore related topics"; feeds the personalized-ordering interest profile | PRD-037 Discovery + RFC-101 §FR5 | P3 |
| **Digest/recap (RFC-023/068)** — diverse recent episodes + topic bands | A personal **"Your Week"** surface: new library episodes + trending topics you follow + what you captured | RFC-101 (new surface) | P3 |
| **MCP server (RFC-095)** — corpus capabilities as agent tools | **Bring-your-own-agent** north-star: expose the user's *personal* corpus so their own LLM/agent answers "what did I learn about X" — keeps D6 intact (the LLM is the user's, not our server) | RFC-101 north-star / future agent bridge | P4+ |
| **Graph toolkit (RFC-069/076/080)** — zoom/minimap/filters/progressive expand | Optional consumer **knowledge-graph browser** scoped to the user's episodes (visual exploration beyond the inline panel) | RFC-099 optional surface | P2+ |
| **LITM context packs (RFC-093)** — token-budgeted briefing packs | Structured context for the BYO-agent path above | pairs with RFC-095 | P4+ |

**Decision (2026-06-23):** build on top of **RFC-075** (clustering), **RFC-023/068** (digest → "Your Week"),
**RFC-095** (MCP BYO-agent north-star), and **RFC-069** (consumer graph browser, P2+). These are folded into
RFC-099/101 and PRD-037/041 below.

> **RFC-088 (enrichment) is being built by a separate effort in parallel.** The platform does **not**
> re-spec it — it *consumes* enrichment outputs (`grounding_rate`, `temporal_velocity`, `topic_cooccurrence`,
> contradictions). Our RFC-099/101 reference RFC-088 as the producer and **must stay in sync** with whatever
> that effort lands (field names, artifact paths). If RFC-088's contract shifts, update the consumption
> points in RFC-099 §Knowledge Panel and RFC-101 §enrichment-powered surfaces.

## 3. Prune — recommend to operator (status unchanged here)

| RFC | Status | Recommendation | Why |
| --- | --- | --- | --- |
| **RFC-051** Database projection (GIL/KG) | Draft | **Leave as-is — waits for the right moment** (operator directive 2026-06-23) | It is the future-DB thinking and will be revisited when persistence is re-assessed. Not pursued under the platform now, **not re-labeled, not touched.** |
| **RFC-070** Semantic search future backends (Qdrant/pgvector) | Draft | **Park until scale** | Platform uses shipped RFC-090 (LanceDB). Scale-out is post-MVP. |
| **RFC-091** KG proximity signal | Rejected | Leave (already settled) | Confirms answers come from edges (RFC-094), not proximity. |
| **RFC-092** ML query router | Draft | **Optional, non-blocking** | RFC-098 ships with the rules router; ML classifier is a later precision upgrade. |

Out-of-platform-scope housekeeping (mention only): several stale general drafts (RFC-027/038/043/053/054/074)
appear abandoned but are unrelated to the platform — flag for a separate doc-hygiene pass, operator's call.

## 4. Gaps in our own RFC-098–101 (close while at requirements stage)

**Should add now (cheap, prevents rework):**

1. **Scrape-completion notification model** (RFC-098/099) — poll vs. long-poll vs. browser push when a
   queued episode finishes. Undefined today; affects P1 UX.
2. **Highlight grounding contract** (PRD-040 + RFC-098) — `segment_id` + char offsets + `[start_ms,end_ms]`,
   and **what happens on re-transcription** (offsets shift). Capture is worthless if it can't survive a
   re-scrape.
3. **Library-wide search** (RFC-098) — today we have *episode-scoped* search and *corpus-scoped recall* but
   no "search my whole library" in between. Natural; wraps RFC-090 with the user's episode-set filter.
4. **Routing/auth topology** (RFC-098) — how `/api/app/*` is isolated from the operator API (one app + two
   routers vs. separate mounts; where auth middleware sits). Currently implicit.
5. **Rate-limit concrete defaults** (RFC-098) — "minimal" needs numbers (scrapes/hr, concurrent/user,
   429-vs-queue behaviour).

**Acknowledge + scope (don't fully build now):**

6. **Consumer observability** (RFC-099) — the original vision named Sentry/Grafana/UX analytics; RFC-099 says
   "basic analytics" with no event taxonomy. Define at least the event list + error reporting; likely a
   small monitoring note.
7. **Account deletion + data export** (RFC-098) — multi-user implies a delete/export story (even if "delete
   the user's files" + a JSON export). State it, even if minimal.
8. **PWA offline scope** (RFC-099) — app-shell only for v2.7, or cache last-played transcript? Decide.
9. **Backend i18n scope** (RFC-098/099) — UI i18n is specced; are artifacts (insights/entities) English-only
   or is there an `Accept-Language` story? Clarify "English-only for v2.7; translation is a future pipeline
   feature" so it's explicit.
10. **Slug collision algorithm** (RFC-098 G4) — concrete suffix scheme + a quick collision audit on the prod
    corpus.
11. **No-store proxy limits** (RFC-100) — max concurrent pass-through streams, timeout, where it runs.
12. **Resurfacing algorithm + "heard" threshold** (RFC-101) — fixed interval ladder vs. adaptive; what % of
    playback counts as "heard" (default e.g. 30% or any capture).

## 5. Recommended actions (for your pick)

- **A. Fold in now:** enrichment signals (RFC-088) + topic clustering (RFC-075) into RFC-101/PRD-037; add
  the BYO-agent MCP north-star (RFC-095) to RFC-101; name the RFC-094 queries in RFC-099/101.
- **B. Close P0-relevant gaps:** notification model, highlight grounding contract, library-wide search,
  routing/auth topology, rate-limit defaults (gaps 1–5).
- **C. Add a short "Consumer observability" section/mini-note** (gap 6) — it was in the original vision.
- **D. Prune recommendation:** re-label RFC-051 as future-persistence reference; park RFC-070; note RFC-092
  optional. (Operator applies status changes.)
- **E. New surface candidate:** a **"Your Week" personal digest** (RFC-101) — small, high-delight, reuses
  the digest engine.

## References

- `docs/rfc/RFC-098…101`, `docs/wip/player/SERVER-SIDE-GAP-ANALYSIS.md`, `docs/prd/PRD-035…041`
- Harvest sources: RFC-075, RFC-088, RFC-023/068, RFC-095, RFC-069/076/080, RFC-093
- Prune candidates: RFC-051, RFC-070, RFC-091, RFC-092
