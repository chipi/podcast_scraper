# #1069 scrape-on-demand — scope analysis (keep-P0 / re-scope / close)

- **Status**: **Decided 2026-07-06 — phased (curated first, user self-serve second)**
- **Date**: 2026-07-06
- **Issue**: #1069 (the last open P0 on umbrella #911)
- **Context**: LEARNING-PLATFORM-GAP-ANALYSIS-2026-07 gap item 3

---

## TL;DR — decision

The product is **curated corpus first, user-bring-your-own-shows second** — both, phased (operator
confirmed 2026-07-06). Both phases need the **same core primitive**: *ingest one specific
feed/episode into the corpus, as a tracked job, idempotent + globally deduped.* So the decision is
neither "close" nor "build the whole consumer surface" — it's:

1. **Build now — the ingestion primitive** (the durable write-path spine). Grows the curated
   corpus today; is the load-bearing foundation the self-serve layer sits on later. Includes an
   **authorization/policy seam** (operator path no-ops it; the future user path enforces
   guardrails through it) so adding guardrails later is a policy, not a rewrite.
2. **Build later — the consumer self-serve surface** (its own epic): the Podcast Index
   `DiscoverySource`, add-to-library + scrape-progress UI, and the guardrail *implementations*
   (rate limits, per-user quotas, cost bounds, abuse prevention). Gated on the deps it actually
   needs — real persistence (currently plain per-user files), the PWA (RFC-099), and real users.

Split #1069 accordingly: the primitive is a build-now issue; the consumer surface is a later epic.
This is **not** the throwaway "thin admin endpoint" stopgap that was floated and rejected — the
primitive is built as the permanent foundation both paths depend on, with the right seams.

---

## What #1069 is (PRD-037 + PLATFORM_API)

The discovery entry point for *adding external content*: search an open catalog (Podcast Index,
~4M shows), add a show to a per-user library, and **request a scrape** of a specific episode/feed
(`POST /api/app/scrape`), globally deduped. Sources are pluggable via a `DiscoverySource`
abstraction (Podcast Index first, then OPML / manual RSS).

## What already shipped WITHOUT #1069

Discovery *over the local corpus* is done: entity search (#1097), personalized ranking (#1098,
now with the #1139 eval), Home / search (#1090), recommendations. Users can find, rank, and open
anything **already in the corpus**. #1069's unique addition is bringing in shows the corpus
**doesn't have yet**.

## What's already built / scaffolded (so #1069 is smaller than it looks)

- **Single-feed pipeline** — the pipeline already targets one feed/episode
  (`resolve_cli_feed_targets`, `single_feeds_spec_output_dir`, `--feeds-spec` / `--rss`; #807
  single-feed run). The PLATFORM_API note "the pipeline currently runs the whole corpus" is
  **stale** — feed-targeted runs exist.
- **`ContentSource` seam** — `app_content_source.py` defines the `ContentSource` Protocol;
  `LocalCorpusSource` is the MVP; `get_content_source()` already has the `DiscoverySource` hook.
  The API contract needs **no reshape** to add discovery.
- **Jobs API** — `server/jobs.py` + `routes/jobs.py` + `routes/scheduled_jobs.py` give async job
  execution + progress, which scrape-on-demand progress can hang on directly.

## What's actually still missing

1. **Scrape orchestration endpoint** — `POST /api/app/scrape` → kick a single-feed pipeline run
   (existing primitive) as a job (existing infra) → merge the result into the corpus, deduped.
2. **`DiscoverySource` impl** — Podcast Index client (search + resolve RSS), the net-new external
   integration.
3. **Consumer UI** — search → add-to-library → per-episode Ready/Requestable state → scrape
   progress. A real surface (PRD-037 FR1–FR4, PRD-038 catalog states).
4. **Guardrails** — rate limits, per-user quotas, abuse prevention, cost bounds on unbounded
   user-triggered scraping.

Items 1 is small (glue over existing primitives). Items 2–4 are the large, genuinely-new surface.

## Options considered (for the record)

| Option | What | Why not chosen |
| --- | --- | --- |
| Thin admin `POST /api/app/scrape` stopgap | A throwaway operator-only endpoint, guardrails "later". | Rejected — a half-built seam that half-commits and rots; adding guardrails later means rewriting the ingest path. |
| Close / fold into scheduled ingestion only | Grow the corpus purely via a scheduled curated feed list; drop self-serve. | Rejected — the product *will* let users bring their own shows; closing self-serve outright contradicts the vision. |
| Keep full monolithic P0 | Build the whole thing now (ingest + Podcast Index + UI + guardrails). | Rejected — the consumer surface depends on persistence + PWA + real users the product doesn't have yet; building it now is speculative. |
| **Phased around the shared primitive (chosen)** | Build the durable ingestion primitive now (with the guardrail seam); defer the consumer surface to a deps-gated epic. | Both phases need the same write-path spine; build it once, correctly, and layer discovery on top when the product is ready. |

## Decision & reasoning (2026-07-06)

**Phased around the shared ingestion primitive.**

- Curated-now and self-serve-later need the *same* primitive — ingest one feed/episode, as a job,
  idempotent + deduped. Build that spine once, as a permanent foundation, not a stopgap.
- Enabling primitives already exist and this composes them: single-feed pipeline runs (#807), the
  `ContentSource` / `DiscoverySource` seam (`app_content_source`), and the jobs API for progress.
- The guardrail/policy **seam** goes in now (an authorization hook); the guardrail *implementations*
  (rate/quota/cost/abuse) land with the self-serve epic — so that later work slots in, no rewrite.
- The consumer surface (Podcast Index `DiscoverySource`, add-to-library + progress UI, guardrail
  impls) is a separate later epic, gated on real persistence, the PWA (RFC-099), and real users.

## Build plan

**Now (this work):** the ingestion primitive.

- `ingest(feed_url | episode)` → runs the single-feed pipeline as a **job** (jobs API) → merges the
  resulting artifacts into the corpus, **globally deduped** (an already-ingested feed/episode is a
  no-op, per PRD-037 "globally deduped" + PRD-035 shared-artifacts principle).
- An **authorization/policy seam** on the trigger — operator/admin path no-ops; the shape is ready
  for per-user quotas/rate limits later.
- Operator entry point (CLI and/or admin-gated endpoint) — *not* a public consumer route yet.

**Later (separate epic, deps-gated):** the consumer self-serve surface — Podcast Index
`DiscoverySource`, add-to-library + scrape-progress UI, and the guardrail implementations.

## Housekeeping

- Split #1069: the ingestion primitive = build-now; the consumer-discovery surface = later epic.
  (Operator opens the issues.)
- Reconcile PRD-037: it is **curated first, self-serve second — phased**, not "self-serve in v2.7".
- Update the PLATFORM_API "blocked on a pipeline enhancement" note — the single-feed primitive
  (#807) exists; what remains is the ingest orchestration + dedup, built here.
