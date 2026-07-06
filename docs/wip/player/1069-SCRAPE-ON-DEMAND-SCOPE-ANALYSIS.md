# #1069 scrape-on-demand — scope analysis (keep-P0 / re-scope / close)

- **Status**: Decision needed (operator)
- **Date**: 2026-07-06
- **Issue**: #1069 (the last open P0 on umbrella #911)
- **Context**: LEARNING-PLATFORM-GAP-ANALYSIS-2026-07 gap item 3

---

## TL;DR — recommendation

**Re-scope.** Split #1069 into a small **operator-ingest slice** (build now) and a large
**consumer-discovery slice** (defer). The moat — a corpus that grows over weeks/months — is
served by the small slice using primitives that already exist. The large slice (Podcast Index
search + add-to-library + public scrape UI + abuse guardrails) is premature while the product is
operator-curated and pre-user-scale.

Do **not** keep it as a single monolithic P0, and do **not** close it outright — the ingest
primitive is genuinely useful and cheap.

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

## Options

| Option | What | Cost | When it's right |
| --- | --- | --- | --- |
| **A. Re-scope (recommended)** | Ship item 1 (operator/admin-gated `POST /api/app/scrape` over the existing single-feed pipeline + jobs API). Defer 2–4. | Small | Corpus growth is operator-curated today; you want to add shows without a full-corpus rerun, but there are no external users to self-serve. |
| **B. Close / fold into scheduled ingestion** | Close #1069; grow the corpus via operator-scheduled feed runs (cron/CI over a curated feed list). Reopen on real user demand. | ~0 | If even item 1 isn't needed — a scheduled feed list covers corpus growth and nobody needs interactive scrape. |
| **C. Keep full P0** | Build 1–4 including Podcast Index + consumer discovery UI + guardrails. | Large | Only if "users add their own shows" is launch-blocking. |

## Recommendation & reasoning

**A (re-scope), with B as the fallback if item 1 isn't even wanted yet.**

- Discovery-over-local already shipped, so #1069 no longer blocks the *discovery* experience —
  only *corpus growth*.
- The moat (a corpus that grows) is served by item 1, which is cheap: the single-feed pipeline
  (#807), the `ContentSource` seam, and the jobs API already exist. It's glue, not a new subsystem.
- Items 2–4 (Podcast Index integration, consumer scrape UI, abuse guardrails) are a large surface
  whose value depends on **external users self-serving shows** — which the product doesn't have
  yet (per-user state is still plain files; the gap analysis flags persistence as a pre-scale
  cliff). Building them now is speculative.
- Reconcile PRD-037 (mark FR1/FR3/FR4 as the deferred consumer slice) and update the PLATFORM_API
  "blocked on a pipeline enhancement" note (the primitive exists).

## Decision needed

Pick A / B / C. If **A**: I can scope the thin `POST /api/app/scrape` slice (endpoint → single-feed
job → deduped merge, admin-gated) as its own issue under #911. If **B**: close #1069 with a note and
capture the scheduled-ingestion approach. If **C**: this becomes a multi-PR epic (DiscoverySource +
UI + guardrails) and needs its own plan.
