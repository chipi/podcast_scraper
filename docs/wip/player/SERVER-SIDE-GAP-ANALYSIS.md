# Server-side gap analysis — delivering the Learning Platform on the core platform

- **Status**: Analysis (feeds the foundation RFC)
- **Date**: 2026-06-23
- **Scope**: What the existing server (`src/podcast_scraper/server/`) already provides vs. what is
  missing to deliver PRD-035–041, grounded in a code inventory (not speculation).
- **Companion**: `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md` (2026-06-07) — confirms the same shape;
  this doc is vision-specific and phase-mapped.

---

## TL;DR

The hard half exists. The pipeline + read API + relational layer + job queue are a **solid, single-corpus,
single-operator skeleton**. Everything the *learning platform* adds is **greenfield and mostly orthogonal**:
identity, a per-user store, a dedicated consumer API, an audio *bridge* (vs. today's local-byte serving),
and a stable slug contract. The "ask about this episode" and "what have I learned about X" features are
served by the **existing hybrid search layer** as extractive grounded retrieval — **no request-time LLM**
(D6), so no online LLM route, no server-side provider credentials, and no CI-LLM concern for these.

Critically, our design choice (one shared corpus + per-user overlay) means we **do not** need the biggest
item in the prior analysis — per-user corpus tenancy. That stays out.

---

## What already exists (reuse, don't rebuild)

| Capability | Where | Notes |
| --- | --- | --- |
| FastAPI app factory + `app.state` | `server/app.py` (`create_app`) | Single corpus bound at startup (`app.state.output_dir`). Fine for us — corpus is shared. |
| Read API: search, explore, relational, CIL | `server/routes/{search,explore,relational,cil}.py` | Hybrid BM25+dense (RFC-090); 8 relational traversals (RFC-094); person/topic timelines. |
| Corpus library/catalog reads | `routes/corpus_library.py` | episodes, feeds, similar, digest, topic-clusters, coverage, top persons. |
| Transcript / segments artifacts | `routes/corpus_text_file.py`, `*.segments.json` | Segment timing + speaker IDs produced by the pipeline. |
| GI / KG / bridge artifacts | `routes/artifacts.py`, `metadata/*.{gi,kg,bridge}.json` | Grounded insights + quotes + entities; read-only. |
| Job queue / scrape | `server/jobs.py` (renamed from `pipeline_jobs.py` in #1101 chunk 1), `pipeline_job_registry.py`, `.viewer/jobs.jsonl` | Enqueue, status, file-locked registry, concurrency cap (`PODCAST_VIEWER_MAX_PIPELINE_JOBS`, default 1), stale reconcile, scheduled sweeps (APScheduler). Flag-gated (`--enable-jobs-api`), **unguarded**. |
| Audio (local) | `routes/corpus_media.py` (`GET /api/corpus/media`) | Streams **local file bytes** from `media/`. This is operator-internal — **not** the consumer bridge. |
| Config split | CLI flags + env + `viewer_operator.yaml` | Operator vs runtime separation already exists. |
| Test infra incl. LLM-free guarantee | `tests/{unit,integration,e2e}/server/` | `importorskip` + mocked `/api/*`; CI never calls a real LLM. |

---

## Gap table (what's missing, by area)

Status: **ABSENT** (confirmed by search) · **PARTIAL** (exists but not in the shape we need) · effort is
rough (S/M/L). Phase = where it lands (P0–P3 from PRD-035).

| # | Gap | Status today | Needed for | Phase | Effort |
| --- | --- | --- | --- | --- | --- |
| G1 | **Identity / auth** — no OAuth, sessions, `Depends(get_current_user)`, user model (confirmed absent) | ABSENT | all per-user features | P0 | L |
| G2 | **Per-user state** — today only immutable/shared file state exists. Per-user overlay (library, playback, queue, highlights, notes, interests) is written as **plain per-user files** using existing helpers (`atomic_write`, `filelock`), exactly like `jobs.jsonl` — **no new abstraction, no interfaces, no persistence-layer work**. A real persistence layer (a database + any abstraction) is a **separate future refactor, out of scope here** (D2) | ABSENT | library, playback, capture, consolidation | P0 | S |
| G3 | **Dedicated consumer API surface** (D5) — separate namespace/mount, auth-gated, slug-addressed, BFF-shaped, independent of operator `/api/corpus/*` | ABSENT | clean evolution; all surfaces | P0 | M |
| G4 | **Stable episode slug contract** — corpus addresses episodes by `episode_id` (UUID/string), not a slug; `segments.json` contract uses `episode_slug`. Need a canonical, stable, URL-safe slug ↔ episode_id mapping | PARTIAL | every `/api/app/episodes/{slug}/*` route | P0 | M |
| G5 | **Audio bridge subsystem** — a dedicated new subsystem (own RFC-E, complete new work), not a metadata field. Must: map episode→origin enclosure, keep URLs fresh (expiry/redirect/tracking-prefix handling), validate playability, hand the client a ready-to-play URL, optionally pass-through-proxy *without storing* only where a host forces it (mixed-content/CORS/signed URLs), and stay fully decoupled from the pipeline-internal `media/` bytes. Today: only local-byte serving (`/api/corpus/media`), no origin concept | ABSENT | Player (Principle 4) | P0/P1 | L |
| G6 | **`segments.json` serving in contract shape** — segments exist as artifacts; a clean `GET …/segments` returning the frozen PRD-036 shape (by slug) is not established | PARTIAL | Player transcript sync | P0 | S–M |
| G7 | **Consumer scrape-on-demand** — job queue exists but is operator-flagged, unguarded, single-concurrency. Need consumer enqueue with global dedup + per-user quota + status, reusing `jobs.py` (renamed from `pipeline_jobs.py` in #1101 chunk 1) | PARTIAL | Discovery/Catalog | P1 | M |
| G8 | **Discovery source (Podcast Index)** — no external catalog integration; needs server-side proxy + key handling + `DiscoverySource` abstraction | ABSENT | Discovery | P1 | M |
| G9 | **Per-user library + status join** — catalog reads exist but must be filtered to the user's subscriptions and joined with job state to compute ready/pending/not-scraped | PARTIAL | Catalog | P1 | M |
| G10 | **Playback position + queue state** | ABSENT (needs G2) | Player resume + queue | P1 | S |
| G11 | **Highlights + notes** — per-user, grounded to offsets/timestamps (grounding data is available from artifacts) | ABSENT (needs G2) | Capture | P2 | M |
| G12 | **~~Server-side LLM route~~ → dropped (D6).** Player "ask" and recall are served by the existing hybrid search layer (RFC-090) + relational traversal (RFC-072) as **extractive grounded retrieval — no request-time LLM**. Net new work is just episode-scoped and corpus-scoped query wiring (rides RFC-A / RFC-D), not an LLM route. Generative layer parked | N/A (parked) | Player ask, recall | P1/P3 | S |
| G13 | **Personal knowledge corpus projection** — per-user projection over `CorpusGraph`/relational layer, scoped to episodes the user heard/captured; the moat | ABSENT | Consolidation | P3 | L |
| G14 | **Spaced resurfacing** — per-user scheduling of highlight re-surfacing (APScheduler pattern exists but is corpus-level, not per-user) | ABSENT | Consolidation | P3 | M |
| G15 | **Interest profile + personalized ordering** | ABSENT | Consolidation → Catalog | P3 | M |
| G16 | **Rate limiting / abuse controls** (D4) — none today; need a config-driven per-user limiter + concurrent-scrape cap | ABSENT | platform-wide | P0/P1 | S–M |
| G17 | **Authz on existing write endpoints** — 5 unguarded operator writes (`PUT /api/feeds`, `PUT /api/operator-config`, `POST /api/jobs` plus cancel/reconcile). The consumer API (G3) is separate and auth-gated from day one, so this is about the *operator* surface in any shared deployment | ABSENT | safe shared hosting | P0 | S–M |

---

## Explicitly NOT needed (de-scoped by design)

- **Per-user corpus tenancy / multi-corpus routing.** One shared corpus serves all users; only the overlay
  is per-user (Principle 3). This removes the single largest item from the 2026-06-07 analysis.
- **Rehosting / transcoding audio for consumers.** Bridge-only (Principle 4); the local `media/` bytes stay
  a pipeline-internal asset.
- **Content translation, orgs/roles/sharing, social.** Out for v2.7.

---

## Cross-cutting risks & constraints

- **No request-time LLM (D6).** Q&A/recall use the existing hybrid search + relational layer (extractive,
  grounded). This removes the need for an online LLM route, server-side provider credentials, and the
  no-LLM-in-CI concern for these features. A future generative layer would reintroduce all three.
- **Latency / caching at consumer scale.** Today `CorpusGraph` and the LanceDB handle are built/opened
  **per request** (fine for one operator). Consumer traffic over a shared corpus may need a cached graph /
  index handle on `app.state`. Flag as a scale risk; measure before optimizing.
- **Slug stability (G4) is contract-critical.** The frozen `segments.json` contract and every consumer
  route key on slug; the mapping must be stable across re-scrapes and collision-safe.
- **Enclosure persistence (G5) — CONFIRMED (2026-06-23).** The origin enclosure URL is already persisted
  per episode at `content.media_url` (+ `media_id`, `media_type`) in `*.metadata.json`. No pipeline change
  for the bridge; only a possible backfill of pre-existing metadata.

---

## How the gaps map to RFCs (proposed)

Fewer RFCs than PRDs — group by technical concern, keep each comprehensible:

- **RFC-A — Platform foundation** (G1, G2, G3, G4, G6, G16, G17): identity, per-user persistence (file-based), consumer API
  surface, slug contract, rate limiting. Includes the episode-/corpus-scoped retrieval wiring for "ask"
  and recall over the existing search layer (no request-time LLM, D6). The keystone.
- **RFC-B — Consumer client architecture**: the new top-level PWA app (D1/D3), transcript-sync engine,
  queue, capture UI, a11y/i18n foundations. Mostly client-side; thin server touchpoints.
- **RFC-C — Audio bridge subsystem** (G5): origin enclosure resolution + freshness + playability +
  optional no-store pass-through. Complete new work; own subsystem. Player depends on it.
- **RFC-D — Personal knowledge corpus** (G13, G14, G15): the per-user projection, retrieval-based recall
  scoping, resurfacing, interest profile. The moat; depends on A.

(The previously-listed online-LLM-serving RFC is dropped per D6 — no request-time LLM. Audio bridge moved
from RFC-E to RFC-C so the set stays A–D.)

Discovery and Catalog (G7–G10) are mostly foundation + a thin integration; they ride RFC-A rather than
needing their own RFC.

---

## References

- `docs/prd/PRD-035-learning-platform.md` (+ child PRDs 036–041)
- `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md`
- `src/podcast_scraper/server/` (app.py, routes/, jobs.py)
- `docs/api/HTTP_API.md`, `docs/guides/SERVER_GUIDE.md`
- `docs/rfc/RFC-090-hybrid-retrieval.md`, `RFC-094-search-powered-surfaces-query-layer.md`,
  `RFC-096-audio-pipeline-separation-and-viewer-media.md`
