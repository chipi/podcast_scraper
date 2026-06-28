# RFC-098: Learning Platform Foundation

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Server API, Core Pipeline, Consumer App
- **Related PRDs**:
  - `docs/prd/PRD-035-learning-platform.md` (parent)
  - `docs/prd/PRD-036-foundation-identity.md`
  - `docs/prd/PRD-037-discovery.md`, `docs/prd/PRD-038-catalog.md`
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` (search the consumer API reuses)
  - `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md` (relational layer reused)
  - `docs/rfc/RFC-099-learning-platform-consumer-client.md` (client of this API)
  - `docs/rfc/RFC-100-audio-bridge-subsystem.md` (audio resolution)
  - `docs/rfc/RFC-101-personal-knowledge-corpus.md` (per-user projection)
- **Related Documents**:
  - `docs/wip/player/SERVER-SIDE-GAP-ANALYSIS.md` (gaps G1–G4, G6, G16, G17)

## Abstract

This RFC defines the server-side foundation for the multi-user Learning Platform: a single-provider
OAuth identity, per-user state stored as plain files (no new persistence layer), a **dedicated consumer API surface** (`/api/app/*`)
independent of the operator/viewer API, a stable episode **slug** contract, and the per-episode/per-corpus
**retrieval wiring** that powers "ask" and recall over the existing hybrid search layer — with **no
request-time LLM** (PRD-035 D6).

**Architecture Alignment:** Preserves the shared-corpus / per-user-overlay boundary (PRD-035 D3): episode
artifacts stay file-based and shared; per-user overlay state is persisted in its own per-user files. The consumer API mounts as a separate
sub-application so the operator routes (`/api/corpus/*`, `/api/relational/*`) and the viewer remain
untouched (RFC-062/RFC-077).

## Problem Statement

The current server (`src/podcast_scraper/server/`) is single-operator, single-corpus, with **no auth and no
per-user persistence** (gap analysis G1/G2/G5). It addresses episodes by opaque `episode_id`,
not a stable slug (G4). To deliver an end-user platform we need identity, per-user persistence, a
consumer-shaped API, and a slug contract — without forking the corpus per user and without overloading the
operator API.

**Use Cases:**

1. **Sign in**: a user authenticates with Google and gets a stable identity + session across devices.
2. **Personal state**: their library, playback, queue, highlights, and notes persist and are private.
3. **Consume intelligence**: the client reads episodes, segments, insights, and entities by slug, and runs
   episode-scoped grounded search — all over one decoupled API.

## Goals

1. **Identity**: single-provider OAuth (Google to start), session-gated `/api/app/*`.
2. **Per-user state as plain files**: write per-user overlay state as per-user files using existing helpers;
   **no new persistence layer, abstraction, or interfaces** (a real persistence layer is a deferred, separate
   effort — see Key Decisions).
3. **Consumer API surface**: `/api/app/*`, auth-gated, slug-addressed, BFF-shaped, versioned independently.
4. **Slug contract**: deterministic, stable, collision-safe episode slug ↔ `episode_id` mapping.
5. **Retrieval wiring**: episode-scoped grounded search via RFC-090 (no LLM); corpus-scoped recall hand-off
   to RFC-101.
6. **Guardrails**: minimal per-user rate limit + concurrent-scrape cap (PRD-035 D4).

## Constraints & Assumptions

**Constraints:**

- No request-time LLM and no LLM provider credentials on the server (PRD-035 D6).
- Episode artifacts remain read-only and shared; only the per-user files are per-user.
- CI must never hit a real OAuth provider, real LLM, or real network (project rule); all are stubbed.
- The operator API and viewer must keep working unchanged.

**Assumptions:**

- One shared corpus serves all users (no multi-corpus tenancy).
- The existing hybrid index (RFC-090) supports feed/episode filtering — verified in the gap analysis.

## Design & Implementation

### 1. Consumer API as a mounted sub-application

A new package `src/podcast_scraper/server/app_api/` exposes an `APIRouter` (or sub-`FastAPI`) mounted at
`/api/app`. Every route depends on `get_current_user`. It reads shared artifacts through the existing
services (`corpus_library`, `search`, `relational`) and per-user state through the new store, returning
**consumer DTOs** (slug-addressed) — never the raw operator models.

```text
server/
  app_api/
    __init__.py        # router factory, mounted in app.py behind --enable-app-api
    auth.py            # OAuth login/callback/logout, session, get_current_user
    deps.py            # current user + rate-limit dependencies
    userstate.py       # read/write per-user files (reuses atomic_write + filelock, like jobs.jsonl)
    episodes.py        # GET /episodes, /{slug}, /{slug}/segments|insights|entities|search
    library.py         # library subscriptions, catalog status join
    playback.py        # playback position + queue
    scrape.py          # scrape-on-demand (wraps jobs) + per-user quota
    slugs.py           # slug <-> episode_id resolution
    dto.py             # pydantic consumer response models
```

### 2. Identity & sessions

- OAuth via **Authlib**; one provider configured by env (`APP_OAUTH_*`). `provider` is swappable.
- On callback, upsert `users` + `oauth_identities`, set a **signed, httponly session cookie** (itsdangerous
  / Starlette `SessionMiddleware`). No JWT, no token store — simplest that works.
- `get_current_user` resolves the session → `User`; missing/invalid → 401. Anonymous read of catalog/search
  is a config flag (default off).

### 3. Per-user state as plain files (no new persistence layer)

- Per-user overlay state is written as **plain per-user files** under a per-user directory
  (`<data_dir>/users/<user_id>/…`), reusing the helpers the codebase already uses for `jobs.jsonl`:
  `filelock` for concurrent writes and `atomic_write` for whole-file writes. No new abstraction, **no
  repository/interface layer, no schema, no migration tooling.**
- File types: `profile`, `oauth_identity`, `library` (subscriptions), `playback` (per-episode position),
  `queue`, `highlights`, `notes`, `interests`, plus a shared `episode_slugs` index. (Highlights/notes
  detailed in PRD-040; recall projections in RFC-101.)
- Each file is owned by a `user_id`; the route dependencies (`deps.py`) confine reads/writes to the current
  user's directory — per-user isolation by path.
- Lives **outside** the corpus artifact tree so the boundary stays physical: corpus artifacts = shared
  files; per-user state = per-user files.
- **Explicitly out of scope:** a real persistence layer — a database and any abstraction/interfaces around
  it — is a separate, potentially large refactor, assessed only *after* these requirements are locked. This
  phase does **no** persistence-layer work; it writes files the way the project already does.

### 4. Slug contract (G4)

- A canonical slug is derived deterministically from **stable** inputs — the feed slug + the episode's RSS
  GUID (falling back to publish-date + title when no GUID) — and stored in `episode_slugs(slug PK,
  episode_id, feed_id, created_at)`. Keying on the immutable GUID makes the slug **stable across
  re-scrapes** and collision-safe (suffix on collision).
- The mapping is populated by a small pipeline hook at artifact-write time and backfilled for the existing
  corpus by a one-shot command. All `/api/app/episodes/{slug}/*` routes resolve slug→episode_id via this
  table.

### 5. Episode-scoped grounded search (no LLM)

- `GET /api/app/episodes/{slug}/search?q=...` calls the existing hybrid search (RFC-090) with an
  episode/feed filter, returning ranked **grounded passages** (segment/insight hits) with timestamps for
  jump-to-moment. This is the entirety of the player's "ask" feature — extractive, no generation (D6).
- Corpus-scoped recall ("what have I learned about X") is the same engine scoped to the user's episode set;
  its projection/scoping lives in RFC-101.

### 6. Scrape-on-demand + guardrails

- `POST /api/app/scrape` wraps the existing `jobs` enqueue (already globally deduped) and adds a
  per-user **concurrent-scrape cap** + **token-bucket rate limit** (in-process, config-driven). `GET
  /api/app/scrape/{job_id}` proxies RFC-065 status. No new queue.

### 7. P0 contract details (added per landscape review)

- **Scrape-completion notification (gap 1):** v2.7 uses **client polling** of `GET /api/app/scrape/{job_id}`;
  the catalog/queue flips to Ready on terminal status without a manual refresh. SSE / browser push is a
  deferred enhancement (Open Questions) — no new server-push infrastructure now.
- **Library-wide search (gap 3):** `GET /api/app/search?q=…` runs hybrid retrieval (RFC-090) filtered to the
  **user's episode set** (library ∪ heard) — the tier between episode-scoped search (§5) and personal recall
  (RFC-101). Same grounded-passage shape; no LLM.
- **Routing & auth topology (gap 4):** `/api/app/*` mounts as its own `APIRouter`/sub-app with the auth
  dependency applied at the prefix; operator routers (`/api/corpus/*`, `/api/relational/*`, RFC-077 ops) keep
  their existing chain unchanged. One process, two prefixes, two middleware chains. `/api/corpus/media` is
  **never** mounted under `/api/app`.
- **Rate-limit defaults (gap 5):** initial, config-tunable — scrape **≤2 concurrent/user, ≤20/day**; read API
  **token bucket ≈60 req/min/user**; on limit → **HTTP 429 + `Retry-After`** (reject, not queue). Global
  scrape dedup still applies underneath.
- **Highlight grounding & re-transcription (gap 2):** highlights store
  `episode_slug + segment_id + char_start/char_end + [start_ms,end_ms] + quote_text` (PRD-040 FR3.1).
  **Timestamps are the stable anchor** across re-transcription; `segment_id`/offsets may shift on re-scrape,
  so a highlight re-anchors by timestamp on read (re-locating the nearest segment), keeping `quote_text` for
  display + drift verification. A re-scrape never silently drops a highlight.

## Key Decisions

1. **Separate `/api/app/*` sub-app, not new routes on the operator API**
   - **Decision**: mount an independent, auth-gated consumer API.
   - **Rationale**: PRD-035 D5 — decoupled evolution; operator/viewer untouched; consumer-shaped contracts.
2. **Per-user state as plain files; no persistence-layer work now**
   - **Decision**: write per-user state as per-user files (JSON/JSONL + `filelock`/`atomic_write`), exactly
     as the project already does for `jobs.jsonl`. **No** database, abstraction, repository, or interface.
   - **Rationale**: PRD-035 D2 — this is requirements-gathering; building a persistence layer (and the
     abstraction a future database would need) is a separate, larger refactor, deliberately not started here.
3. **Signed session cookie, no JWT/token store**
   - **Decision**: server-side session via signed cookie.
   - **Rationale**: minimal moving parts for a single-provider login.
4. **Slug keyed on episode GUID**
   - **Decision**: deterministic slug from stable feed+GUID inputs, stored in a mapping table.
   - **Rationale**: stability across re-scrapes; the contract every consumer route keys on.

## Alternatives Considered

1. **Reuse the operator API for consumers** — Rejected: entangles two audiences, blocks independent
   evolution, forces auth retrofit onto operator routes (D5).
2. **Build a persistence layer now (database and/or an abstraction)** — Deferred, not chosen: per-user state
   has query patterns (library×status, recall scoping) a database would serve well, but committing to one
   mid-requirements is premature and a sizeable refactor. We write plain per-user files for now and re-assess
   a real persistence layer as its own separate effort.
3. **Per-user corpus tenancy** — Rejected by design (PRD-035 Principle 3): shared corpus + overlay removes
   the largest lift.

## Testing Strategy

**Test Coverage:**

- **Unit**: slug derivation/stability, authz filtering, rate-limit/quota logic, DTO projection.
- **Integration**: `/api/app/*` against a temp per-user data dir + fixture corpus; **stubbed OAuth** provider; per-user
  isolation (user A cannot read user B's rows); episode-scoped search returns grounded hits.
- **E2E**: sign-in (stub) → library → episode → segments/search, on the consumer client (RFC-099).

**Test Organization:** `tests/integration/app_api/`; temp per-user data-dir fixtures; a fake OAuth + fake
session; **no real network, OAuth, or LLM** in CI.

## Rollout & Monitoring

- **Phase P0**: this RFC behind `--enable-app-api` (off by default). Reference client (RFC-099 thin mode)
  proves the contract.
- **Monitoring**: per-route latency, auth failures, scrape-quota rejections; reuse existing Prometheus hook.
- **Success Criteria**: sign-in on two devices yields identical state; episode-scoped search < ~1s; zero
  cross-user data leakage (test-enforced).

## Relationship to Other RFCs

Part of the Learning Platform initiative (RFC-098–101):

1. **RFC-099 Consumer client** — the PWA that consumes this API.
2. **RFC-100 Audio bridge** — supplies the playable origin URL this API references by slug.
3. **RFC-101 Personal corpus** — the per-user projection scoped via this API's identity + store.

**Key Distinction:** RFC-098 is the *server foundation + contract*; RFC-099 is the *client*; RFC-100 is
*audio*; RFC-101 is *consolidation*.

## Open Questions

1. Anonymous read access for catalog/search — on or off by default (privacy vs. discoverability)?
2. Where does `<data_dir>` live in prod hosting (RFC-082) relative to the corpus bind-mount?
3. Backfill strategy for slugs on the existing prod corpus — one-shot command timing.

## References

- **Related PRDs**: `docs/prd/PRD-035-learning-platform.md`, `docs/prd/PRD-036-foundation-identity.md`
- **Related RFCs**: `docs/rfc/RFC-090-hybrid-retrieval.md`, `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md`
- **Source Code**: `src/podcast_scraper/server/` (app.py, routes/, jobs.py)
- **Analysis**: `docs/wip/player/SERVER-SIDE-GAP-ANALYSIS.md`
