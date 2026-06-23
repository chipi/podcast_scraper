# PRD-036: Foundation / Identity

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.7 (Phase P0)
- **Parent PRD**: `docs/prd/PRD-035-learning-platform.md`
- **Related analysis**: `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md`
- **Related RFCs**: RFC-049, RFC-055, RFC-072, RFC-090, RFC-097 (intelligence the read API exposes)

---

## Summary

The Foundation layer is the minimal multi-user substrate every other segment depends on:
a single-provider OAuth identity, per-user state persistence, the consumer-facing read API
over shared episode artifacts, the frozen `segments.json` contract, and original-host audio
enclosure resolution. It ships with a deliberately thin **reference player** whose only job
is to prove the contract end-to-end before any rich UI is built.

This is Phase P0. It de-risks the whole platform by locking identity and the data contract first.

## Background & Context

- The existing server (`src/podcast_scraper/server/`) is built for one operator, one corpus,
  no auth (`MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md`). The consumer platform needs identity and
  per-user state — but **not** per-user corpus tenancy.
- Per Principle 3 (shared artifacts / personal overlay), episode-derived artifacts stay global
  and read-only; only the personal overlay is per-user. This is what avoids the "Large"
  multi-tenancy lift and keeps P0 small.
- The read surface largely exists already (`/api/corpus/episodes`, `/api/search`,
  `/api/relational/*`, `/api/artifacts/*`). P0 mostly *adapts and stabilises* it for consumer use
  (slug-addressed, artifact-flagged), rather than building new ML.

## Goals

- Authenticate users via one OAuth provider (Google to start) with session-gated per-user routes.
- Define and persist the per-user data model: library, playback, queue, highlights, notes, interests.
- Expose a stable, slug-addressed read API for episode metadata, segments, insights, entities, summary.
- Freeze the `segments.json` scraper→player contract and serve it per episode.
- Resolve original-host audio enclosure URLs (bridge, never proxy/rehost).
- Provide scrape-on-demand request + status APIs (globally deduped), with a per-user quota hook.
- Ship a thin reference player that exercises auth → fetch segments → resolve enclosure → play + sync.

## Non-Goals

- No rich UI (Discovery/Catalog/Player UX is P1).
- No per-user corpus forking; artifacts are shared and read-only.
- No orgs, roles, sharing, or social.
- No third-party audio storage or proxying.
- No content translation.

## Personas

Inherits PRD-035 personas. P0 is validated primarily by the **active learner** signing in on a
second device and finding their state intact, and by the **researcher** hitting grounded read routes.

## User Stories

- _As any user, I can sign in with Google and have a stable identity across devices._
- _As any user, my library, queue, playback position, highlights, and notes are private to me and
  follow me across sessions._
- _As the platform, I can serve an episode's `segments.json` and intelligence artifacts by slug so the
  player never parses raw corpus files._
- _As a listener, I can play an episode from its original host, not a copy stored by us._

## Functional Requirements

### FR1: Identity & sessions

- **FR1.1**: OAuth sign-in via a single configured provider (Google to start). Provider is swappable
  by config.
- **FR1.2**: Authenticated sessions gate all per-user routes (`/api/user/*`). Anonymous users may read
  shared, non-personal routes (catalog/search) if enabled, but cannot mutate per-user state.
- **FR1.3**: Sign-out invalidates the session. Account deletion removes all per-user overlay rows
  (not shared artifacts).

### FR2: Per-user data model

- **FR2.1**: Entities — `user`, `library_subscription` (user→podcast), `playback_position`
  (user→episode→seconds+updated_at), `queue_item` (user→episode→order), `highlight`, `note`,
  `interest_topic` (user→topic canonical id). Highlights/notes detailed in PRD-040.
- **FR2.2**: All per-user rows are keyed by user id and never leak across users (authz on every read/write).
- **FR2.3**: Shared corpus artifacts are read-only via the API; no per-user route mutates them.

### FR3: Consumer read API (slug-addressed)

- **FR3.1**: `GET /api/episodes` — paginated episode list across the user's library, with artifact flags
  and status (ready/pending/not-scraped).
- **FR3.2**: `GET /api/episodes/{slug}` — episode detail + artifact availability flags + enclosure ref.
- **FR3.3**: `GET /api/episodes/{slug}/segments` — the `segments.json` transcript contract.
- **FR3.4**: `GET /api/episodes/{slug}/insights` — GIL insights (grounded) for the knowledge panel.
- **FR3.5**: `GET /api/episodes/{slug}/entities` — KG entities (persons/topics) for the knowledge panel.
- **FR3.6**: Search + relational routes (RFC-090, RFC-072) are exposed read-only to the consumer client.

### FR4: Audio bridge / enclosure resolution

- **FR4.1**: `GET /api/episodes/{slug}/audio-source` returns the original enclosure URL (+ metadata),
  not audio bytes. The client plays from the host.
- **FR4.2**: Internal pipeline audio is never reachable from any `/api/episodes/**` route.

### FR5: Scrape-on-demand

- **FR5.1**: `POST /api/scrape` enqueues processing for an episode/feed; globally deduped (RFC-065
  status.json), so concurrent requests for the same episode share one job.
- **FR5.2**: `GET /api/scrape/status/{job_id}` reports pipeline stage + progress.
- **FR5.3**: A per-user quota hook exists (enforcement policy decided at RFC stage; see PRD-035 OQ4).

### FR6: Reference player (contract proof)

- **FR6.1**: A minimal client: sign in → list episodes → load segments → resolve enclosure → play with
  segment highlight + tap-to-seek + resume. No design polish; correctness only.

## `segments.json` contract (frozen here)

```json
{
  "version": "1.0",
  "episode_slug": "lex-fridman-420",
  "segments": [
    { "id": "seg_001", "start": 12.4, "end": 18.7,
      "text": "The thing about transformer architecture is...",
      "speaker": "person:lex-fridman" }
  ]
}
```

`speaker` is optional; canonical `person:{slug}` (RFC-072) when identity resolution has run, else a raw
label. Breaking this shape requires amending PRD-035 FR1.

## API summary

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/auth/login` / `/callback` / `/logout` | OAuth flow |
| `GET` | `/api/me` | Current user |
| `GET` | `/api/episodes` | Library episode list + status + artifact flags |
| `GET` | `/api/episodes/{slug}` | Episode detail |
| `GET` | `/api/episodes/{slug}/segments` | Transcript contract |
| `GET` | `/api/episodes/{slug}/insights` | GIL insights |
| `GET` | `/api/episodes/{slug}/entities` | KG entities |
| `GET` | `/api/episodes/{slug}/audio-source` | Original enclosure URL (no bytes) |
| `POST` | `/api/scrape` | Request scrape (deduped) |
| `GET` | `/api/scrape/status/{job_id}` | Scrape progress |

## Success Metrics

- A user signs in on device A, queues an episode, signs in on device B → state is identical.
- Reference player plays from the original host, highlights the current segment, and seeks within 0.5s
  of a tapped line.
- Zero per-user data leakage across users (authz verified by test on every `/api/user/*` route).
- The read API serves segments + insights + entities by slug at interactive latency on the prod corpus.

## Dependencies

- Existing read API + artifacts (RFC-049/055/072/090/097).
- Per-user overlay state stored as **plain per-user files** (existing `atomic_write`/`filelock` helpers,
  like `jobs.jsonl`); no new persistence layer, abstraction, or interfaces. A real persistence layer is a
  deferred, separate effort (PRD-035 D2).
- OAuth provider registration (Google).

## Open Questions

- A real persistence layer (a database and/or any abstraction around it) — a separate future refactor, out
  of scope here; this phase writes plain per-user files (PRD-035 D2).
- Anonymous read access on/off for catalog/search (privacy + abuse trade-off).
- Scrape quota enforcement model (PRD-035 OQ4).

## References

- `docs/prd/PRD-035-learning-platform.md`
- `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md`
- `docs/api/HTTP_API.md`, `docs/rfc/RFC-065-agent-observable-instrumentation.md`
