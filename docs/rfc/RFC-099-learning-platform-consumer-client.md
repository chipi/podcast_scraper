# RFC-099: Learning Platform Consumer Client

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Frontend, Server API, Design/UX, Accessibility
- **Related PRDs**:
  - `docs/prd/PRD-035-learning-platform.md` (parent)
  - `docs/prd/PRD-037-discovery.md`, `docs/prd/PRD-038-catalog.md`, `docs/prd/PRD-039-player.md`,
    `docs/prd/PRD-040-capture.md`
- **Related RFCs**:
  - `docs/rfc/RFC-098-learning-platform-foundation.md` (the API this client consumes)
  - `docs/rfc/RFC-100-audio-bridge-subsystem.md` (audio source)
  - `docs/rfc/RFC-062-gi-kg-viewer.md` (operator viewer — primitives reused, kept separate)

## Abstract

This RFC defines the **new top-level consumer application** (PRD-035 D1/D3): a mobile-first, installable
**PWA** that delivers the Discovery → Catalog → Player → Capture experience over the `/api/app/*`
foundation (RFC-098). It specifies the transcript-sync engine, the queue, the capture UX, and the
accessibility + internationalisation foundations that are non-negotiable from the first commit.

**Architecture Alignment:** A separate top-level app (`app/`), distinct from `web/gi-kg-viewer` (operator
only). It reuses extracted UI primitives where sensible but ships its own shell, routing, and state.

## Problem Statement

The intelligence layer and (after RFC-098) the consumer API exist, but there is **no end-user client**. The
operator viewer is the wrong surface — different audience, no playback, no auth, no capture. We need a
polished, Spotify-grade player whose differentiator is transcript-synced playback + inline grounded
intelligence + frictionless capture, accessible and localisable from day one.

**Use Cases:**

1. **Listen**: queue an episode, play it bridged from the origin host, follow the synced transcript, tap a
   line to seek.
2. **Understand**: open the Knowledge Panel for grounded insights/entities and episode-scoped grounded
   search — no leaving the player.
3. **Capture**: highlight the current moment or a transcript span in one interaction; add a note.

## Goals

1. **New top-level PWA** (`app/`): Vue 3 + TypeScript + Vite, installable, mobile-first, offline app-shell.
2. **Transcript-sync engine**: highlight + autoscroll the active segment; tap-to-seek; resume.
3. **Queue**: add/reorder/auto-advance; enqueue triggers scrape-on-demand (RFC-098).
4. **Knowledge Panel + in-episode grounded search** (no request-time LLM, D6).
5. **Capture UX**: highlights + notes (PRD-040) with one-interaction capture.
6. **a11y (WCAG 2.1 AA) + i18n** built in from the first commit; **voice control** as a later north-star.

## Constraints & Assumptions

**Constraints:**

- Audio is played from the origin URL provided by RFC-100; the client never fetches our stored media bytes.
- No request-time LLM; "ask" is `GET /api/app/episodes/{slug}/search` (extractive).
- a11y and i18n are acceptance criteria, not follow-ups.

**Assumptions:**

- `/api/app/*` (RFC-098) is available and session-authenticated.
- Cross-origin `<audio>` playback with `timeupdate` works for transcript sync (standard browser behaviour).

## Design & Implementation

### 1. App shell & stack

```text
app/                      # new top-level project (sibling of web/)
  index.html
  src/
    main.ts
    router/               # Discovery, Catalog, Player, Library, Capture, Corpus(RFC-101)
    stores/               # Pinia: auth, player, queue, capture
    services/api.ts       # typed client for /api/app/*
    features/{discovery,catalog,player,capture}/
    i18n/                 # vue-i18n catalogs (en first; RTL-ready)
    a11y/                 # focus mgmt, live regions, skip links
    primitives/           # shared UI (some extracted from gi-kg-viewer)
```

- **PWA**: service worker caches the app shell + GET API responses (stale-while-revalidate); **audio is
  never cached/proxied** by the SW. Web App Manifest for install.
- **Auth**: unauthenticated → redirect to `/api/app/auth/login`; session cookie carries the rest.

### 2. Transcript-sync engine

- Load `segments.json` once; maintain an index sorted by `start`. On `audio.timeupdate`, binary-search the
  active segment, apply the highlight class, and autoscroll into view (disabled on manual scroll, re-enabled
  after ~5s idle). Tap a segment → `audio.currentTime = segment.start`.
- Speaker labels from `segment.speaker` (canonical `person:{slug}` when present).

### 3. Queue

- Pinia `queue` store mirrored to `GET/PUT /api/app/queue`. Auto-advance on `ended`. "Play next" / "add to
  queue" from Catalog cards. Enqueuing an unprocessed episode calls `POST /api/app/scrape` and shows
  progress inline; it becomes playable when Ready.

### 4. Knowledge Panel & in-episode search

- Collapsible panel: Summary, Topics, Insights (grounded cards with timestamp jump), Persons.
- "Ask / find in this episode" → `GET /api/app/episodes/{slug}/search` → ranked grounded passages with
  jump-to-moment. No generation, no disclaimer (results are verbatim).
- **Relational context** uses the RFC-094 queries by name: Insights via `who_said` + `cross_show_synthesis`,
  Persons via `positions_of` — scoped to this episode.
- **Enrichment signals (consumes RFC-088, built in parallel — stay in sync):** related-topic chips from
  `topic_cooccurrence` and a credibility badge from `grounding_rate` ("N% grounded"), shown when present.

### 5. Capture

- One-tap "highlight current moment" (anchors to active segment); transcript span selection; "save insight".
- Notes attach to highlight/insight/episode. Persisted via PRD-040 routes on `/api/app/*`.

### 6. Accessibility & i18n

- **a11y**: full keyboard operability of the listen→capture flow; ARIA roles; live region for "now playing"
  segment; visible focus; reduced-motion respect for autoscroll; target WCAG 2.1 AA.
- **i18n**: all copy via `vue-i18n`; no hard-coded strings; locale-aware dates/numbers; layout RTL-ready.
  (Content/transcript translation is out — a future pipeline feature.)

### 7. Observability & analytics (consumer)

- **Errors/crashes** → Sentry (client). **Web-vitals** (LCP/INP/TTI, main-thread block measured on the worst
  common device — retina/DPR-2, throttled) → a metrics endpoint.
- **Event taxonomy** (privacy-light: no PII, no transcript text — in the spirit of the existing
  `query_log`): play, pause, seek, segment-jump, search, recall, capture-highlight, add-note, queue-add,
  resurface-shown/acted.
- **Backend** `/api/app/*` latency/errors → Prometheus (reuse the existing `PODCAST_METRICS_ENABLED` hook) →
  Grafana, aligning the consumer layer with the operator's existing observability.
- **GDPR-light:** per-user analytics are deletable with the account (RFC-098).

### 8. Optional consumer knowledge-graph browser (P2+)

- A read-only visual explorer reusing the RFC-069 graph toolkit (zoom/minimap/filters) + RFC-094 queries,
  scoped to the user's episode set. Distinct from the operator viewer; off the critical path — ships only
  after the core listen→capture flow.

## Key Decisions

1. **New top-level `app/`, not an extension of `gi-kg-viewer`**
   - **Decision**: separate consumer app.
   - **Rationale**: PRD-035 D3 — operator vs consumer concerns stay separate; viewer untouched.
2. **PWA, not native**
   - **Decision**: installable responsive PWA in v2.7.
   - **Rationale**: PRD-035 D1 — mobile + a11y + i18n with no app-store friction; native is north-star.
3. **Client-side transcript sync, audio direct from origin**
   - **Decision**: no server in the sync hot path; bridge URL plays directly.
   - **Rationale**: lowest latency; aligns with bridge-never-rehost.

## Alternatives Considered

1. **Extend the operator viewer** — Rejected (D3): conflates audiences, bloats the operator tool.
2. **Native app first** — Rejected for v2.7: app-store friction, slower a11y/i18n iteration; revisit for
   background audio (north-star).
3. **Server-driven transcript sync (push current segment)** — Rejected: needless latency/complexity; the
   client already has segments + audio time.

## Testing Strategy

**Test Coverage:**

- **Unit (vitest)**: sync index/active-segment math; queue auto-advance; capture anchoring.
- **Component**: Knowledge Panel degradation (missing artifacts), search results rendering.
- **E2E (Playwright)**: sign-in (stub) → play → segment highlight + tap-seek → highlight capture; **a11y
  checks** (axe) in the listen→capture path.

**Test Organization:** in `app/` (cd into it before vitest/playwright); mocked `/api/app/*`; no real audio
fetched in CI (stub source / silent clip).

## Rollout & Monitoring

- **P0**: thin reference mode (auth → segments → play) to validate RFC-098 contract.
- **P1**: full Discovery + Catalog + Player + queue. **P2**: Capture. **P3**: Corpus surface (RFC-101).
- **Monitoring**: client perf (TTI, main-thread block on worst-case retina/throttled per project pref),
  Sentry, basic UX analytics.
- **Success**: listen→capture flow fully keyboard/screen-reader operable; sync has no perceptible lag.

## Open Questions

1. Exact top-level dir name (`app/` vs `web-app/`) and whether to share a workspace/build with `web/`.
2. Which `gi-kg-viewer` primitives are worth extracting vs. reimplementing for the consumer aesthetic.
3. Offline scope of the PWA beyond app-shell (cache last-played transcript?).

## References

- **Related PRDs**: `docs/prd/PRD-039-player.md`, `docs/prd/PRD-038-catalog.md`, `docs/prd/PRD-040-capture.md`
- **Related RFCs**: `docs/rfc/RFC-098-learning-platform-foundation.md`, `docs/rfc/RFC-100-audio-bridge-subsystem.md`
- **Source Code**: new `app/`; primitives from `web/gi-kg-viewer/`
