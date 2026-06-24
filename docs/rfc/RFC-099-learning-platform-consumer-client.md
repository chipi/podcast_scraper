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
- **Related UX spec**:
  - `docs/uxs/UXS-011-consumer-learning-app.md` — the **Editorial Bold** design system + Player visual
    contract this client implements (a **separate** design system from the operator viewer's UXS-001)

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
3. **Queue**: add/reorder/auto-advance over local-ready episodes (scrape-on-demand on enqueue is
   post-#1069 — see §4).
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
- **Visual design**: implements **UXS-011** (Editorial Bold, dark-primary). The single token layer is
  `app/src/styles/tokens.css`; the Player's now-playing artwork zone applies the per-show adaptive accent
  (contrast-clamped per UXS-011 Accessibility). This is a **separate** design system from `gi-kg-viewer`.

#### Local mocked OAuth provider (dev / e2e)

Production requires **Google** OAuth (RFC-098 `GoogleProvider`). For **local development and e2e tests**
we add a **`MockOAuthProvider`** alongside it (same `OAuthProvider` protocol, RFC-098 §2) that completes
the code flow **without any network call**, minting a fixed dev identity (e.g. `dev@localhost`). It is
selected only when explicitly enabled (e.g. `APP_OAUTH_PROVIDER=mock`, never in prod config) so a
developer can sign in with a fake account and Playwright e2e can drive the full authed flow
deterministically — the same path CI already uses for the stub in unit/integration tests (no real LLM,
no real OAuth in CI, per project rule). The provider boundary already exists in RFC-098; this is the
required second implementation, not a new abstraction.

### 2. Transcript-sync engine

- Load `segments.json` once; maintain an index sorted by `start`. On `audio.timeupdate`, binary-search the
  active segment, apply the highlight class, and autoscroll into view (disabled on manual scroll, re-enabled
  after ~5s idle). Tap a segment → `audio.currentTime = segment.start`.
- Speaker labels from `segment.speaker` (canonical `person:{slug}` when present).

### 3. Catalog & the pluggable ContentSource

- The Catalog (PRD-038) lists episodes via **net-new** read endpoints `GET /api/app/episodes` and
  `GET /api/app/podcasts/{id}/episodes` (Epic-1 shipped only episode *detail* by slug — these lists are
  the central net-new server work for Epic 2).
- These endpoints read through a **pluggable `ContentSource`**. The **MVP** backend is a
  **`LocalCorpusSource`** enumerating the **already-processed local corpus** — no scrape, no discovery;
  every catalogued episode is effectively **Ready**. When #1069/#1070 land, a **`DiscoverySource`**
  extends the same contract to surface not-yet-processed content and provide the "add content" entry
  point. The client and the `/api/app/episodes*` response shape are **unchanged** across that swap.

### 4. Queue

- Pinia `queue` store mirrored to `GET/PUT /api/app/queue`. Auto-advance on `ended`. "Play next" / "add to
  queue" from Catalog cards.
- **MVP**: the queue holds **local-ready** episodes only (no scrape-on-demand). Enqueuing an
  **unprocessed** episode (calling `POST /api/app/scrape`, showing inline progress, flipping to playable
  when Ready) is **post-#1069** — built when scrape-on-demand and the `DiscoverySource` arrive.

### 5. Knowledge Panel & in-episode search

- Collapsible panel: Summary, Topics, Insights (grounded cards with timestamp jump), Persons.
- "Ask / find in this episode" → `GET /api/app/episodes/{slug}/search` → ranked grounded passages with
  jump-to-moment. No generation, no disclaimer (results are verbatim).
- **Relational context** uses the RFC-094 queries by name: Insights via `who_said` + `cross_show_synthesis`,
  Persons via `positions_of` — scoped to this episode.
- **Enrichment signals (consumes RFC-088, built in parallel — stay in sync):** related-topic chips from
  `topic_cooccurrence` and a credibility badge from `grounding_rate` ("N% grounded"), shown when present.

### 6. Capture

- One-tap "highlight current moment" (anchors to active segment); transcript span selection; "save insight".
- Notes attach to highlight/insight/episode. Persisted via PRD-040 routes on `/api/app/*`.

### 7. Accessibility & i18n

- **a11y**: full keyboard operability of the listen→capture flow; ARIA roles; live region for "now playing"
  segment; visible focus; reduced-motion respect for autoscroll; target WCAG 2.1 AA.
- **i18n**: all copy via `vue-i18n`; no hard-coded strings; locale-aware dates/numbers; layout RTL-ready.
  (Content/transcript translation is out — a future pipeline feature.)

### 8. Observability & analytics (consumer)

- **Errors/crashes** → Sentry (client). **Web-vitals** (LCP/INP/TTI, main-thread block measured on the worst
  common device — retina/DPR-2, throttled) → a metrics endpoint.
- **Event taxonomy** (privacy-light: no PII, no transcript text — in the spirit of the existing
  `query_log`): play, pause, seek, segment-jump, search, recall, capture-highlight, add-note, queue-add,
  resurface-shown/acted.
- **Backend** `/api/app/*` latency/errors → Prometheus (reuse the existing `PODCAST_METRICS_ENABLED` hook) →
  Grafana, aligning the consumer layer with the operator's existing observability.
- **GDPR-light:** per-user analytics are deletable with the account (RFC-098).

### 9. Optional consumer knowledge-graph browser (P2+)

- A read-only visual explorer reusing the RFC-069 graph toolkit (zoom/minimap/filters) + RFC-094 queries,
  scoped to the user's episode set. Distinct from the operator viewer; off the critical path — ships only
  after the core listen→capture flow.

### 10. Deployment & API boundary (mobile-future)

- **Separate Docker container.** `app/` builds to a static bundle served by its own lightweight image
  (the PWA shell + assets), independent of the API image. It talks to the backend purely over
  `/api/app/*` — no shared process, no server-rendered coupling. This keeps the consumer surface
  independently deployable/scalable and lets the operator API and pipeline images stay untouched.
  Concretely (#1086): `app/Dockerfile` (node build → nginx; `app/nginx.conf` = SPA fallback +
  `/api` proxy, no audio) + the `compose/docker-compose.app.yml` overlay (adds `learning-app` to
  the stack network) + `make app-docker-build` / `app-stack-up`.
- **API is the only contract.** Because the client consumes `/api/app/*` exclusively (session cookie
  auth, JSON), the **same API supports a future native mobile app** with no server changes — the mobile
  client would swap the cookie session for a token grant against the same OAuth boundary (RFC-098) and
  reuse every read/write route. The web PWA and a future mobile app are **two clients of one API**.
- **No request-time LLM, bridge-never-rehost** hold at the deployment boundary too: the container serves
  UI + proxies nothing audio; audio plays from the origin host (RFC-100).

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

**Epic 2 = a player-first vertical slice over local content, shipped quickly as the MVP.** Scope is
**Catalog (local corpus) + Player + Knowledge Panel + queue**, with Google OAuth (+ the local mocked
provider for dev/e2e). Discovery/scrape-on-demand (#1069), the audio proxy (#1070), and Capture
(PRD-040) are explicitly **after** the app ships.

- **P0**: thin reference mode (auth → segments → play) to validate the RFC-098 contract — *already
  proven by the Epic-1 reference client (`app_reference_client.py`)*.
- **P1 (Epic 2 MVP)**: net-new catalog list endpoints (`LocalCorpusSource`) → Catalog → **Player**
  (transcript-sync, Knowledge Panel, in-episode search) → queue → auth (Google + mock). Full vertical
  slice, top-to-bottom, MVP-fast.
- **P2**: Capture (PRD-040). **P3**: Discovery/scrape-on-demand (#1069, `DiscoverySource`) + Corpus
  surface (RFC-101).
- **Monitoring**: client perf (TTI, main-thread block on worst-case retina/throttled per project pref),
  Sentry, basic UX analytics.
- **Success**: listen→capture flow fully keyboard/screen-reader operable; sync has no perceptible lag.

## Open Questions

1. **Resolved**: top-level dir is **`app/`**, its **own** Vue 3 + Vite + Pinia build and **own Docker
   image** (§10) — *not* sharing a workspace/build with `web/` (operator viewer stays separate, D3).
2. **Resolved (direction)**: aesthetic is **Editorial Bold** (UXS-011); given the distinct design system,
   default to **reimplementing** consumer components against UXS-011 tokens and only extract a
   `gi-kg-viewer` primitive when it is genuinely design-neutral (e.g. a focus-trap util), not for styled
   surfaces.
3. Offline scope of the PWA beyond app-shell (cache last-played transcript?).

## References

- **Related PRDs**: `docs/prd/PRD-039-player.md`, `docs/prd/PRD-038-catalog.md`, `docs/prd/PRD-040-capture.md`
- **Related RFCs**: `docs/rfc/RFC-098-learning-platform-foundation.md`, `docs/rfc/RFC-100-audio-bridge-subsystem.md`
- **Source Code**: new `app/`; primitives from `web/gi-kg-viewer/`
