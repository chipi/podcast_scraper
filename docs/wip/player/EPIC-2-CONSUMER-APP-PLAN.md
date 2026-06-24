# Epic 2 — Consumer Learning App (P1): plan & task breakdown

> **Status:** approved breakdown (2026-06-24). Local planning doc (WIP). Mirrors GH Epic 2.
> **Parent:** umbrella epic #1062 · **Sibling:** Epic 1 (#911, Foundation P0 — shipped, CI green).
> **Specs:** [PRD-038](../../prd/PRD-038-catalog.md) · [PRD-039](../../prd/PRD-039-player.md) ·
> [RFC-099](../../rfc/RFC-099-learning-platform-consumer-client.md) ·
> [UXS-011](../../uxs/UXS-011-consumer-learning-app.md)

## Goal

Ship a **player-first vertical slice**, fast, as the consumer MVP: a mobile-first, installable PWA
(`app/`) that lets a signed-in user **browse the local catalog → play an episode with transcript-sync →
read grounded intelligence inline → queue more**. Over **local content only** (already-processed corpus).
Discovery/scrape-on-demand (#1069), audio proxy (#1070), and Capture (PRD-040) are **after** the app ships.

Identity: **Editorial Bold** (UXS-011), dark-primary, per-show adaptive accent on the now-playing zone.

## Cross-cutting Definition of Done (EVERY task)

These are not a final task — they are acceptance criteria on **each** task from day one:

- **Mobile-optimized in all aspects.** Designed mobile-first (≤599px is the baseline, not an afterthought):
  touch targets ≥44px, thumb-reachable controls, full-viewport player, no hover-only affordances, fluid
  type, performance budget met on the **worst common device** (retina/DPR-2, throttled) per project pref
  (TTI + main-thread block, with headroom). Tested at mobile breakpoints in component/e2e.
- **Easily translatable (i18n) from line one.** **Zero hard-coded user-facing strings** — all copy via
  `vue-i18n` message catalogs; locale-aware dates/numbers; layout **RTL-ready**. A lint/CI check guards
  against raw string literals in templates. English ships first; adding a locale must require no code change.
- **Accessible (WCAG 2.1 AA).** Visible focus ring, full keyboard operability of the task's flow, ARIA
  roles/live regions where relevant, `prefers-reduced-motion` respected. axe checks in e2e for the path.
- **Full test pyramid** (like the viewer): vitest unit + component, Playwright e2e; mocked `/api/app/*`;
  **no real LLM / OAuth / network / audio in CI** (stub providers, silent clip). UXS-011 token compliance.

## Task breakdown

### Server (net-new, small — the only real Epic-1 gap)

| # | Task | Notes |
| - | ---- | ----- |
| S1 | **Catalog list endpoints** `GET /api/app/episodes`, `GET /api/app/podcasts/{id}/episodes` behind a pluggable `ContentSource` / `LocalCorpusSource` | The central net-new server work. Episode-summary shape per PRD-036. `DiscoverySource` (#1069) extends the same contract later — no UI/API reshape. Full pytest pyramid. |
| S2 | **`MockOAuthProvider`** (dev/e2e) alongside Google | Same `OAuthProvider` protocol (RFC-098 §2); env-selected (`APP_OAUTH_PROVIDER=mock`), never prod. Unblocks deterministic local sign-in + e2e. |

### Client (`app/` — new top-level Vue 3 project)

| # | Task | Notes |
| - | ---- | ----- |
| C1 | **Scaffold** | Vue 3 + TS + Vite + Pinia; `src/styles/tokens.css` (UXS-011); `vue-i18n` + RTL scaffolding; a11y primitives (focus mgmt, live regions, skip links); typed `/api/app` client; PWA shell + manifest; vitest + Playwright + **CI wiring**; own `Dockerfile`. |
| C2 | **Auth flow** | unauth → `/api/app/auth/login` → callback → session cookie. Google in prod, **mock provider** in dev/e2e. |
| C3 | **Catalog surface** (PRD-038) | global all-episodes + per-podcast views; Editorial Bold cards; graceful degradation. Local-content = effectively all Ready. |
| C4 | **Player surface** (PRD-039) — *the hero* | transcript-sync engine (binary-search active segment, autoscroll, tap-to-seek), controls/scrubber/speed/resume, **intelligent artwork zone** (speaking-now, grounding badge, surfacing insight, per-show adaptive accent), balanced split (1-col mobile / 2-col desktop). |
| C5 | **Knowledge Panel + in-episode grounded search** | summary/topics/insights/persons; "ask this episode" → `GET /api/app/episodes/{slug}/search` (extractive, **no request-time LLM**); jump-to-moment. |
| C6 | **Queue** | Pinia store ↔ `GET/PUT /api/app/queue`; auto-advance; play-next/add-to-queue from cards. Local-ready only (scrape-on-enqueue is post-#1069). |
| C7 | **Deployment** | separate Docker container (static PWA bundle) + compose wiring; talks to backend only via `/api/app/*`; one API → web PWA + future native mobile. |

> The cross-cutting DoD (mobile + i18n + a11y + tests) applies to **C1–C7 and S1–S2**, so there is no
> separate "a11y/i18n task" — it ships inside every slice.

## Sequencing & branch strategy

- **One themed branch** (continue on `feat/learning-platform` or a fresh `feat/consumer-app`), bisectable
  per task — per bundling preference.
- Order: **S2 + C1** (scaffold + mock auth so e2e works) → **S1** (catalog endpoints) → **C2 → C3 → C4 →
  C5 → C6** → **C7**. Player (C4) is the priority; catalog is the minimal on-ramp to it.
- Each task: commit with `#<task-issue>`; close via PR `Closes #…` (never close manually).

## Out of scope (explicit)

- Scrape-on-demand / Discovery (#1069), audio no-store proxy (#1070), Capture (PRD-040), consumer KG
  browser (RFC-099 §9), per-user corpus tenancy, request-time LLM, rehosting audio, persistence-layer/DB work.

## Open items to confirm with operator

- Whether to spin the individual **task issues** (S1–S2, C1–C7) now, or create them just-in-time as we
  start each. (This doc + the GH epic capture the plan either way.)
