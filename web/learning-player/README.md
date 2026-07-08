# Learning Player (consumer app)

The end-user **Learning Platform** client (Epic 2 / #1077) — a mobile-first, installable PWA
that plays podcasts (audio bridged from the origin host) with transcript-synced playback over
the existing grounded intelligence. A **separate** app from the operator viewer
(`web/gi-kg-viewer`) with its own design system (Editorial Bold — UXS-011), its own build,
and its own Docker image.

- **Spec:** [PRD-035](../docs/prd/PRD-035-learning-platform.md) ·
  [PRD-038 Catalog](../docs/prd/PRD-038-catalog.md) ·
  [PRD-039 Player](../docs/prd/PRD-039-player.md) ·
  [RFC-099](../docs/rfc/RFC-099-learning-platform-consumer-client.md) ·
  [UXS-011](../docs/uxs/UXS-011-consumer-learning-app.md)
- **API:** thin client of `/api/app/*` (RFC-098) — see [PLATFORM_API](../docs/api/PLATFORM_API.md).
  The same API supports a future native mobile client (RFC-099 §10).

## Stack

Vue 3 + TypeScript + Vite + Pinia + Vue Router + vue-i18n, **Tailwind 3 mapped to CSS tokens**
(`src/theme/tokens.css`, same tokens→Tailwind bridge as the viewer), PWA via `vite-plugin-pwa`.
Tests: Vitest (unit/component) + Playwright (e2e). Dark-primary; a11y + i18n from line one.

## Develop

```bash
cd web/learning-player
npm install
npm run dev          # http://localhost:5174 (proxies /api → 127.0.0.1:8000; override VITE_API_TARGET)
```

Run the backend with the **mock OAuth provider** for a local dev account (no real Google):

```bash
APP_OAUTH_PROVIDER=mock APP_SESSION_SECRET=dev-secret <start the API>
```

## Test & build

```bash
cd app
npm run test:unit    # Vitest
npm run test:e2e     # Playwright (builds + previews, then runs e2e)
npm run build        # vue-tsc -b && vite build  (catches strict-mode TS errors)
```

CI runs three app-scoped jobs on any `app/**` change (see
`.github/workflows/python-app.yml` :: `app-unit`, `app-e2e`,
`app-lighthouse`). The Lighthouse gate hard-fails on missing/broken
manifest, service worker, maskable icon, apple-touch-icon, viewport,
or themed omnibox — configured in `web/learning-player/lighthouserc.json`.

## PWA behavior

- **Install / offline shell.** Manifest + service worker via
  `vite-plugin-pwa`. Precache = SHELL ONLY (JS/CSS/HTML/woff2/ico/svg,
  plus the manifest icons). Audio is NEVER cached by the SW (bridge-never-rehost);
  artwork is runtime-cached (`CacheFirst`, 500 entries × 30d); other
  shared GET `/api/app/*` reads are `StaleWhileRevalidate` (200 entries
  × 7d). Per-user endpoints (`/me`, `/queue`, `/playback`, `/auth`)
  are excluded from cache — auth-safety across sign-in/out.

- **Updates.** `registerType: 'prompt'` — when a new build ships, users
  see a "New version available — Reload" toast instead of silently
  running the old SW forever (avoided the guide-flagged silent-update
  stall trap). The composable also proactively calls
  `registration.update()` on tab refocus and every 15 min.

- **Build identity.** `window.__buildInfo = { sha, time }` is injected
  by `vite.config.ts` `define:` block. When triaging "PWA isn't
  updating" reports, ask the reporter to open DevTools console:
  `console.info` logs the sha/time at boot, and `window.__buildInfo`
  is inspectable.

- **Subpath deploys.** Set `APP_BASE=/some-prefix/` at build time (see
  `vite.config.ts`) — manifest `start_url`/`scope`, workbox
  `navigateFallback`, and the Vue Router base all pick it up
  automatically.

## Docker

Its own static image (node build → nginx), independent of the API/viewer images:

```bash
make app-docker-build        # build podcast-scraper-learning-app:latest
# serves the SPA on :80 and proxies /api → the api service (see nginx.conf)
```

Run alongside the stack (api + viewer + this app on one network) via the compose overlay:

```bash
make app-stack-up            # APP_PORT default 8081; api proxied over the compose network
# working sign-in (dev only): APP_OAUTH_PROVIDER=mock APP_SESSION_SECRET=dev-secret make app-stack-up
make app-stack-down
```

> Compose overlay: `compose/docker-compose.app.yml` (use with `docker-compose.stack.yml`).
> Validate config with `make app-stack-config`. A full stack boot is required to validate
> end-to-end before the deployment PR merges (compose changes → real-boot gate).

## Status

**C1 scaffold (#1080):** shell, routing, typed API client, auth store, i18n, theme tokens,
PWA, Docker, test harness. Catalog + Player are minimal scaffolds — full surfaces land in
C3 (#1082) / C4 (#1083). See `docs/wip/player/EPIC-2-CONSUMER-APP-PLAN.md`.
