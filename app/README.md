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
cd app
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

## Docker

```bash
docker build -t learning-player app/
# serves the SPA on :80 and proxies /api → the api service (see nginx.conf)
```

## Status

**C1 scaffold (#1080):** shell, routing, typed API client, auth store, i18n, theme tokens,
PWA, Docker, test harness. Catalog + Player are minimal scaffolds — full surfaces land in
C3 (#1082) / C4 (#1083). See `docs/wip/player/EPIC-2-CONSUMER-APP-PLAN.md`.
