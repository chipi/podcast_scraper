# Brief for the orrery agent — add client error tracking → self-hosted GlitchTip

**From:** podcast_scraper infra (owns the shared public edge, ADR-114).
**Date:** 2026-07-22.
**Goal:** un-skip orrery's browser-side error tracking and point it at the
self-hosted GlitchTip (homelab), the same one the podcast app now uses. Your
o11y handover (`agentic-ai-homelab/docs/wip/orrery-o11y-handover.md` §3) skipped
this because tailnet-only GlitchTip is unreachable from a public browser. That
blocker is now solved on our side — read on.

## What changed on the edge (your dependency is being removed)

Browser SDKs POST errors from the *end user's* browser, which can only reach
public endpoints — never `homelab:8090` (tailnet-only). So we added a public,
**ingest-only** GlitchTip vhost on the shared Caddy edge:

- File: `podcast_scraper-infra:infra/caddy/glitchtip.caddy`.
- It exposes **only** the Sentry/GlitchTip ingest paths
  (`/api/<project_id>/{envelope,store,security}/`) and 404s everything else — the
  GlitchTip dashboard/admin stays tailnet-only.
- Public host: **`telemetry.orrerylearn.com`** (generic name, LIVE with a valid
  Let's Encrypt cert; ingest-only).
- Upstream to `homelab:8090` is resolved on-box (no hardcoded IP anywhere).

**Hard ordering:** the vhost must be **live + cert issued** before you flip
orrery's DSN. A DSN pointed at a not-yet-live host **silently drops** events.
Until then, keep it gated (see below) so a missing DSN is a true no-op.

## Your GlitchTip project + DSN — READY (created 2026-07-22)

- Orrery = **GlitchTip project 4** (a dedicated project; the old "project 2" from
  the earlier handover never actually existed / was deleted — do NOT use `/2`).
- The public ingest edge is **live**: `telemetry.orrerylearn.com` (a generic name,
  ingest-only, valid Let's Encrypt cert). Use this **exact DSN** in the orrery
  browser build:

  ```
  https://9b69bd79-85fa-4ce5-9cec-dbb5c9f7febe@telemetry.orrerylearn.com/4
  ```

  - Scheme is **https** (browsers block mixed-content POSTs from an https page to
    an http DSN; the edge terminates TLS). Validated: a store POST to
    `https://telemetry.orrerylearn.com/api/4/store/` returns 200.
  - The public key is **not a secret** — it ships in your JS bundle by design.
  - Do **not** use `homelab:8090` in the browser build — that's the tailnet host,
    unreachable for public users. `telemetry.orrerylearn.com` is the public path.

## What to add in the orrery repo

Orrery is a static site. Mirror the pattern the podcast **player** now uses —
reference implementation: `podcast_scraper-infra:web/learning-player/src/main.ts`
(commit on `main`, 2026-07-22). Concretely:

1. **Dependency:** add the Sentry browser SDK for your framework
   (`@sentry/browser`, or `@sentry/vue` / `@sentry/react` if orrery uses one).
   Match a recent 8.x (we pinned `@sentry/vue@^8.55`).

2. **Gated init** at app entry — initialise **only when the DSN env var is set**,
   so dev/CI builds without a DSN stay a true no-op:

   ```ts
   import * as Sentry from '@sentry/browser' // or @sentry/vue with { app }
   const dsn = import.meta.env.VITE_SENTRY_DSN_ORRERY // your build tool's env convention
   if (dsn) {
     Sentry.init({
       dsn,
       environment: import.meta.env.PROD ? 'prod' : 'dev',
       release: /* your build sha, if available */ undefined,
       sendDefaultPii: false,
       tracesSampleRate: 0.1,
       initialScope: { tags: { component: 'orrery' } }, // keep streams separable
     })
   }
   ```

3. **Build-time plumbing:** the DSN is baked into the bundle at build time
   (`VITE_*` / your equivalent). Pass it as a build-arg in orrery's Dockerfile +
   the compose/CI that builds the site. Keep the env var **empty by default**.

4. **Secret:** store the DSN as `PROD_SENTRY_DSN_ORRERY` (or your convention) and
   feed it into the build only for the prod/public build. Never commit the DSN
   value (even though the key is public, keep repos clean).

## Verification (once the edge + DSN are live)

- From a browser on the public site, trigger a handled error; confirm it appears
  in GlitchTip **project 4** with `component: orrery`, `environment: prod`.
- Direct ingest smoke (no browser): a well-formed envelope/store POST to
  `https://telemetry.orrerylearn.com/api/4/store/` with the project-4 key returns `200`;
  a non-ingest path (e.g. `/`) returns `404`.

## Non-goals / notes

- **Metrics/traces stay out of scope** for orrery (static site) — this brief is
  errors only, matching your handover's §"out of scope".
- No server-side component for orrery — it's browser-only, so there's nothing to
  wire on the tailnet side.
- Coordinate the DSN flip with the podcast-infra owner so it lands **after** the
  edge is LIVE + validated. The project-4 DSN above is
  ready to use as-is.
