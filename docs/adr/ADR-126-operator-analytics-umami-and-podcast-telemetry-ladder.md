# ADR-126: Operator analytics on self-hosted Umami + the podcast dev→prod telemetry ladder

**Status:** Accepted
**Date:** 2026-07-24

## Context & Problem Statement

The podcast estate emits telemetry from three surfaces: the **backend** (api +
pipeline, Python Sentry SDK → self-hosted GlitchTip), the **player** (public
consumer app — Sentry/Vue → GlitchTip + Umami page/event analytics), and the
**operator viewer** (`gi-kg-viewer`, tailnet-only knowledge-graph console —
Sentry/Vue → GlitchTip, but analytics on **PostHog Cloud**).

Two problems:

1. **Analytics inconsistency + a third-party cloud dependency.** The player and
   orrery already run self-hosted **Umami** (cookieless, PII-free, on homelab).
   Only the operator viewer used **PostHog Cloud** (`eu.i.posthog.com`) — the odd
   surface out, and the one place production-derived operator data left our
   infrastructure for a third party.

2. **No dev rung.** Local `vite dev` / `make serve` sent telemetry **nowhere**
   (DSN/token unset → silent). There were no dedicated dev projects, so a
   developer got no local signal and dev noise could never be separated from
   prod even if a shared DSN were used.

The operator viewer is served **HTTPS over `tailscale serve`** (tailnet-only),
which adds a constraint: a `http://homelab:3001` Umami script is
**mixed-content-blocked** by the browser on an HTTPS page.

## Decision

1. **Move the operator viewer off PostHog Cloud onto self-hosted Umami.** The
   whole estate is now Umami; nothing leaves for a third-party analytics cloud.
   The 12 `posthog.capture(name, props)` sites became `track(name, props)` via a
   new typed helper `web/gi-kg-viewer/src/lib/analytics.ts` (Umami `track()` + an
   event-name registry). Event names are unchanged, so dashboards keep working.
   `posthog-js` is removed.

2. **Establish a dev→prod telemetry ladder** (the podcast surfaces have no
   staging rung). Each surface reports to a **dedicated dev project** by default
   in local dev — GlitchTip `player-dev` / `operator-dev`, Umami `player-dev` /
   `operator-dev` — so dev and prod telemetry never cross-pollute.

3. **Reach the dev backends over the Tailscale host `homelab`, never a fixed
   IP.** Only a device on the tailnet resolves `homelab`, so a fork/stranger who
   runs the repo sends nothing (the transport just fails) — fork-safe by
   construction. `VITE_ANALYTICS_OFF=1` (vitest + playwright) and a
   pytest/CI/prod guard (backend) suppress the dev default where it must be
   silent.

4. **For operator PROD Umami, expose Umami over an HTTPS entry point *inside the
   tailnet*** via `tailscale serve` (`https://homelab.<tailnet>/umami`) — a real
   Tailscale-provisioned cert, tailnet-only, **no public surface** (serve, not
   funnel). This resolves the mixed-content block without publishing Umami. A
   matching tailnet ACL grant (`autogroup:admin → tag:homelab-host:443`) lets the
   operator's browser reach it.

## Rationale

- **Estate consistency + data sovereignty.** One analytics system (Umami,
  self-hosted) across player, viewer, and orrery; operator data stays on homelab.
- **Dev/prod separability is structural, not conventional.** Dedicated dev
  projects + a `homelab`-only transport mean isolation can't be forgotten.
- **`tailscale serve` beats a public edge for a tailnet-only app.** The viewer
  has no public domain (ADR-116 is the future public cutover); a tailnet HTTPS
  endpoint keeps the whole path private and needs no CORS or cert plumbing of our
  own.

## Alternatives Considered

- **Keep PostHog Cloud for the operator.** Rejected: leaves the estate
  inconsistent and keeps operator data on a third-party cloud that can't be
  tailnet-gated (a dev PostHog project would leak from forks/CI).
- **Public Umami edge for the operator viewer** (like the player's
  `analytics.<domain>`). Rejected for now: the viewer has no public domain and is
  tailnet-only; a public edge is unnecessary surface. Revisit at the ADR-116
  public cutover.
- **Serve Umami over plain `http://homelab:3001` in prod.** Rejected:
  mixed-content-blocked on the HTTPS-over-tailscale viewer.

## Consequences

- **Positive:** single self-hosted analytics; dev telemetry visible + isolated;
  fork-safe by construction; no third-party analytics cloud; no public Umami
  surface.
- **Negative / cost:** operator PROD Umami depends on a `tailscale serve` config
  on homelab (runtime state, pinned in the ACL but not otherwise IaC) and a
  tailnet ACL grant that ships via the GitOps ACL action (ADR-128; was `tofu apply`).
  The live operator image must be
  rebuilt to bake the new `VITE_UMAMI_*` build-args.
- **Neutral:** the browser DSN keys + Umami website ids committed in code are
  PUBLIC ids (they ship in the bundle) — safe to commit by design.

## Implementation Notes

- **Viewer analytics:** `web/gi-kg-viewer/src/lib/analytics.ts` (`track()` +
  `EVENT_NAMES`); call sites in `App.vue` + `stores/{search,explore,graphHandoff,shell}.ts`.
- **Dev defaults (tailnet, `homelab`):** GlitchTip `player-dev`(project 8) /
  `operator-dev`(9); Umami `player-dev` / `operator-dev`. Wired in the frontends'
  `main.ts` and `src/podcast_scraper/utils/sentry_init.py` (guarded off under
  pytest/CI/prod).
- **Prod build args:** `docker/viewer/Dockerfile` + `.github/workflows/stack-test.yml`
  take `VITE_UMAMI_SRC` / `VITE_UMAMI_WEBSITE_ID` (GH vars `OPERATOR_UMAMI_*`);
  empty ⇒ silent image.
- **Tailnet HTTPS entry point:**
  `tailscale serve --bg --set-path=/umami http://127.0.0.1:3001` on homelab;
  ACL grant added in `tailscale/policy.hujson` (`autogroup:admin →
  tag:homelab-host:443`), shipped by `infra/terraform/tailscale.tf` on
  `tofu apply`.

## References

- ADR-116 — operator viewer going public (future public analytics edge).
- ADR-0005 — self-hosted observability stack on homelab.
- RFC-082 — tailnet ACL as code (`tailscale/policy.hujson`).
- Supersedes the PostHog telemetry described in ADR-094 / RFC-085 (graph-handoff).
