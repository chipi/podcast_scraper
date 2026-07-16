# Goal-1 critical path — orrery on minimal common infra

- **Status**: Active (scope-of-focus tracker)
- **Date**: 2026-07-09

The operator's real goal order (2026-07-09):

1. **orrery goes live** on top of **minimal common infra**. Once the common infra is done,
   orrery adds its part (orrery#381) and we launch orrery.
2. **podcast player** (the consumer app).
3. **podcast operator surface** (the kg/gi console).

**Focus is Goal 1.** A large part of what was built is Goal-2 (player) work — done + tested
but **parked**, off the Goal-1 path. This doc pins the trimmed scope so we don't
live-apply or keep polishing player-specific pieces before orrery.

## What Goal 1 needs (the lean list)

The common infra is **built**; the remaining work is mostly *application*, not building:

- [ ] **Live rollout** — apply the edge + firewall 80/443 + host hardening to the running
      box (step-by-step, backup-verified-first; see the rollout options + data-loss guards
      in `project_shared_public_edge` memory / the future rollout runbook).
- [x] **T-11 security alerting** (+ observability GitOps) — common plane built (ADR-117):
      Alloy security-log pipeline + `config/grafana/alerts/common/security.yaml` + `make
      obs-sync`. Fires once the box + log sources are live.
- [ ] **orrery vhost coordination** (#1158 ↔ orrery#381) — orrery drops its `<app>.caddy`
      once our Caddy engine is live.
- [ ] **T-12** HSTS/CSP verify on orrery's first live vhost.
- [x] **fail2ban Caddy-log jail** — built: `caddy-access` filter+jail bans an IP on a burst
      of 30 4xx / 2m (path scans, auth probes, 429 abuse) → 1h ban in the INPUT chain. Caddy
      log pinned to JSON+ISO8601 so the filter is version-stable. Applies on rebuild / live.
- [x] **T-05** Cloudflare (D4 accepted) — origin prep landed (ADR-118): Caddy CF real-IP
      trust, CF-safe fail2ban `client_ip` key, optional `cloudflare_origin_lock` TF firewall.
      Operator does the CF account/DNS + flips the lock per the ADR rollout steps.

## Built = common infra (ready, needs live application)

Caddy edge (ADR-114) + firewall (TF) + host hardening (Phases 1–3: SSH/fail2ban/metadata)
+ threat model + pre-public gate. All committed on `production`; `caddy validate` /
structure tests green; **not yet live**.

## Parked = Goal-2 player (built + tested, NOT on the Goal-1 path)

Do **not** live-apply these for orrery:

- `PODCAST_SERVE_APP_ONLY` app-only backend + its tests
- `compose/docker-compose.player-public.yml`, player nginx (path-filter + rate-limit)
- `deploy-player.sh` / `deploy-player.yml`, `backup-player-appdata-prod.yml`
- OAuth provider switch, forwarded-header handling, player runtime e2e

## Deferred = Goal-3 operator surface

RBAC across `/api/*` (#1164) + the public/control privilege-split (#1165, ADR-116). Design
only; kg/gi stays tailnet until built.

## Not needed for Goal 1

- **ADR-115 secret delivery** — orrery is a static PWA (no backend secrets).
- **B+C cutover**, **OpenBao** (#1162) — secret-infra tail, Goal-2+.
