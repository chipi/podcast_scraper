# ADR-129: Config-layer deploy by surface (decouple config from app image deploys)

- **Status**: Accepted
- **Date**: 2026-07-29
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (shared edge),
  [ADR-121](ADR-121-one-node-alloy-per-box-app-rule-dropins.md) (per-app Alloy drop-ins),
  [ADR-128](ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md) (same "decouple + phase" pattern)

## Context

The **config layer** — declarative, non-image files the box runs — is deployed today as a
**side-effect of full app deploys**:

- Caddy **vhosts** (`infra/caddy/*.caddy`) → `/etc/caddy/sites/` + `systemctl restart caddy`,
  cp'd inside `deploy.sh` / `deploy-player.sh` / `deploy-operator.sh`.
- Alloy **drop-ins** (`infra/observability/*.alloy`) → `/opt/vps-observability/config.d/` +
  `docker kill -s HUP alloy`, also inside those deploys.
- The base `Caddyfile` → `/etc/caddy/Caddyfile`, cp'd by `apply-edge.sh` (root).

Consequence: **changing one config file (a header, a log rule, a vhost) forces a full
privileged-stack redeploy** (docker.sock + LLM keys + pipeline-llm) — disproportionate blast
radius, and no way to ship *just* config. There is also no operator control over *which*
surface's config moves.

## Decision

**A dedicated `deploy-config.yml` deploys the config layer by SURFACE, independently of app
images.** Operator picks the surface; nothing else moves.

**Surface map** (podcast_scraper-owned config only). Names align across surfaces:
`<surface>.caddy` = the app vhost, `<surface>-telemetry.caddy` = GlitchTip ingest,
`<surface>-analytics.caddy` = Umami ingest, `<surface>.alloy` = the log drop-in.

| Surface | Caddy (deploy-writable `/etc/caddy/sites/`) | Alloy | Templating |
| --- | --- | --- | --- |
| **player** — the app + its own ingest | `player.caddy`, `player-telemetry.caddy` (→ GlitchTip), `player-analytics.caddy` (→ Umami) | `player.alloy` | `PLAYER_DOMAIN` + `PLAYER_PREVIEW_COOKIE`; `GLITCHTIP_UPSTREAM` (Caddy env, box-side) |
| **operator** — the app | `operator.caddy` | `operator.alloy` | `OPERATOR_DOMAIN` + `OPERATOR_PREVIEW_COOKIE` |
| **orrery** — orrery's vhosts, owned here for now | `orrery.caddy`, `orrery-telemetry.caddy` (→ GlitchTip), `orrery-analytics.caddy` (→ Umami) | — | none (hardcoded `orrerylearn.com`; `GLITCHTIP_UPSTREAM` env) |

**Ownership principle (operator's rule):** each **app owns its own serving vhost, its own
observability/analytics ingest vhosts, and its own log drop-in** — one aligned naming scheme
per surface (`player-*`, `operator*`, `orrery-*`). There is no shared "infra" surface: the
GlitchTip/Umami *engines* are homelab-owned, but each app's ingest *vhost* rides that app's
surface. The operator surface consumes analytics/telemetry over the homelab tailnet directly,
so it has no ingest vhost of its own.

**orrery is a temporary surface here.** orrery's Caddy routing currently lives + deploys from
podcast_scraper (the shared-edge owner). Orrery's own repo (`agentic-ai-homelab`/orrery)
deploys its *app* + its Alloy drop-in, but has **no** Caddy-vhost deploy today. So for now
`deploy-config` owns and deploys the orrery vhosts as their **own surface** (full isolation —
deploying orrery config never touches player/operator).

**Deferred to a separate task** (do NOT conflate with this ADR): moving the orrery vhosts to
the orrery repo (its routing per orrery's own edge-ownership ADR — requires building a vhost
deploy there).

**Mechanics:** the workflow templates each vhost in the runner (the domain rewrite + the
`__*_PREVIEW_COOKIE__` secret, exactly as the deploy scripts' `sed`), scp's to the box, cp's
into `/etc/caddy/sites/`, runs `caddy validate` (with the box's `CADDY_BIND_ADDRS` /
`GLITCHTIP_UPSTREAM` systemd env), and **restarts caddy only if valid** (rollback the vhost
files if not — caddy never restarts on a broken config). Alloy drop-ins scp to `config.d/` +
HUP. A `dry_run` toggle stages + validates without restarting. Gated: `environment: prod` +
typed `DEPLOY_CONFIG`.

**The base `Caddyfile` stays in `apply-edge.sh`.** It is root-owned (`deploy@` can write
`/etc/caddy/sites/` but not `/etc/caddy/Caddyfile`), and it is bundled with the host-level
systemd env (`CADDY_BIND_ADDRS`, `GLITCHTIP_UPSTREAM`) that `apply-edge` also manages. It is
edge-engine config, not an ingest vhost — the clean permission + ownership boundary keeps it
with the host-hardening path.

**Phasing (mirrors ADR-128 — add, prove, then remove):**

- **Phase 1 (this ADR):** add `deploy-config.yml` — additive; the app deploys keep cp'ing
  config as before, so nothing breaks while the new path is proven.
- **Phase 2 (follow-up):** remove the vhost/alloy cp side-effects from `deploy.sh` /
  `deploy-player.sh` / `deploy-operator.sh`, leaving them to deploy only app images.

## Consequences

**Positive:** a config change ships in seconds via one gated, surface-scoped workflow — no
image, no compose recreate, no full-stack redeploy, minimal blast radius. Operator has full
control (pick `player` / `operator` / `orrery` / `all`). One place for config-layer deploys.

**Negative:** the vhost `sed` templating is now in two places until Phase 2 (the workflow +
the app scripts) — a divergence risk, mitigated by Phase 2 removing it from the scripts.
The base Caddyfile remains a separate path (`apply-edge`) — "Caddy config" isn't 100% in one
place, but the permission boundary makes that the correct split.

**Neutral:** orrery is a surface here **for now** — a temporary home until its routing moves
to the orrery repo (the deferred task); deploying it is fully isolated from the podcast surfaces.

## Alternatives considered

- **Deploy by config-type** (`target: caddy|alloy|all`) instead of by surface. Rejected —
  the operator wants surface isolation (ship player's log rule without touching operator).
- **Include the base Caddyfile** in `deploy-config`. Rejected for Phase 1 — needs a new root
  sudoers grant for `/etc/caddy/Caddyfile`; `apply-edge` already owns it correctly.
- **Three separate workflows** (one per surface). Rejected — one workflow with a `surface`
  input is one place to operate with the same isolation.

## References

- `.github/workflows/deploy-config.yml` · `infra/caddy/*.caddy` · `infra/observability/*.alloy`
- [EDGE_CONVERGENCE_RUNBOOK](../guides/EDGE_CONVERGENCE_RUNBOOK.md) (the base Caddyfile / host path)
