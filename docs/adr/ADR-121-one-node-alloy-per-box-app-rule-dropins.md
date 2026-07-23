# ADR-121: One node Alloy per box + per-app label-rule drop-ins

- **Status**: Accepted
- **Date**: 2026-07-23
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**: [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (the
  Caddy shared-edge pattern this mirrors), [ADR-117](ADR-117-multi-tenant-observability-gitops.md)
  (multi-tenant o11y — this refines its log-collection mechanism),
  [ADR-119](ADR-119-vendor-neutral-event-emission.md) (emit/ship split)
- **Tracking**: orrery log-mislabel incident (2026-07-23); #1158 edge programme

## Context

The prod VPS runs **two** log-shipping agents side by side:

1. **`alloy`** (grafana/alloy v1.x) — the podcast/box agent. Reads the Docker socket,
   journal, and Caddy access log; ships to homelab VictoriaLogs. Modern, keep-filters
   work. Config lives at `/opt/vps-observability/config.alloy` — **untracked** (in no repo).
2. **`orrery-grafana-agent`** (grafana/agent **v0.43.4**) — orrery's own agent, a
   leftover from the Grafana-Cloud era (Grafana-Cloud-named env vars), later repointed
   at homelab. grafana-agent is **EOL (Nov 2025)**.

Two problems this caused:

- **Mislabeling (the incident):** both agents can see *every* container via the shared
  Docker socket. Orrery's agent has a `keep` filter for `orrery-*`, but v0.43.4's
  `docker_sd` doesn't reliably drop non-matching containers — so it also tailed the
  podcast API + cAdvisor logs and stamped them `app=orrery`. ~48% of `app=orrery`
  volume was actually prod-podcast infra. (The podcast Alloy was innocent — it never
  sets `app=orrery`; the label fingerprint `{app=orrery, env=prod}` with no
  `instance/cluster` proved the source was orrery's agent.)
- **Redundancy + drift:** two agents on one box (both read the socket), one on dead
  software, and the box Alloy config is deploy-only drift.

The **root pattern** is wrong: each app running its own node-level log agent that
scrapes the shared socket and self-filters. That is exactly the anti-pattern
ADR-114 solved for the public edge (one shared Caddy engine; apps drop vhosts).

## Decision

**One Alloy per box (the "node agent" / router); each app contributes its own
label rules as a drop-in from its own repo. Retire per-app log agents.**

Concretely, mirroring the ADR-114 Caddy edge:

| Caddy edge (ADR-114) | Alloy node agent (this ADR) |
| --- | --- |
| One shared Caddy engine per box | One shared Alloy per box |
| Base `Caddyfile` (infra-owned, repo-tracked) | Base `base.alloy`: `discovery.docker "all"` (socket reader) + `loki.write "sink"` (→ homelab) + journal + caddy sources |
| Drop-in dir `/etc/caddy/sites/*.caddy` (deploy-writable) | Config dir `/etc/alloy/config.d/*.alloy` (deploy-writable) — Alloy runs against the **directory** and merges all files into one config |
| App drops `sites/<app>.caddy` | App drops `config.d/<app>.alloy` |
| `sudo systemctl reload caddy` (narrow grant) | `docker kill -s HUP alloy` (deploy is in the docker group — NO sudoers grant needed) |

**Mechanism.** Alloy natively loads a *directory* (`alloy run /etc/alloy/config.d/`)
and merges every `.alloy` file into one namespace, so an app fragment can reference
the shared components. Each app fragment:

```alloy
// <app>.alloy — lives in the APP's repo, dropped on the app's deploy
discovery.relabel "<app>" {
  targets = discovery.docker.all.targets            // the shared router's output
  rule { source_labels = ["__meta_docker_container_name"], regex = "/<app>-.*", action = "keep" }
  rule { source_labels = ["__meta_docker_container_name"], target_label = "container", regex = "/(.*)", replacement = "$1" }
}
loki.source.docker "<app>" {
  host       = "unix:///var/run/docker.sock"
  targets    = discovery.relabel.<app>.output       // only this app's containers
  forward_to = [loki.write.sink.receiver]           // the shared sink
  labels     = { app = "<app>" }                    // the app's own label
}
```

**Ownership:** infra owns `base.alloy` (repo-tracked); each app owns its `<app>.alloy`
(in its own repo, delivered by its own deploy). Same split as the vhosts.

## Consequences

**Positive**

- One agent per box — no redundant socket readers, no dead software (grafana-agent gone).
- Labels correct by construction: `app=<x>` is only on that app's scoped source, so the
  cross-app leak is structurally impossible.
- Apps own their rules in their own repos, plugged in via drop-ins — consistent with
  the Caddy edge; the orrery/homelab agents have a clear contract.
- The base config becomes repo-tracked (kills the `/opt` drift).

**Negative / neutral**

- **Trust:** a deploy-writable `config.d/` lets any app drop-in define *any* Alloy
  component (Alloy can exec/read files) — same trust surface as apps dropping Caddy
  vhosts. Acceptable at single-operator scale; revisit if tenants aren't mutually trusted.
- One mechanical wrinkle vs Caddy: Alloy's directory-merge (not a dumb glob) means a
  malformed fragment fails the whole config — validate (`alloy fmt`/`alloy run --dry-run`
  equivalent) before reload, and reload keeps the last-good config running.
- Cross-file component references require unique component labels per app (namespace by
  app name) to avoid collisions.

## Alternatives considered

- **Each app keeps its own agent (just modernize orrery's to Alloy).** Fixes the leak
  but keeps two agents + the redundant socket reads, and doesn't establish the pattern.
  Rejected — treats the symptom, not the anti-pattern.
- **Infra owns all app rules** (all fragments in the infra repo, like `orrery.caddy`
  currently is). Simpler delivery, but the app no longer owns its o11y — breaks the
  tenant boundary and forces an infra PR for every app label change. Rejected.
- **Keep grafana-agent, add a hard drop-empty-container rule.** A band-aid on EOL
  software; leaves the redundancy + drift. Rejected.

## Migration (execution checklist)

1. **Base:** move `/opt/vps-observability/config.alloy` into a repo (agentic-ai-homelab
   `hosts/prod-podcast/`), split the shared router bits into `base.alloy`, point the
   Alloy unit at `/etc/alloy/config.d/`, drop `base.alloy` there.
2. **Drop-in dir + grant:** create `/etc/alloy/config.d/` (deploy-writable) + a narrow
   `systemctl reload alloy` sudoers grant (mirror `99-caddy-reload`); add both to
   cloud-init + `apply-edge`.
3. **Orrery:** add `ops/observability/orrery.alloy` to the orrery repo; orrery's deploy
   `scp`s it to `config.d/` + reloads Alloy; **delete** `orrery-grafana-agent` from
   orrery's compose.
4. **Verify:** in VictoriaLogs, `app=orrery` carries only orrery-web/pipeline (no bare
   `{app=orrery,env=prod}` infra logs); podcast logs unchanged.
5. **Player (later):** ships `player.alloy` the same way when it launches.
