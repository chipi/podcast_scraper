# ADR-117: Multi-tenant observability — common box/edge plane + per-tenant app telemetry, GitOps

- **Status**: Proposed
- **Date**: 2026-07-09
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**: [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (edge
  ownership split this mirrors), [ADR-115](ADR-115-multi-tenant-secret-delivery-sops-tmpfs-files.md)
  (secret delivery), [ADR-096](ADR-096-dgx-spark-prod-primary-with-fallback.md) / RFC-081 (o11y layer)
- **Security SSOT**: [Threat model](../security/THREAT_MODEL.md) **T-11** (detection gap) — this ADR is its home
- **Tracking**: #1160 (hardening); #1158 / orrery#381 (edge programme)

## Context

The VPS ships observability — Alloy (host metrics), grafana-agent (api/DGX metrics +
container logs → Grafana Cloud Prom/Loki), Sentry (app errors), dashboards in
`config/grafana/`. Two gaps:

1. **No security detection (T-11)** — sshd/fail2ban/**Caddy access log** are host services,
   so the container-log tail never sees them, and no alert rules exist.
2. **Not tenant-aware** — the box is going multi-tenant (orrery, then the podcast player,
   then the operator UI). o11y must follow the **same ownership split** the edge (ADR-114)
   and secrets (ADR-115) already use: **infra owns the engine; each tenant brings its part.**

## Decision

### 1. Two planes

- **Common (box/edge) — owned by infra.** The shipping pipeline (Alloy/agent → Grafana
  Cloud), the Grafana Cloud + Sentry connection, the **GitOps sync mechanism**, and
  detection for the **shared substrate**: host metrics, sshd/fail2ban/Caddy-access-log
  security logs, firewall. This is **T-11 for the box + edge**.
- **Per-tenant (app) — owned by each tenant.** Its app metrics/logs, its dashboards +
  alert rules, its Sentry project, its app-level detection.

### 2. The four cases (initial)

| `tenant` label | Kind | Ships / watches | Account (§3a) |
| --- | --- | --- | --- |
| `common` | generic infra / box + edge | host metrics; sshd/fail2ban/Caddy/firewall security logs + alerts; box health | **infra's own** (always) |
| `orrery` | static PWA (Goal 1) | its Caddy vhost access/error logs, uptime; minimal | start in `common`; own account if it wants full independence |
| `podcast-player` | consumer app (Goal 2) | app-only api metrics, `/api/app` latency, auth-fail rate, rate-limit `429`s | its own |
| `podcast-operator` | kg/gi operator UI (Goal 3, tailnet) | operator api metrics, jobs/pipeline, control-plane | its own |

### 3. Convention (the load-bearing mechanism)

Mirrors `import /etc/caddy/sites/*.caddy` and per-tenant `secrets.enc.yaml`:

- **Every signal carries a `tenant=<name>` label** (relabel at the Alloy/agent scrape/log
  stage; each app exposes its own `tenant`). The label is the routing key — to a folder
  within one account, or to a separate per-tenant account (see §3a).
- **Grafana: one folder per tenant** *within a given account* (`Common (Box & Edge)`,
  `orrery`, …); dashboards + alert rules live under the tenant's folder.
- **Sentry: one project per tenant** (in the tenant's own account/org — see §3a).
- A tenant leaving = drop its folder/project (or point-of-presence in its account) + stop
  its label; nothing shared changes (clean removal, like orrery-can-move-to-its-own-VPS).

### 3a. Account granularity — a per-tenant choice

Each tenant may live in a **shared account** (folder/project isolation) **or its own
account** (full isolation). `common` (box/edge/security) is **always infra's own account**.
The two backends differ:

- **Sentry — per-account is the default.** Sentry isolates by DSN → project → org, so a
  tenant using **its own DSN** already lands in **its own Sentry account**. No special
  routing; each tenant configures its DSN. `common`/infra has its own (or none).
- **Grafana — the shipping pipeline routes by the `tenant` label.** Alloy/agent fan out to
  **per-tenant `remote_write` / `loki.write` targets** (each = that tenant's Grafana Cloud
  endpoint + token): `tenant=common` → infra's account, `tenant=orrery` → orrery's, etc.
  A tenant that shares `common`'s account just gets a folder instead of a separate target.

**Trade-off (recorded in Consequences):** separate accounts give stronger isolation +
independent billing/access/retention + clean leave, at the cost of more credentials/routing
and **harder cross-account correlation** — when the shared box has an incident, `common`
(infra account) and the affected tenant's app (its account) are in *different* Grafanas.

**Initial placement:** `common` = infra account (always). **orrery** (static PWA — its
signal is mostly the shared Caddy's view of its vhost) starts in `common` and graduates to
its own account only if it wants full independence. **podcast-player / podcast-operator**
(real backends) → their own accounts.

### 4. GitOps mechanism (config-as-code → upload)

Option A — **config-as-code + API sync scripts, no new dependency** (curl/jq already in
use; matches the vanilla / minimal-infra ethos and the "version here, upload there" ask):

```text
config/obs/tenants.yaml                        # tenant -> {grafana: shared|<account-id>, sentry_org, folder}
config/grafana/dashboards/<tenant>/*.json      # dashboards (existing ones move under common/)
config/grafana/alerts/<tenant>/*.json          # alert rules (Grafana alerting provisioning fmt)
config/grafana/contact-points/*.json           # where alerts route (operator supplies targets)
config/sentry/<tenant>/alerts.yaml             # Sentry issue-alert rules per project
scripts/obs/grafana_sync.sh                     # per tenant: POST folder+dashboards+rules -> that tenant's Grafana account
scripts/obs/sentry_sync.sh                      # per tenant: POST alert rules -> that tenant's Sentry org/project
```

`make obs-sync` loops the tenants in `tenants.yaml` and syncs each to **its own account**
(or the shared `common` account for tenants configured that way) — idempotent upsert. Each
tenant's token (`GRAFANA_<TENANT>_TOKEN`, `SENTRY_<TENANT>_TOKEN`) comes from env/GH
secrets — never committed. A workflow can run `obs-sync` on merge.

### 5. Ownership split

| Piece | Owner | Lives in |
| --- | --- | --- |
| Shipping pipeline (Alloy/agent), Grafana Cloud + Sentry connection | infra | this repo cloud-init + compose |
| GitOps sync mechanism (`scripts/obs/*`, `make obs-sync`) | infra | this repo |
| `common` dashboards + **security alert rules** (T-11: sshd/fail2ban/Caddy/firewall) | infra | `config/grafana/{dashboards,alerts}/common/` |
| Security-log shipping (sshd/fail2ban/Caddy → Loki) | infra | Alloy config (cloud-init) |
| **Per-tenant** dashboards + alert rules + Sentry project | **each tenant** | that tenant's repo, synced via the shared mechanism |
| Contact points / routing | operator | `config/grafana/contact-points/` |

## Consequences

**Positive**

- Same mental model as the edge + secrets: infra owns the engine, tenants plug in — one
  consistent multi-tenant story across ingress, secrets, and observability.
- Tenant-scoped from day one → no painful retrofit when the player/operator o11y arrive.
- T-11 gets a home; security detection is versioned + reviewable, not clicked-in.
- Vanilla + no new dependency; you see the actual Grafana/Sentry APIs.

**Negative**

- Hand-rolled sync idempotency (upsert-by-uid) vs a provider's managed state; must handle
  "delete a rule" explicitly (prune step) so removed config actually disappears.
- Alert rules only fire once the box + its log sources are live (fail2ban/Caddy).
- **Separate accounts (§3a) make cross-account correlation harder** — when the *shared box*
  has an incident, `common` (infra account) and the affected tenant's app (its own account)
  are in different Grafanas, so correlating spans two logins. Mitigations: keep the `common`
  host/edge signal rich enough to diagnose box-level issues alone; a tenant that needs tight
  box↔app correlation can choose the shared `common` account instead.
- More credentials + pipeline routing (per-tenant `remote_write`/`loki.write` targets +
  per-tenant sync tokens) than a single shared account.

**Neutral**

- Existing dashboards move under `config/grafana/dashboards/common/` (or their tenant).
- Grafana Cloud + Sentry remain the SaaS backends (no self-hosting).

## Alternatives considered

- **Terraform providers (grafana/grafana + a Sentry provider)** — robust, drift-detecting,
  `apply` uploads, and the repo already runs OpenTofu. Rejected *for now*: two new provider
  deps + another state file; heavier than the "just upload" ask. Kept as the upgrade if
  strict drift-detection is later wanted.
- **Grizzly (`grr`)** — purpose-built Grafana-as-code, but a new tool and Grafana-only
  (Sentry still needs the API path). Rejected.
- **Single-plane, non-tenant o11y** — simplest, but bakes in a retrofit the moment a second
  tenant needs its own dashboards/alerts/routing. Rejected — contradicts the whole
  multi-tenant direction.

## Sequencing

Goal 1 builds the **common plane only** (GitOps mechanism + Alloy security-log pipeline +
common security alert rules + the tenant-scoping scaffold), plus **orrery's minimal part**.
`podcast-player` and `podcast-operator` o11y land with Goals 2 and 3.
