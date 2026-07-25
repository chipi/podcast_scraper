# GitHub Actions → observability (CI/ops events)

Everything that ships **CI/ops telemetry from GitHub Actions** into the self-hosted
observability stack lives here: the emit tooling, the Grafana dashboard, and the
roadmap. App telemetry (traces/logs/Umami from the running VPS) lives elsewhere —
this folder is only the **GitHub-sourced** signal.

## Why

GH runners are ephemeral and don't ship their logs anywhere. To see deploy/backup
health (and, in Tier 2, CI/DORA trends) next to app telemetry on one Grafana pane,
workflows **push discrete events** to the homelab sink over the tailnet.

Design follows [ADR-119](../../../docs/adr/ADR-119-vendor-neutral-event-emission.md)
(vendor-neutral event emission) and [ADR-117](../../../docs/adr/ADR-117-multi-tenant-observability-gitops.md).

## Sink

- **VictoriaLogs** on the homelab: `http://homelab:9428`, ingest `POST /insert/jsonline`.
- **Tailnet-only, no token** — auth is the network (VictoriaLogs is not publicly exposed).
- Reachability: `tag:gha-deployer → tag:homelab-host:9428` ACL grant
  ([`tailscale/policy.hujson`](../../../tailscale/policy.hujson)).
- URL is the repo **variable** `HOMELAB_VICTORIALOGS_URL` (not a secret — it's a
  tailnet hostname). Empty ⇒ emit is skipped (non-fatal).

## Event schema (`ops_event/v1`)

Canonical envelope, one JSON object per event:

| field | kind | example |
| --- | --- | --- |
| `_time` | VL time field | `2026-07-25T09:10:10Z` |
| `schema` | version | `ops_event/v1` |
| `event_type` | **stream** | `deploy` \| `backup` |
| `app` | **stream** | `podcast_scraper` |
| `env` | **stream** | `prod` \| `test` |
| `status` | field | `success` \| `failure` |
| `surface` | field | `operator` \| `player` \| `corpus` \| `player-appdata` |
| `duration_ms` | field | `42000` |
| `dry_run` | field | `true` \| `false` |
| `sha` / `triggered_by` | field | `7da782a` / `octocat` |
| `_msg` | summary | `prod deploy success sha-7da782a` |

Stream fields are low-cardinality (`{app, env, event_type}`); everything else is a
searchable field. LogsQL does numeric range filters on the string-stored numbers
(`duration_ms:>60000`).

## Emit tooling

- [`scripts/ops/emit_ops_event.sh`](../../../scripts/ops/emit_ops_event.sh) — the
  emitter. `--event-type T [--env E] [--msg M] [--field k=v ...]`, `VICTORIALOGS_URL`
  in env. Non-fatal.
- [`.github/actions/emit-ops-event`](../../../.github/actions/emit-ops-event) — a
  composite action wrapping the script (DRY, injection-safe, non-fatal).

## Wired workflows (Tier 1 — push)

Only workflows that **already join the tailnet** push directly (zero added join cost):

| workflow | event_type | when |
| --- | --- | --- |
| `deploy-prod` | `deploy` (`surface=operator`) | always (success + failure) |
| `deploy-player` | `deploy` (`surface=player`) | always |
| `backup-corpus-prod` | `backup` (`surface=corpus`) | always |
| `backup-player-appdata-prod` | `backup` (`surface=player-appdata`) | always |

## Dashboard

[`dashboards/ci-ops-overview.json`](dashboards/ci-ops-overview.json) — **CI / Ops —
GitHub Actions** (`uid: podcast-ci-ops-overview`). Grafana `http://homelab:3000`,
datasource `victorialogs`. Panels: deploy/backup counts, deploy failures, avg deploy
duration, event volume by type, deploys-by-status over time, a type×surface×status
breakdown table, and a live event tail. `Env` template var (Prod/Test/All).

> Provisioning: this dashboard is dashboard-as-code like the app dashboards in
> [`../grafana/`](../grafana/); the homelab Grafana provisioning should include this
> `dashboards/` path (or import the JSON once).

## Not here — Tier 2 (pull / DORA)

CI outcomes, `infra-drift`, and drill **cycle** results are **not** pushed — those
workflows don't join the tailnet (or the orchestrator's finalize job doesn't), and
adding an OAuth join per run is a poor trade. They'll be captured by a **homelab-side
GitHub Actions API poller** → VictoriaLogs, which also yields DORA metrics (deploy
frequency, lead time, CI pass-rate, flaky-rate, queue time). That poller lives in the
homelab repo (operator-owned); this folder will gain its dashboard when it lands.
