# Handover → homelab agent: CI/ops observability (Tier 1 done, Tier 2 to build)

**From:** podcast_scraper-infra · **Date:** 2026-07-25 · **Repo for homelab work:** `agentic-ai-homelab`

Tier-1 CI/ops observability shipped in `podcast_scraper` (#1297): the 4 already-on-tailnet
prod workflows (deploy-prod/player, backup-corpus/player-appdata) now push canonical
`ops_event/v1` events to **VictoriaLogs `:9428`** over the tailnet, and there's a
dashboard-as-code. Two things are **homelab-side** and need you.

## Event schema (already flowing)

`POST http://homelab:9428/insert/jsonline`, tailnet-only, no token. Envelope:

```json
{"_time":"…Z","schema":"ops_event/v1","event_type":"deploy|backup","app":"podcast_scraper",
 "env":"prod","status":"success|failure","surface":"operator|player|corpus|player-appdata",
 "duration_ms":41000,"sha":"…","dry_run":false,"triggered_by":"…","_msg":"…"}
```

Stream fields `{app, env, event_type}`. ACL grant `tag:gha-deployer → tag:homelab-host:9428`
is live.

## TODO 1 — provision the dashboard (small)

The dashboard is version-controlled at
`podcast_scraper:infra/observability/github-actions/dashboards/ci-ops-overview.json`
(uid `podcast-ci-ops-overview`, datasource `victorialogs`). I API-imported it into the
**General** folder for validation. Please add it to Grafana **provisioning**
(`.../grafana/provisioning/dashboards/dashboards.yml`, `foldersFromFilesStructure`) so it
lands in a proper folder (suggest **"CI / Ops"**) and is git-managed like the app
dashboards. Then delete the General-folder API copy.

## TODO 2 — Tier 2: DORA poller (the real ask)

Tier-1 only covers workflows that already join the tailnet. **CI, infra-drift, and drill
cycle-level** results don't (adding an OAuth join per CI run is a poor trade). Capture them
by **pull**, not push:

- A scheduled homelab job (cron / systemd timer, ~every 15 min) hits the **GitHub Actions
  API** (`GET /repos/chipi/podcast_scraper/actions/runs?created=>…`) with a read-only token.
- For each run, emit an `ops_event/v1` line to VictoriaLogs `:9428` (same schema, new
  `event_type`s: `ci_run`, `drift`, `drill`) with `status` (conclusion), `duration_ms`
  (updated_at − run_started_at), `workflow`, `branch`, `event` (push/PR/schedule), `sha`.
- Dedup on `run_id` (store last-seen; VictoriaLogs is append-only, so don't re-emit).
- **DORA** falls out of this: deploy frequency + lead time (from `deploy` events, Tier-1),
  change-failure-rate (deploy `status:failure` ÷ total), MTTR (already have deploy events),
  and CI health (pass-rate, flaky-rate = reruns, queue time = `run_started_at − created_at`).

Then a **Tier-2 dashboard** (DORA panels) — I can draft the JSON if you want; it queries the
same `victorialogs` datasource.

**Why homelab, not this repo:** the poller is a standing service with a GH token, owned by
your stack; keeping it in `agentic-ai-homelab` (GPG-signed) matches ownership.

## Pointers

- Emit contract + full docs: `podcast_scraper:infra/observability/github-actions/README.md`
- Emitter (reuse the exact JSONL shape): `scripts/ops/emit_ops_event.sh`
- Tracking issue: (Tier-2 issue link — see podcast_scraper issues)
