# Alloy double-ship of pipeline container logs — fix instructions (2026-08-25)

## Finding (verified live)

Every pipeline-container log line lands in VictoriaLogs TWICE (box log file holds it
once): two Alloy docker-sources both capture pipeline containers on prod-podcast:

1. `operator.alloy` (podcast repo, `infra/observability/operator.alloy`) — canonical:
   labels `app=podcast, surface=pipeline, container=<name>`.
2. `base.alloy` `loki.source.docker "pipeline"` block — legacy: label
   `job=podcast-pipeline`. Source of truth:
   `agentic-ai-homelab/infra/observability/hosts/prod-podcast/config.d/base.alloy`
   (~line 90-104); on-box copy `/opt/vps-observability/config.d/base.alloy` (root:root).

Evidence: event ts `2026-08-25T06:10:43.110957` — 1 hit in the box log, 2 in VL with
the two `_stream` label sets above. Every cost/consumption dashboard summing log events
double-counts until fixed.

## Fix (two halves)

- PODCAST repo (done on `feat/enrichment-telemetry-hardening`): dashboards' llm_cost
  exprs repointed `job:podcast-pipeline` → `app:podcast AND surface:pipeline`.
  Remaining references: the dashboards' template-variable lists still mention
  `podcast-pipeline` as a job filter option — harmless post-removal, tidy later.
- HOMELAB repo (needs operator/root): delete the `discovery.relabel "pipeline"` +
  `loki.source.docker "pipeline"` blocks from
  `infra/observability/hosts/prod-podcast/config.d/base.alloy`, re-stage to
  `/opt/vps-observability/config.d/base.alloy` (root-owned), then
  `docker compose up -d alloy` in `/opt/vps-observability`.
  NOTE: the on-box copy has drifted from the homelab repo copy
  (`forward_to = logs_sink` vs `homelab_std`) — reconcile while touching it.

## Post-fix verification

Pick any fresh llm_cost line's ts from the newest reprocess log; VL count for that ts
must be exactly 1, stream carrying `app=podcast, surface=pipeline`.
