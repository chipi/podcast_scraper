# RFC-083: Production Failover — Orchestration, Spare Stack, and Traffic Cutover

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Operator (infra), on-call (future)
- **Related PRDs**: —
- **Related ADRs** (decisions extracted from this RFC; ratified as **Accepted**):
  - [ADR-089](../adr/ADR-089-prod-failover-orchestrator-separate-from-drill.md) — orchestrator boundary vs DR drill
  - [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md) — DNS-first cutover on tailnet
  - [ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md) — GHA triggers and safety gates
- **Related RFCs**:
  - [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) — always-on hosting, drill workspace
  - [RFC-084](RFC-084-corpus-backup-manifest-and-version-aware-restore.md) — snapshot manifest and
    newest-compatible restore (**Completed** on this branch; [#763](https://github.com/chipi/podcast_scraper/issues/763))
- **Related documents**:
  - [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764) — problem statement, DNS path, runbook intent
  - [GitHub #762](https://github.com/chipi/podcast_scraper/issues/762) — stack contract vs adapters ([ADR-093](../adr/ADR-093-canonical-stack-contract-and-environment-adapters.md); operator hub under `docs/guides/STACK_CONTRACT.md`)
  - [ADR-081](../adr/ADR-081-drill-opentofu-workspace-tailscale-acl-ownership.md), [ADR-082](../adr/ADR-082-gitops-app-deploy-via-stack-test-and-gha.md), [ADR-083](../adr/ADR-083-tailscale-private-ingress-always-on-vps.md), [ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)
  - [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) — shared restore entry points
  - [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md) — tailnet health, `tailscale serve` (dedicated failover chapter still open)

## Abstract

When **production** is unhealthy or untrusted, we need a **repeatable** path to **stand up a spare** environment (same stack contract as prod), **restore corpus**, **validate** (including browser/API gates), **move traffic** to the spare, then **fail back** and decommission—**without** tearing down the spare at the end of the run (unlike **`drill-exercise`**).

This RFC specifies **GitHub Actions orchestration** (manual and, later, automated triggers), **phased gates**, **DNS-first cutover** on the tailnet, **OpenTofu prerequisites** for a distinct spare host, and **explicit non-goals** for v1. **ADRs 089–091** lock the architectural decisions; this RFC holds the **full technical design** and implementation checklist.

## Problem Statement

**DR drill** ([`drill-exercise.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/drill-exercise.yml)) optimizes for **proof** and **cost**: it ends in **`drill-infra-destroy`**. That is correct for drills and wrong for a **live incident**, where the spare must **stay up** until failback.

Today we lack:

1. A **documented and automatable** sequence from “declare incident” to “traffic on spare” aligned with **#764**.
2. A **single CI entry point** that operators trust under stress (instead of ad hoc SSH + DNS).
3. A clear split between **stand up + validate** (safe to drive from automation) and **cutover** (high blast radius).

**Impact:** slower recovery, higher error rate, and risk of confusing **drill** workflows with **prod failover**.

## Goals

1. **Orchestrated GHA parent workflow** (new family, names in implementation PR) with phases: provision spare infra (when Terraform supports it) → deploy app to spare → restore corpus → validate (HTTPS + `tests/stack-test` subset pattern as on drill) → **gated** cutover → (later) failback job family.
2. **Manual trigger** via `workflow_dispatch` with **typed confirms** per dangerous phase.
3. **Optional automated trigger** via `repository_dispatch` (monitoring → GitHub) with **secret-verified** payload; v1 **does not** auto-run DNS cutover without human approval path (see [ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md)).
4. **DNS-first cutover** for the **canonical tailnet hostname** (`https://<tailnet_hostname>.<tailscale_tailnet>/` per
   `infra/terraform/outputs.tf`), with TTL/TLS prerequisites documented in runbooks ([ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md)).
   Issue **#764** frames this as a **stable hostname** flip; in this repo that is **MagicDNS on the tailnet**, not a public
   internet A/AAAA record ([ADR-083](../adr/ADR-083-tailscale-private-ingress-always-on-vps.md)).
5. **Spare footprint** reuses the existing **drill** Hetzner project, OpenTofu **`drill`** workspace, and drill Actions secrets
   (**`HCLOUD_TOKEN_DRILL`**, deploy/restore keys) — **no** new cloud project or token family ([ADR-089](../adr/ADR-089-prod-failover-orchestrator-separate-from-drill.md)).
6. **Same stack contract** on spare as prod ([ADR-093](../adr/ADR-093-canonical-stack-contract-and-environment-adapters.md)): compose
   overlays, in-container `/api/health` on port **8000**, and `tests/stack-test` semantics for behavioral gates.
7. **Corpus restore on spare** uses the **same** `scripts/ops/corpus_snapshot/` and **newest-compatible** tag selection as prod/drill
   ([ADR-092](../adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)); explicit `backup_tag` override remains
   available under incident stress.
8. **Operator-owned timing** for cutover, failback, and spare teardown: automation **prepares** and **validates**; it does **not**
   schedule decommission or unattended traffic moves ([ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md)).

## Non-goals (v1)

- **Scheduled warm spare** that runs full stack on a cron without an incident (separate future RFC/issue).
- **Fully unattended DNS cutover** on first external alert.
- **Automated spare teardown** at the end of a failover run (decommission stays a **manual** operator action).
- **Multi-region** active-active.

## Constraints & Assumptions

- **Tailnet-first prod** per existing ops docs; public ingress is out of scope for cutover mechanics.
- **Incident concurrency (split-brain):** treat **scheduled feed ingestion** as the main dual-writer risk. Routine ingestion is **not** continuous — it fires only from **`scheduled_jobs:`** in the corpus **`viewer_operator.yaml`** (in-process APScheduler; see [SERVER_GUIDE.md](../guides/SERVER_GUIDE.md)). **Pragmatic policy:** before cutover, **freeze** the prod instance (disable schedules / stop in-process scheduler); on the spare, **do not** enable schedules on first bring-up even when the restored backup had **`enabled: true`** entries. Manual **`POST /api/jobs`** remains possible on either side — operators avoid overlapping manual runs during overlap.
- **OpenTofu / Hetzner**: spare is the existing **`drill`** workspace footprint in the **same** Hetzner project and token scope as
  **`drill-exercise`** (**ADR-081**, **ADR-089**). On demand, **`drill-infra-apply`** stands up the throwaway VPS; there is **no**
  always-on spare and **no** second server in prod OpenTofu state. ACL rows for **`tag:dr-drill`** / drill MagicDNS remain owned by the
  prod workspace per **ADR-081**.
- **Secrets**: reuse drill Actions secrets for spare bring-up, deploy, and restore; cutover and teardown remain **manual** operator steps
  (runbook + optional typed confirms), not new secret families.

### Branch status (2026-05-13)

**Landed (this branch, not prod-failover-specific):**

- Corpus snapshot manifest, validation, and **newest-compatible** restore (**RFC-084** / **ADR-092**); prod/drill restore workflows and
  `scripts/ops/corpus_snapshot/` + `resolve_latest_snapshot_prod_tag.sh`.
- Cross-surface **stack contract** hub and **ADR-093**; shared VPS restore host script path (**#762**).
- Drill reference orchestrator **`drill-exercise.yml`** (plan → apply → deploy → restore → e2e → Playwright → **destroy**) and
  **`drill-stack-playwright.yml`** HTTPS gate pattern to mirror on spare validation.

**Not landed (tracked by #764 / this RFC):**

- **`prod-failover-*`** workflow family composing existing drill reusables **without** **`drill-exercise`** or automated destroy.
- Dedicated **prod failover** chapter in [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md) and [WORKFLOWS.md](../ci/WORKFLOWS.md) index rows.
- Optional `repository_dispatch` helpers for **stand-up / validate** only (**ADR-091**).

## Design & Implementation

### 1. Orchestrator shape (parent workflow)

- **New** top-level workflow(s), e.g. `prod-failover-stand-up.yml` (exact name in implementation), **`concurrency: prod-failover`**, `cancel-in-progress: false`.
- **Jobs** compose **existing** drill reusables where safe: **`drill-infra-plan`**, **`drill-infra-apply`**, **`drill-deploy`**, **`drill-restore-corpus`**, **`drill-e2e`**, **`drill-stack-playwright`** — same Hetzner project, workspace, and secrets as **`drill-exercise`**, with **failover-specific** typed confirms and run summaries.
- **Never** `uses` **`drill-exercise`** as the parent graph and **never** `workflow_call` **`drill-infra-destroy`** from this tree (**ADR-089**). Spare may stay up for minutes or days until the operator runs destroy **manually**.
- **Cutover** (canonical MagicDNS / hostname flip) and **failback** are **runbook + manual** steps in v1; the orchestrator may stop after phase D with spare FQDN and checklist output (**ADR-091**).

### 2. Phases and typed confirms (illustrative)

| Phase | Purpose | Example confirm | Notes |
| ----- | ------- | --------------- | ----- |
| A | Provision spare (**`drill-infra-apply`**, workspace **`drill`**) | `PROD_FAILOVER_PROVISION` | Skipped if drill VPS already exists from a prior stand-up |
| B | Deploy image + compose on spare (**`drill-deploy`**) | inherit or `PROD_FAILOVER_DEPLOY` | Reuse `deploy.sh` on drill host |
| C | Restore corpus from backup repo (**`drill-restore-corpus`**) | `PROD_FAILOVER_RESTORE` or input `backup_tag` | Default tag via `resolve_latest_snapshot_prod_tag.sh` (**ADR-092**); shared `restore_corpus_from_tarball_host.sh` |
| D | Validate: in-container `api` `/api/health` on **8000**, refresh `tailscale serve`, HTTPS probe on drill MagicDNS, then Playwright `stack-viewer.spec.ts` (**`drill-e2e`** + **`drill-stack-playwright`**) | automatic after C or sub-confirm | Schedules **off** on spare; prod frozen per **§6** before cutover |
| E | Cutover: canonical hostname / MagicDNS | **Manual** runbook only in v1 | [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md); operator decides **when** to flip |
| F | Failback + decommission spare | **Manual** only | Repair prod, revert traffic if needed, then **`drill-infra-destroy.yml`** with **`DRILL_DESTROY`** when spare is no longer needed — **not** chained from the failover parent |

### 3. Cutover (DNS path)

Full operator narrative (TTL where applicable, TLS on spare before flip, propagation checks, rollback) lives in **#764** and
[PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md); normative **decision** is [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md).

**v1 operator path (matches #764 non-goals):** manual checklist and copy-paste checks for the hostname flip; automation **stops**
after phase D with spare target, rollback notes, and cutover checklist — **no** unattended flip. **Optional later job:** DNS API token +
script behind **environment** protection — not required for first runbook delivery.

**Spare validation name:** exercise the spare on the **drill** MagicDNS hostname (`DRILL_TAILNET_FQDN` / drill `tailnet_hostname`) before
repointing the **canonical** prod name; do not learn TLS or serve misconfig only at cutover.

### 4. Triggers

Normative list: [ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md).

### 5. Observability & audit

- Log **backup tag**, **image SHA**, **resolved spare FQDN**, and **cutover timestamp** in workflow summaries.
- Open a single **GitHub issue** per incident or attach to existing incident thread.

### 6. Ingestion freeze and spare scheduler policy

**Why this is enough for v1:** without enabled **`scheduled_jobs`**, neither host runs unattended feed sweeps. Corpus **backup** and **restore** are **operator-triggered** workflows, not background timers on the API. The remaining overlap window is **manual** job starts and **misfire catch-up** (APScheduler may fire within **1 h** after a host wakes if schedules were still enabled — another reason to freeze prod before overlap).

**Prod (before cutover):**

1. **Freeze** — disable every **`scheduled_jobs`** entry (set **`enabled: false`** in **`viewer_operator.yaml`** via Configuration save / **`PUT /api/operator-config`**, or equivalent host edit) so the in-process scheduler stops firing. Confirm with **`GET /api/scheduled-jobs`** (`scheduler_running: false` or no enabled jobs with future **`next_run_at`**).
2. **Optional** — wait for in-flight **`POST /api/jobs`** runs to finish; cancel stale jobs if the operator model supports it.
3. **Avoid** starting a new **prod** backup tarball while the spare is being validated for cutover unless the runbook explicitly calls for a final snapshot.

**Spare (after restore, before cutover):**

1. **Do not** turn on **`scheduled_jobs`** on first bring-up — even when the restored tarball carried **`enabled: true`** rows. Prefer a **failover adapter** step (workflow or runbook) that forces all schedule entries **`enabled: false`** before **`compose up`** / API restart, or keeps **`PODCAST_SERVE_ENABLE_JOBS_API`** off until post-cutover review.
2. **Validate** read-only / smoke paths on drill MagicDNS with schedules **off**; enable schedules on the spare only after **failback** or an explicit operator decision to make the spare primary for ingestion.

**After failback:** re-enable schedules on **one** host only; keep the other frozen until decommission.

## Testing & Validation

- **Dry-run mode** (optional input): run through SSH reachability + `tofu plan` only where supported.
- **Phase D** must fail if HTTPS :443 is not reachable from runner (same class of bug as drill stack Playwright before serve refresh).

## Implementation checklist (engineering)

- [ ] Parent `prod-failover-*.yml` composing drill reusables through validate; **never** call **`drill-exercise`** or **`drill-infra-destroy`** (**ADR-089**).
- [ ] `docs/guides/PROD_RUNBOOK.md` failover chapter (phases, drill vs incident table, **ingestion freeze**, manual cutover/decommission, DNS/TLS checklist) + [WORKFLOWS.md](../ci/WORKFLOWS.md) + #764 cross-links to **RFC-083** and ADR-089–091.
- [ ] Failover stand-up adapter: after restore, force **`scheduled_jobs`** **off** on spare before API serves traffic (workflow step or documented host edit).
- [ ] **ADR-093** / stack contract audit row clarifying **incident spare** reuses the **drill VPS** row when stood up.
- [ ] Optional `repository_dispatch` for **stand-up / validate** only (**ADR-091**).

Corpus restore contract and shared ops scripts (**RFC-084** / **ADR-092**) and stack contract hub (**ADR-093**) are **landed on this branch**; see **Branch status** above. Drill OpenTofu workspace, Hetzner project, and Actions secrets are **reused** for on-demand spare (**ADR-089**); no new cloud project or token family.

## References

- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
- [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [ADR-089](../adr/ADR-089-prod-failover-orchestrator-separate-from-drill.md), [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md), [ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md)
