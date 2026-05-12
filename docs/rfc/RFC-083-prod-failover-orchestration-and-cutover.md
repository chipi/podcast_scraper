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
- **Related documents**:
  - [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764) — problem statement, DNS path, runbook intent
  - [ADR-081](../adr/ADR-081-drill-opentofu-workspace-tailscale-acl-ownership.md), [ADR-082](../adr/ADR-082-gitops-app-deploy-via-stack-test-and-gha.md), [ADR-083](../adr/ADR-083-tailscale-private-ingress-always-on-vps.md)
  - [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md) — tailnet health, `tailscale serve`

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
4. **DNS-first cutover** for the canonical hostname used on the tailnet, with TTL/TLS prerequisites documented in runbooks ([ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md)).
5. **No coupling** to drill teardown or drill Hetzner token ([ADR-089](../adr/ADR-089-prod-failover-orchestrator-separate-from-drill.md)).

## Non-goals (v1)

- **Scheduled warm spare** that runs full stack on a cron without an incident (separate future RFC/issue).
- **Fully unattended DNS cutover** on first external alert.
- **Multi-region** active-active.

## Constraints & Assumptions

- **Tailnet-first prod** per existing ops docs; public ingress is out of scope for cutover mechanics.
- **Single writer** to external side effects at a time: runbook / workflow text must state **split-brain** avoidance (only one env mutates feeds/backups during incident unless explicitly designed).
- **OpenTofu**: today `infra/terraform/main.tf` has one `hcloud_server.prod`. A **spare** requires **separate state** (new workspace and/or second server resource, new `tailnet_hostname`, ACL rows for `tag:gha-deployer` → spare tag). Exact Terraform design is a **dependency** of the orchestrator; this RFC assumes it lands in a sibling PR or phase.
- **Secrets**: spare SSH identity, `TS_AUTHKEY`, backup token, and optional DNS API token live in **GitHub Environments** with **required reviewers** for cutover.

## Design & Implementation

### 1. Orchestrator shape (parent workflow)

- **New** top-level workflow(s), e.g. `prod-failover-exercise.yml` (exact name in implementation), **`concurrency: prod-failover`**, `cancel-in-progress: false`.
- **Jobs** call **`uses: ./.github/workflows/...`** reusable workflows where reuse is safe; **new** reusables for “deploy to **spare** FQDN” and “restore to **spare**” that mirror `deploy-prod.yml` / `prod-restore-corpus.yml` but parameterize **SSH target**, **secrets**, and **confirm** strings.
- **Never** `uses` **`drill-infra-destroy`** or drill apply/plan from this tree ([ADR-089](../adr/ADR-089-prod-failover-orchestrator-separate-from-drill.md)).

### 2. Phases and typed confirms (illustrative)

| Phase | Purpose | Example confirm | Notes |
| ----- | ------- | --------------- | ----- |
| A | Provision spare (Terraform apply in prod or dedicated workspace) | `PROD_FAILOVER_PROVISION` | Skipped if spare is pre-provisioned and idle |
| B | Deploy image + compose on spare | inherit or `PROD_FAILOVER_DEPLOY` | Reuse `deploy.sh` patterns |
| C | Restore corpus from backup repo | `PROD_FAILOVER_RESTORE` or input `backup_tag` | Align with [#763](https://github.com/chipi/podcast_scraper/issues/763) when manifest exists |
| D | Validate: SSH refresh `tailscale serve`, runner `curl -k https://<spare>/api/health`, then Playwright `stack-viewer.spec.ts` against spare base URL | automatic after C or sub-confirm | Same lesson as [`drill-stack-playwright.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/drill-stack-playwright.yml) |
| E | Cutover: DNS or floating IP | `PROD_FAILOVER_CUTOVER` | [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md) |

**Failback** (v1 runbook + follow-up workflow): restore prod host from spare or backup, validate, flip DNS back, then decommission spare—**no** destroy of prod; spare teardown only after explicit phase.

### 3. Cutover (DNS path)

Full operator narrative (TTL, TLS on spare, propagation checks, rollback) lives in **#764** and [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md); normative **decision** is [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md).

Automation v1 options:

- **Pause job** after phase D success: output “update DNS A/AAAA to X”; operator confirms in DNS UI; **manual** `workflow_dispatch` continuation or checkbox job.
- **Optional job** with `DNS_API_TOKEN` and script (Cloudflare / other) behind **environment** protection.

### 4. Triggers

Normative list: [ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md).

### 5. Observability & audit

- Log **backup tag**, **image SHA**, **resolved spare FQDN**, and **cutover timestamp** in workflow summaries.
- Open a single **GitHub issue** per incident or attach to existing incident thread.

## Testing & Validation

- **Dry-run mode** (optional input): run through SSH reachability + `tofu plan` only where supported.
- **Phase D** must fail if HTTPS :443 is not reachable from runner (same class of bug as drill stack Playwright before serve refresh).

## Implementation checklist (engineering)

- [ ] Terraform: spare server + tailnet name + firewall; ACL updates in `tailscale/policy.hujson` (via prod workspace apply per ADR-081).
- [ ] Reusable workflows: `deploy-prod-spare.yml`, `prod-restore-spare.yml` (names TBD), shared scripts with prod where possible ([#762](https://github.com/chipi/podcast_scraper/issues/762) alignment).
- [ ] Parent `prod-failover-*.yml` with phases and Environment gates.
- [ ] `docs/guides/PROD_RUNBOOK.md` + [WORKFLOWS.md](../ci/WORKFLOWS.md) + #764 cross-links to **RFC-083** and ADR-089–091.
- [ ] `repository_dispatch` payload contract + test with `gh api`.

## References

- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
- [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [ADR-089](../adr/ADR-089-prod-failover-orchestrator-separate-from-drill.md), [ADR-090](../adr/ADR-090-prod-failover-dns-first-cutover.md), [ADR-091](../adr/ADR-091-prod-failover-gha-triggers-and-gates.md)
