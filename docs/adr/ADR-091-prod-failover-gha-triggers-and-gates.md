# ADR-091: Prod Failover — GitHub Actions Triggers and Gates

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)
- **Related ADRs**: [ADR-082](ADR-082-gitops-app-deploy-via-stack-test-and-gha.md), [ADR-089](ADR-089-prod-failover-orchestrator-separate-from-drill.md), [ADR-092](ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md)

## Context

Failover must be **callable manually** under stress. The operator may run on the spare for **minutes or longer** while prod is repaired, then **manually** fail back and decommission. **`schedule`**-driven cutover or teardown is unsafe for v1 (no incident context).

## Decision

1. **Primary trigger** is **`workflow_dispatch`** on the prod-failover orchestrator: typed confirms per high-risk **stand-up** phase (strings defined in **RFC-083** and workflow inputs).
2. **Optional automation** uses **`repository_dispatch`** with a **shared secret** in the payload (or `X-Hub-Signature` validation) for **stand-up / validate** only; workflows **must** reject events without valid authentication.
3. **v1 policy**: **cutover** (canonical hostname / MagicDNS flip), **failback**, and **spare teardown** are **manual** operator actions (runbook + optional separate dispatches). The failover parent **does not** auto-run cutover, auto-failback, or **`drill-infra-destroy`**. **`repository_dispatch`** must not trigger unattended cutover or destroy.
4. **`schedule`** is **not** used for cutover, spare bring-up, or spare teardown in v1.
5. **Ingestion during incidents:** before cutover, operators **freeze** prod by disabling all **`scheduled_jobs`** (in-process scheduler). The spare **must not** start with schedules **enabled** after restore, even when the backup had **`enabled: true`** entries. Unattended ingestion is the primary dual-writer risk; manual **`POST /api/jobs`** overlap is avoided by operator discipline.

## Consequences

- **Positive**: Operator controls traffic move and spare lifetime; audit trail in Actions for stand-up/validate; aligns with on-demand spare (**ADR-089**).
- **Negative**: Cutover and decommission latency depend on operator availability; automation is **prepare and validate**, not end-to-end incident closure.

## References

- [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)
- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
