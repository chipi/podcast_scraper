# ADR-091: Prod Failover — GitHub Actions Triggers and Gates

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)
- **Related ADRs**: [ADR-082](ADR-082-gitops-app-deploy-via-stack-test-and-gha.md), [ADR-089](ADR-089-prod-failover-orchestrator-separate-from-drill.md)

## Context

Failover must be **callable manually** under stress and, when mature, from **external automation** (monitoring, pager) without bypassing safety. **`schedule`**-driven cutover is unsafe for v1 (no incident context).

## Decision

1. **Primary trigger** is **`workflow_dispatch`** on the prod-failover orchestrator: typed confirms per high-risk phase (strings defined in **RFC-083** and workflow inputs).
2. **Optional automation** uses **`repository_dispatch`** with a **shared secret** in the payload (or `X-Hub-Signature` validation); workflows **must** reject events without valid authentication.
3. **v1 policy**: automated events may run **through validate** (stand up spare, deploy, restore, HTTPS + stack-test) only if explicitly configured; **cutover** jobs require **GitHub Environment** protection (**required reviewers**) or a **second** manual dispatch—**no** fully unattended DNS cutover on first alert in v1.
4. **`schedule`** is **not** used for cutover or full spare bring-up in v1.

## Consequences

- **Positive**: Reduces false-positive traffic moves; audit trail in Actions; aligns with existing `environment: prod` / drill patterns.
- **Negative**: On-call must still approve cutover unless later RFC relaxes this with additional safeguards.

## References

- [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)
- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
