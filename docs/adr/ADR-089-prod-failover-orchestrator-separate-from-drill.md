# ADR-089: Prod Failover Orchestrator Is Separate from DR Drill

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md), [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- **Related ADRs**: [ADR-081](ADR-081-drill-opentofu-workspace-tailscale-acl-ownership.md)

## Context

**`drill-exercise`** proves infra and app paths on a **throwaway** workspace and **always** runs **`drill-infra-destroy`**. **Production failover** must **keep** the spare running, use **different** confirms and secrets scope, and must **never** be confused with drill automation.

## Decision

1. **Prod failover** is implemented as a **separate GitHub Actions workflow family** (parent + reusables), not a flag on **`drill-exercise`**.
2. That family **must not** `workflow_call` **`drill-infra-destroy`**, **`drill-infra-apply`**, or other drill-only jobs tied to **`HCLOUD_TOKEN_DRILL`** / drill state.
3. **Reuse** is limited to **patterns** (deploy script, restore tarball flow, stack-test probe) and **documentation**, not to reusing the drill orchestrator graph as-is.

## Consequences

- **Positive**: Clear mental model for operators; no accidental drill teardown during an incident.
- **Negative**: Some duplication of YAML structure versus drill; mitigated by shared scripts and small reusable jobs.

## References

- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
- [`drill-exercise.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/drill-exercise.yml)
