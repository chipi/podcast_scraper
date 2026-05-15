# ADR-089: Prod Failover Orchestrator Is Separate from DR Drill

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md), [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- **Related ADRs**: [ADR-081](ADR-081-drill-opentofu-workspace-tailscale-acl-ownership.md)

## Context

**`drill-exercise`** proves infra and app paths on a **throwaway** workspace and **always** runs **`drill-infra-destroy`**. **Production failover** must **keep** the spare running for an **operator-chosen** interval, use **different** confirms and run summaries, and must **never** be confused with the drill **exercise** parent workflow.

The spare is **not** always on. When needed, it is stood up **on demand** in the **same** Hetzner project and OpenTofu **`drill`** workspace already used for orchestrated DR drill, reusing existing drill Actions secrets — **no** new cloud project or token family.

## Decision

1. **Prod failover** is implemented as a **separate GitHub Actions workflow family** (parent + composed reusables), not a flag on **`drill-exercise`**.
2. **Spare infra** uses the existing **drill** footprint: OpenTofu workspace **`drill`**, Hetzner project scoped to **`HCLOUD_TOKEN_DRILL`**, drill deploy/restore/e2e/Playwright reusables, and drill MagicDNS for **pre-cutover** validation. **No** second server in prod OpenTofu state and **no** new Hetzner project or secrets for spare bring-up.
3. The failover parent **may** `workflow_call` **`drill-infra-plan`**, **`drill-infra-apply`**, **`drill-deploy`**, **`drill-restore-corpus`**, **`drill-e2e`**, and **`drill-stack-playwright`** as phased tools. It **must not** `workflow_call` **`drill-exercise`** or **`drill-infra-destroy`**. **Decommission** runs only when the operator manually dispatches **`drill-infra-destroy.yml`** (typed **`DRILL_DESTROY`**) after failback or when the spare is no longer needed.
4. Failover **phases** that touch the stack (deploy, restore, validate) must follow the **stack contract** and shared `scripts/ops/` paths documented in [STACK_CONTRACT.md](../guides/STACK_CONTRACT.md) and **ADR-093**; only **adapters** (confirms, run summaries, manual cutover timing) differ from routine drill.

## Consequences

- **Positive**: Reuses proven drill infra and secrets; clear separation from **`drill-exercise`** auto-destroy; operator controls spare lifetime.
- **Negative**: Failover and drill **share** one throwaway workspace — only one full spare stack at a time; operators must not run **`drill-exercise`** and incident stand-up concurrently on the same state.

## References

- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
- [`drill-exercise.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/drill-exercise.yml)
- [`drill-infra-destroy.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/drill-infra-destroy.yml)
