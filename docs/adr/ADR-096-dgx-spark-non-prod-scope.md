# ADR-096: DGX Spark — non-prod augmentation only, no prod request path

- **Status**: Accepted
- **Date**: 2026-05-23
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md), [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- **Related ADRs**: [ADR-093](ADR-093-canonical-stack-contract-and-environment-adapters.md)
- **Related**: ADR-097 (companion — self-hosted GHA runner policy on public repos)

## Context

The operator owns an NVIDIA DGX Spark Founders Edition (GB10 Grace Blackwell, 128 GB unified memory) that joins the existing tailnet as `tag:dgx-llm-host`. The hardware is capable of serving 70B-class models locally with usable latency — competitive with cloud APIs on quality, dominant on cost for autoresearch and pre-prod workloads.

The architectural question this ADR resolves: **may DGX sit in the prod request path?** The temptation is real — substantial cloud-LLM spend could move from Gemini/OpenAI to DGX at zero marginal cost. But DGX lives at the operator's residence, on residential power + ISP, with no SLA.

[RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) defines the prod blast-radius contract: prod ingress is tailnet-only, prod compute is on a managed Hetzner VPS, prod's failure modes are controlled by infrastructure choices the operator can observe and replace. Introducing a residential-network dependency would silently re-shape that contract.

## Decision

DGX is **scoped to non-prod workloads only** and **must not** sit in the prod request path. Specifically:

1. **ACL enforcement is the boundary.** The Tailscale ACL has **no rule** allowing `tag:prod-app` → `tag:dgx-llm-host`. Adding such a rule is a one-line change that would silently break this contract; the absence is intentional and load-bearing. Any future ACL edit that adds prod→DGX traffic requires this ADR's explicit revision.
2. **No code path in the prod-deployed application** may reference DGX endpoints. The `tailnet_dgx` provider exists in the codebase but is only configured into non-prod profiles (`local_dgx_balanced`, `local_dgx_full`) and the future pre-prod LLM backend. Prod profiles (`cloud_balanced`, `cloud_thin`) stay on Gemini + OpenAI.
3. **DGX consumers must implement cloud fallback.** Every consumer of DGX (provider, autoresearch, pre-prod) must degrade to a cloud provider on DGX unavailability. This is operationally important for non-prod surfaces; for prod, it's structurally guaranteed by the fact that prod never targets DGX.
4. **DGX is not a backup or failover target.** Backup-corpus lives in `chipi/podcast_scraper-backup`; failover lives on the DR drill row ([ADR-089](ADR-089-prod-failover-orchestrator-separate-from-drill.md)). DGX has no role in either.

## Rationale

- **Residential SPOF is structurally different from cloud SPOF.** A Hetzner VPS outage is handleable: another VPS in the same Tier-2 spec costs €5/month and can be spun up by `tofu apply` in minutes. A DGX outage means an in-home power blip, an ISP hiccup, or a hardware fault on a unique machine. The recovery surface is operator-specific; the failure-mode analysis is operator-specific. Prod's contract is "any operator (or recovering operator) can restore service"; DGX-in-prod would break that.
- **Blast-radius isolation is the cheap win.** Keeping DGX out of prod costs nothing — the cloud LLM spend at current scale is modest, the cost-arbitrage win is concentrated in autoresearch (which is non-prod by definition), and the qualitative gains (capacity, dev ergonomics, realistic pre-prod) are all upstream of prod.
- **Future contributors will ask "why doesn't prod use DGX?"** Without this ADR, the answer is buried in a 250-line RFC section. With this ADR, the question has a discoverable answer that future RFCs can cite.

## Consequences

- **Positive**: Prod blast-radius story stays intact. RFC-082's "prod fails-over by replacing the VPS" remains true. DGX downtime never pages an operator about prod. The decision is mechanically enforced by an absent ACL rule.
- **Negative**: Prod continues paying cloud LLM spend that DGX could (technically) absorb. Acceptable given the current spend scale; will revisit if monthly cloud-LLM cost grows >10× (e.g., production-scale corpus rebuilds).
- **Neutral**: Pre-prod gets DGX as its LLM backend, which is the closest the architecture comes to "DGX in serving" — but pre-prod has different SLA expectations than prod by design.

## Reversal criteria

This ADR is revisitable only if **all four** conditions hold:

1. Cloud LLM monthly spend exceeds $200/month sustained over 3 months (≥ ~10× current).
2. Operator has documented evidence that DGX uptime in their environment exceeds 99.9% over a 6-month window.
3. A second DGX (or comparable cloud-GPU instance the operator manages) is available as failover for the first.
4. A new RFC supersedes this ADR with explicit blast-radius analysis covering residential-network dependencies.

Until all four conditions hold: prod stays on cloud LLM providers.

## References

- [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) — full DGX integration design
- [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) — prod hosting + blast-radius contract
- [ADR-089](ADR-089-prod-failover-orchestrator-separate-from-drill.md) — failover is on the drill row, not DGX
- [ADR-093](ADR-093-canonical-stack-contract-and-environment-adapters.md) — environment-adapter discipline this ADR builds on
- Tailscale ACL (current absence of prod→DGX rule): `tailscale/policy.hujson`
