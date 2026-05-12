# ADR-090: Prod Failover — DNS-First Cutover on Tailnet

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)
- **Related ADRs**: [ADR-083](ADR-083-tailscale-private-ingress-always-on-vps.md)

## Context

Operators need a **stable name** for prod traffic on the tailnet. Spare has a **different** node/IP until cutover. Alternatives: **DNS record change**, **Hetzner floating IP** attach, or **Tailscale-only** name tricks. The team discussed tradeoffs in [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764).

## Decision

1. **Primary cutover mechanism** for prod failover (v1) is **DNS**: change the **canonical hostname** (A/AAAA or CNAME target) used by clients on the tailnet so it resolves to the **spare** after validation.
2. **Hetzner floating IP** (or provider attachable IP) is **optional** and documented as a **secondary** path in the runbook; automation may add it later without changing the primary decision.
3. **Prerequisites** (TTL, TLS certificates valid for the hostname on the **spare** before flip, propagation checks, rollback by reverting DNS) are **normative runbook** content referenced by **RFC-083**; operators must not rely on IP literals for cutover.

## Consequences

- **Positive**: Matches tailnet-first posture; works across providers; rollback is often “revert DNS”.
- **Negative**: Propagation delay unless TTL is kept low in steady state; TLS must be prepared on spare before cutover.

## References

- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
- [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md) (`tailscale serve`, MagicDNS)
