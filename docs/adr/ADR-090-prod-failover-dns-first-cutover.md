# ADR-090: Prod Failover — DNS-First Cutover on Tailnet

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)
- **Related ADRs**: [ADR-083](ADR-083-tailscale-private-ingress-always-on-vps.md)

## Context

Operators need a **stable name** for prod traffic on the tailnet. Spare has a **different** node/IP and MagicDNS label until cutover. Alternatives: **DNS record change** (tailnet MagicDNS or operator-managed DNS for the same hostname), **Hetzner floating IP** attach, or **Tailscale-only** name tricks. The team discussed tradeoffs in [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764).

## Decision

1. **Primary cutover mechanism** for prod failover (v1) is **DNS**: change the **canonical hostname** used by clients on the tailnet so it resolves to the **spare** after validation. In this repository that hostname is **`https://<tailnet_hostname>.<tailscale_tailnet>/`** ([ADR-083](ADR-083-tailscale-private-ingress-always-on-vps.md)), not a public internet A/AAAA front door.
2. **Hetzner floating IP** (or provider attachable IP) is **optional** and documented as a **secondary** path in the runbook; automation may add it later without changing the primary decision.
3. **Prerequisites** (TLS valid for the canonical name on the **spare** before flip, spare validation on a **temporary** MagicDNS name, propagation checks where operator DNS is involved, rollback by reverting the name target) are **normative runbook** content referenced by **RFC-083**; operators must not rely on IP literals for cutover.

## Consequences

- **Positive**: Matches tailnet-first posture; works across providers; rollback is often “revert DNS”.
- **Negative**: Propagation delay unless TTL is kept low in steady state; TLS must be prepared on spare before cutover.

## References

- [GitHub #764](https://github.com/chipi/podcast_scraper/issues/764)
- [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md) (`tailscale serve`, MagicDNS)
