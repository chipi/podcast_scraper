# ADR-083: Tailscale as Private Ingress for Always-On VPS

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)

## Context & Problem Statement

The always-on stack serves an operator-facing **api** and **viewer** with TLS expectations, but the
project does not want a **public internet** attack surface on application ports. Home automation,
phones, and collaborators should reach services only after **tailnet** membership and ACL policy.

## Decision

1. **Production api and viewer are reached primarily via Tailscale** (MagicDNS / tailnet FQDN), not
   via open public bind to app ports. Hetzner firewall policy remains **default deny inbound** for
   the app as described in RFC-082; SSH from the open internet may exist for bootstrap only where
   the operator explicitly allows it, then preference is tailnet SSH for operations.
2. **GitHub Actions deploy jobs** join the tailnet as **`tag:gha-deployer`** and SSH to **`deploy@`**
   on the target host over the tailnet using dedicated deploy keys (**`PROD_SSH_PRIVATE_KEY`** /
   **`DRILL_DEPLOY_SSH_PRIVATE_KEY`**), matching ACL rules such as **`tag:gha-deployer` → `tag:prod:22`**
   and **`tag:gha-deployer` → `tag:dr-drill:22`** in **`tailscale/policy.hujson`**.
3. **TLS for HTTPS viewer** uses Tailscale’s HTTPS story on the tailnet hostname (see RFC-082 and
   prod runbook); the app does not require a separate public CA front door for hobby operation.

## Rationale

- **Smallest exposed surface** — aligns with RFC-082 non-goals (no public open ports for the app).
- **Stable operator URLs** — MagicDNS names survive IP changes and match mobile bookmark workflows.
- **Network-level access control** — revoking a collaborator is an ACL change, not a VPS user matrix.

## Alternatives Considered

1. **Public reverse proxy + OAuth on every request** — Rejected for v1 hobby prod; higher moving
   parts than tailnet membership for this threat model.
2. **WireGuard manual only (no Tailscale)** — Rejected; Tailscale already encodes mesh identity,
   DNS, and ACL distribution for the operator tailnet.
3. **Cloudflare Tunnel without tailnet** — Possible future additive; not the chosen primary ingress in
   RFC-082.

## Consequences

- **Positive**: Headless callers (Home Assistant, scripts) use stable tailnet routes; CI deploys
   without exposing GHCR pull secrets on the public listener.
- **Negative**: Every operator or automation path needs a tailnet identity; debugging "from a random
   laptop" requires Tailscale or bastion patterns.
- **Neutral**: Pre-prod (Codespaces) remains a separate ingress story from prod tailnet hosting.

## Implementation Notes

- **Paths**: `tailscale/policy.hujson`, `infra/terraform/` (firewall, Tailscale auth key material),
  `.github/workflows/deploy-prod.yml`, `.github/workflows/drill-deploy.yml`
- **Docs**: [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md)

## References

- [RFC-082 — Goals and non-goals](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
