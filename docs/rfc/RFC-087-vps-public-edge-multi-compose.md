# RFC-087: VPS public edge and multi–Docker Compose hosting

- **Status**: Draft
- **Authors**: Marko + assistant
- **Stakeholders**: Operator (infra), security review for public exposure
- **Related RFCs**:
  - [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) — prod VPS, Tailscale-only default, GitOps deploy
  - [RFC-081](RFC-081-pre-prod-environment-and-control-plane.md) — Codespaces pre-prod (orthogonal)
- **Related guides**:
  - [Prod runbook](../guides/PROD_RUNBOOK.md)
  - [VPS multi-app onboarding](../guides/VPS_MULTI_APP_ONBOARDING.md) — same host, additional compose roots, GitOps pattern

## Abstract

[RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) standardizes **operator-facing**
production on a Hetzner VPS with **Tailscale-only ingress**: stable tailnet URL, SSH and
`tailscale serve` for the viewer, and **no public TCP 80/443** on the cloud firewall.

This RFC proposes an **optional second plane**: a **public TLS edge** (one reverse proxy
terminating HTTPS) that routes **multiple hostnames** (subdomains or apex domains) to **one or
more** Docker Compose stacks on the **same** VPS — including **other repositories’** compose
apps, using the isolation and GitOps patterns in
[VPS multi-app onboarding](../guides/VPS_MULTI_APP_ONBOARDING.md).

**Operators keep using Tailscale** for SSH, deploy workflows, internal health checks, and
sensitive surfaces; **end users** reach only vhosts explicitly published on the public edge.

## Problem Statement

- Tailscale gives a low-friction **private** control plane but does not replace a **public**
  origin for arbitrary internet users without installing TS clients.
- Running several hobby or small production web services on one VPS is economical, but needs:
  - clear **per-app isolation** (paths, env, ports, volumes);
  - one **routing and TLS** story so each service gets its own hostname without port hacks;
  - a **security model** that does not accidentally expose admin APIs, metrics, or job endpoints.
- Today’s prod runbook **forbids** opening public 80/443 without deliberate firewall, proxy,
  and auth changes. That work should be **designed** rather than ad-hoc.

## Goals

1. **Split trust domains**: Tailscale remains the default path for **operator** access;
   **public** traffic is opt-in per vhost or per app.
2. **Single public edge**: one reverse proxy (or managed equivalent) terminates TLS and maps
   `Host` → upstream (local ports or internal Docker network).
3. **Compose-extensible host**: additional apps under `/srv/<app-slug>` (or equivalent) follow
   [VPS multi-app onboarding](../guides/VPS_MULTI_APP_ONBOARDING.md); edge config is the only
   **shared** coupling layer.
4. **Documented firewall and DNS**: Hetzner rules, DNS records, and ACME strategy are explicit.
5. **Rollback**: disable a vhost or revert proxy config without reprovisioning the VM.

## Non-goals

- Multi-region HA, Kubernetes, or dedicated “edge fleet” (out of scope for hobby scale).
- Replacing Tailscale for **operators** (public edge is for **users**, not a mandate to drop TS).
- Product changes inside `podcast_scraper` beyond what is needed to sit behind a shared proxy
  (optional follow-up issues).

## Constraints and assumptions

- **Assumption**: operator controls DNS for at least one zone (direct A/AAAA or via a CDN).
- **Constraint**: opening **public 443** (and usually **80** for ACME HTTP-01) increases attack
  surface; each exposed app must have appropriate **auth** and **rate** posture.
- **Constraint**: podcast_scraper prod today relies on **tailnet + optional basic auth**; public
  exposure may require **different** auth or WAF rules — not a lift-and-shift without review.

## Design overview

### Two planes on one host

| Plane | Audience | Typical access | Ingress |
| ----- | -------- | -------------- | ------- |
| **Operator** | Human + GHA deploy | Tailscale SSH, tailnet HTTPS where used today | Tailscale ACL + MagicDNS |
| **Public** | Internet users | Browser HTTPS to `app.example.com` | Internet → VPS :443 → edge proxy → upstream |

The **edge proxy** is the only process that should bind **public** 443 (and 80 if required for
redirects or ACME). Application containers should listen on **loopback** or an **internal
Docker network** reachable only from the host or from the proxy container.

### Upstream mapping

- Each compose stack exposes one or more HTTP services on **distinct internal ports**
  (e.g. `127.0.0.1:8080`, `127.0.0.1:9001`).
- Proxy **server blocks** or routers match `server_name` / SNI and `proxy_pass` to the correct
  upstream.
- **WebSockets** and large uploads: configure timeouts and body size per app if needed.

### TLS

**Option A — On-box ACME (e.g. Caddy, nginx + certbot, Traefik)**

- Pros: simple mental model, full control on VPS.
- Cons: must open **80/443** on Hetzner firewall; operator maintains renewal and proxy config.

**Option B — Cloudflare Tunnel (`cloudflared`)**

- Pros: **no inbound** ports to VPS if tunnel-only; CDN/WAF optional.
- Cons: Cloudflare account and tunnel lifecycle; egress still from VPS for origins.

**Option C — Separate tiny edge VM**

- Deferred unless isolation requirements justify cost.

**Recommendation for hobby scale:** start with **A** if comfortable opening 443, or **B** if
minimizing open ports is priority. Record choice in runbook addendum once implemented.

### Firewall (Hetzner)

- **Today**: Tailscale UDP + ICMP; no public web.
- **With public edge**: add **TCP 443** (and **TCP 80** if HTTP-01 or redirect required).
- Restrict **22** to tailnet-only where possible; never duplicate RFC-082 posture accidentally.

### Integration with multi-app onboarding

[VPS multi-app onboarding](../guides/VPS_MULTI_APP_ONBOARDING.md) defines **per-app** roots,
env files, systemd units, and **per-repo** GitHub Actions deploy. This RFC adds:

- a **shared edge stack** or host-level proxy **config repo** (or a documented directory on the
  host, e.g. `/srv/edge/`) versioned in git;
- a **naming table**: hostname → upstream:port → owning repo;
- optional **shared monitoring** (access logs shipped to existing Grafana/Loki patterns).

## GitOps

- **Per-app deploy** stays in each repository (clone of `deploy-prod.yml` pattern).
- **Edge config** changes should be **PR-reviewed** (dedicated repo or `infra/edge/` in a chosen
  repo) and applied via **SSH + git pull + reload proxy**, or **Ansible**/script — exact
  mechanism is an open question below.
- **Do not** hand-edit production-only proxy state without capturing it in git.

## Security

- **No admin routes** on public hostnames unless behind strong auth (OIDC, mutual TLS, or
  Tailscale-only split DNS — prefer TS-only for admin).
- **Rate limiting** and **WAF** (especially if Cloudflare fronts the zone).
- **Separate credentials** per app; no shared `.env` across unrelated stacks.
- **Regular review** of exposed vhosts when adding apps.

## Testing and validation

- **Smoke**: `curl -fsS https://vhost/.well-known` or app health path after each edge change.
- **Regression**: operator path still works over Tailscale (SSH + internal checks).
- **CI**: optional `actionlint` / config linter for proxy configs if stored as static files.

## Rollout (phased)

1. **Design-only** (this RFC + checklist in runbook/onboarding): no production change.
2. **Edge MVP**: one proxy + one non-critical vhost (or staging hostname) → one upstream.
3. **Generalize**: document hostname table; onboard second compose app per onboarding guide.
4. **Optional**: migrate podcast_scraper “user” URL to public vhost while keeping ops on TS.

## Open questions

1. **Edge implementation**: Caddy vs Traefik vs nginx vs Cloudflare Tunnel only?
2. **Config ownership**: single `infra` monorepo vs `edge` repo vs host path not in podcast_scraper?
3. **ACME**: HTTP-01 on VPS vs DNS-01 (Cloudflare API token on host — secret handling)?
4. **podcast_scraper public cutover**: single marketing/login vhost vs keep viewer tailnet-only
   indefinitely?
5. **Observability**: access logs to Loki with PII scrubbing policy?

## Relationship to other RFCs

- **RFC-082** defines **tailnet-first** prod; this RFC is an **optional extension** for **public**
  readership and does not invalidate RFC-082’s default posture.
- **VPS multi-app onboarding** is the **operational** companion for **many compose roots**; this
  RFC specifies the **shared public edge** that onboarding defers.

## References

- [RFC-082 — always-on hosting](RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [VPS multi-app onboarding](../guides/VPS_MULTI_APP_ONBOARDING.md)
- [Prod runbook](../guides/PROD_RUNBOOK.md)
- [Tailscale serve](https://tailscale.com/kb/1242/tailscale-serve) (operator plane today)
