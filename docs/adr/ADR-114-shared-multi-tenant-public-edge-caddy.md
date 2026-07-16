# ADR-114: Shared multi-tenant public edge (host-level Caddy + Let's Encrypt)

- **Status**: Accepted
- **Date**: 2026-07-08
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) (always-on hosting), [RFC-087](../rfc/RFC-087-vps-public-edge-multi-compose.md) (broader public-edge ambition — remains deferred; this ADR ships only the shared-engine slice)
- **Related ADRs**: [ADR-083](ADR-083-tailscale-private-ingress-always-on-vps.md) (tailnet private ingress — unchanged), [ADR-084](ADR-084-full-stack-docker-compose-topology.md) (compose topology)
- **Cross-repo**: orrery **ADR-078** (tenant-side decision), issues [#1158](https://github.com/chipi/podcast_scraper/issues/1158) (this repo — engine) / [chipi/orrery#381](https://github.com/chipi/orrery/issues/381) (first tenant)
- **Security SSOT**: [Threat model — VPS + public edge](../security/THREAT_MODEL.md) — the living risk register + pre-public gate this decision is measured against

## Context

The always-on Hetzner VPS ([RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md))
is tailnet-only today: the Hetzner firewall opens `22/SSH` (conditional) + `41641/udp`
(Tailscale) + ICMP, and both podcast_scraper and orrery are served privately via
`tailscale serve`. ADR-083's private-ingress posture is unchanged for the admin/deploy
plane.

A **public** ingress is now needed. orrery is going public on its own registered domain
(orrery ADR-078). Behind it, the operator will stand up **at least three more public
endpoints** over the coming weeks:

| Tenant | Kind | Public host |
| --- | --- | --- |
| orrery | website | its **own** registered domain (isolated) |
| gi/kg viewer | website | podcast-family domain (TBD) |
| podcast player | mobile app | stable HTTPS **API host**, podcast-family domain (TBD) |
| app #3 | mobile app | podcast-family domain (TBD, later) |

This repo owns the VPS's Terraform state and cloud-init, so it owns the **shared public
edge engine**. orrery's ADR-078 is a *tenant's* view and an input — the engine is designed
here, as a reusable N-tenant platform, not shaped around any single app.

The operator may **move orrery to its own VPS later** if budget allows. The design must let
a tenant leave with zero blast radius on the others.

## Decision

### 1. One shared host-level Caddy, per-app drop-in vhosts

A single vanilla Caddy process (official apt repo, no plugins) owns `:80` (→ 301 https)
and `:443` (TLS via Let's Encrypt / ACME — an open standard, self-hosted, swappable). It
routes by `Host` header to each app's container on host loopback. The load-bearing
multi-tenant mechanism is `import /etc/caddy/sites/*.caddy`: the engine is owned once here;
each app drops its own `<app>.caddy` into the sites dir. **No repo ever edits another
repo's routing.** Because Caddy routes purely on `Host`, the engine is **topology-agnostic**
— apex domains, subdomains, or a mix all work; each tenant brings its own hostname(s). The
podcast-family domain layout (one domain + subdomains, or two domains) is deliberately
**not** decided here and does not block the engine.

### 2. Ownership split (mirrors the tailscale-serve precedent)

| Piece | Owner | Lives in |
| --- | --- | --- |
| Firewall: open 80 + 443 | infra | this repo — `hcloud_firewall.main` (`infra/terraform/main.tf`) |
| Caddy install + systemd unit | infra | this repo — `infra/cloud-init/prod.user-data` |
| Base `Caddyfile` (ACME email + shared snippets + `import`) | infra | this repo — cloud-init |
| Reusable snippets (HSTS, compression, access log) | infra | this repo — base Caddyfile |
| Narrow sudoers: `deploy@` may `reload caddy` | infra | this repo — cloud-init |
| Host hardening (fail2ban on Caddy access log) | infra | this repo — cloud-init |
| ACME cert issuance + storage | Caddy (self-managed) | host data dir |
| **Per-app vhost** `<app>.caddy` | **each app** | that app's repo (e.g. orrery `infra/caddy/orrery.caddy`) |
| Deploy: scp vhost → `/etc/caddy/sites/` + validate + reload | each app | that app's deploy workflow |
| DNS A record | operator | registrar |

### 3. Define shared policy once, apps import it

Rather than each app copy-pasting HSTS/headers (as the orrery draft does inline), the base
Caddyfile defines reusable snippets and apps import them. One place to ramp HSTS or change
logging platform-wide:

```caddyfile
{
    email marko.dragoljevic@gmail.com   # ACME account contact (expiry notices; not public)
}

(hardened) {
    # Start short; ramp to max-age=31536000; includeSubDomains; preload once proven.
    header Strict-Transport-Security "max-age=86400; includeSubDomains"
    encode zstd gzip
    log {
        output file /var/log/caddy/access.log
    }
}

import /etc/caddy/sites/*.caddy
```

A tenant vhost then reduces to its host block + `import hardened` + `reverse_proxy
127.0.0.1:<port>`. The upstream container remains the authority for CSP / cache-control;
Caddy passes those through untouched and only adds the shared edge policy.

### 4. Loopback port registry (this repo allocates)

Collision avoidance across N upstreams is infra's job. Reserved map:

| Port | Tenant |
| --- | --- |
| 8080 | podcast api (existing stack) |
| 8090 | orrery nginx (live) |
| 8091 | gi/kg viewer |
| 8092 | podcast player API |
| 8093 | app #3 (reserved) |

### 5. Deploy contract: validate before reload

Every tenant deploy MUST `caddy validate` with its new drop-in staged, and roll back its
own `<app>.caddy` on failure, before `sudo -n systemctl reload caddy`. Caddy reload is
atomic — it rejects an invalid merged config and keeps the last-good config serving — so a
broken drop-in from one tenant cannot take the others *down*; it merely fails to apply.
Validate-before-reload turns that silent no-op into a caught deploy failure. A single shared
sudoers line (`deploy ALL=(root) NOPASSWD: /usr/bin/systemctl reload caddy`, file
`/etc/sudoers.d/99-caddy-reload`) covers every tenant that deploys as the `deploy@` user.

### 6. Terraform apply semantics — two very different changes

- **(A) Firewall (`hcloud_firewall.main`)** — a real TF resource. Adding inbound 80 + 443
  shows an **in-place update (`~`)** in `tofu plan`. **Verify it is NOT a server `-/+`
  replace**, then apply. Immediate, reversible.
- **(B) cloud-init (Caddy install / base Caddyfile / sudoers / fail2ban)** — `write_files` /
  `runcmd` run on **first boot only**. `hcloud_server.prod` sets
  `lifecycle { ignore_changes = [user_data, ssh_keys] }` (`main.tf`), so committing the
  cloud-init edit produces **no plan diff at all** — not a replace. The commit is therefore
  purely for **future rebuilds**; the live box gets Caddy via the **imperative-once-as-root**
  path (the tailscale-serve precedent). Never apply a plan that replaces the VPS
  (2026-05-29 prod-destruction guardrail).

### 7. Tenant lifecycle — a tenant can leave cleanly

A tenant is a pure drop-in: its `<app>.caddy`, its loopback port, its DNS, its cert. Removal
= pull the vhost + reload (frees the port); nothing shared changes. In particular **orrery
carries zero shared-config entanglement** and its domain is isolated, so a future migration
of orrery to its own VPS is `rm /etc/caddy/sites/orrery.caddy` + reload here, plus DNS
re-point. The podcast-family platform domain excludes orrery for the same reason.

## Consequences

**Positive**

- Reusable N-tenant edge: each new public app is one drop-in vhost + one reserved port, no
  new infra. Shared HSTS/logging policy lives in one place.
- Vendor-neutral: standard ACME / Let's Encrypt, self-hosted, swappable; no account tie-in.
- Tenants are isolated and independently removable; orrery can migrate off with no blast
  radius. Rollback is clean (stop Caddy / pull vhost → public down, tailnet still up).

**Negative**

- The VPS public IP becomes visible (the trade-off vs. a tunnel). Mitigated by fail2ban, a
  small static-file attack surface, and keeping the box patched.
- One shared edge is a single point of failure for all public apps on the VPS. Acceptable at
  this scale; Caddy is battle-tested and reload is zero-downtime.
- New supply-chain surface in cloud-init: the Caddy apt repo + GPG key and the `caddy`
  package (a conscious dependency addition).

**Neutral**

- ADR-083's tailnet private ingress is unchanged — Tailscale stays the admin/deploy plane;
  Funnel is not used.
- fail2ban ownership, unassigned in both tracking issues, is claimed here as host infra.

## Alternatives considered

- **Tailscale Funnel** — rejected: Funnel only presents a valid cert for the node's
  `*.ts.net` name, so a custom domain gets a cert-name mismatch; it also exposes the tailnet
  node. Correct posture is to keep Tailscale private and stand a separate public ingress
  beside it.
- **Cloudflare Tunnel / cloudflared as primary ingress** — rejected: couples the primary
  path to a vendor, against the operator's vanilla constraint. Retained only as a possible
  *additive* CDN front later (zero origin change, since the origin serves its own valid
  cert).
- **nginx + certbot** — rejected: same standard ACME outcome but with manual renewal
  plumbing (timer, cert mounts, reload hooks). Caddy gives automatic HTTPS with none of it,
  still 100% standard ACME.
- **Wildcard cert (`*.domain`) via DNS-01** — rejected for now: DNS-01 needs a DNS-provider
  API token (vendor coupling the operator dislikes). Per-host HTTP-01 issuance is
  vendor-neutral and well within Let's Encrypt rate limits at this tenant count. Revisit if
  the subdomain count grows large.
- **Per-app Caddy processes** — rejected: defeats the shared-edge goal, multiplies port/cert
  management, and a single reload-validated engine already isolates tenant failures.
