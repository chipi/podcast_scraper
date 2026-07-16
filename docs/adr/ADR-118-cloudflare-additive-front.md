# ADR-118: Cloudflare additive front for the public edge (real-IP recovery + origin lock)

- **Status**: Accepted
- **Date**: 2026-07-09
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**: [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (the
  Caddy origin this fronts — designed to sit behind an additive proxy),
  [ADR-115](ADR-115-multi-tenant-secret-delivery-sops-tmpfs-files.md) (secret
  delivery, needed only for Alternative B)
- **Security SSOT**: [Threat model](../security/THREAT_MODEL.md) **T-05** (no
  DDoS / WAF / edge rate-limit) — this ADR is its home
- **Tracking**: #1158 / orrery#381 (edge programme)

## Context

The shared Caddy edge (ADR-114) is a single small VPS. **T-05** is unaddressed:
no WAF, no edge rate-limit, no volumetric-DDoS absorption. The `caddy-access`
fail2ban jail (2026-07-09) bans 4xx-burst scanners at the INPUT chain, but does
nothing against an L7 flood or a volumetric attack — a small origin cannot.

ADR-114 was deliberately designed to sit behind an additive proxy. Cloudflare's
free tier gives proxy + managed WAF + rate-limiting + DDoS absorption + global
TLS termination at **zero origin-code change**. Constraints: Caddy stays
plugin-free (no CF-DNS plugin); secrets go through ADR-115; SSH is tailnet-only,
so the admin plane survives any edge/firewall misconfig (no lockout risk).

The one thing CF changes is *who Caddy sees as the peer*: once proxied, the TCP
peer is a Cloudflare edge, and the real visitor arrives in `CF-Connecting-IP` /
`X-Forwarded-For`. That single fact drives every origin-side change below.

## Decision (D4)

Accept **Cloudflare free as the additive front** for public vhosts. The operator
owns the CF account + DNS; the origin-side prep is minimal and lands here:

1. **Real-IP recovery** — Caddy `trusted_proxies static <CF ranges>` +
   `client_ip_headers Cf-Connecting-Ip X-Forwarded-For` in the global `servers`
   block. `client_ip`, the access log, and the XFF forwarded to downstream
   tenants (e.g. player nginx) all reflect the real visitor. **No-op until CF
   actually fronts the box** — direct clients are never in the CF ranges.

2. **CF-safe ban target** — the `caddy-access` fail2ban filter keys on
   **`client_ip`, not `remote_ip`**. Correct in both regimes:
   - pre-CF: `client_ip` == the direct TCP peer → a real iptables drop of the
     scanner (as today);
   - post-CF: `client_ip` == the visitor *behind* CF → the iptables ban is a
     harmless no-op (that IP never connects directly), and — critically — it
     **never bans a Cloudflare shared edge IP**, which keying on `remote_ip`
     would do, taking the whole site down. Volumetric / L7 banning moves to CF's
     rules once fronted (step 6 below).

3. **Origin lock (optional, flip after DNS is live)** — TF
   `cloudflare_origin_lock` narrows the **:443** firewall rule to CF's published
   ranges, so an attacker can't bypass CF by hitting the origin IP directly.
   **:80 stays world-open** for ACME HTTP-01 (Let's Encrypt validates from its
   own IPs, not through CF).

4. **TLS** — keep Let's Encrypt auto-HTTPS on the origin; CF SSL/TLS mode
   **Full (strict)** validates the real cert. No new secret. (Cloudflare Origin
   CA + closing :80 is the documented hardening upgrade — Alternative B.)

## Consequences

**Positive**

- A real DDoS / WAF / edge-rate-limit front at zero origin-code change.
- Real visitor IP preserved end-to-end (logs, rate-limits, per-user authZ).
- Origin-lock kills the direct-IP bypass; the app becomes reachable only via CF.
- Fully reversible in <5 min (grey-cloud DNS, or `cloudflare_origin_lock=false`).
- SSH is tailnet-only → no lockout risk from any CF/firewall misstep.

**Negative**

- CF terminates TLS → CF sees plaintext. Acceptable for public **static /
  consumer** content; the **operator / api plane stays tailnet-only and is never
  CF-fronted** (T-01). This ADR must not be read as license to expose api.
- CF IP ranges are baked in **two** places (Caddyfile `trusted_proxies` + TF
  `cloudflare_ip_ranges`) and need a ~yearly refresh from
  <https://www.cloudflare.com/ips/>.
- Under Alternative A, :80 stays world-open — but it only serves the ACME
  challenge + an HTTP→HTTPS redirect, no app content.
- Adds an external dependency (CF availability) in the request path.

**Neutral**

- Free-tier limits (managed WAF only, few page rules) are sufficient for the
  current surface; revisit at scale.

## Alternatives considered

- **A — Caddy-only + fail2ban jail (status quo).** Already in place; handles
  scanners, **not** volumetric DDoS. Insufficient alone for a public consumer
  surface. This ADR is *additive* to it, not a replacement.
- **B — Cloudflare Origin CA cert + close :80 entirely.** Strictly stronger
  (zero world-open ports, fully CF-only). Deferred as an upgrade: needs a static
  cert+key delivered via ADR-115 + per-vhost `tls` directives, diverging from
  auto-HTTPS. Adopt if the :80 residue or CF-only enforcement becomes a hard
  requirement.
- **C — Caddy rate-limit plugin.** Rejected — breaks the plugin-free constraint
  (ADR-114) and still gives no volumetric-DDoS absorption.

## Operator rollout steps (this order avoids any outage)

1. Create/verify the Cloudflare account; add the zone; point the registrar's
   nameservers at CF.
2. Add DNS records for each public hostname and **orange-cloud (proxy)** them.
3. SSL/TLS mode → **Full (strict)** (origin already serves a valid LE cert).
4. Verify each hostname loads **through CF** while :443 is still world-open:
   confirm a `cf-ray` response header, and confirm the Caddy access log shows the
   real `client_ip` (not a CF edge IP).
5. **Only after step 4 is green:** set `cloudflare_origin_lock = true` and
   `tofu apply` (narrows :443 to CF ranges). Re-verify the site loads, and that a
   direct `curl https://<origin-ip>` now times out.
6. (Optional but recommended) Configure CF rate-limiting + the managed WAF
   ruleset — the L7 / volumetric control that replaces the edge fail2ban jail for
   CF-fronted traffic.

**Rollback (<5 min):** grey-cloud the DNS record (instant, DNS-only) **or**
`cloudflare_origin_lock=false` + `tofu apply` (re-open :443). SSH stays reachable
over tailnet throughout.
