# Edge convergence runbook (`apply-edge.sh` / `verify-edge.sh`)

**What this is:** how to push **base-edge** changes to the live VPS — the shared Caddy
engine + base `Caddyfile` (security headers, HSTS, Cloudflare real-IP ranges), fail2ban
jails, SSH hardening, and the metadata-egress SSRF guard.

> **The key fact:** app **deploys do NOT push the base Caddyfile.** `deploy-player` /
> `deploy-operator` only drop their **per-vhost** `.caddy` file into `/etc/caddy/sites/` and
> reload Caddy. Anything in the base `infra/cloud-init/Caddyfile` — the `(hardened)` snippet
> (headers/HSTS), CF `trusted_proxies` ranges, the `admin` socket, log format — reaches the box
> **only** via `apply-edge.sh` (imperative) or a full VPS rebuild (cloud-init). If you edit
> the base Caddyfile and only deploy, **your change never goes live.**

## When to run it

- You changed `infra/cloud-init/Caddyfile` (headers, HSTS ramp, CF IP ranges, log format).
- You changed the fail2ban jails / SSH hardening / metadata-egress guard in `prod.user-data`
  (the `apply-edge` snippets mirror it) and don't want to rebuild.
- After a rebuild, to re-verify convergence.

Not for: per-app vhosts (those ship via their deploy), the firewall (Terraform / `infra-apply`,
Phase 4), or the app stack.

## Access model

Runs **on the box as root-equivalent, over the tailnet**. The `deploy@` user has **passwordless
sudo only** for the exact allowlisted paths `/usr/local/sbin/apply-edge.sh` +
`/usr/local/sbin/verify-edge.sh` (not a general root shell). The installed
`/usr/local/sbin/apply-edge.sh` is placed by cloud-init from `scripts/ops/apply-edge.sh` and is
**refreshed only on rebuild** — if you changed the script itself, reinstall it first (root):
`install -m0755 /srv/podcast-scraper/scripts/ops/apply-edge.sh /usr/local/sbin/apply-edge.sh`.

## Procedure

1. **Get the change onto the box's repo checkout.** `apply-edge` copies the Caddyfile from the
   checkout. Merge to `main`, then on the box: `cd /srv/podcast-scraper && git pull` (or let the
   next app deploy's `git reset --hard origin/main` bring it).
2. **Dry-run first (changes nothing):**

   ```sh
   ssh deploy@prod-podcast \
     'sudo -n /usr/local/sbin/apply-edge.sh --dry-run --repo-dir /srv/podcast-scraper'
   ```

   Read the diff. Ideal: the only yellow line is `cp .../Caddyfile -> /etc/caddy/Caddyfile` +
   `reload/restart caddy`; everything else `(already converged)`. If it shows SSH/fail2ban/
   metadata changes, the box was never converged — expected on a fresh box, but eyeball it.
3. **Apply:**

   ```sh
   ssh deploy@prod-podcast \
     'sudo -n /usr/local/sbin/apply-edge.sh --repo-dir /srv/podcast-scraper'
   ```

   It **validates before (re)starting** (`caddy validate`, `sshd -t`) — an invalid config is
   **not** applied, so it can't brick the edge. **The firewall is never touched** (box stays
   tailnet-only; opening ports is Terraform/Phase-4). SSH stays reachable over the tailnet throughout.
4. **Verify (go/no-go gate):**

   ```sh
   ssh deploy@prod-podcast \
     'sudo -n /usr/local/sbin/verify-edge.sh --repo-dir /srv/podcast-scraper'
   ```

   Expect all PASS (SSH key-only, fail2ban jails, metadata DROP, Caddy owns :443, Caddyfile
   matches repo, o11y collector container up).
5. **Confirm externally** (the box can't prove reachability):

   ```sh
   curl -sI https://operator.closelistening.app/ | grep -iE 'x-frame-options|strict-transport|x-content-type'
   ```

## Reload, not restart (2026-10-15)

Until 2026-10-15 the base Caddyfile set **`admin off`** (T-02), so `caddy reload` had no endpoint
(bit us 2026-07-28) and every config or vhost change **restarted** Caddy, dropping all tenants'
traffic for seconds (a 521 on the player smoke on 2026-10-15). The admin endpoint is now a
**0600 unix socket** in caddy's home (`admin unix//var/lib/caddy/admin.sock|0600`): never on the
network, and `systemctl reload caddy` swaps the config with no dropped request (400/400 across a
reload, caddy v2.11.4). Deploys reload and restart only if the reload is refused.

- **First apply on a box still running `admin off`:** the reload is refused (no endpoint), so
  `apply-edge` falls back to one restart, which brings the socket up. Every later change
  reloads. Do the first apply at a quiet moment, never during a smoke or deploy.
- **A failed reload keeps the old config running** (no outage, but the change isn't live). Check
  `journalctl -u caddy` for the reason; the allowlisted fallback is still
  `ssh deploy@prod-podcast 'sudo -n /usr/bin/systemctl restart caddy'`.
- **A service-env change** (`CADDY_BIND_ADDRS`, `GLITCHTIP_UPSTREAM`) still needs a restart:
  systemd reads `Environment` at start.
- **`CADDY_BIND_ADDRS` holds the public IPv4 and IPv6, and the line is quoted** in
  `/etc/caddy/…/10-public-bind.conf`: `Environment="CADDY_BIND_ADDRS=<v4> <v6>"`. Unquoted, systemd
  keeps only the first address and Caddy silently listens on IPv4 only (2026-10-17). `apply-edge.sh`
  and `deploy_config_caddy.sh` read both forms; `verify-edge.sh` now checks that **every**
  configured address has a `:443` listener (`ss -tlnH`), where the owner check alone passed on the
  first address. A failure reads `no listener on [<v6>]:443`; check
  `systemctl show caddy -p Environment`.

## What `apply-edge` does NOT manage

- **The o11y collector** — that's the Alloy **container** at `/opt/vps-observability/` shipping
  to homelab (ADR-128-era; Grafana Cloud retired). `apply-edge` no longer installs a systemd Alloy.
- **The firewall / port exposure** — Terraform (`infra-apply`), the deliberate Phase-4 exposure.
- **Secrets** — ADR-115 tmpfs delivery, operator-owned.

## References

- [ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md) (shared edge) ·
  [ADR-118](../adr/ADR-118-cloudflare-additive-front.md) (CF real-IP / origin-lock) ·
  [THREAT_MODEL](../security/THREAT_MODEL.md) (T-02/T-05/T-07/T-11)
- `scripts/ops/apply-edge.sh` · `scripts/ops/verify-edge.sh` · `infra/cloud-init/Caddyfile`
