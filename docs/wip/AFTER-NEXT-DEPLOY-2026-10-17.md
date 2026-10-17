# After the next deploy: post-deploy steps (opened 2026-10-17)

Steps that can only run once the next deploy is on the box. Each prod step needs the operator's
go. The deploy itself and the long runs before it stay in
[POST-DEPLOY-BOARDS-AND-DGX-GATEWAY.md](POST-DEPLOY-BOARDS-AND-DGX-GATEWAY.md).

## 1. Caddy listens on the public IPv4 address only, not IPv6

**Found 2026-10-17** (prod session, while verifying #1877's units with `systemd-analyze` on the box):

```text
/etc/systemd/system/caddy.service.d/10-public-bind.conf:2: Invalid environment assignment, ignoring: 2a01:4f8:1c1b:8cd4::1
```

**Checked on prod, read-only:**

- The drop-in holds both public addresses, unquoted:
  `Environment=CADDY_BIND_ADDRS=178.105.182.17 2a01:4f8:1c1b:8cd4::1`.
- systemd splits an unquoted value on spaces and keeps only the first address. `systemctl show caddy
  -p Environment` gives `CADDY_BIND_ADDRS=178.105.182.17`.
- `ss -ltn` on :80/:443: Caddy listens on `178.105.182.17:80` and `:443` only. The other two :443
  listeners are tailscale serve (tailnet v4 and v6). Nothing listens on `2a01:4f8:1c1b:8cd4::1`.
- The intent was both addresses. `prod.user-data` and `apply-edge.sh` collect the IPv4 AND the IPv6
  address into the variable. The Hetzner firewall opens 80/443 to `::/0`, or to Cloudflare's ranges
  under origin lock. The Caddyfile says "bind ONLY the box's public interface(s)".

**Impact today: none seen.** All 13 public names (closelistening.app, operator., mcp., obs.,
telemetry., analytics., www., and the orrerylearn.com set) resolve to Cloudflare (A 188.114.96/97.0,
AAAA 2a06:98c1:3120/3121::), so no client reaches the box's IPv6 directly. Cloudflare reaches the
origin over IPv4 today.

**Not verified:** whether Cloudflare holds an AAAA origin record for any of these names. I did not
read the Cloudflare zone. If it does, Cloudflare could try the origin over IPv6 and fail (52x).

**Cause:** the two writers emit the value unquoted:

- `infra/cloud-init/prod.user-data` (the `printf '[Service]\nEnvironment=CADDY_BIND_ADDRS=%s\n'`
  line);
- `scripts/ops/apply-edge.sh:272` (`Environment=CADDY_BIND_ADDRS=$BIND_ADDRS`).

**Fix (built 2026-10-17, operator "build fix now"; ships with the next deploy):**

1. Both writers now quote the value: `Environment="CADDY_BIND_ADDRS=<v4> <v6>"`. The cloud-init
   file gained 2 bytes (29,422 of 29,491).
2. Both readers (`deploy_config_caddy.sh`, `verify-edge.sh`) accept the quoted and the old unquoted
   form, so a box with the old drop-in still validates.
3. `verify-edge.sh` fails when a configured bind address has no :443 listener. Run on prod as
   deploy before the fix: `PASS listening on 178.105.182.17:443`, `FAIL no listener on
   [2a01:4f8:1c1b:8cd4::1]:443`. (Its `Caddyfile INVALID` line as deploy comes from `caddy
   validate` opening `/var/log/caddy/access.log`; the unmodified script fails it the same way.
   Run it as root.)
4. Caddy accepts the pair. `caddy adapt` on the box with both addresses lists `178.105.182.17:443`
   and `[2a01:4f8:1c1b:8cd4::1]:443`; with the IPv4 alone, only the first.
5. Tests: `tests/unit/scripts/test_caddy_bind_env.py`, the writers' real lines and both readers
   against both forms. One mutant per writer and reader, each fails a test.
6. Not verified until applied: that systemd hands Caddy both addresses from the quoted line
   (systemd's documented quoting; macOS cannot run it).

**Apply, after the deploy** (the box checkout then has the fixed apply-edge.sh). With the operator's
go:

1. Tell the orrery session first: all public sites, podcast and orrery, drop for the seconds of the
   restart. Then, as root: `apply-edge.sh --dry-run` (expect "write …10-public-bind.conf" and
   "restart caddy (service env changed)"), then `apply-edge.sh`. It rewrites the drop-in, runs
   `daemon-reload` and restarts Caddy. A bind change does not apply on reload; see
   `docs/guides/EDGE_CONVERGENCE_RUNBOOK.md`.
2. Check: `systemctl show caddy -p Environment` lists both addresses; `ss -ltn` shows Caddy on
   `[2a01:4f8:1c1b:8cd4::1]:80/:443` as well; `curl -6 -sk --resolve
   closelistening.app:443:[2a01:4f8:1c1b:8cd4::1] https://closelistening.app/ -o /dev/null -w
   '%{http_code}'` from the box answers like the IPv4 address; `verify-edge.sh` passes.
3. Undo: put back the old drop-in line, `daemon-reload`, restart Caddy.

## 2. Ship `unscoped.alloy` (logs of containers no app drop-in claims)

Pushed in `76bec9274`; it goes live on the next `deploy-config.yml` run (any surface). Operator's
go. Check: VictoriaLogs has `app:unscoped` streams for `alloy`, `cadvisor` and any one-off
container, and none for an app container (no double shipping). The first run sends about 19k lines
of old history once. cadvisor's "failed to collect filesystem stats" errors will show; they are
real.

## 3. Install the prod-ops-health timer (#1877)

The commit (`8acc95d72`, local until it is pushed WITH the deploy, since the push stops the GitHub
cron) needs the wrapper in the box checkout, which the deploy brings. As root, the three install
lines in `infra/systemd/README.md`, then `systemctl start prod-ops-health.service` and read
`journalctl -u prod-ops-health`: every check listed, "metrics pushed", exit 0. The homelab page
then shows a fresh run. If a 06:00 passes between the push and the install, the 27 h stale rule
fires; that is the safety net working.
