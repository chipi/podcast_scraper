# Tailnet & networking runbook

**Audience:** any agent or human touching the tailnet — this repo (`podcast_scraper`),
the homelab repo (`agentic-ai-homelab`), orrery, or a future project sharing this tailnet.
**Goal:** know how the network is wired, where the source of truth lives, and the exact
procedure for the common changes (grant access, expose a service over HTTPS, add a device).

If you are an agent from another repo and you need a host to reach another host over the
tailnet, or to expose a service on it: **you change one file here and open a PR.** Details below.

---

## TL;DR

- The tailnet is a private WireGuard mesh (Tailscale, tailnet `tail6d0ed4.ts.net`). Nothing
  here is public — reachability requires being on the tailnet **and** an ACL grant.
- **The ACL is the single source of truth in this repo: `tailscale/policy.hujson`.** It is
  default-deny; only listed `src → dst:port` grants are allowed.
- **It is applied by GitOps, not OpenTofu** ([ADR-128](../adr/ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md)):
  open a PR → the workflow runs a dry-run **`test`**; merge to `main` → it **`apply`**s to the
  live tailnet. Workflow: `.github/workflows/tailscale-acl.yml`.
- **OpenTofu manages the cloud infra** (the Hetzner VPS, firewall, network, volumes) and the
  VPS's own tailnet **join key** — but **not** the ACL. So an ACL edit never triggers a
  full-estate `tofu apply`.

## What's managed by what

| Thing | Managed by | Source | Applied by |
| --- | --- | --- | --- |
| Tailnet **ACL** (who reaches what) | **GitOps** | `tailscale/policy.hujson` | `tailscale-acl.yml` (PR→test, merge→apply) |
| Hetzner VPS + firewall + network + volumes | OpenTofu | `infra/terraform/*.tf` | `infra-apply.yml` (manual, gated) |
| VPS tailnet **join key** (`tailscale_tailnet_key`) | OpenTofu | `infra/terraform/tailscale.tf` | `infra-apply.yml` |
| Per-host **`tailscale serve`** HTTPS mounts | Host-local runtime state | on the host (not IaC) | `tailscale serve` on that host |

## Hosts and tags

| Tag | Host | Tailnet IP | Notes |
| --- | --- | --- | --- |
| `tag:prod` | prod VPS (`prod-podcast`) | `100.124.111.115` | player/operator/api; public via Cloudflare→Caddy |
| `tag:homelab-host` | homelab Mac mini (`homelab`) | `100.87.33.61` | self-hosted o11y (Grafana, VictoriaMetrics/Logs/Traces, GlitchTip, Umami, Langfuse) |
| `tag:dgx-llm-host` | DGX Spark (`dgx-llm-1`) | `100.69.49.126` | Ollama + inference + GPU/host exporters |
| `tag:gha-deployer` | ephemeral GitHub Actions runners | (dynamic) | deploy / backup / drill / ops-event push |
| `tag:dr-drill` | throwaway DR-drill VPS | (dynamic) | disaster-recovery rehearsals only |

`autogroup:admin` = the operator's own devices (laptop/phone/iPad); they reach everything.
Tag ownership is declared in `policy.hujson` `tagOwners`.

---

## Runbook 1 — grant a host access to another host's ports

**Use when:** a host needs to reach ports it currently can't (e.g. homelab → DGX exporters).

1. Edit `tailscale/policy.hujson`, add (or extend) an `acls` entry:

   ```jsonc
   {
     "action": "accept",
     "src":    ["tag:homelab-host"],
     "dst":    ["tag:dgx-llm-host:9400", "tag:dgx-llm-host:8080"]
   }
   ```

   Keep the comment convention above each rule (what/why). Ports are comma-separated per dst.
2. Open a PR. The **`test`** run validates the policy + posts a dry-run. Review it.
3. Merge to `main`. The **`apply`** run syncs it. (On-demand: `gh workflow run "Tailscale ACL" -f mode=apply`.)
4. Verify from the *source* host (the grant is directional):

   ```sh
   nc -z -w4 <dst-tailnet-ip> <port> && echo ":<port> open"
   ```

## Runbook 2 — expose a service over HTTPS on the tailnet

**Use when:** a host runs a service on loopback and you want other tailnet devices to reach it
over HTTPS (tailnet-only — never public). This is the homelab o11y pattern.

**Mechanism:** `tailscale serve` on the host terminates TLS with the tailnet cert and proxies to
the local service. **`serve`, never `funnel`** (funnel is public). Two patterns:

- **Path mount on `:443`** — for APIs / path-tolerant apps. Strips the path prefix before proxying:

  ```sh
  # on the host (no sudo needed for serve):
  tailscale serve --bg --https=443 --set-path=/grafana http://127.0.0.1:3000
  # → https://homelab.tail6d0ed4.ts.net/grafana  (proxies to :3000, prefix stripped)
  ```

  Breaks web UIs that emit root-absolute asset URLs — for those, either set the app's
  external-URL/base (e.g. Grafana `root_url`, Langfuse `AUTH_TRUST_HOST`) **or** use ⬇.
- **Dedicated TLS port** — for web UIs with root-absolute assets:

  ```sh
  tailscale serve --bg --https=8443 --set-path=/ http://127.0.0.1:4000
  # → https://homelab.tail6d0ed4.ts.net:8443/  (Langfuse, root path)
  ```

  A non-443 port **needs an ACL grant** (Runbook 1) — add `<port>` to the host's dst list, e.g.
  `"dst": ["tag:homelab-host:...,8443,..."]`. `:443` is usually already granted.

To change a mount: re-run with the new target; remove: `tailscale serve --https=443 --set-path=/x off`.
Note: the serve map lives in **tailscaled state, not the repo** — it persists across reboots but
isn't captured as code (a fresh host must re-apply it).

## Runbook 3 — add a new tailnet device or tag

1. New **tag**: add it to `policy.hujson` `tagOwners` (usually `["autogroup:admin"]`) + the
   comment block, and add the `acls` grants it needs. PR → merge (applies).
2. New **device**: join it with a tagged auth key (`tailscale up --advertise-tags=tag:<name>`).
   For the prod VPS this is `tailscale_tailnet_key` in OpenTofu; for a personal device, an
   admin-console auth key. The ACL grants then govern what it can reach.

## Verify after any ACL change

```sh
# reachability (from the SOURCE host of the grant):
nc -z -w4 <dst-ip> <port> && echo open
# for a homelab o11y target, confirm data is landing:
curl -sG "http://localhost:8428/api/v1/query" --data-urlencode 'query=count({instance="<host>"})'
```

---

## Cross-repo note (for the homelab / orrery / other-project agent)

The **tailnet ACL lives in `podcast_scraper`** (`tailscale/policy.hujson`) — it is the one
tailnet-wide policy, so it can't be split per project. If your project needs a tailnet grant
(e.g. homelab → DGX, or a new service port), **propose the change here**: edit `policy.hujson`,
open a PR against `podcast_scraper`, and the GitOps action applies it on merge. Do the
host-side work (the `tailscale serve` mount, the service, the exporter) in your own repo; the
**ACL grant is the only thing that lives here.** A clean handover names the exact rule + the
verify command (see the runbooks above).

Direct pushes of `policy.hujson` to `main` **auto-apply** (push→apply) and skip the PR `test`
gate — prefer a PR so the dry-run runs first.

## Gotchas (learned the hard way)

- **`--set-path=/x` strips the `/x` prefix** before proxying — great for APIs, breaks web UIs
  that emit root-absolute asset URLs (fix via app base-URL or a dedicated TLS port).
- **A non-443 serve port needs a matching ACL grant** — the serve mount alone isn't enough;
  other devices are ACL-denied until the port is granted.
- **`tailscale serve` needs no sudo** and no GUI consent over SSH (the real binary is
  `/Applications/Tailscale.app/Contents/MacOS/Tailscale` on the mini).
- **The ACL is tailnet-wide** — a grant to `tag:homelab-host:9428` opens that port from *every*
  homelab-tagged host; scope `src`/`dst` deliberately.
- **`policy.hujson` is HuJSON** (JSON + comments + trailing commas). The GitOps `test` catches
  syntax errors before merge.

## References

- [ADR-128](../adr/ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md) — ACL decoupled from OpenTofu → GitOps
- [ADR-083](../adr/ADR-083-tailscale-private-ingress-always-on-vps.md) — Tailscale private ingress
- [ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md) — shared public edge (what's public vs tailnet)
- [HOSTING_AND_INFRASTRUCTURE](../architecture/HOSTING_AND_INFRASTRUCTURE.md) — the full estate architecture
- `tailscale/policy.hujson` · `.github/workflows/tailscale-acl.yml`
