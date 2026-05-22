# VPS multi-app onboarding (Docker Compose)

Small guide for running **additional** Docker Compose applications on the same
Hetzner + Tailscale host as
[podcast_scraper prod](PROD_RUNBOOK.md), **without** provisioning new IaaC.
New apps use the same GitOps shape: **CI → SSH over tailnet → `compose pull && up`**.

## Prerequisites

- Prod VPS already exists; you reach it as `deploy@prod-podcast.<tailnet>.ts.net`
  (see [Prod runbook](PROD_RUNBOOK.md)).
- You can add **GitHub Actions** secrets (`TS_AUTHKEY`, SSH key or reuse deploy)
  and optional **Tailscale ACL** updates in
  [`tailscale/policy.hujson`](https://github.com/chipi/podcast_scraper/blob/main/tailscale/policy.hujson).

## 1. Isolate each app on disk

Pick a dedicated root per app (example):

| App | Suggested path | Notes |
| --- | --- | --- |
| podcast_scraper (existing) | `/srv/podcast-scraper` | Do not share `.env` or data dirs with other apps. |
| Other app | `/srv/<app-slug>` | Clone or `rsync` repo; keep `compose/` and `.env` under this tree. |

Rules:

- One **`.env` per app**, mode `600`, owned by the user that runs `docker compose`.
- **Named volumes** and bind mounts must not collide with podcast_scraper (`docker volume ls`, `docker compose ps`).

## 2. Ports and Tailscale exposure

- Assign each stack **distinct host ports** (for example app A on `8081`, app B on `8082`).
- **Do not** open Hetzner public TCP 80/443 for hobby stacks; keep ingress **tailnet-only**
  (same security model as [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)).

Expose over Tailscale:

- Per-app **`tailscale serve`** (separate local port per app), or
- Extra MagicDNS names if you register additional hostnames (operational detail in Tailscale admin).

Document the stable URL you chose (`https://<name>.<tailnet>.ts.net/`) in the app repo README.

## 3. Compose invocation (match podcast_scraper lessons)

If the **first** `-f` file lives under `compose/`, Compose’s project dir may resolve to
`.../compose/` — see the [Prod runbook](PROD_RUNBOOK.md) FAQ section *Why `--env-file`?*.

When running compose by hand on the VPS, prefer:

```bash
cd /srv/<app-slug>
docker compose --env-file /srv/<app-slug>/.env -f compose/docker-compose.yml up -d
```

Adjust `-f` list to your project. **systemd** `EnvironmentFile=` for that unit avoids the pitfall for boot-time `up`.

## 4. systemd (optional but recommended)

Add a **separate** unit per app, mirroring `podcast-scraper.service`:

- `WorkingDirectory=/srv/<app-slug>`
- `EnvironmentFile=/srv/<app-slug>/.env`
- `ExecStart=/usr/bin/docker compose ... up -d --remove-orphans`
- `ExecStop=/usr/bin/docker compose ... down`

Enable after `.env` exists and a one-time `docker compose pull` has succeeded.

## 5. GitOps from the **other** repository

Copy the pattern from
[`deploy-prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/deploy-prod.yml):

1. Job joins tailnet: `tailscale/github-action@v2` with `secrets.TS_AUTHKEY`.
2. SSH as `deploy` to the **same** VPS FQDN.
3. `cd /srv/<app-slug> && git pull && docker compose ... pull && up -d`.
4. Optional: health-check `curl` against tailnet URL or in-container probe.

Keep deploy keys or `deploy` user access scoped; prefer **read-only** deploy where possible.

## 6. Tailscale ACL

Ensure the `tag:gha-deployer` (or your runner identity) can SSH to `tag:prod` (already required for podcast_scraper).
No change needed if you reuse the same SSH target and user.

## 7. Observability and cost

- **RAM/CPU**: extra stacks compete with podcast_scraper; watch `docker stats` after cutover.
- **Logs/metrics**: either piggyback Grafana Agent patterns from this repo or accept “logs on host only” for small apps.
- **Backups**: define per-app backup (same idea as corpus snapshots — script + GHA + object store or Releases).

## 8. Rollback

Same idea as prod: pin image tags or prior `git` SHA on the host, then `compose up -d`.
Document `PODCAST_IMAGE_TAG`-style variable names per app.

## References

- [Prod runbook](PROD_RUNBOOK.md) — bootstrap, health checks, constraints.
- [Prod operator cheat sheet](PROD_OPERATOR_CHEAT_SHEET.md) — quick commands.
- [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) — hosting and GitOps decisions.
- [RFC-087](../rfc/RFC-087-vps-public-edge-multi-compose.md) — optional public TLS edge + multi-vhost
  design (operators stay on Tailscale).
