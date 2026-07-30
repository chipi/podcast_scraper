# container-metrics — prod container inventory for the homelab landing page

Emits the per-box **container inventory** the homelab landing page
(`agentic-ai-homelab/infra/homelab-home`) renders — the **Containers table**
(name · traffic light · uptime · port) and the per-app colour rollup — for the
**prod VPS**.

## Why this exists (prod is the odd one out)

The mini reads its own containers (`agentic-ai-homelab/infra/mini-metrics`) and
the DGX's are read from the mini over keyless Tailscale SSH
(`agentic-ai-homelab/infra/dgx-scrape`). **Neither works for prod:** there is no
mini→prod SSH (prod keeps classic key-based SSH, out of the Tailscale-SSH ACL),
and prod's cAdvisor exposes only cgroup ids (no container names). So the same
logic has to run **on prod** and push to the mini's VictoriaMetrics. This dir is
that collector.

It is the exact `ctr.py` + emit logic used on the mini/DGX, ported to a Linux
push loop — so all three boxes produce identical metrics and the landing page
treats them the same.

## What it emits (box="prod")

| Metric | Meaning |
| --- | --- |
| `compose_app_up/running/total{app,box="prod"}` | Per compose-project rollup (traffic light). `total` **excludes cleanly-exited one-shots** (`Exited (0)` — migrate/init), falling back to the raw count when nothing runs so a fully-stopped project still reads red. |
| `container_uptime_seconds{app,name,port,state,box="prod"}` | Per container: name, state (colour), uptime seconds, published host port (blank for host-network). |

Push target: `http://homelab:8428/api/v1/import/prometheus` over the tailnet.
`tag:prod → tag:homelab-host:8428` is already granted in the ACL, and prod's node
Alloy already pushes metrics there. Override with `VM_URL` if needed.

## Deploy (prod owns this — pick your mechanism)

Needs on the box: `docker` (socket access), `python3`, `curl`.

systemd (recommended) — install alongside the other vps-observability bits:

```sh
sudo install -Dm755 push.sh /opt/vps-observability/container-metrics/push.sh
sudo install -Dm644 ctr.py /opt/vps-observability/container-metrics/ctr.py
sudo install -Dm644 container-metrics.service /etc/systemd/system/container-metrics.service
sudo systemctl daemon-reload
sudo systemctl enable --now container-metrics.service
sudo journalctl -u container-metrics -f   # verify it's pushing
```

If the install path differs from `/opt/vps-observability/container-metrics`, edit
`ExecStart` in the unit accordingly.

Alternatively, wire it into the existing prod deploy (matches the alloy drop-ins
in `infra/deploy/deploy.sh`, which `cp` + reload on each deploy): copy both
`push.sh` and `ctr.py` into place and `systemctl restart container-metrics` from
the deploy script, so a redeploy ships updates.

## Verify (from any tailnet host)

```sh
curl -sG "http://homelab:8428/api/v1/query" \
  --data-urlencode 'query=count(container_uptime_seconds{box="prod"})'
```

A result `> 0` means the prod Containers table on the landing page lights up
automatically. The landing-page UI is already prepped (`agentic-ai-homelab` — the
prod `pctr` table + `ctable('pctr','prod')`); it shows "no data" until this
collector runs, then populates with no further UI change.

## Relation to the node Alloy

Deliberately standalone — Alloy has no compose-project / uptime exporter, and the
mini/DGX use the same standalone-loop pattern. It does not touch `operator.alloy`
/ `player.alloy` or the shared `base.alloy`.
