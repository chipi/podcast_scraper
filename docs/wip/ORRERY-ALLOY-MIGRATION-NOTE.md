# Note for the orrery agent — migrate off grafana-agent to the shared node Alloy

**From:** podcast_scraper infra. **Date:** 2026-07-23. **Ref:** ADR-121, issue #1268.

## Why (context)

The prod VPS now runs **one** node Alloy (the shared log "router" — reads the Docker
socket) against a config **directory** `/etc/alloy/config.d/`, into which each app drops
its own rules — exactly like the Caddy edge (`/etc/caddy/sites/<app>.caddy`). This is
already **deployed and live** (base config in place, podcast unaffected).

Your `orrery-grafana-agent` (grafana/agent **v0.43.4**, EOL) is now redundant **and** is
the source of a log-mislabel bug: its `docker_sd` keep-filter leaks, so it tails the
podcast API + cAdvisor containers and stamps them `app=orrery` (~48% of `app=orrery`
volume was actually prod-podcast infra). **Retire it** and contribute an Alloy rule
fragment instead.

On the box, `/opt/vps-observability/config.d/` is **deploy-writable** (owned by `deploy`,
in the `docker` group), so your deploy can drop the fragment + reload Alloy **without root**.

## What to do — exactly

### 1. Add `ops/observability/orrery.alloy` to the orrery repo
Validated on the box with the real `grafana/alloy:v1.17.0` binary (`fmt` + directory
`run`). **Alloy blocks are newline-separated, NOT comma-separated** (don't Prometheus-YAML it):

```alloy
// orrery.alloy — orrery log rules (ADR-121). Dropped into /etc/alloy/config.d/ on deploy.
// References the shared components in base.alloy: discovery.docker "app" + loki.write "logs_sink".
discovery.relabel "orrery" {
  targets = discovery.docker.app.targets
  rule {
    source_labels = ["__meta_docker_container_name"]
    regex         = "/(orrery-web|orrery-pipeline-runner-.*)"
    action        = "keep"
  }
  rule {
    source_labels = ["__meta_docker_container_name"]
    regex         = "/orrery-web"
    target_label  = "surface"
    replacement   = "web"
  }
  rule {
    source_labels = ["__meta_docker_container_name"]
    regex         = "/orrery-pipeline-runner-.*"
    target_label  = "surface"
    replacement   = "pipeline"
  }
  rule {
    source_labels = ["__meta_docker_container_name"]
    target_label  = "container"
    regex         = "/(.*)"
    replacement   = "$1"
  }
}
loki.source.docker "orrery" {
  host       = "unix:///var/run/docker.sock"
  targets    = discovery.relabel.orrery.output
  forward_to = [loki.write.logs_sink.receiver]
  labels     = { app = "orrery" }
}
```
(If you still want the `data-refresh.log` file source, add a `local.file_match` +
`loki.source.file` for `/rootfs/var/log/orrery/data-refresh.log` forwarding to
`loki.write.logs_sink.receiver` with `job = "orrery-data-refresh"`.)

### 2. Wire it into orrery's deploy (replaces the old agent)
In your deploy script/workflow, **in the same deploy** (so logs don't double-collect):
- **Drop the fragment** (deploy-writable dir, no root):
  ```
  scp ops/observability/orrery.alloy  deploy@<box>:/opt/vps-observability/config.d/orrery.alloy
  ssh deploy@<box>  docker kill -s HUP alloy      # reload the node Alloy (deploy is in docker group)
  ```
- **Delete the old agent** from `compose/docker-compose.prod.yml`: remove the entire
  `grafana-agent:` service (the `orrery-grafana-agent` container), and stop it on the box:
  ```
  ssh deploy@<box>  docker rm -f orrery-grafana-agent
  ```

### 3. Clean up (same PR)
Delete the now-unused files: `ops/observability/grafana-agent.yaml`,
`grafana-agent.silent.yaml`, `agent-entrypoint.sh` — they belonged to the retired agent.

**Ordering matters:** drop `orrery.alloy` + remove `orrery-grafana-agent` **together**.
If you drop the fragment while the old agent still runs, orrery logs are collected twice
(once correct via node Alloy, once mislabeled via the old agent).

## How to test

**Pre-deploy (you can run this in CI / locally with Docker):**
```
docker run --rm -v "$PWD/ops/observability:/cfg:ro" grafana/alloy:v1.17.0 fmt /cfg/orrery.alloy
# exit 0 + no output = syntax valid
```

**Post-deploy, on the box (deploy SSH — you don't need root or tailnet):**
```
ssh deploy@<box> '
  docker ps --format "{{.Names}}" | grep orrery-grafana-agent && echo "STILL RUNNING (bad)" || echo "old agent gone ✓"
  docker inspect alloy --format "alloy: {{.State.Status}} restarts={{.RestartCount}}"     # running, restarts=0
  ls /opt/vps-observability/config.d/     # base.alloy + orrery.alloy present
  docker logs alloy --tail 20 2>&1 | grep -iE "orrery|could not perform|level=error.*load" | head
'
```
Expect: old agent gone, `alloy running restarts=0`, `orrery.alloy` present, no config-load errors.

**The label check (needs tailnet access to VictoriaLogs — ping the operator / prod agent if you're origin-locked out):**
```
# these run against homelab VictoriaLogs :9428 (tailnet-only)
# 1) the mislabeled stream should DROP to ~0 within a few minutes:
_time:10m app:orrery -container:orrery-web -surface:pipeline | stats count() as n     # expect ~0
# 2) app=orrery should carry ONLY orrery-web / pipeline:
_time:10m app:orrery | stats by (container, surface) count() as n
# 3) sanity — orrery-web logs still arriving:
_time:5m app:orrery container:orrery-web | stats count() as n     # > 0
```

**Success =** old agent gone, Alloy healthy, orrery-web logs still flowing under
`{app=orrery, container=orrery-web, surface=web}`, and the bare `{app=orrery,env=prod}`
infra-noise stream is gone.

## Notes
- You do **not** touch `base.alloy` or the podcast sources — only your `orrery.alloy`.
- Reload is `docker kill -s HUP alloy` (hot reload, no restart, no gap). If Alloy ever
  rejects your fragment, it keeps the last-good config running — check `docker logs alloy`.
- Future: the player will ship `player.alloy` the same way.
