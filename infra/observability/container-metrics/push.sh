#!/bin/bash
# prod container-metrics collector — emits the per-box container inventory the
# homelab landing page renders (Containers table + apps colouring), for the prod
# VPS. Mirrors the homelab mini's mini-metrics + the DGX's dgx-scrape, which can't
# see prod (no mini->prod SSH; prod's cadvisor exposes only cgroup ids). So this
# runs ON prod and pushes to the mini's VictoriaMetrics over the tailnet.
#
# Emits (box="prod"):
#   compose_app_up/running/total{app,box}   — per compose-project rollup (traffic
#       light); total EXCLUDES cleanly-exited one-shots (Exited (0)), falling back
#       to the raw count when nothing runs so a fully-stopped project still reads red.
#   container_uptime_seconds{app,name,port,state,box}  — per container (via ctr.py):
#       name, state (colour), uptime, published port.
#
# Push target: the mini's VM import endpoint over the tailnet (ACL: tag:prod ->
# tag:homelab-host:8428 already granted; prod's Alloy pushes here too). Override
# with VM_URL if the hostname/port ever changes.
set -u
SELF="$(cd "$(dirname "$0")" && pwd)"
DOCKER=${DOCKER:-docker}
VM_URL=${VM_URL:-http://homelab:8428/api/v1/import/prometheus}
INTERVAL=${INTERVAL:-20}

while true; do
  # --- per-project rollup (compose_app_*) ---
  "$DOCKER" ps -a --format '{{.Label "com.docker.compose.project"}}|{{.State}}|{{.Status}}' 2>/dev/null \
    | awk -F'|' '$1!=""{a[$1]++; if($2=="running")r[$1]++; else if($3 ~ /^Exited \(0\)/)e0[$1]++}
        END{for(p in a){run=r[p]+0; tot=a[p]-(e0[p]+0); if(run==0)tot=a[p]; u=(run==tot && tot>0)?1:0;
          printf "compose_app_up{app=\"%s\",box=\"prod\"} %d\ncompose_app_running{app=\"%s\",box=\"prod\"} %d\ncompose_app_total{app=\"%s\",box=\"prod\"} %d\n",p,u,p,run,p,tot}}' \
    | curl -s -m8 -o /dev/null --data-binary @- "$VM_URL" || true

  # --- per-container detail (container_uptime_seconds) ---
  CFMT=$(printf '{{.Name}}\t{{.State.Status}}\t{{.State.StartedAt}}\t{{index .Config.Labels "com.docker.compose.project"}}\t{{json .NetworkSettings.Ports}}')
  "$DOCKER" inspect $("$DOCKER" ps -aq) --format "$CFMT" 2>/dev/null \
    | python3 "$SELF/ctr.py" prod \
    | curl -s -m8 -o /dev/null --data-binary @- "$VM_URL" || true

  sleep "$INTERVAL"
done
