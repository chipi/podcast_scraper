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

  # --- per-container CPU% (container_cpu_percent) — hang detection BY NAME (P1) ---
  # A wedged container still "runs": its CPU either pegs near 100% (spin) or flatlines at 0%
  # while it holds the corpus lock. The uptime/rollup metrics above can't tell "working" from
  # "stuck" — per-name CPU can. `docker stats` reports only running containers, which is exactly
  # the set that can hang. The `---` sentinel splits the id->project map from the stats lines.
  # We join on the container ID, not the name: `docker ps` .Names can be a comma list for an
  # aliased container while `docker stats` .Name is the single primary — joining on the stable
  # ID keeps {app} correct in that case. A CPUPerc of "--" (container has no cgroup stats yet,
  # e.g. mid-restart) is SKIPPED: emitting `... --` writes an invalid Prometheus value that
  # VictoriaMetrics silently drops, which would blind exactly this hang-detection metric.
  {
    "$DOCKER" ps --format '{{.ID}}|{{.Label "com.docker.compose.project"}}'
    echo '---'
    "$DOCKER" stats --no-stream --format '{{.ID}}|{{.Name}}|{{.CPUPerc}}'
  } 2>/dev/null | awk -F'|' '
      /^---$/ { seen_sep=1; next }
      seen_sep==0 { proj[$1]=$2; next }
      {
        id=$1; name=$2; cpu=$3; gsub(/%/,"",cpu);
        if (cpu == "--" || cpu == "") next;
        app=(id in proj)?proj[id]:"";
        printf "container_cpu_percent{app=\"%s\",name=\"%s\",box=\"prod\"} %s\n", app, name, cpu
      }' \
    | curl -s -m8 -o /dev/null --data-binary @- "$VM_URL" || true

  sleep "$INTERVAL"
done
