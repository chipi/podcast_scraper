#!/usr/bin/env bash
# Does prod need its secrets restaged / containers recreated? READ-ONLY. Prints one JSON object.
#
# The signal for the homelab remediation fleet (agentic-ai-homelab RFC-0005). The fleet reaches this
# over SSH with a key whose authorized_keys entry is pinned to exactly this command:
#
#   restrict,command="/srv/podcast-scraper/scripts/ops/prod_recovery_check.sh" ssh-ed25519 … remediation-fleet
#
# so the key can do nothing else. When "needs_recovery" is true the fleet dispatches
# restage-prod-secrets.yml (surfaces=all, recreate=true) and approves that one run.
#
# Why a direct check and not metrics: on 2026-09-30 the box rebooted and alloy did not come back,
# so every metric was dark — indistinguishable from "box down". This looks at the box itself.
#
# The "is it broken" rules come from prod_health_lib.sh, the same file restage_prod_recreate.sh
# acts on, so this never reports something the recovery would not fix.
#
# Nothing here writes, restarts or reads a secret's CONTENT: only whether the secret files exist.
set -euo pipefail

SHM="${SHM_DIR:-/dev/shm}"
PROC_STAT="${PROC_STAT:-/proc/stat}"
# shellcheck source=scripts/ops/prod_health_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/prod_health_lib.sh"

boot_time="$(awk '/^btime /{print $2}' "$PROC_STAT" 2>/dev/null || true)"

needs=false
secrets_json=""
for s in $PROD_SURFACES; do
    n="$(prod_secret_file_count "$s")"
    [ "$n" -eq 0 ] && needs=true
    secrets_json="${secrets_json:+${secrets_json},}\"${s}\":${n}"
done

down_json=""
keyless=false
for s in $PROD_SURFACES; do
    proj="$(prod_project_for "$s")"
    while IFS='|' read -r name svc; do
        [ -n "$name" ] || continue
        needs=true
        down_json="${down_json:+${down_json},}\"${name}\""
        if [ "$proj" = compose ] && [ "$svc" = api ]; then
            status="$(docker ps -a --filter "name=^${name}$" --format '{{.Status}}' 2>/dev/null || true)"
            case "$status" in Up*) keyless=true ;; esac
        fi
    done <<< "$(prod_broken_containers "$proj" || true)"
done

alloy=false
[ -n "$(docker ps --filter 'name=^alloy$' --filter status=running --format '{{.Names}}' 2>/dev/null || true)" ] && alloy=true

printf '{"schema":"prod_recovery_check/v1","boot_time":%s,"secrets_dirs":{%s},"down_containers":[%s],"keyless_control_plane":%s,"alloy_running":%s,"needs_recovery":%s}\n' \
    "${boot_time:-null}" "$secrets_json" "$down_json" "$keyless" "$alloy" "$needs"
