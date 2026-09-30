#!/usr/bin/env bash
# What "prod needs recovery" means — ONE definition, sourced by both sides.
#
#   scripts/ops/restage_prod_recreate.sh   ACTS on it (recreates what is down)
#   scripts/ops/prod_recovery_check.sh     REPORTS it (read-only; the homelab remediation fleet's
#                                          signal, homelab RFC-0005)
#
# If the check and the recreate disagreed, the fleet could dispatch a recovery that then recreates
# nothing, forever. Keeping the rules here is what stops that.
#
# Sourced, never executed. Callers set SHM (default /dev/shm) before calling.

# shellcheck disable=SC2034  # PROD_SURFACES / PROD_SECRET_DIRS are read by the sourcing scripts
PROD_SURFACES="podcast operator player"

# surface -> compose project name
prod_project_for() {
    case "$1" in
        podcast)  echo compose ;;
        operator) echo operator ;;
        player)   echo player ;;
        *)        return 1 ;;
    esac
}

# Number of non-empty files in a surface's secret dir; 0 when the dir is missing.
prod_secret_file_count() {
    local d="${SHM:-/dev/shm}/$1-secrets"
    [ -d "$d" ] || { echo 0; return 0; }
    find "$d" -maxdepth 1 -type f -size +0c 2>/dev/null | wc -l | tr -d ' '
}

prod_container_tag() {
    docker inspect --format '{{.Config.Image}}' "$1" 2>/dev/null \
        | grep -oE 'sha-[0-9a-f]{7}' | head -1 || true
}

prod_has_secret_mounts() {
    docker inspect --format '{{range .Mounts}}{{.Destination}} {{end}}' "$1" 2>/dev/null \
        | grep -q '/run/secrets/'
}

# "name|service" for each container of a compose project that needs recreating. Status strings
# look like "Exited (127) 9 hours ago" / "Restarting (1) 3 seconds ago" / "Up 2 minutes (healthy)".
#
# "Up" is not "healthy" for the control plane: after a reboot podcast-scraper.service recreates
# compose-api-1 WITHOUT the secrets overlay — running, passing its health check, no keys.
prod_broken_containers() {
    local proj="$1" name svc status
    docker ps -a --filter "label=com.docker.compose.project=${proj}" \
        --format '{{.Names}}|{{.Label "com.docker.compose.service"}}|{{.Status}}' 2>/dev/null \
        | while IFS='|' read -r name svc status; do
            case "$status" in
                Exited*|Restarting*|Created*) echo "${name}|${svc}" ;;
                *)
                    if [ "$proj" = compose ] && [ "$svc" = api ] && ! prod_has_secret_mounts "$name"; then
                        echo "${name}|${svc}"
                    fi
                    ;;
            esac
        done
}
