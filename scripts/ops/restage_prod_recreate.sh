#!/usr/bin/env bash
# Recreate prod containers that are DOWN, each at the image tag it is already on.
#
# Runs ON prod, invoked by .github/workflows/restage-prod-secrets.yml after the tmpfs
# secrets have been restaged. Separate from the workflow so it is shellcheckable and can be
# run by hand during an incident:
#
#   SELECTED="operator player" bash scripts/ops/restage_prod_recreate.sh
#
# Invariants, all learned the hard way:
#
#   1. NEVER move the image. Recovery must be code-neutral. The tag is read off the broken
#      CONTAINER ITSELF, never off a sibling: a project can hold several tags at once (an
#      obs-only deploy left player-obs on a newer sha than player-api), and on 2026-09-30 reading
#      "the project's first running container" silently moved player-api and learning-app to an
#      untested image in the middle of an incident.
#   2. NEVER blanket-recreate. Only broken containers are touched (--no-deps).
#   3. "Up" is not "healthy" for the control plane. After a reboot, podcast-scraper.service
#      recreates compose-api-1 WITHOUT the secrets overlay: it runs, it passes its health check,
#      and it has no provider keys. An api container with no /run/secrets mounts is broken.
#   4. An unknown surface is an error, not a skip. On 2026-09-30 a quoting bug turned
#      "podcast operator player" into "podcast\" "operator\" "player"; the first two were skipped
#      as unknown and the run reported success with operator-api still down.
#
# See #2080.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/srv/podcast-scraper}"
SELECTED="${SELECTED:-podcast operator player}"
SHM="${SHM_DIR:-/dev/shm}"
cd "$REPO_DIR"

container_tag() {
    docker inspect --format '{{.Config.Image}}' "$1" 2>/dev/null \
        | grep -oE 'sha-[0-9a-f]{7}' | head -1 || true
}

has_secret_mounts() {
    docker inspect --format '{{range .Mounts}}{{.Destination}} {{end}}' "$1" 2>/dev/null \
        | grep -q '/run/secrets/'
}

# "name|service" for each container of a project that needs recreating. Status strings look like
# "Exited (127) 9 hours ago" / "Restarting (1) 3 seconds ago" / "Up 2 minutes (healthy)".
broken_containers() {
    local proj="$1" name svc status
    docker ps -a --filter "label=com.docker.compose.project=${proj}" \
        --format '{{.Names}}|{{.Label "com.docker.compose.service"}}|{{.Status}}' 2>/dev/null \
        | while IFS='|' read -r name svc status; do
            case "$status" in
                Exited*|Restarting*|Created*) echo "${name}|${svc}" ;;
                *)
                    if [ "$proj" = compose ] && [ "$svc" = api ] && ! has_secret_mounts "$name"; then
                        echo "${name}|${svc}"
                    fi
                    ;;
            esac
        done
}

rc=0
for s in $SELECTED; do
    case "$s" in
        podcast)
            proj=compose
            if [ ! -d "$SHM/podcast-secrets" ]; then
                echo "::error::compose: /dev/shm/podcast-secrets is missing — restage it first; recreating now would start the control plane without keys"
                rc=1
                continue
            fi
            C=(docker compose --env-file .env
               -f compose/docker-compose.stack.yml
               -f compose/docker-compose.prod.yml
               -f compose/docker-compose.vps-prod.yml
               -f compose/docker-compose.secrets.yml)
            ;;
        operator)
            proj=operator
            C=(docker compose -p operator --env-file .env.operator
               -f compose/docker-compose.operator-public.yml)
            [ -d "$SHM/operator-secrets" ] && C+=(-f compose/docker-compose.operator-secrets.yml)
            ;;
        player)
            proj=player
            C=(docker compose -p player --env-file .env.player
               -f compose/docker-compose.player-public.yml)
            [ -d "$SHM/player-secrets" ] && C+=(-f compose/docker-compose.player-secrets.yml)
            ;;
        *)
            echo "::error::unknown surface '${s}' (want: podcast|operator|player)"
            rc=1
            continue
            ;;
    esac

    broken="$(broken_containers "$proj" || true)"
    if [ -z "$broken" ]; then
        echo "  ${proj}: nothing down — skipping"
        continue
    fi

    while IFS='|' read -r name svc; do
        tag="$(container_tag "$name")"
        if [ -z "$tag" ]; then
            echo "::warning::${name}: no sha- tag on the container; skipping rather than guessing an image"
            rc=1
            continue
        fi
        echo "  ${proj}: recreating ${svc} (${name}) at PODCAST_IMAGE_TAG=${tag}"
        # --no-deps: leave healthy siblings alone. No --pull: the image must not move.
        if ! PODCAST_IMAGE_TAG="$tag" "${C[@]}" up -d --no-deps --force-recreate "$svc" </dev/null; then
            echo "::warning::${name}: recreate reported a failure — see the AFTER state"
            rc=1
        fi
    done <<< "$broken"
done

exit "$rc"
