#!/usr/bin/env bash
# Recreate prod containers that are DOWN, at the image tag they are already on.
#
# Runs ON prod, invoked by .github/workflows/restage-prod-secrets.yml after the tmpfs
# secrets have been restaged. Separate from the workflow so it is shellcheckable and can be
# run by hand during an incident:
#
#   SELECTED="operator player" bash scripts/ops/restage_prod_recreate.sh
#
# Two invariants, both learned the hard way:
#
#   1. NEVER move the image. Recovery must be code-neutral — resolving "newest from main"
#      here would ship an untested image in the middle of an incident. The tag is read off
#      the container that is already there.
#   2. NEVER blanket-recreate. Only containers actually Exited/Restarting/Created are
#      touched; healthy siblings are left alone (--no-deps).
#
# See #2080.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/srv/podcast-scraper}"
SELECTED="${SELECTED:-podcast operator player}"
cd "$REPO_DIR"

# The deployed tag of a project, read from a container that already exists. Prefers a
# running one, falls back to any (including the Exited container we are about to replace —
# which is exactly the tag we want to put back).
running_tag() {
    local proj="$1" t
    t=$(docker ps --filter "label=com.docker.compose.project=${proj}" \
          --format '{{.Image}}' 2>/dev/null | grep -oE 'sha-[0-9a-f]{7}' | head -1 || true)
    if [ -z "$t" ]; then
        t=$(docker ps -a --filter "label=com.docker.compose.project=${proj}" \
              --format '{{.Image}}' 2>/dev/null | grep -oE 'sha-[0-9a-f]{7}' | head -1 || true)
    fi
    printf '%s' "$t"
}

# Services of a project that are not running. Status strings look like
# "Exited (127) 9 hours ago" / "Restarting (1) 3 seconds ago" / "Up 2 minutes (healthy)".
broken_services() {
    docker ps -a --filter "label=com.docker.compose.project=$1" \
        --format '{{.Label "com.docker.compose.service"}}|{{.Status}}' 2>/dev/null \
        | grep -E '\|(Exited|Restarting|Created)' \
        | cut -d'|' -f1 | sort -u
}

rc=0
for s in $SELECTED; do
    case "$s" in
        podcast)  proj=compose  ;;
        operator) proj=operator ;;
        player)   proj=player   ;;
        *) echo "  skipping unknown surface '$s'"; continue ;;
    esac

    svcs="$(broken_services "$proj" || true)"
    if [ -z "$svcs" ]; then
        echo "  ${proj}: nothing down — skipping"
        continue
    fi

    tag="$(running_tag "$proj")"
    if [ -z "$tag" ]; then
        echo "::warning::${proj}: no deployed sha- tag found; skipping rather than guessing an image"
        rc=1
        continue
    fi

    # shellcheck disable=SC2086
    echo "  ${proj}: recreating [$(echo $svcs | tr '\n' ' ')] at PODCAST_IMAGE_TAG=${tag}"

    case "$proj" in
        compose)
            C=(docker compose --env-file .env
               -f compose/docker-compose.stack.yml
               -f compose/docker-compose.prod.yml
               -f compose/docker-compose.vps-prod.yml
               -f compose/docker-compose.secrets.yml)
            ;;
        operator)
            C=(docker compose -p operator --env-file .env.operator
               -f compose/docker-compose.operator-public.yml)
            [ -d /dev/shm/operator-secrets ] && C+=(-f compose/docker-compose.operator-secrets.yml)
            ;;
        player)
            C=(docker compose -p player --env-file .env.player
               -f compose/docker-compose.player-public.yml)
            [ -d /dev/shm/player-secrets ] && C+=(-f compose/docker-compose.player-secrets.yml)
            ;;
    esac

    # --no-deps: leave healthy siblings alone. No --pull: the image must not move.
    # shellcheck disable=SC2086
    if ! PODCAST_IMAGE_TAG="$tag" "${C[@]}" up -d --no-deps --force-recreate $svcs; then
        echo "::warning::${proj}: recreate reported a failure — see the AFTER state"
        rc=1
    fi
done

exit "$rc"
