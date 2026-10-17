#!/usr/bin/env bash
# Read-only access to the prod VPS, with the details that keep going wrong pinned in one place.
#
#   prod.sh run '<command>'                    one shell command on the box, as deploy@
#   prod.sh compose <compose args...>          docker compose of the podcast stack, --env-file included
#   prod.sh fetch-archive <archive key> <dest>  one original from the audio archive -> local <dest>
#
# Read-only is the caller's promise, not something this script can check: anything that changes
# prod needs the operator's go, and deploys belong to the prod session (see SKILL.md).
#
# Pinned here:
#   - deploy@ with ~/.ssh/podcast_prod_operator and IdentitiesOnly. Never try other usernames:
#     the box runs fail2ban and failed logins ban this machine's tailscale IP (2026-09-19).
#   - compose needs --env-file /srv/podcast-scraper/.env. Compose reads .env beside the first
#     compose file (compose/), not the cwd, and fails with "required variable
#     PODCAST_DOCKER_PROJECT_DIR is missing a value" without it (2026-10-17).
#   - fetch-archive works in a uniquely named /tmp dir on the box and deletes it on every exit
#     path, then checks it is gone.
set -euo pipefail

HOST=${PROD_HOST:-prod-podcast}
KEY=${PROD_SSH_KEY:-$HOME/.ssh/podcast_prod_operator}
SSH=(ssh -o IdentitiesOnly=yes -o ConnectTimeout=10 -i "$KEY" "deploy@$HOST")
APP=/srv/podcast-scraper
COMPOSE="docker compose --env-file $APP/.env -f compose/docker-compose.stack.yml \
-f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml"

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

cmd=${1:-}
[ -n "$cmd" ] || usage
shift

case "$cmd" in
  run)
    [ $# -eq 1 ] || usage
    "${SSH[@]}" "$1"
    ;;
  compose)
    [ $# -ge 1 ] || usage
    "${SSH[@]}" "cd $APP && $COMPOSE $(printf '%q ' "$@")"
    ;;
  fetch-archive)
    [ $# -eq 2 ] || usage
    key=$1 dest=$2
    case "$key" in *..*|/*) echo "refusing archive key '$key'"; exit 2 ;; esac
    tmp="/tmp/prod-readonly-$(date +%s)-$$"
    # The pipeline image the box already runs: it carries rclone and reads the archive creds from
    # .env. Whatever tag is newest locally on the box — nothing is pulled.
    image=$("${SSH[@]}" "docker images --format '{{.Repository}}:{{.Tag}}' \
      | grep -m1 'podcast-scraper-stack-pipeline-llm:sha-'") \
      || { echo "FAIL: no pipeline-llm image on the box"; exit 1; }
    cleanup() {
      "${SSH[@]}" "rm -rf '$tmp'; [ ! -e '$tmp' ] && echo 'OK: removed $tmp on the box'" \
        || echo "WARN: could not confirm $tmp is gone on the box — remove it by hand"
    }
    trap cleanup EXIT
    echo "--> $key via $image"
    "${SSH[@]}" "mkdir -p '$tmp' && chmod 777 '$tmp' && docker run --rm \
      --env-file $APP/.env -v '$tmp':/out --entrypoint rclone '$image' \
      copyto 'audioarchive:$key' /out/file 2>&1 | grep -v NOTICE; ls -la '$tmp/file'"
    mkdir -p "$(dirname "$dest")"
    scp -q -o IdentitiesOnly=yes -i "$KEY" "deploy@$HOST:$tmp/file" "$dest"
    echo "OK: $(wc -c < "$dest" | tr -d ' ') bytes -> $dest"
    echo "    real episode audio: keep it out of git, delete it when done"
    ;;
  *) usage ;;
esac
