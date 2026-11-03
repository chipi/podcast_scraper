#!/usr/bin/env bash
# Local Postgres for the test suite: `make test-db-up` / `make test-db-down`.
#
# Starts ONE container named after this checkout (so worktrees do not collide), bound to
# 127.0.0.1 on a free port chosen by docker, and prints the URL to export. `down` removes only
# that container: no prune, no pkill, nothing else on the engine is touched.
set -euo pipefail

IMAGE="${PODCAST_TEST_DB_IMAGE:-postgres:18}"
NAME="podcast-scraper-test-db-$(basename "$(git rev-parse --show-toplevel)")"
USER_NAME=podcast
PASSWORD=podcast

url() {
  local hostport
  hostport=$(docker port "$NAME" 5432/tcp | head -n1)
  echo "postgresql://${USER_NAME}:${PASSWORD}@127.0.0.1:${hostport##*:}/postgres"
}

exists() {
  [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]
}

case "${1:-}" in
  up)
    if exists; then
      docker start "$NAME" >/dev/null
    else
      # fsync off + tmpfs: the data is disposable and the suite is write-heavy.
      docker run -d --name "$NAME" \
        -e POSTGRES_USER="$USER_NAME" -e POSTGRES_PASSWORD="$PASSWORD" \
        -p 127.0.0.1::5432 \
        --tmpfs /var/lib/postgresql \
        --label podcast-scraper-test-db=1 \
        "$IMAGE" -c fsync=off -c synchronous_commit=off -c full_page_writes=off \
        -c max_connections=300 >/dev/null
    fi
    for _ in $(seq 1 60); do
      if docker exec "$NAME" pg_isready -U "$USER_NAME" -h 127.0.0.1 >/dev/null 2>&1; then
        echo "Test database ready. Export:"
        echo "export PODCAST_TEST_DATABASE_URL=$(url)"
        exit 0
      fi
      sleep 1
    done
    echo "Test database did not become ready; logs:" >&2
    docker logs --tail 30 "$NAME" >&2
    exit 1
    ;;
  down)
    if exists; then
      docker rm -f "$NAME" >/dev/null
      echo "Removed $NAME"
    else
      echo "$NAME is not running"
    fi
    ;;
  url)
    url
    ;;
  *)
    echo "usage: $0 up|down|url" >&2
    exit 2
    ;;
esac
