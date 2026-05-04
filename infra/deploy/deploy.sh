#!/usr/bin/env bash
# infra/deploy/deploy.sh — invoked by .github/workflows/deploy-prod.yml after
# the GHA runner has joined the tailnet. Runs ON the VPS as the deploy user.
#
# Pulls latest GHCR images, rolls the compose stack, smoke-tests /api/health.
#
# Exit codes:
#   0  success — stack pulled + up + healthy
#   1  pull failed
#   2  compose up failed
#   3  /api/health did not return 200 within the timeout
set -euo pipefail

REPO_DIR=/srv/podcast-scraper
STACK_FILES=(-f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml)

cd "$REPO_DIR"

# Refresh repo so compose files (not the images) match the deployed tag.
git fetch --depth=50 origin main
git reset --hard origin/main

echo "[$(date -u +%FT%TZ)] pulling latest images..."
if ! docker compose "${STACK_FILES[@]}" pull; then
  echo "ERROR: docker compose pull failed" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] rolling stack..."
if ! docker compose "${STACK_FILES[@]}" up -d --remove-orphans; then
  echo "ERROR: docker compose up failed" >&2
  exit 2
fi

# Local smoke check — the calling workflow also probes externally over Tailscale,
# but a local fail here aborts the SSH session early so the workflow surfaces red
# without waiting another timeout cycle.
echo "[$(date -u +%FT%TZ)] waiting for /api/health (up to 60s)..."
for i in $(seq 1 12); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] /api/health OK after $((i * 5))s"
    exit 0
  fi
  sleep 5
done

echo "ERROR: /api/health did not return 200 within 60s. Last 50 api logs:" >&2
docker logs --tail 50 compose-api-1 >&2 || true
exit 3
