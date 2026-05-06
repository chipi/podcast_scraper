#!/usr/bin/env bash
# infra/deploy/deploy.sh — invoked by .github/workflows/deploy-prod.yml after
# the GHA runner has joined the tailnet. Runs ON the VPS as the deploy user.
#
# Pulls latest GHCR images, rolls the compose stack, smoke-tests /api/health
# from inside the api container (GH-745: api port 8000 is not published on
# the host; host loopback :8000 is a false negative).
#
# Exit codes:
#   0  success — stack pulled + up + healthy
#   1  pull failed
#   2  compose up failed
#   3  /api/health did not return 200 within the timeout
set -euo pipefail

REPO_DIR=/srv/podcast-scraper
# vps-prod overlay (#713) layers basic-auth nginx + .htpasswd bind-mount on
# top of the stack + prod overlays. Codespaces deliberately omits it.
STACK_FILES=(-f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml)

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

# Container-local smoke check (authoritative on the VPS). The workflow also
# probes externally over Tailscale; a failure here aborts SSH early.
# Do not curl http://127.0.0.1:8000 on the host — compose only exposes :8000
# inside the Docker network. See docs/guides/PROD_RUNBOOK.md
# "API health checks by context (GH-745)".
echo "[$(date -u +%FT%TZ)] waiting for /api/health inside api container (up to 60s)..."
for i in $(seq 1 12); do
  if docker compose "${STACK_FILES[@]}" exec -T api \
    curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] /api/health OK after $((i * 5))s (container-local)"
    exit 0
  fi
  sleep 5
done

echo "ERROR: /api/health did not return 200 within 60s (container-local exec probe)." >&2
echo "ERROR: If tailnet curl still works, you were likely misled by host loopback :8000 — api is not published there (GH-745)." >&2
echo "ERROR: From this host, try viewer proxy: curl -fsS http://127.0.0.1:\${VIEWER_PORT:-8080}/api/health (prod nginx allows /api/health without auth)." >&2
echo "ERROR: From your laptop, use https://<prod-tailnet-fqdn>/api/health (see PROD_RUNBOOK.md)." >&2
docker logs --tail 50 compose-api-1 >&2 || true
exit 3
