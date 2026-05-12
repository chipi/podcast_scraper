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
# vps-prod overlay (#713) layers prod nginx template + landing page on
# top of the stack + prod overlays. Codespaces deliberately omits it.
STACK_FILES=(-f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml)

cd "$REPO_DIR"

# ``docker-compose.prod.yml`` requires ``PODCAST_DOCKER_PROJECT_DIR`` for api
# volume interpolation. PROD_RUNBOOK stages it in ``.env``; drill GHA may only
# append ``PODCAST_RELEASE``, so ensure the bind-mount path exists before compose.
if [ ! -f .env ]; then
  install -m 600 /dev/null .env
fi
if ! grep -qE '^PODCAST_DOCKER_PROJECT_DIR=' .env; then
  echo "PODCAST_DOCKER_PROJECT_DIR=$REPO_DIR" >> .env
  chmod 600 .env
fi
# Same pattern as ``.devcontainer/start.sh`` (Codespaces): prod compose binds
# corpus_data to ``PODCAST_CORPUS_HOST_PATH`` on the host → ``/app/output`` in
# containers. VPS default matches cloud-init (``/srv/podcast-scraper/corpus``).
if ! grep -qE '^PODCAST_CORPUS_HOST_PATH=' .env; then
  echo "PODCAST_CORPUS_HOST_PATH=$REPO_DIR/corpus" >> .env
  chmod 600 .env
fi
mkdir -p "$REPO_DIR/corpus"

# Compose resolves the project directory from the first `-f` path (`compose/`), so
# it does not load `/srv/podcast-scraper/.env` by default — pass explicitly (VPS + GHA).
COMPOSE=(docker compose --env-file .env)

# Refresh repo so compose files (not the images) match the deployed tag.
git fetch --depth=50 origin main
git reset --hard origin/main

echo "[$(date -u +%FT%TZ)] pulling latest images..."
if ! "${COMPOSE[@]}" "${STACK_FILES[@]}" pull; then
  echo "ERROR: docker compose pull failed" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] rolling stack..."
if ! "${COMPOSE[@]}" "${STACK_FILES[@]}" up -d --remove-orphans; then
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
  if "${COMPOSE[@]}" "${STACK_FILES[@]}" exec -T api \
    curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] /api/health OK after $((i * 5))s (container-local)"
    if command -v sudo >/dev/null 2>&1 && [ -x /usr/local/sbin/podcast-tailscale-serve.sh ]; then
      sudo -n /usr/local/sbin/podcast-tailscale-serve.sh >/dev/null 2>&1 || true
    fi
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
