#!/usr/bin/env bash
# Codespace postStartCommand — bring up the pre-prod compose stack.
#
# Pulls api / viewer / pipeline-llm from GHCR (per
# compose/docker-compose.prod.yml) and starts them. Pipeline-llm runs
# on demand only — the api job factory spawns it via ``docker compose
# run --rm pipeline-llm`` when a job is triggered.
#
# Gracefully degrades when GHCR images don't exist yet (e.g., before
# the first main push triggers the Stack-test publish job): logs a
# TODO and exits 0 so the codespace boot doesn't fail.

set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE_FILES=(
  -f compose/docker-compose.stack.yml
  -f compose/docker-compose.prod.yml
)

echo "==> Codespace pre-prod stack startup"

# stack.yml's viewer service publishes to ``${VIEWER_PORT:-8080}:80``.
# devcontainer.json forwards 8090 (operator-facing convention from
# RFC-081), so pin the host-side port here to match. Without this,
# ``compose up`` lands on host:8080 while the codespace's port-forward
# targets :8090 → operator's browser sees a "Bad Gateway" from GitHub's
# port forwarder. Override via VIEWER_PORT env if the operator wants
# a different layout (must also update devcontainer.json forwardPorts).
export VIEWER_PORT="${VIEWER_PORT:-8090}"

# Ensure the corpus bind-mount source exists. ``docker-compose.prod.yml``
# overrides the ``corpus_data`` volume as a bind mount onto this dir so
# that (a) operators can edit ``feeds.spec.yaml`` from the codespace
# shell, (b) the api / pipeline containers see the same files at
# ``/app/output``, and (c) ``backup-corpus.yml`` can tar the host path
# directly. Docker bind-mounts fail at startup if the source dir is
# missing — create it before ``compose up``.
CORPUS_HOST_PATH="${PODCAST_CORPUS_HOST_PATH:-/workspaces/podcast_scraper/.codespace_corpus}"
mkdir -p "$CORPUS_HOST_PATH"

# Stale-volume defense: ``compose up`` does NOT recreate an existing named
# volume even when the YAML definition changes. Codespaces booted before
# the prod overlay's bind-mount config landed (or with a different
# ``PODCAST_CORPUS_HOST_PATH``) end up wedged on a Docker-managed volume
# whose ``/app/output`` is invisible from the codespace shell + invisible
# to ``backup-corpus.yml``.
#
# Fix: probe the existing ``compose_corpus_data`` volume's bind device. If
# it doesn't match the path we expect, take the stack down + remove the
# stale volume so ``compose up`` recreates it with the right config. The
# bind path is empty by definition (codespaces are persistent on the host
# workspace dir, not on Docker volumes — no real data lives in the
# Docker-managed corpus_data volume), so removing it loses nothing.
if docker volume inspect compose_corpus_data >/dev/null 2>&1; then
  EXISTING_DEVICE=$(docker volume inspect compose_corpus_data 2>/dev/null \
    | grep -oE '"device":[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/' || true)
  if [ "$EXISTING_DEVICE" != "$CORPUS_HOST_PATH" ]; then
    if [ -z "$EXISTING_DEVICE" ]; then
      echo "==> Stale corpus_data volume detected (no bind device set — Docker-managed)."
    else
      echo "==> Stale corpus_data volume detected (device=$EXISTING_DEVICE; expected=$CORPUS_HOST_PATH)."
    fi
    echo "    Bringing stack down + removing volume so the new bind config takes effect..."
    docker compose "${COMPOSE_FILES[@]}" down 2>&1 | tail -5 || true
    docker volume rm compose_corpus_data 2>/dev/null || true
  fi
fi

# Pull all images first. Quiet mode avoids spamming the boot log with
# layer-by-layer progress.
if ! docker compose "${COMPOSE_FILES[@]}" pull --quiet 2>&1 | tee /tmp/compose-pull.log; then
  echo ""
  echo "::warning::Could not pull GHCR images. Likely causes:"
  echo "  1. The Stack-test publish job has not run yet on main"
  echo "     (the first main push after this branch merges populates"
  echo "     ghcr.io/chipi/podcast-scraper-stack-{api,viewer,pipeline-llm})."
  echo "  2. The GHCR packages are still private and the Codespace's"
  echo "     docker daemon is not authenticated. Make them public via"
  echo "     each package's Settings → Manage Actions access, OR"
  echo "     authenticate the daemon with a GHCR token."
  echo ""
  echo "Until that's resolved, the stack will not auto-start. Re-run:"
  echo "  bash .devcontainer/start.sh"
  exit 0
fi

# Up -d brings api + viewer + grafana-agent up. pipeline-llm is profile-
# gated and stays off until the api spawns it via docker compose run.
docker compose "${COMPOSE_FILES[@]}" up -d --remove-orphans

echo ""
echo "==> Stack up. Forwarded ports:"
echo "  http://localhost:8000  api (FastAPI direct)"
echo "  http://localhost:8090  viewer (Nginx + reverse proxy) ← operator URL"
echo ""
echo "Health check:"
for i in $(seq 1 30); do
  if curl -fsS "http://localhost:8090/api/health" >/dev/null 2>&1; then
    echo "  /api/health → 200 OK"
    exit 0
  fi
  sleep 2
done
echo "  ::warning::/api/health did not respond within 60s. Run \`docker compose ${COMPOSE_FILES[*]} logs api\` to investigate."
