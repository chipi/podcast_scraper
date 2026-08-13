#!/usr/bin/env bash
# Run the OPERATOR e2e suite against the fixture-bootstrapped API — the same corpus the consumer
# suite uses, and the same backend the app talks to in production.
#
# Until #1619 this suite had no backend at all: its webServer was Vite alone, so every API-dependent
# spec had to route-fulfil its own payloads. The endpoints it needs (/api/corpus/*, /api/search,
# /api/index/stats, /api/artifacts) are all served by the same image the consumer suite uses, so the
# mocks were never a necessity — just the only thing available when the suite was written.
#
# Two notes specific to this suite:
#   * it runs on FIREFOX (`npx playwright install firefox` once), not Chromium;
#   * Vite proxies /api to VITE_API_TARGET, so pointing it at the container is one env var.
#
# Usage:  e2e/run-local-stack.sh [playwright args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CORPUS="$REPO_ROOT/tests/fixtures/app-validation-corpus/v3"
IMAGE="${E2E_API_IMAGE:-podcast-api:e2e-local}"
CONTAINER=viewer-e2e-api
VOLUME=viewer-e2e-appdata
PORT=8012

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

[ -d "$CORPUS" ] || { echo "missing fixture corpus: $CORPUS" >&2; exit 1; }

cleanup
docker volume rm "$VOLUME" >/dev/null 2>&1 || true
docker volume create "$VOLUME" >/dev/null

docker run -d --name "$CONTAINER" \
  -p "127.0.0.1:$PORT:$PORT" \
  -v "$CORPUS:/corpus" \
  -v "$VOLUME:/appdata" \
  -e APP_OAUTH_PROVIDER=mock \
  -e APP_SESSION_SECRET=e2e-secret \
  -e APP_DATA_DIR=/appdata \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python "$IMAGE" \
  -m podcast_scraper.cli serve --output-dir /corpus --port "$PORT" --host 0.0.0.0 >/dev/null

printf 'waiting for the api'
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then echo " — up"; break; fi
  printf '.'; sleep 1
done
curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null || { echo; docker logs "$CONTAINER" | tail -30; exit 1; }

VITE_API_TARGET="http://127.0.0.1:$PORT" npx playwright test "$@"
