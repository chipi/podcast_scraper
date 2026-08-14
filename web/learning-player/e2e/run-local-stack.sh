#!/usr/bin/env bash
# Bring up the fixture-backed API the consumer e2e suite runs against, then run Playwright.
#
# Why this script exists: the suite is NOT hermetic across runs. The API keeps per-user state
# (follows, queue, captures) in APP_DATA_DIR, which lives in a docker volume. Re-running against a
# surviving volume produces failures that look like code regressions but are stale state — e.g.
# follow-show.spec asserts aria-pressed="false" on a show a previous run already followed, and
# retries 34 times before failing. Recreating the volume is the whole point of this script; do not
# "optimise" it away.
#
# Usage:  e2e/run-local-stack.sh [playwright args...]
#         e2e/run-local-stack.sh --project=mobile-chrome e2e/follow-show.spec.ts
#
# Only the API needs Docker here, and only because the two-tier search index globalSetup builds
# requires the [search] extras (lancedb / sentence-transformers / torch), which have no macOS
# x86_64 wheels. Everything else runs natively: the app's runtime deps are in the venv, and
# Playwright starts the mock podcast host itself (webServer), exactly as it does in CI.
#
# Requires the API image built once:  see docs/wip/2026-08-13-e2e-on-intel-mac.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CORPUS="$REPO_ROOT/tests/fixtures/app-validation-corpus/v3"
IMAGE="${E2E_API_IMAGE:-podcast-api:e2e-local}"
CONTAINER=lp-e2e-api
VOLUME=lp-e2e-appdata
PORT=8011

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
  -e APP_SIGNUP_MODE=open \
  -e APP_PERSONALIZED_RANKING=true \
  -e APP_TRENDING_NOW=2026-07-20T00:00:00Z \
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

npx playwright test "$@"
