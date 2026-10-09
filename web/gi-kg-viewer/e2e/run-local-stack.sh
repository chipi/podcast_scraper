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
#   * it runs on Chrome (`npx playwright install chromium` once), like the player suite;
#   * Vite proxies /api to VITE_API_TARGET, so pointing it at the container is one env var.
#
# Usage:  e2e/run-local-stack.sh [playwright args...]
#
# Requires the image, which NO make target used to build. Build it with:
#   make e2e-api-image          # -> podcast-api:e2e-local
set -euo pipefail

VIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Found the way the Playwright configs find it (platform-root.mjs), so this runs from the public
# repo and from the private studio mount alike (ADR-162).
REPO_ROOT="$(node "$VIEWER_ROOT/platform-root.mjs")"
CORPUS_SRC="$REPO_ROOT/tests/fixtures/app-validation-corpus/v3"
IMAGE="${E2E_API_IMAGE:-podcast-api:e2e-local}"
# Named per checkout, so two worktrees (or this viewer and Studio) can run side by side on one
# engine. The host port is whatever Docker gives the container: a fixed one collides with other
# checkouts' runs and with the native server the Playwright config starts on 8012.
CHECKOUT="$(basename "$REPO_ROOT")-$(basename "$(dirname "$VIEWER_ROOT")")"
CONTAINER="viewer-e2e-api-$CHECKOUT"
VOLUME="viewer-e2e-appdata-$CHECKOUT"
CORPUS_VOLUME="viewer-e2e-corpus-$CHECKOUT"
API_PORT=8012  # inside the container
# Every container gets a memory cap (the shared dev engine's contract). The API holds the
# embedding model and a LanceDB index; 3g is the size the engine's owners suggested for it.
API_MEMORY="${E2E_API_MEMORY:-3g}"

# ── The corpus is served from a COPY in a volume, never from the tracked fixture ──────────────
#
# The operator plane WRITES into whatever corpus directory it is given: `GET /api/operator-config`
# *creates* `viewer_operator.yaml` when it is missing, and enabling the jobs API creates
# `.viewer/jobs.jsonl.lock`. `.gitignore` deliberately force-includes
# `tests/fixtures/app-validation-corpus/**`, so mounting the fixture directly leaves a dirty
# tracked tree after every run — and worse, a second run starts from a corpus the first one
# mutated, so "fresh corpus" assertions quietly stop being fresh.
#
# The copy lives in a named volume, re-created on every run, not in a bind-mounted host directory.
# On Colima a bind mount crosses the VM boundary for every stat and read: `GET /api/artifacts`,
# which walks the corpus, took 7.3 s against the bind mount and 0.1–0.3 s against the same files
# inside the container (measured 2026-10-08). That walk runs on every graph load, so the
# bind-mounted corpus pushed graph-loading specs past their timeouts.
#
# The container runs as uid 1000 (PODCAST_UID in docker/api/Dockerfile), so the seeded tree is
# chowned to it. Symptom if it is not: `GET /api/jobs` → 500 with
# `PermissionError: '/corpus/.viewer/jobs.jsonl.lock'`, and the Dashboard jobs card sits on
# "Loading…" forever — which reads as a hung frontend, not a permissions problem.

cleanup() { docker rm -f "$CONTAINER" "$CONTAINER-seed" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

[ -d "$CORPUS_SRC" ] || { echo "missing fixture corpus: $CORPUS_SRC" >&2; exit 1; }

cleanup
docker volume rm "$VOLUME" "$CORPUS_VOLUME" >/dev/null 2>&1 || true
docker volume create "$VOLUME" >/dev/null
docker volume create "$CORPUS_VOLUME" >/dev/null

echo "seeding a disposable corpus copy into the $CORPUS_VOLUME volume"
docker run -d --name "$CONTAINER-seed" --memory 512m --user root -v "$CORPUS_VOLUME:/w" \
  --entrypoint sleep "$IMAGE" 300 >/dev/null
docker cp "$CORPUS_SRC/." "$CONTAINER-seed:/w"
docker exec "$CONTAINER-seed" chown -R 1000:1000 /w
docker rm -f "$CONTAINER-seed" >/dev/null

# ── Env the suite actually needs ───────────────────────────────────────────────────────────────
#
# The five vars after APP_DATA_DIR are not optional extras; without them whole surfaces are
# unreachable and the specs fail in ways that look like app bugs:
#
#   APP_SIGNUP_MODE=open                        `/api/app/auth/login?as=…` 403s, so `signInIsolated`
#                                               cannot create a session and anything reading
#                                               `/api/app/preferences` gets 401.
#   APP_ADMIN_EMAILS=ada-admin@e2e.local        `signInAsAdmin` lands in `creator`, so admin-only
#                                               surfaces never render.
#   PODCAST_SERVE_ENABLE_FEEDS_API              /api/feeds is NOT MOUNTED without it (404, not 403).
#   PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API    likewise /api/operator-config.
#   PODCAST_SERVE_ENABLE_JOBS_API               likewise /api/jobs + /api/scheduled-jobs.
#
# Mounting them is safe here because the corpus is a throwaway copy (see above).
docker run -d --name "$CONTAINER" --memory "$API_MEMORY" \
  -p "127.0.0.1::$API_PORT" \
  -v "$CORPUS_VOLUME:/corpus" \
  -v "$VOLUME:/appdata" \
  -e APP_OAUTH_PROVIDER=mock \
  -e APP_SESSION_SECRET=e2e-secret \
  -e APP_DATA_DIR=/appdata \
  -e APP_SIGNUP_MODE=open \
  -e APP_ADMIN_EMAILS=ada-admin@e2e.local \
  -e PODCAST_SERVE_ENABLE_FEEDS_API=1 \
  -e PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API=1 \
  -e PODCAST_SERVE_ENABLE_JOBS_API=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python "$IMAGE" \
  -m podcast_scraper.cli serve --output-dir /corpus --port "$API_PORT" --host 0.0.0.0 >/dev/null
PORT="$(docker port "$CONTAINER" "$API_PORT/tcp" | head -1 | sed 's/.*://')"
echo "api container on host port $PORT"

printf 'waiting for the api'
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then echo " — up"; break; fi
  printf '.'; sleep 1
done
curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null || { echo; docker logs "$CONTAINER" | tail -30; exit 1; }

# Fail loudly rather than let specs mis-report a missing capability as an app bug.
health="$(curl -sf "http://127.0.0.1:$PORT/api/health")"
for cap in feeds_api operator_config_api jobs_api; do
  case "$health" in
    *"\"$cap\":true"*) ;;
    *) echo "api is up but $cap is false — the PODCAST_SERVE_ENABLE_* env did not take" >&2; exit 1 ;;
  esac
done

# ── Wait for SEARCH to be ready, not just for the port to answer ──────────────────────────────
#
# `/api/health` returns the moment uvicorn binds, but the app cannot serve its main feature yet:
# the embedding model loads on first use — ~40 s on a cold container here, ~5 s once warm. The
# server now warms it at startup on a background thread, so this is a race rather than a permanent
# state, but a spec that queries immediately blocks on the SAME model singleton the warmup holds
# and burns its 30 s budget waiting. That is exactly what failed `workspace.spec.ts` at its first
# result assertion, and it read as an app bug rather than a not-ready backend.
#
# So gate on the capability the suite actually needs: issue one real query and wait until it comes
# back without an error field. A corpus with no index (or no `[search]` extras) can never satisfy
# that, so this gives up and continues rather than stalling the run — the specs that need search
# will report it themselves.
printf 'waiting for search to warm'
for _ in $(seq 1 90); do
  probe="$(curl -sf "http://127.0.0.1:$PORT/api/search?q=warm&top_k=1" 2>/dev/null || true)"
  if [ -n "$probe" ]; then
    case "$probe" in
      *'"error":null'*) echo " — ready"; break ;;
      *'"error":"'*)    echo " — unavailable (continuing; search specs will report it)"; break ;;
    esac
  fi
  printf '.'; sleep 2
done

# ── Default to ONE Playwright worker ──────────────────────────────────────────────────────────
#
# The API is single-process: `podcast_scraper.cli serve` calls `uvicorn.run()` with no `--workers`,
# and the CLI exposes no flag for it. A live `/api/search` is a query embedding plus a LanceDB
# search — seconds of CPU that hold that one process — so a second browser worker mostly just
# queues behind the first and makes wall-clock, and therefore timeouts, non-deterministic. That was
# the entire source of the flakes on this suite: every failure was a TIMEOUT, with the same test
# passing in ~10s alone and taking 45s+ under contention.
#
# Browser-side parallelism buys little against a serial backend, and costs determinism. Pass
# `--workers=N` explicitly to override (CI sets its own via playwright.config.ts).
case " $* " in
  *" --workers"*) WORKER_ARGS=() ;;
  *) WORKER_ARGS=(--workers=1) ;;
esac

# Run from the viewer so npx finds its own pinned Playwright and config. From anywhere else
# (`make test-ui-e2e-live` runs at the repo root) npx fetched an unpinned Playwright and collected
# every spec in the repo.
cd "$VIEWER_ROOT"
E2E_API_PORT="$PORT" npx playwright test "${WORKER_ARGS[@]}" "$@"
