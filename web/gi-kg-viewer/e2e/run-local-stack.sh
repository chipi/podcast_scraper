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
#
# Requires the image, which NO make target used to build. Build it with:
#   make e2e-api-image          # -> podcast-api:e2e-local
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CORPUS_SRC="$REPO_ROOT/tests/fixtures/app-validation-corpus/v3"
IMAGE="${E2E_API_IMAGE:-podcast-api:e2e-local}"
CONTAINER=viewer-e2e-api
VOLUME=viewer-e2e-appdata
PORT=8012

# ── The corpus is served from a COPY, never from the tracked fixture ──────────────────────────
#
# The operator plane WRITES into whatever corpus directory it is given: `GET /api/operator-config`
# *creates* `viewer_operator.yaml` when it is missing, and enabling the jobs API creates
# `.viewer/jobs.jsonl.lock`. `.gitignore` deliberately force-includes
# `tests/fixtures/app-validation-corpus/**`, so mounting the fixture directly leaves a dirty
# tracked tree after every run — and worse, a second run starts from a corpus the first one
# mutated, so "fresh corpus" assertions quietly stop being fresh.
#
# Copying is cheap (~6 MB) and makes each run hermetic. Override the location with
# E2E_CORPUS_WORKDIR if the default is inconvenient (it must be a path your container runtime can
# bind-mount — on Colima only paths under $HOME are visible inside the VM, so a $TMPDIR path
# silently mounts as an EMPTY directory).
WORKDIR="${E2E_CORPUS_WORKDIR:-$REPO_ROOT/web/gi-kg-viewer/.e2e-corpus}"
CORPUS="$WORKDIR/v3"

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

[ -d "$CORPUS_SRC" ] || { echo "missing fixture corpus: $CORPUS_SRC" >&2; exit 1; }

echo "seeding a disposable corpus copy at $CORPUS"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cp -R "$CORPUS_SRC" "$CORPUS"
# The container runs as a non-root uid (PODCAST_UID=1000 in docker/api/Dockerfile) while the copy is
# owned by whoever ran this script, so the operator plane cannot create the files it needs inside
# the mount. Symptom if you skip this: `GET /api/jobs` → 500 with
# `PermissionError: '/corpus/.viewer/jobs.jsonl.lock'`, and the Dashboard jobs card sits on
# "Loading…" forever — which reads as a hung frontend, not a permissions problem.
# Safe to blanket-chmod: this tree is a throwaway copy, re-seeded above on every run.
chmod -R a+rwX "$CORPUS"

cleanup
docker volume rm "$VOLUME" >/dev/null 2>&1 || true
docker volume create "$VOLUME" >/dev/null

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
docker run -d --name "$CONTAINER" \
  -p "127.0.0.1:$PORT:$PORT" \
  -v "$CORPUS:/corpus" \
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
  -m podcast_scraper.cli serve --output-dir /corpus --port "$PORT" --host 0.0.0.0 >/dev/null

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

VITE_API_TARGET="http://127.0.0.1:$PORT" npx playwright test "${WORKER_ARGS[@]}" "$@"
