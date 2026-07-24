#!/usr/bin/env bash
# capture-surface-perf.sh — boot dedicated api + viewer on isolated ports, run
# the Playwright/CDP UI trace capturer (scripts/dev/capture-surface-perf.mjs)
# for the "other" viewer surfaces — Library, Digest, entity-load — and tear
# everything down. perf-traces Chunk 3; sibling of capture-search-perf.sh.
#
# See docs/guides/perf-traces/surfaces.md + index.md for the framework.
#
# Usage:
#   scripts/dev/capture-surface-perf.sh \
#       --corpus /abs/path/to/corpus \
#       --label <release>-surfaces \
#       [--output-dir data/perf/traces/surfaces] \
#       [--api-port 8621] [--viewer-port 5621] \
#       [--wait-ms 3000]
#
# Isolated ports 8621/5621 — different from the search (8601/5601) and graph
# (8600/5600) capturers so all three can run without collision.

set -euo pipefail

CORPUS=""
LABEL=""
OUTPUT_DIR="data/perf/traces/surfaces"
API_PORT="8621"
VIEWER_PORT="5621"
WAIT_MS="3000"
RUNS="3"
VIEWPORT_WIDTH="1440"
VIEWPORT_HEIGHT="900"
VIEWPORT_DPR="2"

while [ $# -gt 0 ]; do
  case "$1" in
    --corpus)         CORPUS="$2"; shift 2 ;;
    --label)          LABEL="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --api-port)       API_PORT="$2"; shift 2 ;;
    --viewer-port)    VIEWER_PORT="$2"; shift 2 ;;
    --wait-ms)        WAIT_MS="$2"; shift 2 ;;
    --runs)           RUNS="$2"; shift 2 ;;
    --viewport-w)     VIEWPORT_WIDTH="$2"; shift 2 ;;
    --viewport-h)     VIEWPORT_HEIGHT="$2"; shift 2 ;;
    --viewport-dpr)   VIEWPORT_DPR="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | head -30
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$CORPUS" ] || { echo "FATAL: --corpus <path> required" >&2; exit 2; }
[ -n "$LABEL" ]  || { echo "FATAL: --label <name> required" >&2; exit 2; }
[ -d "$CORPUS" ] || { echo "FATAL: corpus dir not found: $CORPUS" >&2; exit 2; }

CORPUS_ABS="$(cd "$CORPUS" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

command -v node >/dev/null || { echo "FATAL: node not on PATH"; exit 2; }
[ -x ".venv/bin/python" ] || { echo "FATAL: .venv/bin/python missing — run 'make dev-setup' first"; exit 2; }

for port in "$API_PORT" "$VIEWER_PORT"; do
  if lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -qE ":${port}\b"; then
    role="api"; [ "$port" = "$VIEWER_PORT" ] && role="viewer"
    echo "FATAL: ${role} port ${port} already in use. Pass --${role}-port <free-port>." >&2
    exit 3
  fi
done

VIEWER_DIR="${REPO_ROOT}/web/gi-kg-viewer"
[ -d "${VIEWER_DIR}/node_modules" ] || {
  echo "FATAL: ${VIEWER_DIR}/node_modules missing — run 'cd web/gi-kg-viewer && env -u NODE_OPTIONS npm ci'" >&2
  exit 2
}

LOG_DIR="$(mktemp -d -t capture-surface-perf.XXXXXX)"
API_LOG="${LOG_DIR}/api.log"
VIEWER_LOG="${LOG_DIR}/viewer.log"

cleanup() {
  set +e
  if [ -n "${API_PID:-}" ]; then kill "${API_PID}" 2>/dev/null; fi
  if [ -n "${VIEWER_PID:-}" ]; then kill "${VIEWER_PID}" 2>/dev/null; fi
  wait 2>/dev/null
  echo "[capture-surface-perf] logs kept at ${LOG_DIR}"
}
trap cleanup EXIT INT TERM

echo "[capture-surface-perf] booting api on :${API_PORT}"
env -u NODE_OPTIONS KMP_DUPLICATE_LIB_OK=TRUE ".venv/bin/python" \
  -m podcast_scraper.cli serve --output-dir "${CORPUS_ABS}" --port "${API_PORT}" \
  > "${API_LOG}" 2>&1 &
API_PID=$!

for i in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${API_PORT}/api/health"; then break; fi
  sleep 1
done
if ! curl -fsS -o /dev/null "http://127.0.0.1:${API_PORT}/api/health"; then
  echo "FATAL: api did not become healthy in 30s. See ${API_LOG}." >&2
  exit 4
fi

echo "[capture-surface-perf] booting viewer on :${VIEWER_PORT}"
# The viewer proxies /api via vite.config.ts using VITE_API_TARGET (default
# :8000). Point it at our isolated api port so the viewer sees a healthy API.
env -u NODE_OPTIONS bash -c "cd ${VIEWER_DIR} && \
  VITE_API_TARGET=http://127.0.0.1:${API_PORT} \
  node_modules/.bin/vite --host 127.0.0.1 --port ${VIEWER_PORT} --strictPort" \
  > "${VIEWER_LOG}" 2>&1 &
VIEWER_PID=$!

for i in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${VIEWER_PORT}/"; then break; fi
  sleep 1
done
if ! curl -fsS -o /dev/null "http://127.0.0.1:${VIEWER_PORT}/"; then
  echo "FATAL: viewer did not become healthy in 30s. See ${VIEWER_LOG}." >&2
  exit 4
fi

echo "[capture-surface-perf] running mjs capturer"
env -u NODE_OPTIONS bash -c "cd ${VIEWER_DIR} && \
  node ${REPO_ROOT}/scripts/dev/capture-surface-perf.mjs \
    --viewer http://127.0.0.1:${VIEWER_PORT} \
    --corpus '${CORPUS_ABS}' \
    --label '${LABEL}' \
    --output-dir '${OUTPUT_DIR}' \
    --wait-ms '${WAIT_MS}' \
    --runs '${RUNS}' \
    --viewport-w '${VIEWPORT_WIDTH}' \
    --viewport-h '${VIEWPORT_HEIGHT}' \
    --viewport-dpr '${VIEWPORT_DPR}'"
