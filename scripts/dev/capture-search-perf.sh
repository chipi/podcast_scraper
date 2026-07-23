#!/usr/bin/env bash
# capture-search-perf.sh — boot dedicated api + viewer on isolated ports, run
# the Playwright/CDP UI trace capturer (scripts/dev/capture-search-perf.mjs)
# for Search v3, and tear everything down. Complements capture-search-api.sh
# (API-only, no browser).
#
# Search v3 §S1 stabilization pass (2026-07-20). Captures the 3 UI scenarios
# that exist TODAY on the merged Search launcher (compact-launcher shape after
# S1). The 3 remaining scenarios from RFC-107 §P2 (workspace-open, cmdk-open,
# operator-cluster) stay marked NOT_APPLICABLE_YET in the mjs — they need S2
# (Query Workspace shell, #1232) + S3 (Cmd-K, #1233) + S4 (operator bar,
# #1234) UI to exist.
#
# See docs/guides/SEARCH_PERF_TRACE_RUNBOOK.md for the full recipe.
#
# Usage:
#   scripts/dev/capture-search-perf.sh \
#       --corpus /abs/path/to/corpus \
#       --label search-v3-s1-tip \
#       [--output-dir docs/wip/search-v3/traces] \
#       [--api-port 8601] [--viewer-port 5601] \
#       [--wait-ms 3000]
#
# Isolated ports 8601/5601 — deliberately different from graph capturer
# (8600/5600) so both can run without collision.

set -euo pipefail

CORPUS=""
LABEL=""
OUTPUT_DIR="docs/wip/search-v3/traces"
API_PORT="8601"
VIEWER_PORT="5601"
WAIT_MS="3000"
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

# Port collision check.
for port in "$API_PORT" "$VIEWER_PORT"; do
  if lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -qE ":${port}\b"; then
    role="api"; [ "$port" = "$VIEWER_PORT" ] && role="viewer"
    echo "FATAL: ${role} port ${port} already in use. Pass --${role}-port <free-port>." >&2
    exit 3
  fi
done

# Ensure viewer deps are installed.
VIEWER_DIR="${REPO_ROOT}/web/gi-kg-viewer"
[ -d "${VIEWER_DIR}/node_modules" ] || {
  echo "FATAL: ${VIEWER_DIR}/node_modules missing — run 'cd web/gi-kg-viewer && env -u NODE_OPTIONS npm ci'" >&2
  exit 2
}

# Log locations
LOG_DIR="$(mktemp -d -t capture-search-perf.XXXXXX)"
API_LOG="${LOG_DIR}/api.log"
VIEWER_LOG="${LOG_DIR}/viewer.log"

cleanup() {
  set +e
  if [ -n "${API_PID:-}" ]; then kill "${API_PID}" 2>/dev/null; fi
  if [ -n "${VIEWER_PID:-}" ]; then kill "${VIEWER_PID}" 2>/dev/null; fi
  wait 2>/dev/null
  echo "[capture-search-perf] logs kept at ${LOG_DIR}"
}
trap cleanup EXIT INT TERM

echo "[capture-search-perf] booting api on :${API_PORT}"
env -u NODE_OPTIONS KMP_DUPLICATE_LIB_OK=TRUE ".venv/bin/python" \
  -m podcast_scraper.cli serve --output-dir "${CORPUS_ABS}" --port "${API_PORT}" \
  > "${API_LOG}" 2>&1 &
API_PID=$!

# Wait api healthy.
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${API_PORT}/api/health"; then break; fi
  sleep 1
done
if ! curl -fsS -o /dev/null "http://127.0.0.1:${API_PORT}/api/health"; then
  echo "FATAL: api did not become healthy in 30s. See ${API_LOG}." >&2
  exit 4
fi

echo "[capture-search-perf] booting viewer on :${VIEWER_PORT}"
env -u NODE_OPTIONS bash -c "cd ${VIEWER_DIR} && \
  VITE_API_BASE=http://127.0.0.1:${API_PORT} \
  node_modules/.bin/vite --host 127.0.0.1 --port ${VIEWER_PORT} --strictPort" \
  > "${VIEWER_LOG}" 2>&1 &
VIEWER_PID=$!

# Wait viewer up.
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:${VIEWER_PORT}/"; then break; fi
  sleep 1
done
if ! curl -fsS -o /dev/null "http://127.0.0.1:${VIEWER_PORT}/"; then
  echo "FATAL: viewer did not become healthy in 30s. See ${VIEWER_LOG}." >&2
  exit 4
fi

echo "[capture-search-perf] running mjs capturer"
env -u NODE_OPTIONS bash -c "cd ${VIEWER_DIR} && \
  node ${REPO_ROOT}/scripts/dev/capture-search-perf.mjs \
    --viewer http://127.0.0.1:${VIEWER_PORT} \
    --corpus '${CORPUS_ABS}' \
    --label '${LABEL}' \
    --output-dir '${OUTPUT_DIR}' \
    --wait-ms '${WAIT_MS}' \
    --viewport-w '${VIEWPORT_WIDTH}' \
    --viewport-h '${VIEWPORT_HEIGHT}' \
    --viewport-dpr '${VIEWPORT_DPR}'"
