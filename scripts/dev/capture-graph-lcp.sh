#!/usr/bin/env bash
# capture-graph-lcp.sh — boot dedicated api + viewer on isolated ports,
# run the Playwright/CDP trace capturer (scripts/dev/capture-graph-lcp.mjs),
# and tear everything down. Isolated ports so this NEVER collides with the
# operator's dev server.
#
# See docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md for the full recipe.
#
# Usage:
#   scripts/dev/capture-graph-lcp.sh \
#       --corpus /abs/path/to/corpus \
#       --label main-baseline \
#       [--output-dir docs/wip/graph-v3/traces] \
#       [--ref main]            # if set, checks out via git worktree
#       [--api-port 8600] [--viewer-port 5600] \
#       [--wait-ms 5000] \
#       [--load-mode topDown|everything] \
#       [--expand-first-super-theme]
#
# Every arg has a default suitable for the common case. Fails loudly instead
# of guessing when a required tool is missing.

set -euo pipefail

# ---------------------------------------------------------------------- args
CORPUS=""
LABEL=""
OUTPUT_DIR="docs/wip/graph-v3/traces"
REF=""
API_PORT="8600"
VIEWER_PORT="5600"
WAIT_MS="5000"
VIEWPORT_WIDTH="1440"
VIEWPORT_HEIGHT="900"
VIEWPORT_DPR="2"
LOAD_MODE=""
EXPAND_FIRST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --corpus)         CORPUS="$2"; shift 2 ;;
    --label)          LABEL="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --ref)            REF="$2"; shift 2 ;;
    --api-port)       API_PORT="$2"; shift 2 ;;
    --viewer-port)    VIEWER_PORT="$2"; shift 2 ;;
    --wait-ms)        WAIT_MS="$2"; shift 2 ;;
    --viewport-w)     VIEWPORT_WIDTH="$2"; shift 2 ;;
    --viewport-h)     VIEWPORT_HEIGHT="$2"; shift 2 ;;
    --viewport-dpr)   VIEWPORT_DPR="$2"; shift 2 ;;
    --load-mode)      LOAD_MODE="$2"; shift 2 ;;
    --expand-first-super-theme) EXPAND_FIRST="1"; shift 1 ;;
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
if [ -n "$LOAD_MODE" ] && [ "$LOAD_MODE" != "topDown" ] && [ "$LOAD_MODE" != "everything" ]; then
  echo "FATAL: --load-mode must be 'topDown' or 'everything' (got '$LOAD_MODE')" >&2
  exit 2
fi

CORPUS_ABS="$(cd "$CORPUS" && pwd)"
# Output dir must be absolute — the mjs runs with viewer cwd (see below)
# and relative paths would land under web/gi-kg-viewer/ instead.
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# ---------------------------------------------------------------------- deps
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

command -v node >/dev/null || { echo "FATAL: node not on PATH"; exit 2; }
[ -x ".venv/bin/python" ] || { echo "FATAL: .venv/bin/python missing — run 'make dev-setup' first"; exit 2; }

# port collision check — fail loudly rather than silently share ports.
if lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -qE ":${API_PORT}\b"; then
  echo "FATAL: api port ${API_PORT} already in use. Pass --api-port <free-port>." >&2
  lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -E ":${API_PORT}\b" >&2
  exit 3
fi
if lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -qE ":${VIEWER_PORT}\b"; then
  echo "FATAL: viewer port ${VIEWER_PORT} already in use. Pass --viewer-port <free-port>." >&2
  lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -E ":${VIEWER_PORT}\b" >&2
  exit 3
fi

# ------------------------------------------------------------- worktree switch
# If --ref is given, materialise it as a git worktree so we don't touch the
# current worktree's uncommitted state. Otherwise trace the current worktree.
WORKTREE_DIR=""
if [ -n "$REF" ]; then
  WORKTREE_DIR="$(mktemp -d -t lcp-worktree.XXXXXX)"
  echo "[capture-graph-lcp] creating worktree at $WORKTREE_DIR for ref $REF"
  git worktree add "$WORKTREE_DIR" "$REF"
  RUN_ROOT="$WORKTREE_DIR"
  # replicate the venv into the worktree by symlink; api CLI only needs the
  # importable package + config, which sit under the checked-out ref itself.
  ln -s "$REPO_ROOT/.venv" "$RUN_ROOT/.venv"
  # viewer dependencies live under node_modules — install into the worktree
  # only if missing (cache-friendly).
  if [ ! -d "$RUN_ROOT/web/gi-kg-viewer/node_modules" ]; then
    ( cd "$RUN_ROOT/web/gi-kg-viewer" && env -u NODE_OPTIONS npm ci --no-audit --no-fund )
  fi
else
  RUN_ROOT="$REPO_ROOT"
fi

# ---------------------------------------------------------------------- boot
API_LOG="$(mktemp -t lcp-api.XXXXXX.log)"
VIEWER_LOG="$(mktemp -t lcp-viewer.XXXXXX.log)"
API_PID=""
VIEWER_PID=""

cleanup() {
  local rc=$?
  echo "[capture-graph-lcp] teardown (rc=$rc)"
  [ -n "$VIEWER_PID" ] && kill -TERM "$VIEWER_PID" 2>/dev/null || true
  [ -n "$API_PID" ]    && kill -TERM "$API_PID"    2>/dev/null || true
  # give servers 3s to shut down cleanly
  sleep 3
  [ -n "$VIEWER_PID" ] && kill -KILL "$VIEWER_PID" 2>/dev/null || true
  [ -n "$API_PID" ]    && kill -KILL "$API_PID"    2>/dev/null || true
  if [ -n "$WORKTREE_DIR" ]; then
    echo "[capture-graph-lcp] removing worktree $WORKTREE_DIR"
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
  fi
  echo "[capture-graph-lcp] api log:    $API_LOG"
  echo "[capture-graph-lcp] viewer log: $VIEWER_LOG"
}
trap cleanup EXIT INT TERM

# --- api ---
echo "[capture-graph-lcp] booting api on :${API_PORT} against $CORPUS_ABS"
(
  cd "$RUN_ROOT"
  # Serve reads --output-dir; api's routes read the corpus artifacts from there.
  # Bind explicitly to 127.0.0.1 (no external exposure).
  exec .venv/bin/python -m podcast_scraper.cli serve \
    --output-dir "$CORPUS_ABS" \
    --host 127.0.0.1 \
    --port "$API_PORT"
) > "$API_LOG" 2>&1 &
API_PID=$!

# wait up to 30s for api to accept connections
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
    echo "[capture-graph-lcp] api is up (pid $API_PID)"
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
  echo "FATAL: api did not become healthy in 30s. Tail of $API_LOG:" >&2
  tail -30 "$API_LOG" >&2
  exit 4
fi

# --- viewer ---
echo "[capture-graph-lcp] booting viewer on :${VIEWER_PORT} (proxying api to :${API_PORT})"
(
  cd "$RUN_ROOT/web/gi-kg-viewer"
  # env -u NODE_OPTIONS — strip cmux/agent env poison that breaks node preload
  # scripts (see docs/wip/graph-v3/HARDEN-FOLLOWUPS-2026-07-17.md).
  exec env -u NODE_OPTIONS \
    VITE_API_TARGET="http://127.0.0.1:${API_PORT}" \
    ./node_modules/.bin/vite dev --port "$VIEWER_PORT" --host 127.0.0.1
) > "$VIEWER_LOG" 2>&1 &
VIEWER_PID=$!

# wait up to 30s for vite to serve index
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${VIEWER_PORT}/" >/dev/null 2>&1; then
    echo "[capture-graph-lcp] viewer is up (pid $VIEWER_PID)"
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:${VIEWER_PORT}/" >/dev/null 2>&1; then
  echo "FATAL: viewer did not become healthy in 30s. Tail of $VIEWER_LOG:" >&2
  tail -30 "$VIEWER_LOG" >&2
  exit 5
fi

# ---------------------------------------------------------------------- capture
# The graph route reads ?path= from the query string to select the corpus.
TARGET_URL="http://127.0.0.1:${VIEWER_PORT}/?path=${CORPUS_ABS}"

echo "[capture-graph-lcp] capturing trace: $TARGET_URL"
# The mjs imports @playwright/test, which lives under web/gi-kg-viewer/node_modules.
# Run node from that dir so resolver finds it, but pass the absolute path to
# the mjs so we still consume the top-level script (single source of truth).
(
  cd "$RUN_ROOT/web/gi-kg-viewer"
  env -u NODE_OPTIONS \
    LCP_TARGET_URL="$TARGET_URL" \
    LCP_OUTPUT_DIR="$OUTPUT_DIR" \
    LCP_LABEL="$LABEL" \
    VIEWPORT_WIDTH="$VIEWPORT_WIDTH" \
    VIEWPORT_HEIGHT="$VIEWPORT_HEIGHT" \
    VIEWPORT_DPR="$VIEWPORT_DPR" \
    LCP_WAIT_MS="$WAIT_MS" \
    LCP_LOAD_MODE="$LOAD_MODE" \
    LCP_EXPAND_FIRST_SUPERTHEME="$EXPAND_FIRST" \
    node "$REPO_ROOT/scripts/dev/capture-graph-lcp.mjs"
)

echo "[capture-graph-lcp] done. Metrics + trace under: $OUTPUT_DIR/$LABEL.*"
