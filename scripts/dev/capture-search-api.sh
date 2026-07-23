#!/usr/bin/env bash
# capture-search-api.sh — pure-HTTP perf capturer for the search API.
#
# Search v3 S0(d) / RFC-107 §P (mirrors scripts/dev/capture-graph-lcp.sh
# for the API side). Records p50/p95/p99 for /api/search per intent class,
# per top_k, and under 4-way concurrent load. Emits a per-scenario
# .metrics.json + a summary printout.
#
# The UI-side capturer (capture-search-perf.{sh,mjs}) is separate — it
# lands with slice S2 when the Query Workspace exists. This script is the
# API-only piece that CAN baseline today against any running api.
#
# Usage:
#   scripts/dev/capture-search-api.sh \
#       --api http://localhost:8000 \
#       --corpus /abs/path/to/corpus \
#       --queries tests/fixtures/viewer-validation-corpus/v3/search-queries.json \
#       --label S0-baseline \
#       [--output-dir docs/wip/search-v3/traces] \
#       [--iterations 3]
#
# Assertions:
#  - Each API request returns 200 (non-200 fails the whole run).
#  - The `api-concurrent-4` scenario asserts no SIGSEGV exit code (139)
#    across 4 parallel workers — the runtime companion to the #1205
#    guardrail lint (`make lint-search-v3`).

set -euo pipefail

API=""
CORPUS=""
QUERIES=""
LABEL=""
OUTPUT_DIR="docs/wip/search-v3/traces"
ITERATIONS=3

while [ $# -gt 0 ]; do
  case "$1" in
    --api)          API="$2"; shift 2 ;;
    --corpus)       CORPUS="$2"; shift 2 ;;
    --queries)      QUERIES="$2"; shift 2 ;;
    --label)        LABEL="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --iterations)   ITERATIONS="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | head -30
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$API" ]     || { echo "FATAL: --api <url> required" >&2; exit 2; }
[ -n "$CORPUS" ]  || { echo "FATAL: --corpus <path> required" >&2; exit 2; }
[ -n "$QUERIES" ] || { echo "FATAL: --queries <path> required" >&2; exit 2; }
[ -n "$LABEL" ]   || { echo "FATAL: --label <name> required" >&2; exit 2; }
[ -d "$CORPUS" ]  || { echo "FATAL: corpus dir not found: $CORPUS" >&2; exit 2; }
[ -f "$QUERIES" ] || { echo "FATAL: queries file not found: $QUERIES" >&2; exit 2; }

CORPUS_ABS="$(cd "$CORPUS" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
OUT_JSON="${OUTPUT_DIR}/${LABEL}.api.metrics.json"

# Sanity: api reachable
if ! curl -fsS -o /dev/null "${API%/}/api/health"; then
  echo "FATAL: api unreachable at ${API}/api/health" >&2
  exit 3
fi

# Hand off to the Python capturer (keeps the shell shim tiny; capture logic
# is easier to test in Python + reuses the eval harness's query loader).
REPO_ROOT="$(git rev-parse --show-toplevel)"
[ -x "${REPO_ROOT}/.venv/bin/python" ] || {
  echo "FATAL: .venv/bin/python missing — run 'make dev-setup' first" >&2
  exit 2
}

echo "[capture-search-api] label=${LABEL} api=${API} iterations=${ITERATIONS}"
echo "[capture-search-api] out=${OUT_JSON}"

exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/dev/capture_search_api.py" \
  --api "$API" \
  --corpus "$CORPUS_ABS" \
  --queries "$QUERIES" \
  --label "$LABEL" \
  --out "$OUT_JSON" \
  --iterations "$ITERATIONS"
