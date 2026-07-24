#!/usr/bin/env bash
# capture-graph-api.sh — pure-HTTP perf capturer for the graph LOAD API path.
#
# perf-traces Chunk 2: the API-only companion to capture-graph-lcp.{sh,mjs}
# (browser paint). Measures the endpoint latency the viewer pays before the
# graph renders — /api/artifacts listing, the per-episode GI/KG artifact
# fan-out, /api/corpus/topic-clusters, and a 4-way concurrency guard. No
# browser. Emits a per-run .api.metrics.json + a summary printout.
#
# Like capture-search-api.sh, this targets an ALREADY-RUNNING api (start one
# with `make serve` or `podcast-scraper-api --path <corpus>`), so it can
# baseline against any live api without booting its own.
#
# Usage:
#   scripts/dev/capture-graph-api.sh \
#       --api http://localhost:8000 \
#       --corpus /abs/path/to/corpus \
#       --label <release>-graph-api \
#       [--output-dir data/perf/traces/graph] \
#       [--iterations 3] [--sample 20]
#
# Assertions:
#  - api reachable at /api/health (else FATAL).
#  - The `api-concurrent-4` scenario asserts no socket death across 4 parallel
#    workers — the runtime companion to the #1205 SIGSEGV guardrail.

set -euo pipefail

API=""
CORPUS=""
LABEL=""
OUTPUT_DIR="data/perf/traces/graph"
ITERATIONS=3
SAMPLE=20

while [ $# -gt 0 ]; do
  case "$1" in
    --api)          API="$2"; shift 2 ;;
    --corpus)       CORPUS="$2"; shift 2 ;;
    --label)        LABEL="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --iterations)   ITERATIONS="$2"; shift 2 ;;
    --sample)       SAMPLE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | head -30
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$API" ]    || { echo "FATAL: --api <url> required" >&2; exit 2; }
[ -n "$CORPUS" ] || { echo "FATAL: --corpus <path> required" >&2; exit 2; }
[ -n "$LABEL" ]  || { echo "FATAL: --label <name> required" >&2; exit 2; }
[ -d "$CORPUS" ] || { echo "FATAL: corpus dir not found: $CORPUS" >&2; exit 2; }

CORPUS_ABS="$(cd "$CORPUS" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
OUT_JSON="${OUTPUT_DIR}/${LABEL}.api.metrics.json"

if ! curl -fsS -o /dev/null "${API%/}/api/health"; then
  echo "FATAL: api unreachable at ${API}/api/health" >&2
  exit 3
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
[ -x "${REPO_ROOT}/.venv/bin/python" ] || {
  echo "FATAL: .venv/bin/python missing — run 'make dev-setup' first" >&2
  exit 2
}

echo "[capture-graph-api] label=${LABEL} api=${API} iterations=${ITERATIONS} sample=${SAMPLE}"
echo "[capture-graph-api] out=${OUT_JSON}"

exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/dev/capture_graph_api.py" \
  --api "$API" \
  --corpus "$CORPUS_ABS" \
  --label "$LABEL" \
  --out "$OUT_JSON" \
  --iterations "$ITERATIONS" \
  --sample "$SAMPLE"
