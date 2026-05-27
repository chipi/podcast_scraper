#!/usr/bin/env bash
# Post-deploy smoke — codified viewer-surface probes (#797).
#
# Usage:
#   PROD_TAILNET_FQDN=prod.example.ts.net scripts/ops/post_deploy_smoke.sh
#   scripts/ops/post_deploy_smoke.sh <tailnet-fqdn>
#   scripts/ops/post_deploy_smoke.sh <tailnet-fqdn> --corpus-path /srv/podcast-scraper/corpus
#   scripts/ops/post_deploy_smoke.sh --base-url http://127.0.0.1:8090 --corpus-path /app/output
#
# Hits six critical surfaces (see docs/architecture/CORPUS_ARTIFACTS_AND_SURFACES.md):
#   1. GET /api/health          — status ok + core subsystem flags true
#   2. GET /api/corpus/episodes — at least one episode (Library)
#   3. GET /api/corpus/digest   — structured 200 (Digest; rows may be empty)
#   4. GET /api/artifacts       — at least one GI/KG/bridge file (Graph)
#   5. GET /api/corpus/topic-clusters — clusters when index built (Graph overlay)
#   6. GET /api/search          — 200, non-5xx (Search; hits optional)
#
# Exit codes:
#   0  all green
#   1  /api/health subsystem false or status not ok
#   2  corpus-reading surface returned unexpected 4xx/5xx
#   3  2xx but malformed or empty when populated data was expected
set -euo pipefail

FQDN=""
CORPUS_PATH=""
BASE_URL="${SMOKE_BASE_URL:-}"
EXPECT_POPULATED="${EXPECT_POPULATED:-1}"

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \?//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --base-url)
      shift
      BASE_URL="${1:?--base-url requires a value}"
      ;;
    --corpus-path)
      shift
      CORPUS_PATH="${1:?--corpus-path requires a value}"
      ;;
    --allow-empty-corpus)
      EXPECT_POPULATED=0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      if [ -z "$FQDN" ]; then
        FQDN="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
      fi
      ;;
  esac
  shift
done

FQDN="${FQDN:-${PROD_TAILNET_FQDN:-}}"
if [ -z "$BASE_URL" ]; then
  if [ -z "$FQDN" ]; then
    echo "ERROR: tailnet FQDN required (arg or PROD_TAILNET_FQDN) unless --base-url is set" >&2
    exit 2
  fi
  BASE_URL="https://${FQDN}"
fi
BASE_URL="${BASE_URL%/}"
ENC_PATH=""
if [ -n "$CORPUS_PATH" ]; then
  ENC_PATH=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$CORPUS_PATH")
fi

log() {
  # stderr — retry_probe() is called via command substitution to capture
  # the response body to stdout, so any log() emit from inside it would
  # otherwise be interleaved with the body. jq downstream then sees a log
  # line + JSON and errors with "Invalid numeric literal" trying to parse
  # the leading "[YYYY-MM-DD..." as a JSON array (#830).
  echo "[$(date -u +%FT%TZ)] post_deploy_smoke: $*" >&2
}

retry_probe() {
  local name="$1"
  local url="$2"
  local jq_ok="$3"
  local attempt body
  for attempt in $(seq 1 12); do
    log "probe $name (attempt $attempt/12)"
    if body=$(curl -fsS "$url" 2>/dev/null) && printf '%s' "$body" | jq -e "$jq_ok" >/dev/null 2>&1; then
      log "probe $name OK after $((attempt * 5))s wall (approx)"
      printf '%s' "$body"
      return 0
    fi
    sleep 5
  done
  log "probe $name FAILED after 60s: $url"
  curl -sS "$url" 2>&1 | head -c 800 >&2 || true
  return 1
}

fetch_with_code() {
  local url="$1"
  local out="$2"
  curl -sS -o "$out" -w '%{http_code}' "$url"
}

# --- 1. Health ---
health_json=""
if ! health_json=$(retry_probe "health" "${BASE_URL}/api/health" '.status == "ok"'); then
  echo "ERROR: /api/health did not return status=ok within 60s" >&2
  exit 1
fi

for flag in artifacts_api search_api explore_api index_routes_api corpus_library_api corpus_digest_api corpus_metrics_api; do
  if [ "$(printf '%s' "$health_json" | jq -r ".${flag} // true")" = "false" ]; then
    echo "ERROR: /api/health ${flag}=false (expected true on prod viewer stack)" >&2
    exit 1
  fi
done

if [ -z "$CORPUS_PATH" ]; then
  log "no --corpus-path; health-only smoke complete"
  exit 0
fi

PATH_Q="path=${ENC_PATH}"

# --- 2. Library / episodes ---
episodes_json=""
if ! episodes_json=$(retry_probe "corpus/episodes" \
  "${BASE_URL}/api/corpus/episodes?${PATH_Q}&limit=1" \
  '.items | type == "array"'); then
  echo "ERROR: /api/corpus/episodes failed" >&2
  exit 2
fi
if [ "$EXPECT_POPULATED" = "1" ] && [ "$(printf '%s' "$episodes_json" | jq '.items | length')" -lt 1 ]; then
  echo "ERROR: /api/corpus/episodes returned zero items (expected populated prod corpus)" >&2
  exit 3
fi

# --- 3. Digest ---
digest_json=""
if ! digest_json=$(retry_probe "corpus/digest" \
  "${BASE_URL}/api/corpus/digest?${PATH_Q}&window=all" \
  '.rows | type == "array"'); then
  echo "ERROR: /api/corpus/digest failed" >&2
  exit 2
fi
log "digest rows=$(printf '%s' "$digest_json" | jq '.rows | length')"

# --- 4. Graph / artifacts ---
artifacts_json=""
if ! artifacts_json=$(retry_probe "artifacts" \
  "${BASE_URL}/api/artifacts?${PATH_Q}" \
  '.artifacts | type == "array"'); then
  echo "ERROR: /api/artifacts failed" >&2
  exit 2
fi
if [ "$EXPECT_POPULATED" = "1" ] && [ "$(printf '%s' "$artifacts_json" | jq '.artifacts | length')" -lt 1 ]; then
  echo "ERROR: /api/artifacts returned zero files (expected populated prod corpus)" >&2
  exit 3
fi
log "artifacts count=$(printf '%s' "$artifacts_json" | jq '.artifacts | length')"

# --- 5. Topic clusters ---
tc_tmp=$(mktemp)
tc_code=$(fetch_with_code "${BASE_URL}/api/corpus/topic-clusters?${PATH_Q}" "$tc_tmp")
if [ "$tc_code" -ge 500 ]; then
  echo "ERROR: /api/corpus/topic-clusters HTTP ${tc_code}" >&2
  cat "$tc_tmp" >&2 || true
  rm -f "$tc_tmp"
  exit 2
fi
if [ "$tc_code" = "404" ]; then
  if [ "$EXPECT_POPULATED" = "1" ]; then
    echo "ERROR: topic_clusters.json missing (404) on populated prod corpus" >&2
    cat "$tc_tmp" >&2 || true
    rm -f "$tc_tmp"
    exit 3
  fi
  log "topic-clusters 404 — acceptable with EXPECT_POPULATED=0"
else
  if ! jq -e '.clusters | type == "array"' "$tc_tmp" >/dev/null 2>&1; then
    echo "ERROR: topic_clusters.json malformed" >&2
    cat "$tc_tmp" >&2 || true
    rm -f "$tc_tmp"
    exit 3
  fi
  if [ "$EXPECT_POPULATED" = "1" ] && [ "$(jq '.clusters | length' "$tc_tmp")" -lt 1 ]; then
    echo "ERROR: topic_clusters.clusters empty on populated prod corpus" >&2
    rm -f "$tc_tmp"
    exit 3
  fi
  log "topic-clusters OK (clusters=$(jq '.clusters | length' "$tc_tmp"))"
fi
rm -f "$tc_tmp"

# --- 6. Search ---
search_tmp=$(mktemp)
search_code=$(fetch_with_code "${BASE_URL}/api/search?${PATH_Q}&q=ai&top_k=1" "$search_tmp")
if [ "$search_code" -ge 500 ]; then
  echo "ERROR: /api/search HTTP ${search_code}" >&2
  cat "$search_tmp" >&2 || true
  rm -f "$search_tmp"
  exit 2
fi
if ! jq -e '.query and (.results | type == "array")' "$search_tmp" >/dev/null 2>&1; then
  echo "ERROR: /api/search malformed body" >&2
  cat "$search_tmp" >&2 || true
  rm -f "$search_tmp"
  exit 3
fi
log "search OK (HTTP ${search_code}, results=$(jq '.results | length' "$search_tmp"))"
rm -f "$search_tmp"

log "all post-deploy smoke probes green"
exit 0
