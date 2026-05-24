#!/usr/bin/env bash
# Post-deploy feature-surface smoke over tailnet HTTPS (#796).
#
# Usage:
#   prod_post_deploy_surface_smoke.sh <base_url> <corpus_path>
#
# Example:
#   prod_post_deploy_surface_smoke.sh \
#     "https://prod-podcast.example.ts.net" \
#     "/srv/podcast-scraper/corpus"
set -euo pipefail

BASE_URL="${1:?base URL required (e.g. https://prod-podcast.example.ts.net)}"
CORPUS_PATH="${2:?corpus path required (e.g. /srv/podcast-scraper/corpus)}"

ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$CORPUS_PATH")

probe_json() {
  local name="$1"
  local url="$2"
  local jq_expr="$3"
  echo "[$(date -u +%FT%TZ)] surface smoke: $name"
  if ! curl -fsS "$url" | jq -e "$jq_expr" >/dev/null; then
    echo "ERROR: $name failed ($url)" >&2
    return 1
  fi
}

probe_json "Library (feeds)" \
  "${BASE_URL}/api/corpus/feeds?path=${ENC}" \
  '.feeds | type == "array"'

probe_json "Digest" \
  "${BASE_URL}/api/corpus/digest?path=${ENC}" \
  'type == "object"'

probe_json "Graph (artifacts)" \
  "${BASE_URL}/api/artifacts?path=${ENC}" \
  '.artifacts | type == "array"'

echo "[$(date -u +%FT%TZ)] surface smoke: Search"
search_code=$(curl -s -o /tmp/prod_surface_search.json -w '%{http_code}' \
  "${BASE_URL}/api/search?q=smoke&path=${ENC}&limit=1")
if [ "$search_code" -ge 500 ]; then
  echo "ERROR: Search returned HTTP ${search_code} (5xx)" >&2
  head -c 500 /tmp/prod_surface_search.json >&2 || true
  exit 1
fi
echo "[$(date -u +%FT%TZ)] surface smoke: Search HTTP ${search_code} (non-5xx OK)"

echo "[$(date -u +%FT%TZ)] surface smoke: all four tabs OK"
