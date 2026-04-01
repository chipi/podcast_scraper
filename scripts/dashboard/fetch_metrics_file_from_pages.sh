#!/usr/bin/env bash
# Load a single file from published GitHub Pages metrics/, with git gh-pages fallback.
#
# GitHub Actions "deploy-pages" publishes the site without necessarily updating the gh-pages
# git ref. Workflows that only use "git show gh-pages:metrics/..." therefore see stale or
# empty history and append one line per run forever. Prefer the live site URL first.
#
# Usage:
#   GITHUB_REPOSITORY=owner/repo bash fetch_metrics_file_from_pages.sh <file> <output-path>
#
# Env:
#   GITHUB_REPOSITORY  — required (set automatically in Actions)
#   METRICS_PAGES_BASE — optional override, e.g. https://owner.github.io/repo/metrics (no trailing slash)
#
# Exit 0 always. Writes output file (possibly empty).

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: fetch_metrics_file_from_pages.sh <file-under-metrics/> <local-output-path>" >&2
  exit 2
fi

REL="$1"
OUT="$2"
mkdir -p "$(dirname "$OUT")"

if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
  echo "error: GITHUB_REPOSITORY is not set" >&2
  exit 2
fi

OWNER="$(echo "${GITHUB_REPOSITORY%%/*}" | tr '[:upper:]' '[:lower:]')"
REPO="$(echo "${GITHUB_REPOSITORY#*/}" | tr '[:upper:]' '[:lower:]')"
BASE="${METRICS_PAGES_BASE:-https://${OWNER}.github.io/${REPO}/metrics}"
BASE="${BASE%/}"

if curl -fsSL "${BASE}/${REL}" -o "${OUT}.tmp" 2>/dev/null && [[ -s "${OUT}.tmp" ]]; then
  mv "${OUT}.tmp" "${OUT}"
  echo "✅ metrics/${REL} <- live Pages (${BASE})"
  exit 0
fi
rm -f "${OUT}.tmp"

git fetch origin gh-pages:gh-pages 2>/dev/null || true
if git show "gh-pages:metrics/${REL}" >"${OUT}" 2>/dev/null && [[ -s "${OUT}" ]]; then
  echo "✅ metrics/${REL} <- gh-pages git branch"
  exit 0
fi

: >"${OUT}"
echo "📝 metrics/${REL} not found on live Pages or gh-pages branch; empty ${OUT}"
