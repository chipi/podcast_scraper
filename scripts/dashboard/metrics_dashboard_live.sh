#!/usr/bin/env bash
# Download CI metrics artifacts, validate each bundle, merge with nightly from metrics/,
# regenerate unified index.html, then serve http://127.0.0.1:8777/ (Ctrl+C to stop).
#
# Usage: bash scripts/dashboard/metrics_dashboard_live.sh [LIMIT]
# Env: PYTHON, OUT_DIR, GITHUB_BRANCH — passed through to fetch/validate (see fetch scripts)

set -euo pipefail

LIMIT="${1:-40}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREVIEW="${ROOT}/artifacts/dashboard-preview"

PYTHON_CMD="${PYTHON:-}"
if [[ -z "${PYTHON_CMD}" && -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD="${ROOT}/.venv/bin/python"
fi
PYTHON_CMD="${PYTHON_CMD:-python3}"
export PYTHON="${PYTHON_CMD}"

echo "==> 1/3 Fetch + validate (last ${LIMIT} successful main runs)"
bash "${SCRIPT_DIR}/fetch_and_validate_ci_metrics.sh" "${LIMIT}"

echo ""
echo "==> 2/3 Build dual-source preview (CI + nightly from metrics/)"
bash "${SCRIPT_DIR}/build_local_metrics_preview.sh"

echo ""
echo "==> 3/3 Serving ${PREVIEW} at http://127.0.0.1:8777/"
echo "    Toggle CI vs Nightly in the page. Ctrl+C to stop."
cd "${PREVIEW}"
exec "${PYTHON_CMD}" -m http.server 8777 --bind 127.0.0.1
