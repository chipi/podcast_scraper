#!/usr/bin/env bash
# Download CI metrics artifacts (last N successful runs) and validate each bundle.
# Usage: bash scripts/dashboard/fetch_and_validate_ci_metrics.sh [LIMIT]
# Env: OUT_DIR, GITHUB_BRANCH — same as fetch_ci_metrics_artifacts.sh
#      PYTHON — interpreter for validate_metrics_bundle.py (default: .venv/bin/python or python3)
#
# Validates every run-* under OUT_DIR, including folders left from earlier fetches.
# For only the runs just downloaded, remove OUT_DIR first: rm -rf artifacts/ci-metrics-runs

set -euo pipefail

LIMIT="${1:-40}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

bash "${SCRIPT_DIR}/fetch_ci_metrics_artifacts.sh" "${LIMIT}"

OUT_DIR="${OUT_DIR:-artifacts/ci-metrics-runs}"
PYTHON_CMD="${PYTHON:-}"
if [[ -z "${PYTHON_CMD}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD="${REPO_ROOT}/.venv/bin/python"
fi
PYTHON_CMD="${PYTHON_CMD:-python3}"

shopt -s nullglob
runs=( "${OUT_DIR}"/run-* )
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "error: no run-* directories under ${OUT_DIR}" >&2
  exit 1
fi

failed=0
for d in "${runs[@]}"; do
  [[ -d "${d}" ]] || continue
  echo ""
  echo "======== validate: ${d} ========"
  if ! "${PYTHON_CMD}" "${SCRIPT_DIR}/validate_metrics_bundle.py" "${d}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "" >&2
  echo "error: one or more bundles failed validation" >&2
  exit 1
fi

echo ""
echo "All bundles under ${OUT_DIR}/run-* passed validation."
