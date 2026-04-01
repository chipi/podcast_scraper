#!/usr/bin/env bash
# Build artifacts/dashboard-preview/ so the unified dashboard loads BOTH sources locally:
#   - CI: newest artifacts/ci-metrics-runs/run-*/latest-ci.json + merged history-ci.jsonl
#     (one JSONL line per run-* bundle, chronological), else metrics/
#   - Nightly: metrics/latest-nightly.json + metrics/history-nightly.jsonl, OR merged from
#     artifacts/nightly-metrics-runs/run-* (see fetch_nightly_metrics.sh <N>) when present
#   - dashboard-data.json: single bundle (consolidate_dashboard_data.py) for one-fetch UI
# Then regenerate index.html with generate_dashboard.py --unified.
#
# Usage: bash scripts/dashboard/build_local_metrics_preview.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${ROOT}/artifacts/dashboard-preview"
METRICS="${ROOT}/metrics"
RUNS="${ROOT}/artifacts/ci-metrics-runs"
NY_RUNS="${ROOT}/artifacts/nightly-metrics-runs"

mkdir -p "${DEST}"

PYTHON_CMD="${PYTHON:-}"
if [[ -z "${PYTHON_CMD}" && -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD="${ROOT}/.venv/bin/python"
fi
PYTHON_CMD="${PYTHON_CMD:-python3}"

best=""
shopt -s nullglob
bundle_runs=( "${RUNS}"/run-* )
shopt -u nullglob
if [[ ${#bundle_runs[@]} -gt 0 ]]; then
  best=$(printf '%s\n' "${bundle_runs[@]}" | sort -V | tail -1)
fi

if [[ -n "${best}" && -f "${best}/latest-ci.json" ]]; then
  cp "${best}/latest-ci.json" "${DEST}/"
  "${PYTHON_CMD}" "${ROOT}/scripts/dashboard/merge_ci_metrics_runs_to_history.py" \
    --runs-dir "${RUNS}" --output "${DEST}/history-ci.jsonl"
  echo "CI: latest from ${best}; history-ci.jsonl merged from all run-* under ${RUNS}"
elif [[ -f "${METRICS}/latest-ci.json" ]]; then
  cp "${METRICS}/latest-ci.json" "${DEST}/"
  if [[ -f "${METRICS}/history-ci.jsonl" ]]; then
    cp "${METRICS}/history-ci.jsonl" "${DEST}/"
  else
    : > "${DEST}/history-ci.jsonl"
  fi
  echo "CI: ${METRICS}/ (no ci-metrics-runs bundle)"
else
  echo "error: no latest-ci.json (fetch with: make fetch-ci-metrics)" >&2
  exit 1
fi

use_ny_merge=0
shopt -s nullglob
ny_bundles=( "${NY_RUNS}"/run-* )
shopt -u nullglob
for d in "${ny_bundles[@]}"; do
  if [[ -f "${d}/latest-nightly.json" ]]; then
    use_ny_merge=1
    break
  fi
done

if [[ "${use_ny_merge}" -eq 1 ]]; then
  "${PYTHON_CMD}" "${ROOT}/scripts/dashboard/merge_nightly_metrics_runs_to_history.py" \
    --runs-dir "${NY_RUNS}" \
    --history-output "${DEST}/history-nightly.jsonl" \
    --latest-output "${DEST}/latest-nightly.json"
  echo "Nightly: merged from ${NY_RUNS}/run-* into preview"
else
  for f in latest-nightly.json history-nightly.jsonl; do
    if [[ -f "${METRICS}/${f}" ]]; then
      cp "${METRICS}/${f}" "${DEST}/"
    fi
  done
fi

if [[ ! -f "${DEST}/latest-nightly.json" ]]; then
  echo "error: missing nightly latest — run: make fetch-nightly-metrics, or make fetch-nightly-metrics N=25, or add ${METRICS}/latest-nightly.json" >&2
  exit 1
fi
if [[ ! -f "${DEST}/history-nightly.jsonl" ]]; then
  : > "${DEST}/history-nightly.jsonl"
fi

CONSOLIDATE_ARGS=(--input-dir "${DEST}" --output "${DEST}/dashboard-data.json")
if [[ "${METRICS_PREVIEW_STRICT:-}" == "1" ]]; then
  CONSOLIDATE_ARGS+=(--strict)
fi
"${PYTHON_CMD}" "${ROOT}/scripts/dashboard/consolidate_dashboard_data.py" "${CONSOLIDATE_ARGS[@]}"

"${PYTHON_CMD}" "${ROOT}/scripts/dashboard/generate_dashboard.py" --unified --output "${DEST}/index.html"
echo "Dashboard: ${DEST}/index.html"
echo "Serve: make serve-metrics-dashboard"
