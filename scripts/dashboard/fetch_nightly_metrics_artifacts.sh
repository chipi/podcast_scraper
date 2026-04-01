#!/usr/bin/env bash
# Download the "nightly-metrics" workflow artifact from recent successful nightly.yml runs on main.
# Requires: gh CLI (authenticated), git repo with GitHub remote.
#
# Usage:
#   bash scripts/dashboard/fetch_nightly_metrics_artifacts.sh [LIMIT]
#   LIMIT defaults to 25 (last twenty-five successful runs attempted).
#
# Output directory: artifacts/nightly-metrics-runs/run-<databaseId>/
#
# Env:
#   GITHUB_BRANCH  — branch to filter (default: main)
#   OUT_DIR        — base output directory (default: artifacts/nightly-metrics-runs)

set -euo pipefail

LIMIT="${1:-25}"
BRANCH="${GITHUB_BRANCH:-main}"
OUT_DIR="${OUT_DIR:-artifacts/nightly-metrics-runs}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh (GitHub CLI) not found. Install: https://cli.github.com/" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

WF="nightly.yml"
IDS=()
while IFS= read -r id; do
  [[ -n "${id}" ]] && IDS+=("${id}")
done < <(gh run list --workflow="${WF}" --branch="${BRANCH}" --status=success \
  --limit="${LIMIT}" --json databaseId -q '.[].databaseId' 2>/dev/null || true)

if [[ ${#IDS[@]} -eq 0 ]]; then
  echo "error: no successful runs found for workflow ${WF} on ${BRANCH}" >&2
  exit 1
fi

echo "Fetching nightly-metrics artifact for up to ${LIMIT} run(s) on ${BRANCH} -> ${OUT_DIR}/"
# gh run download fails if --dir already contains extracted files (no overwrite). Clear the
# run directory before each download; treat existing latest-nightly.json as cache (like CI fetch).
usable=0
for id in "${IDS[@]}"; do
  dest="${OUT_DIR}/run-${id}"
  if [[ -f "${dest}/latest-nightly.json" ]]; then
    echo "  have ${id} (already downloaded)"
    usable=$((usable + 1))
    continue
  fi
  rm -rf "${dest}"
  mkdir -p "${dest}"
  if gh run download "${id}" --name nightly-metrics --dir "${dest}" 2>/dev/null; then
    echo "  ok   ${id} -> ${dest}"
    usable=$((usable + 1))
  else
    echo "  skip ${id} (no nightly-metrics artifact)"
    rmdir "${dest}" 2>/dev/null || true
  fi
done

if [[ "${usable}" -eq 0 ]]; then
  echo "error: no nightly-metrics artifacts usable (check workflow uploads this name)" >&2
  exit 1
fi

echo "Done (${usable} bundle(s)). Merge with:"
echo "  \${PYTHON} scripts/dashboard/merge_nightly_metrics_runs_to_history.py \\"
echo "    --runs-dir ${OUT_DIR} --history-output metrics/history-nightly.jsonl \\"
echo "    --latest-output metrics/latest-nightly.json"
