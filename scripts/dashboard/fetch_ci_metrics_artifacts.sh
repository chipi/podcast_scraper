#!/usr/bin/env bash
# Download the "metrics" workflow artifact from recent successful Python-app runs on main.
# Requires: gh CLI (authenticated), git repo with GitHub remote.
#
# Usage:
#   bash scripts/dashboard/fetch_ci_metrics_artifacts.sh [LIMIT]
#   LIMIT defaults to 40 (many successful runs have no metrics artifact — scan more to fill history).
#
# Output directory: artifacts/ci-metrics-runs/run-<databaseId>/
# Runs without a metrics artifact are skipped (older runs, or jobs that did not upload).
#
# Env:
#   GITHUB_BRANCH  — branch to filter (default: main)
#   OUT_DIR        — base output directory (default: artifacts/ci-metrics-runs)

set -euo pipefail

LIMIT="${1:-40}"
BRANCH="${GITHUB_BRANCH:-main}"
OUT_DIR="${OUT_DIR:-artifacts/ci-metrics-runs}"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh (GitHub CLI) not found. Install: https://cli.github.com/" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# workflow file in this repo
WF="python-app.yml"
# macOS ships Bash 3.2 (no mapfile); build IDs with a read loop instead.
IDS=()
while IFS= read -r id; do
  [[ -n "${id}" ]] && IDS+=("${id}")
done < <(gh run list --workflow="${WF}" --branch="${BRANCH}" --status=success \
  --limit="${LIMIT}" --json databaseId -q '.[].databaseId' 2>/dev/null || true)

if [[ ${#IDS[@]} -eq 0 ]]; then
  echo "error: no successful runs found for workflow ${WF} on ${BRANCH}" >&2
  exit 1
fi

echo "Fetching metrics artifact for up to ${LIMIT} run(s) on ${BRANCH} -> ${OUT_DIR}/"
for id in "${IDS[@]}"; do
  dest="${OUT_DIR}/run-${id}"
  if [[ -f "${dest}/latest-ci.json" ]]; then
    echo "  have ${id} (already downloaded)"
    continue
  fi
  mkdir -p "${dest}"
  if gh run download "${id}" --name metrics --dir "${dest}" 2>/dev/null; then
    echo "  ok   ${id} -> ${dest}"
  else
    echo "  skip ${id} (no metrics artifact — run may predate upload or skipped metrics job)"
    rmdir "${dest}" 2>/dev/null || true
  fi
done

echo "Done. Validate a bundle with:"
echo "  .venv/bin/python scripts/dashboard/validate_metrics_bundle.py ${OUT_DIR}/run-<id>"
