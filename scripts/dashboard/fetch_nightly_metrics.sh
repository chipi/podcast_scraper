#!/usr/bin/env bash
# Populate metrics/latest-nightly.json + metrics/history-nightly.jsonl for local dashboard preview.
#
# Strategy (first success wins):
#   1) GitHub CLI: download the "nightly-metrics" artifact from the latest successful nightly.yml run.
#   2) curl: fetch the same files from GitHub Pages (published metrics/ on gh-pages).
#
# Usage:
#   bash scripts/dashboard/fetch_nightly_metrics.sh
#   bash scripts/dashboard/fetch_nightly_metrics.sh 25   # last 25 nightlies: gh download + merge JSONL
#
# Env:
#   GITHUB_BRANCH     — branch for gh run list (default: main)
#   METRICS_DIR       — output directory (default: <repo>/metrics)
#   GHPAGES_METRICS_BASE — override Pages base, e.g. https://chipi.github.io/podcast_scraper/metrics
#                          (no trailing slash). If unset, derived via gh or git remote.
#   FETCH_NIGHTLY_PREFER_PAGES — if set to 1, skip gh and curl GitHub Pages only (often fuller history).
#   PYTHON            — optional; if set or .venv exists, runs repair_metrics_jsonl on history
#
# First argument (or use a numeric-only first arg): if >= 1, runs fetch_nightly_metrics_artifacts.sh
# and merge_nightly_metrics_runs_to_history.py into METRICS_DIR (many chart points locally).
#
# Requires: curl. For (1): gh authenticated. For (2): public Pages or reachable URL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
METRICS_DIR="${METRICS_DIR:-${ROOT}/metrics}"
BRANCH="${GITHUB_BRANCH:-main}"

PYTHON_CMD="${PYTHON:-}"
if [[ -z "${PYTHON_CMD}" && -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD="${ROOT}/.venv/bin/python"
fi
PYTHON_CMD="${PYTHON_CMD:-}"

mkdir -p "${METRICS_DIR}"

LIMIT="${1:-}"
if [[ -n "${LIMIT}" && "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -ge 1 ]]; then
  if [[ -z "${PYTHON_CMD}" ]]; then
    echo "error: set PYTHON or create .venv for merge step" >&2
    exit 1
  fi
  echo "Multi-nightly mode: up to ${LIMIT} successful ${BRANCH} run(s) -> artifacts/nightly-metrics-runs/ + merged metrics/"
  bash "${SCRIPT_DIR}/fetch_nightly_metrics_artifacts.sh" "${LIMIT}"
  "${PYTHON_CMD}" "${ROOT}/scripts/dashboard/merge_nightly_metrics_runs_to_history.py" \
    --runs-dir "${ROOT}/artifacts/nightly-metrics-runs" \
    --history-output "${METRICS_DIR}/history-nightly.jsonl" \
    --latest-output "${METRICS_DIR}/latest-nightly.json"
  echo "OK: merged latest-nightly + history-nightly.jsonl -> ${METRICS_DIR}/"
  echo "Done. Rebuild preview: make build-metrics-dashboard-preview"
  exit 0
fi

repair_history() {
  local f="${METRICS_DIR}/history-nightly.jsonl"
  [[ -f "${f}" ]] || return 0
  if [[ -n "${PYTHON_CMD}" ]]; then
    "${PYTHON_CMD}" "${ROOT}/scripts/dashboard/repair_metrics_jsonl.py" "${f}" --in-place || true
  fi
}

try_gh_artifact() {
  if ! command -v gh >/dev/null 2>&1; then
    return 1
  fi
  local id
  id="$(gh run list --workflow=nightly.yml --branch="${BRANCH}" --status=success \
    --limit=1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"
  if [[ -z "${id}" || "${id}" == "null" ]]; then
    return 1
  fi
  local tmp
  tmp="$(mktemp -d)"
  if gh run download "${id}" --name nightly-metrics --dir "${tmp}" 2>/dev/null; then
    cp "${tmp}/latest-nightly.json" "${METRICS_DIR}/"
    cp "${tmp}/history-nightly.jsonl" "${METRICS_DIR}/"
    rm -rf "${tmp}"
    echo "OK: nightly-metrics artifact (run ${id}) -> ${METRICS_DIR}/"
    return 0
  fi
  rm -rf "${tmp}"
  return 1
}

derive_pages_base() {
  if [[ -n "${GHPAGES_METRICS_BASE:-}" ]]; then
    echo "${GHPAGES_METRICS_BASE%/}"
    return 0
  fi
  local owner repo
  if command -v gh >/dev/null 2>&1; then
    owner="$(gh repo view --json owner -q .owner.login 2>/dev/null || true)"
    repo="$(gh repo view --json name -q .name 2>/dev/null || true)"
    if [[ -n "${owner}" && -n "${repo}" ]]; then
      owner="$(echo "${owner}" | tr '[:upper:]' '[:lower:]')"
      echo "https://${owner}.github.io/${repo}/metrics"
      return 0
    fi
  fi
  local origin
  origin="$(git -C "${ROOT}" config --get remote.origin.url 2>/dev/null || true)"
  if [[ "${origin}" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
    owner="$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
    repo="${BASH_REMATCH[2]}"
    echo "https://${owner}.github.io/${repo}/metrics"
    return 0
  fi
  return 1
}

try_pages_curl() {
  local base
  base="$(derive_pages_base)" || {
    echo "error: set GHPAGES_METRICS_BASE or use gh / a github.com git remote" >&2
    return 1
  }
  local latest_url="${base}/latest-nightly.json"
  local hist_url="${base}/history-nightly.jsonl"
  echo "Trying GitHub Pages: ${base}/"
  curl -fsSL -o "${METRICS_DIR}/latest-nightly.json" "${latest_url}"
  curl -fsSL -o "${METRICS_DIR}/history-nightly.jsonl" "${hist_url}"
  echo "OK: curl -> ${METRICS_DIR}/"
}

if [[ "${FETCH_NIGHTLY_PREFER_PAGES:-}" == "1" ]]; then
  echo "FETCH_NIGHTLY_PREFER_PAGES=1: using GitHub Pages only (skip gh artifact)"
  if try_pages_curl; then
    repair_history
    echo "Done. Rebuild preview: make build-metrics-dashboard-preview"
    exit 0
  fi
  exit 1
fi

if try_gh_artifact; then
  repair_history
  echo "Done. Rebuild preview: make build-metrics-dashboard-preview"
  exit 0
fi

echo "gh artifact unavailable or failed; falling back to GitHub Pages curl..."
if try_pages_curl; then
  repair_history
  echo "Done. Rebuild preview: make build-metrics-dashboard-preview"
  exit 0
fi

exit 1
