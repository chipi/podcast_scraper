#!/usr/bin/env bash
# Single entrypoint for the **Playwright-driven** stack test (same sequence as
# ``.github/workflows/stack-test.yml`` after image build): (optional) build → up →
# health → Playwright → export → pipeline job log assert → artifact assert → down.
#
# This does **not** run ``make stack-test-run`` / ``make stack-test-assert-logs``;
# those validate ``.stack-test/pipeline.log`` from a **one-shot** compose pipeline
# (alternate local path). The UI job leaves traces under ``.viewer/jobs/*.log`` on
# the exported corpus — covered by ``make stack-test-assert-pipeline-job-logs``.
#
# Discoverability: ``make stack-test-ci-local`` (optional ``STACK_TEST_CI_LOCAL_ARGS``),
# ``make stack-test-ci-local-clean`` (``--clean-export``), or
# ``STACK_TEST_CI_LOCAL_CLEAN_EXPORT=1``.
#
# **Full transcript:** keep logs under ``.stack-test/``, not the repo root (avoid
# ``tee .stack-test-*.log`` at ``$PWD``). Example:
#   mkdir -p .stack-test && bash scripts/tools/run_stack_test_ci_local.sh 2>&1 | tee .stack-test/ci-local-run.log
#
# Prereqs: Docker + Compose v2, make, Node/npm (Playwright), Python with
# ``pip install -e '.[dev]'`` for stack-test-assert-artifacts (gil/kg scripts).
# Build step: unless ``--skip-build``, skips ``make stack-test-build`` when all four
# stack images exist locally (``docker image inspect``); use ``--force-build`` to
# always rebuild. If only ``pipeline-llm`` is missing: ``make stack-test-build-pipeline-llm``.
# Heavy ``pipeline-ml`` rebuild: ``make stack-test-build-fast``.
#
# Usage (from repo root):
#   bash scripts/tools/run_stack_test_ci_local.sh
#   bash scripts/tools/run_stack_test_ci_local.sh --verbose --skip-build --no-down
#   bash scripts/tools/run_stack_test_ci_local.sh --force-build
#   make stack-test-ci-local STACK_TEST_CI_LOCAL_ARGS='--skip-build'
#   make stack-test-ci-local-clean
#
# Env (optional): CONFIG_FILE, PODCAST_DOCKER_PROJECT_DIR, STACK_TEST_BASE_URL,
# STACK_TEST_HEALTH_TIMEOUT_SEC (default 180), STACK_TEST_OPERATOR_PROFILE,
# STACK_TEST_JOB_POLL_MS, STACK_PIPELINE_MEM_LIMIT / STACK_PIPELINE_SHM_SIZE,
# STACK_TEST_LOG_DIR (default: ``<repo>/.stack-test`` — created at startup),
# STACK_TEST_CI_LOCAL_FORCE_BUILD=1, STACK_TEST_CI_LOCAL_CLEAN_EXPORT=1, other STACK_TEST_* / make vars.
#
# Dotenv: if ``$ROOT/.env`` exists (or ``ENV_FILE`` path), it is sourced as bash
# with ``set -a``. Skip: ``SKIP_DOTENV=1``. Override path: ``ENV_FILE=/path/.env``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

skip_build=0
no_down=0
verbose=0
clean_export=0
force_build=0

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

load_dotenv() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  log "loading env from ${f}"
  set -a
  # shellcheck disable=SC1090
  source "$f"
  set +a
}

usage() {
  cat <<'EOF'
Run stack test flow locally (same make targets as stack-test CI after build).

Usage:
  bash scripts/tools/run_stack_test_ci_local.sh [options]
  make stack-test-ci-local [STACK_TEST_CI_LOCAL_ARGS='..']

Options:
  --verbose       Trace shell commands (set -x) and extra hints
  --skip-build    Never run make stack-test-build (mutually exclusive with --force-build)
  --force-build   Always run make stack-test-build (default: skip if all stack images exist)
  --no-down       Do not run make stack-test-down at the end
  --clean-export  Remove STACK_TEST_EXPORT_DIR before export; must resolve under repo root
  -h, --help      This message

Logs: use ``mkdir -p .stack-test`` and ``tee .stack-test/<name>.log`` — not ``.stack-test-*.log`` at repo root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) skip_build=1 ;;
    --force-build) force_build=1 ;;
    --no-down) no_down=1 ;;
    --verbose) verbose=1 ;;
    --clean-export) clean_export=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${verbose}" -eq 1 ]]; then
  set -x
fi

if [[ "${SKIP_DOTENV:-0}" != "1" ]]; then
  load_dotenv "${ENV_FILE:-$ROOT/.env}"
fi

export CONFIG_FILE="${CONFIG_FILE:-$ROOT/config/ci/stack-test-config.yaml}"
export PODCAST_DOCKER_PROJECT_DIR="${PODCAST_DOCKER_PROJECT_DIR:-$ROOT}"
export PODCAST_DOCKER_HOST_REPO="${PODCAST_DOCKER_HOST_REPO:-$PODCAST_DOCKER_PROJECT_DIR}"

if [[ "${CONFIG_FILE}" != /* ]]; then
  echo "CONFIG_FILE must be an absolute path (got: ${CONFIG_FILE})" >&2
  exit 2
fi
if [[ "${PODCAST_DOCKER_PROJECT_DIR}" != /* ]]; then
  echo "PODCAST_DOCKER_PROJECT_DIR must be an absolute path (got: ${PODCAST_DOCKER_PROJECT_DIR})" >&2
  exit 2
fi

if [[ "${STACK_TEST_CI_LOCAL_CLEAN_EXPORT:-0}" == "1" ]]; then
  clean_export=1
fi
if [[ "${STACK_TEST_CI_LOCAL_FORCE_BUILD:-0}" == "1" ]]; then
  force_build=1
fi
if [[ "${skip_build}" -eq 1 && "${force_build}" -eq 1 ]]; then
  echo "run_stack_test_ci_local.sh: --skip-build and --force-build are mutually exclusive" >&2
  exit 2
fi

base="${STACK_TEST_BASE_URL:-http://127.0.0.1:8090}"
health_timeout="${STACK_TEST_HEALTH_TIMEOUT_SEC:-180}"

cd "${ROOT}"

STACK_TEST_LOG_DIR="${STACK_TEST_LOG_DIR:-${ROOT}/.stack-test}"
mkdir -p "${STACK_TEST_LOG_DIR}"
export STACK_TEST_LOG_DIR
# Short basenames under ``STACK_TEST_LOG_DIR`` (readable; not ``.stack-test-*.log``).
STACK_TEST_FULL_RUN_LOG="${STACK_TEST_FULL_RUN_LOG:-${STACK_TEST_LOG_DIR}/full-run.log}"
STACK_TEST_CI_LOCAL_LOG="${STACK_TEST_CI_LOCAL_LOG:-${STACK_TEST_LOG_DIR}/ci-local-run.log}"
export STACK_TEST_FULL_RUN_LOG STACK_TEST_CI_LOCAL_LOG

# Match Makefile default ``STACK_TEST_EXPORT_DIR ?= $(PWD)/.stack-test-corpus``.
: "${STACK_TEST_EXPORT_DIR:=${ROOT}/.stack-test-corpus}"
export STACK_TEST_EXPORT_DIR

log "repo root: ${ROOT}"
log "stack-test log dir: ${STACK_TEST_LOG_DIR} (tee e.g. ${STACK_TEST_CI_LOCAL_LOG}; avoid repo-root .stack-test-*.log)"

stack_test_stack_images_present() {
  local img
  for img in \
    podcast-scraper-stack-viewer:latest \
    podcast-scraper-stack-api:latest \
    podcast-scraper-stack-pipeline-ml:latest \
    podcast-scraper-stack-pipeline-llm:latest; do
    if ! docker image inspect "${img}" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

if [[ "${skip_build}" -eq 1 ]]; then
  log "skip stack-test-build (--skip-build)"
elif [[ "${force_build}" -eq 1 ]]; then
  log "make stack-test-build (--force-build)"
  make stack-test-build
elif stack_test_stack_images_present; then
  log "skip stack-test-build (all stack images present); use --force-build to rebuild"
else
  log "make stack-test-build (missing stack image(s); build once or use --skip-build if intentional)"
  make stack-test-build
fi

log "make stack-test-up"
make stack-test-up

log "wait for ${base}/api/health (timeout ${health_timeout}s)"
deadline=$((SECONDS + health_timeout))
last_beat="${SECONDS}"
while true; do
  if curl -fsS "${base}/api/health" >/dev/null 2>&1; then
    log "health OK (${base}/api/health)"
    break
  fi
  if ((SECONDS >= deadline)); then
    echo "Timed out waiting for stack health (${base}/api/health)" >&2
    exit 1
  fi
  if ((SECONDS - last_beat >= 15)); then
    left=$((deadline - SECONDS))
    log "still waiting for health… (~${left}s remaining; is the stack up?)"
    last_beat="${SECONDS}"
  fi
  sleep 2
done

log "make stack-test-playwright (Playwright prints [stack-test] heartbeats during long waits)"
if [[ "${verbose}" -eq 1 ]]; then
  log "tip: in another terminal → docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.stack-test.yml logs -f"
fi
make stack-test-playwright

if [[ "${clean_export}" -eq 1 ]]; then
  root_abs="$(cd "${ROOT}" && pwd)"
  exp="${STACK_TEST_EXPORT_DIR}"
  [[ "${exp}" == /* ]] || exp="${ROOT}/${exp}"
  parent="$(dirname "${exp}")"
  leaf="$(basename "${exp}")"
  mkdir -p "${parent}"
  exp_abs="$(cd "${parent}" && pwd)/${leaf}"
  if [[ "${exp_abs}" == "${root_abs}" || "${exp_abs}" == "${root_abs}/" ]]; then
    echo "run_stack_test_ci_local.sh: refusing --clean-export (export dir is repo root)" >&2
    exit 2
  fi
  if [[ "${exp_abs}" != "${root_abs}"/* ]]; then
    echo "run_stack_test_ci_local.sh: refusing --clean-export (STACK_TEST_EXPORT_DIR must be under repo: ${root_abs}; got: ${exp_abs})" >&2
    exit 2
  fi
  log "clean-export: rm -rf ${exp_abs}"
  rm -rf "${exp_abs}"
  mkdir -p "${exp_abs}"
fi

log "make stack-test-export"
make stack-test-export

log "make stack-test-assert-pipeline-job-logs"
make stack-test-assert-pipeline-job-logs

log "make stack-test-assert-artifacts"
make stack-test-assert-artifacts

if [[ "${no_down}" -eq 0 ]]; then
  log "make stack-test-down"
  make stack-test-down
else
  log "skip stack-test-down (--no-down); tear down with: make stack-test-down"
fi

log "stack test CI local: OK"
