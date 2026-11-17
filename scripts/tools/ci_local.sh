#!/usr/bin/env bash
# Public's Python suites the way main's GitHub CI runs them, locally: `make ci-local`.
#
#   scripts/tools/ci_local.sh [unit|integration|e2e|all]     # default: all
#
# The commands and flags are copied from .github/workflows/python-app.yml (jobs test-unit,
# test-integration, test-e2e); keep them in step when that file changes. The full integration and
# e2e suites run in CI only on a push to main or a release branch (a PR runs their -fast subsets):
# this is how a failure in them shows up before the merge instead of after it.
#
# The environment is this checkout's .venv (`make init` + the `search` extra: the same
# `.[dev,ml,llm,search]` CI's integration and e2e jobs install, no private packages, no pandas).
# The test database is this checkout's postgres:18 from scripts/tools/test_db.sh, started if
# PODCAST_TEST_DATABASE_URL is unset and removed again at the end if this run started it.
#
# Env knobs:
#   CI_LOCAL_WORKERS  xdist workers (default 6: the machine is shared; CI uses cpu_count - 2)
#   CI_LOCAL_TREE     the tree to test (default this checkout), e.g. a detached worktree at an
#                     older commit; this checkout's .venv runs it with the tree's src/ first on
#                     PYTHONPATH, so the tree's code and tests are what run.
#
# Each suite prints `PASS <suite>` or `FAIL <suite>`; the last line is `CI-LOCAL: PASS` or
# `CI-LOCAL: FAIL` and the exit code says the same. Logs: reports/ci-local/<suite>.log.
# The functions below are called through `trap` and `"run_$s"`, which shellcheck cannot follow.
# shellcheck disable=SC2329
set -uo pipefail
unset NODE_OPTIONS

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TREE=$(cd "${CI_LOCAL_TREE:-$ROOT}" && pwd)
PY="$ROOT/.venv/bin/python"
WORKERS="${CI_LOCAL_WORKERS:-6}"
LOGS="$ROOT/reports/ci-local"

suite_arg="${1:-all}"
case "$suite_arg" in
  unit | integration | e2e) suites=("$suite_arg") ;;
  all) suites=(unit integration e2e) ;;
  *) echo "usage: $0 [unit|integration|e2e|all]"; exit 2 ;;
esac

# CI's setup-python reads .python-version (python-version-file); the venv must be on its minor.
want=$(sed -nE 's/^([0-9]+\.[0-9]+).*/\1/p' "$TREE/.python-version" 2>/dev/null | head -1)
if [ ! -x "$PY" ]; then
  echo "Missing $PY. Set up the venv first (CONTRIBUTING.md): python${want:-3} -m venv .venv && make init" \
    "&& .venv/bin/pip install -e '.[dev,ml,llm,search]'"
  echo "CI-LOCAL: FAIL"
  exit 1
fi
have=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ -n "$want" ] && [ "$have" != "$want" ]; then
  echo "$PY is Python $have; .python-version (what CI runs) says $want. Rebuild .venv on $want."
  echo "CI-LOCAL: FAIL"
  exit 1
fi
if "$PY" -c "import pandas" 2>/dev/null; then
  echo "WARNING: pandas is importable in $PY; CI's jobs do not have it, so a pandas import passes here and fails there."
fi

mkdir -p "$LOGS"
cd "$TREE" || exit 2
export PYTHONPATH="$TREE/src:$TREE${PYTHONPATH:+:$PYTHONPATH}"
echo "ci-local: tree $TREE at $(git -C "$TREE" rev-parse --short HEAD), python $("$PY" --version 2>&1), $WORKERS workers"

db_started=0
cleanup() {
  # From ROOT: the container is named after the checkout test_db.sh runs in.
  if [ "$db_started" = 1 ]; then (cd "$ROOT" && bash scripts/tools/test_db.sh down); fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ -z "${PODCAST_TEST_DATABASE_URL:-}" ]; then
  if (cd "$ROOT" && bash scripts/tools/test_db.sh up >/dev/null); then
    db_started=1
    PODCAST_TEST_DATABASE_URL=$(cd "$ROOT" && bash scripts/tools/test_db.sh url)
  else
    echo "Could not start the test database (scripts/tools/test_db.sh up)."
    echo "CI-LOCAL: FAIL"
    exit 1
  fi
fi
export PODCAST_TEST_DATABASE_URL

# The CI step's own check after pytest: tests were collected, and at least 50 passed.
check_counts() {
  local log=$1
  if grep -q "no tests collected" "$log"; then
    echo "ERROR: No tests were collected!"
    return 1
  fi
  local n
  n=$(grep -oE "[0-9]+ passed" "$log" | tail -1 | grep -oE "[0-9]+" || echo 0)
  if [ "${n:-0}" -lt 50 ]; then
    echo "ERROR: Only ${n:-0} tests passed, expected at least 50"
    return 1
  fi
}

# CI's helper steps run with no .env; here the checkout may have one (CACHE_DIR in it moves the
# model cache), and config loads .env outside pytest. TESTING=1 skips that load, as pytest does.
step() { TESTING=1 "$PY" "$@"; }

verify_models() { step scripts/cache/verify_required_models.py; }

run_unit() {
  local log="$LOGS/unit.log"
  {
    step scripts/tools/check_unit_test_imports.py &&
      step scripts/tools/kg_quality_metrics.py tests/fixtures/gil_kg_ci_enforce \
        --enforce --strict-schema --fail-on-errors --min-artifacts 1 --min-avg-nodes 1 \
        --min-avg-edges 0 --min-extraction-coverage 1.0 &&
      "$PY" -m pytest tests/unit/ tests/guardrails/ -v --tb=short -n "$WORKERS" \
        --cov=podcast_scraper --cov-report=term-missing --reruns 2 --reruns-delay 1
  } 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && check_counts "$log"
}

run_integration() {
  local log="$LOGS/integration.log"
  verify_models || return 1
  ML_MODELS_VALIDATED=true "$PY" -m pytest tests/integration/ -v -m integration -n "$WORKERS" \
    --cov=podcast_scraper --cov-append --cov-report=term-missing --cov-fail-under=42 \
    --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && check_counts "$log"
}

# CI gives the e2e job one secret, HF_TOKEN. Take only that from the checkout's gitignored .env:
# the rest of .env (API keys) is not in CI and stays out.
hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    printf '%s' "$HF_TOKEN"
  elif [ -f "$ROOT/.env" ]; then
    sed -n -E 's/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=[[:space:]]*//p' "$ROOT/.env" |
      tail -1 | sed -E "s/^[\"']//; s/[\"'][[:space:]]*$//"
  fi
}

run_e2e() {
  local log="$LOGS/e2e.log"
  verify_models || return 1
  local token
  token=$(hf_token)
  [ -n "$token" ] || echo "HF_TOKEN not set (env or .env); CI's e2e job has it."
  # CI's test-e2e job has no Postgres service and no PODCAST_TEST_DATABASE_URL.
  env -u PODCAST_TEST_DATABASE_URL ${token:+HF_TOKEN="$token"} ML_MODELS_VALIDATED=true HF_HUB_OFFLINE=1 \
    E2E_TEST_MODE=multi_episode "$PY" -m pytest tests/e2e/ -v -m "e2e and not nightly" -n "$WORKERS" \
    --cov=podcast_scraper --cov-config=.coveragerc-e2e --cov-report=term-missing \
    --cov-fail-under=37.5 --disable-socket --allow-hosts=127.0.0.1,localhost \
    --reruns 2 --reruns-delay 1 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && check_counts "$log"
}

failed=0
results=()
for s in "${suites[@]}"; do
  echo "=== ci-local: $s ($(date '+%H:%M:%S'))"
  # Each CI job starts on a fresh runner: a coverage file left by the last suite would be appended
  # to (integration passes --cov-append) and lift the measured coverage above CI's.
  rm -f .coverage .coverage.*
  if "run_$s"; then results+=("PASS $s"); else results+=("FAIL $s"); failed=1; fi
done

echo
printf '%s\n' "${results[@]}"
if [ "$failed" = 0 ]; then echo "CI-LOCAL: PASS"; exit 0; fi
echo "CI-LOCAL: FAIL"
exit 1
