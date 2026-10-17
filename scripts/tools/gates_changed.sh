#!/usr/bin/env bash
# The local gates for what changed, and only those: `make gates-changed` (skill: local-gates).
#
#   scripts/tools/gates_changed.sh                 # vs origin/main: committed + staged + unstaged + untracked
#   scripts/tools/gates_changed.sh --staged        # only what is staged (what the next commit holds)
#   scripts/tools/gates_changed.sh --since <ref>   # vs another base
#
# Each gate prints `PASS <gate>` or `FAIL <gate>`; the last line is `GATES: PASS` or `GATES: FAIL`
# and the exit code says the same. Checks only, never rewrites a file (`make format` does that).
#
# Not the full suite: Python tests here are the changed test files plus the tests that NAME a
# changed module (its dotted path or file stem). That misses a test reaching the module through
# another one; the whole suite (`pytest tests/unit tests/integration -n 6`, ~6 min) is the gate at
# the end of an arc.
set -uo pipefail
unset NODE_OPTIONS  # see the top of the Makefile: an inherited shim path breaks every node step

cd "$(git rev-parse --show-toplevel)" || exit 2
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

mode=since
base=origin/main
case "${1:-}" in
  --staged) mode=staged ;;
  --since) base="${2:?--since needs a ref}" ;;
  "") ;;
  *) echo "usage: $0 [--staged | --since <ref>]"; exit 2 ;;
esac

if [ "$mode" = staged ]; then
  files=$(git diff --cached --name-only --diff-filter=d)
else
  files=$( { git diff --name-only --diff-filter=d "$(git merge-base "$base" HEAD)"; \
             git diff --name-only --diff-filter=d; git diff --cached --name-only --diff-filter=d; \
             git ls-files --others --exclude-standard; } | sort -u)
fi
files=$(printf '%s\n' "$files" | while read -r f; do [ -n "$f" ] && [ -f "$f" ] && echo "$f"; done)
if [ -z "$files" ]; then
  if [ "$mode" = staged ]; then echo "nothing staged"; else echo "nothing changed vs $base"; fi
  echo "GATES: PASS"
  exit 0
fi

failed=()
ran=0
gate() {  # gate <name> <command...>
  local name=$1; shift
  ran=$((ran + 1))
  echo "--> $name"
  if "$@"; then echo "PASS $name"; else echo "FAIL $name"; failed+=("$name"); fi
}
pick() { printf '%s\n' "$files" | grep -E "$1" || true; }

py=$(pick '\.py$' | grep -vE '^(web|infra)/' || true)
src_py=$(printf '%s\n' "$py" | grep -E '^src/' || true)
md=$(pick '\.md$' | grep -vE '(^|/)node_modules/|/vendor/' || true)
docs=$(pick '^(docs/|mkdocs\.yml$)')
lp=$(pick '^web/learning-player/(src|e2e)/.*\.(ts|vue|tsx|js)$')
viewer=$(pick '^web/gi-kg-viewer/(src|e2e)/.*\.(ts|vue|tsx|js)$')
workflows=$(pick '^\.github/workflows/.*\.ya?ml$')

if [ -n "$py" ]; then
  # shellcheck disable=SC2086
  gate black "$PY" -m black --check --quiet $py
  # shellcheck disable=SC2086
  gate isort "$PY" -m isort --check-only --quiet $py
  # shellcheck disable=SC2086
  gate flake8 "$PY" -m flake8 --config .flake8 $py
fi
if [ -n "$src_py" ]; then
  # shellcheck disable=SC2086
  gate mypy "$PY" -m mypy $src_py
  gate docstrings make --no-print-directory docstrings
fi
if [ -n "$py" ]; then
  tests=$(printf '%s\n' "$py" | grep -E '^tests/.*test_[^/]*\.py$' || true)
  for f in $src_py; do
    mod=${f#src/}; mod=${mod%.py}; mod=${mod%/__init__}; mod=${mod//\//.}
    stem=$(basename "${f%.py}")
    [ "$stem" = "__init__" ] && stem=$(basename "$(dirname "$f")")
    tests="$tests"$'\n'$(git grep -lE "${mod//./\\.}|\\b${stem}\\b" -- 'tests/**/test_*.py' 2>/dev/null || true)
  done
  tests=$(printf '%s\n' "$tests" | sed '/^$/d' | sort -u)
  if [ -n "$tests" ]; then
    echo "    $(printf '%s\n' "$tests" | wc -l | tr -d ' ') test file(s) name a changed module or changed"
    # shellcheck disable=SC2086
    gate pytest "$PY" -m pytest -q -n 4 -p no:cacheprovider $tests
  else
    echo "    no test file names a changed module: no pytest gate (the full suite still applies)"
  fi
fi
if [ -n "$md" ]; then
  # shellcheck disable=SC2086
  gate markdownlint markdownlint --config .markdownlint.json $md
fi
if [ -n "$docs" ]; then
  gate docs make --no-print-directory docs
fi
if [ -n "$lp" ]; then
  rel=$(printf '%s\n' "$lp" | sed 's#^web/learning-player/##')
  # shellcheck disable=SC2086
  gate "player vitest (related)" bash -c "cd web/learning-player && node_modules/.bin/vitest related --run ${rel//$'\n'/ }"
  gate "player vue-tsc" bash -c "cd web/learning-player && node_modules/.bin/vue-tsc --noEmit -p tsconfig.app.json && node_modules/.bin/vue-tsc --noEmit -p tsconfig.test.json"
fi
if [ -n "$viewer" ]; then
  rel=$(printf '%s\n' "$viewer" | sed 's#^web/gi-kg-viewer/##')
  # shellcheck disable=SC2086
  gate "viewer vitest (related)" bash -c "cd web/gi-kg-viewer && node_modules/.bin/vitest related --run ${rel//$'\n'/ }"
  gate "viewer vue-tsc" bash -c "cd web/gi-kg-viewer && node_modules/.bin/vue-tsc --noEmit -p tsconfig.app.json"
fi
if [ -n "$workflows" ] && command -v actionlint >/dev/null; then
  # shellcheck disable=SC2086
  gate actionlint actionlint $workflows
fi

echo
echo "$(printf '%s\n' "$files" | wc -l | tr -d ' ') changed file(s), $ran gate(s)"
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}"
  echo "GATES: FAIL"
  exit 1
fi
[ "$ran" -gt 0 ] || echo "no gate applies to these files (none of: py, md, docs, web src, workflows)"
echo "GATES: PASS"
