#!/usr/bin/env bash
# Rehearse the wall-clock bound, the unbuffered log, and the exit-code path that reprocess-prod.yml
# wraps around the remote `docker compose run`.
#
# WHY THIS EXISTS. Those three things are inline shell inside an `ssh` argument. actionlint parses
# it, shellcheck lints it, and neither EXECUTES it — the only place it runs for real is a run that
# rewrites the production corpus and spends real ASR money. On 2026-08-18 that run hit the
# 360-minute GitHub default (a bound nobody had chosen), was killed, and took six hours of output
# with it: the sole copy lived in the Actions log, which GitHub withholds until a job ends and then
# discards on cancel, and Python had block-buffered most of it into a pipe anyway.
#
# The dangerous half of the fix is the `| tee` that now saves those logs on the box. A pipeline's
# exit status is its LAST command's, so `docker compose run ... | tee log` reports tee's success
# even when the pipeline died — a failed repair would report green. `set -o pipefail` is what
# prevents that, and case 3 below is the test that it is actually doing so.
#
# REQUIREMENTS: none. Everything runs in a temp sandbox against `false`/`sleep`, never docker,
# never ssh, never prod.
#
# USAGE:  bash scripts/tools/rehearse_reprocess_bounds.sh
# Keep these cases in sync with the "Reprocess" step in .github/workflows/reprocess-prod.yml.
set -uo pipefail

SANDBOX=$(mktemp -d)
trap 'rm -rf "$SANDBOX"' EXIT
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  FAIL  %s\n' "$1"; }
check() { # check <description> <expected> <actual>
  if [ "$2" = "$3" ]; then ok "$1 (= $3)"; else bad "$1 (expected $2, got $3)"; fi
}

# The workflow calls coreutils `timeout` on the prod box (Ubuntu). macOS ships neither `timeout`
# nor `gtimeout` and this machine's brew cannot install coreutils, so the cases below use the
# real thing when it exists and this six-line equivalent when it does not. It reproduces only
# what these cases actually assert: run a command, kill it after N seconds, exit 124 if it had
# to be killed. It is NOT a `timeout` replacement and nothing outside this script uses it.
BOUNDED_IMPL="unknown"
if command -v timeout >/dev/null 2>&1; then
  BOUNDED_IMPL="timeout(1)"
  bounded() { local secs=$1; shift; timeout --signal=TERM --kill-after=2 "$secs" "$@"; }
elif command -v gtimeout >/dev/null 2>&1; then
  BOUNDED_IMPL="gtimeout(1)"
  bounded() { local secs=$1; shift; gtimeout --signal=TERM --kill-after=2 "$secs" "$@"; }
else
  BOUNDED_IMPL="portable shell fallback (no timeout(1) on this machine)"
  bounded() {
    local secs=$1; shift
    "$@" & local pid=$!
    ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null; sleep 2; kill -KILL "$pid" 2>/dev/null ) &
    local killer=$!
    # NOT `if wait ...; then`: a bash `if` whose condition FAILS and which has no `else` returns
    # 0, so the killed process's status was being thrown away and every overrun looked clean.
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$killer" 2>/dev/null
    wait "$killer" 2>/dev/null
    # 128+SIGTERM(15)=143, 128+SIGKILL(9)=137: we killed it, which coreutils reports as 124.
    if [ "$rc" -eq 143 ] || [ "$rc" -eq 137 ]; then return 124; fi
    return "$rc"
  }
fi
echo "bounded-run implementation: ${BOUNDED_IMPL}"
echo

echo "== 1. the remote bound is derived from the job bound, and is SHORTER =="
# Verbatim from the workflow: REMOTE_TIMEOUT_S=$(( (TIMEOUT_MINUTES - 1) * 60 ))
for m in 240 60 16; do
  TIMEOUT_MINUTES="$m"                       # arrives from the step env as a STRING
  REMOTE_TIMEOUT_S=$(( (TIMEOUT_MINUTES - 1) * 60 ))
  check "job ${m}m -> remote bound" "$(( (m - 1) * 60 ))" "$REMOTE_TIMEOUT_S"
  if [ "$REMOTE_TIMEOUT_S" -lt "$(( m * 60 ))" ]; then
    ok "remote bound is shorter than the job bound (so the box kills the container first)"
  else
    bad "remote bound is NOT shorter than the job bound"
  fi
done

echo
echo "== 2. \`timeout\` actually terminates work that overruns =="
start=$(date +%s)
bounded 1 sleep 30
rc=$?
elapsed=$(( $(date +%s) - start ))
check "timeout returns 124 on overrun" "124" "$rc"
if [ "$elapsed" -lt 5 ]; then
  ok "overrunning command was killed promptly (${elapsed}s, not 30s)"
else
  bad "overrunning command ran ${elapsed}s — timeout did not terminate it"
fi

echo
echo "== 3. THE ONE THAT MATTERS: a failing run must stay failing THROUGH the tee =="
# This is the regression the `| tee` introduces and `set -o pipefail` removes. Both directions are
# asserted, because a green assertion that would also be green without the fix proves nothing.
LOG="$SANDBOX/reprocess.log"

# 3a. WITHOUT pipefail — the pre-fix shape. Demonstrates the bug is real, not hypothetical.
( set +o pipefail; false 2>&1 | tee -a "$LOG" >/dev/null )
check "without pipefail, a failed run reports SUCCESS (the bug)" "0" "$?"

# 3b. WITH pipefail — the shape the workflow now uses.
( set -o pipefail; false 2>&1 | tee -a "$LOG" >/dev/null )
check "with pipefail, a failed run reports FAILURE" "1" "$?"

# 3c. and a successful run must still report success.
( set -o pipefail; true 2>&1 | tee -a "$LOG" >/dev/null )
check "with pipefail, a successful run still reports SUCCESS" "0" "$?"

# 3d. the timeout's own exit code must survive the tee too — otherwise hitting the wall clock
#     would look like a clean finish, which is precisely how the 2026-08-18 run was reported.
( set -o pipefail; bounded 1 sleep 30 2>&1 | tee -a "$LOG" >/dev/null )
check "a run killed by the remote timeout reports 124 through the tee" "124" "$?"

echo
echo "== 4. the log file is written, and is readable independently of the job =="
printf 'episode 1 done\n' > "$SANDBOX/out.txt"
( set -o pipefail; tee -a "$LOG" < "$SANDBOX/out.txt" >/dev/null )
if grep -q 'episode 1 done' "$LOG"; then
  ok "output reached the box-local log"
else
  bad "output did NOT reach the box-local log"
fi

echo
echo "== 5. PYTHONUNBUFFERED is what makes a killed run's log non-empty =="
# Python block-buffers stdout into a pipe (4-8 KiB). Killed before the buffer flushes, the log is
# empty even though the work happened. With PYTHONUNBUFFERED=1 each line lands immediately.
PYBIN=$(command -v python3 || true)
if [ -z "$PYBIN" ]; then
  echo "  SKIP  no python3 on PATH"
else
  cat > "$SANDBOX/emit.py" <<'PYEOF'
import sys, time
for i in range(200):
    print(f"line {i} " + "x" * 60)
time.sleep(30)
PYEOF
  # Buffered: killed mid-sleep, the 200 lines are still in the buffer.
  ( bounded 1 "$PYBIN" "$SANDBOX/emit.py" > "$SANDBOX/buffered.log" 2>&1 ) || true
  # Unbuffered: the same 200 lines are already on disk when the kill lands.
  ( bounded 1 env PYTHONUNBUFFERED=1 "$PYBIN" "$SANDBOX/emit.py" \
      > "$SANDBOX/unbuffered.log" 2>&1 ) || true
  b=$(wc -l < "$SANDBOX/buffered.log" | tr -d ' ')
  u=$(wc -l < "$SANDBOX/unbuffered.log" | tr -d ' ')
  echo "  buffered log: ${b} lines / unbuffered log: ${u} lines"
  if [ "$u" -gt "$b" ]; then
    ok "PYTHONUNBUFFERED=1 preserves output a kill would otherwise discard"
  else
    bad "PYTHONUNBUFFERED made no difference (buffered=${b} unbuffered=${u})"
  fi
fi

echo
echo "================================"
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
