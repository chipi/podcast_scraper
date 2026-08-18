#!/usr/bin/env bash
# Rehearse deploy-prod's "prove before promote" gateway-key gate against a REAL LiteLLM gateway.
#
# WHY THIS EXISTS. The gate is inline shell inside a `ssh` heredoc in deploy-prod.yml. CI cannot
# execute it, actionlint only parses it, and the one place it runs for real is a production
# deploy — which is exactly where a mistake is most expensive. On 2026-08-18 a deploy replaced a
# working LITELLM_API_KEY with a secret value that did not authenticate; D5 caught it a minute
# later, by which point the working key was already gone. The gate exists so that cannot recur,
# and this script exists so the gate itself is tested rather than trusted.
#
# It found two real defects on its first run: `|| echo 000` double-appended curl's own 000 on an
# unreachable host, and a trailing space in the key turned out to be TOLERATED by LiteLLM
# (HTTP 200) rather than fatal — eliminating whitespace as a candidate cause of that incident.
#
# REQUIREMENTS: a reachable LiteLLM gateway and a key valid at it. Defaults to the homelab
# gateway on 127.0.0.1:4001 and reads the key from the repo .env. Nothing is written outside a
# temp sandbox; the repo .env is only READ, never modified.
#
# USAGE:  bash scripts/tools/rehearse_gateway_key_gate.sh
# Keep the `gate()` body in sync with the step in .github/workflows/deploy-prod.yml.
set -uo pipefail

SANDBOX=$(mktemp -d)
trap 'rm -rf "$SANDBOX"' EXIT
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GOOD=$(sed -n 's/^LITELLM_API_KEY=//p' "$REPO_ROOT/.env" | head -1)
if [ -z "$GOOD" ]; then echo "no LITELLM_API_KEY in $REPO_ROOT/.env — cannot rehearse"; exit 2; fi

# The gate, verbatim in behaviour from deploy-prod.yml (unescaped for local run).
gate() {
  cd "$SANDBOX" || return 9
  GW="$1"
  NEWKEY=$(sed -n 's/^LITELLM_API_KEY=//p' .env.deploy-staged | head -1)
  if [ -z "$NEWKEY" ] || [ ! -f .env ] || [ -z "$GW" ]; then
    echo "  gate: SKIPPED (nothing to protect)"
  else
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 20 \
      "${GW%/}/models" -H "Authorization: Bearer $NEWKEY") || true
    CODE=${CODE:-000}
    echo "  gate: HTTP $CODE"
    if [ "$CODE" != 200 ]; then
      rm -f .env.deploy-staged
      echo "  REFUSED — live .env untouched"
      return 1
    fi
  fi
  mv .env.deploy-staged .env
  echo "  PROMOTED"
  return 0
}

GW_OK="http://127.0.0.1:4001/v1"
pass=0; fail=0
check() { # name expected_rc expected_env_marker
  if [ "$2" = "$3" ]; then echo "  ✓ $1"; pass=$((pass+1)); else echo "  ✗ $1 (got '$2' want '$3')"; fail=$((fail+1)); fi
}

echo "=== 1. GOOD key -> promotes ==="
printf 'LITELLM_API_KEY=%s\nMARK=live\n' "$GOOD" > "$SANDBOX/.env"
printf 'LITELLM_API_KEY=%s\nMARK=staged\n' "$GOOD" > "$SANDBOX/.env.deploy-staged"
gate "$GW_OK"; rc=$?
check "returns 0" "$rc" "0"
check "env replaced" "$(sed -n 's/^MARK=//p' "$SANDBOX/.env")" "staged"

echo "=== 2. BAD key -> refuses, live .env survives ==="
printf 'LITELLM_API_KEY=%s\nMARK=live\n' "$GOOD" > "$SANDBOX/.env"
printf 'LITELLM_API_KEY=sk-definitely-not-a-real-key\nMARK=staged\n' > "$SANDBOX/.env.deploy-staged"
gate "$GW_OK"; rc=$?
check "returns 1" "$rc" "1"
check "live .env PRESERVED" "$(sed -n 's/^MARK=//p' "$SANDBOX/.env")" "live"
check "staged file removed" "$([ -f "$SANDBOX/.env.deploy-staged" ] && echo yes || echo no)" "no"

# MEASURED 2026-08-18: LiteLLM TOLERATES a trailing space in the Authorization header
# (HTTP 200), so whitespace is NOT a way to break the key. Recorded as the expectation so
# nobody re-derives it — and so this stops being a candidate cause for the prod 401.
echo "=== 3. TRAILING SPACE on a good key -> still authenticates (whitespace is tolerated) ==="
printf 'LITELLM_API_KEY=%s\nMARK=live\n' "$GOOD" > "$SANDBOX/.env"
printf 'LITELLM_API_KEY=%s \nMARK=staged\n' "$GOOD" > "$SANDBOX/.env.deploy-staged"
gate "$GW_OK"; rc=$?
check "returns 0 (tolerated)" "$rc" "0"
check "env replaced" "$(sed -n 's/^MARK=//p' "$SANDBOX/.env")" "staged"

echo "=== 4. UNREACHABLE gateway -> refuses (fail closed) ==="
printf 'LITELLM_API_KEY=%s\nMARK=live\n' "$GOOD" > "$SANDBOX/.env"
printf 'LITELLM_API_KEY=%s\nMARK=staged\n' "$GOOD" > "$SANDBOX/.env.deploy-staged"
gate "http://127.0.0.1:59999/v1"; rc=$?
check "returns 1" "$rc" "1"
check "live .env PRESERVED" "$(sed -n 's/^MARK=//p' "$SANDBOX/.env")" "live"

echo "=== 5. NO live .env (bootstrap) -> promotes anyway ==="
rm -f "$SANDBOX/.env"
printf 'LITELLM_API_KEY=sk-bogus\nMARK=staged\n' > "$SANDBOX/.env.deploy-staged"
gate "$GW_OK"; rc=$?
check "returns 0" "$rc" "0"
check "env created" "$(sed -n 's/^MARK=//p' "$SANDBOX/.env")" "staged"

echo "=== 6. NO staged LITELLM_API_KEY (tmpfs overlay) -> promotes ==="
printf 'MARK=live\n' > "$SANDBOX/.env"
printf 'MARK=staged\nOTHER=x\n' > "$SANDBOX/.env.deploy-staged"
gate "$GW_OK"; rc=$?
check "returns 0" "$rc" "0"
check "env replaced" "$(sed -n 's/^MARK=//p' "$SANDBOX/.env")" "staged"

echo
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
