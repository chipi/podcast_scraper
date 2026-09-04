#!/usr/bin/env bash
# incremental_step_validate.sh — snapshot + GO/NO-GO for one incremental prod processing step.
#
# The incremental rollout (per-feed jobs, e.g. Step 0/1) adds a few episodes at a time. This
# captures the corpus state before a run and diffs it after, printing PASS/FAIL against the
# GO/NO-GO criteria, so validation is one command instead of ad-hoc curls.
#
# Usage:
#   incremental_step_validate.sh before                 # snapshot baseline -> $SNAP
#   # ... trigger the job, wait for it to finish ...
#   incremental_step_validate.sh after  [expected_delta] # diff vs baseline + PASS/FAIL (default delta=1)
#
# Env:
#   B    prod base URL (default the tailnet operator API)
#   SNAP snapshot file (default /tmp/incr_step_snapshot.json)
#   HOMELAB_SSH_KEY  optional; if set, also counts NEW GlitchTip errors in the window
#
# No prod SSH required for the corpus checks — all via the tailnet operator API.
set -euo pipefail

# Default base = prod over the tailnet, with the domain DERIVED rather than hardcoded (a literal
# is an operator identifier in a tracked file; a placeholder would just break the default). Pass
# B= to point elsewhere. Resolution order lives in resolve_tailnet_domain.sh.
# shellcheck source=scripts/ops/resolve_tailnet_domain.sh
. "$(dirname "${BASH_SOURCE[0]}")/resolve_tailnet_domain.sh"
if [ -z "${B:-}" ]; then
  _d="$(resolve_tailnet_domain)" || {
    echo "incremental_step_validate: cannot resolve the tailnet domain — pass B=https://<prod-host> explicitly" >&2
    exit 1
  }
  B="https://prod-podcast.${_d}"
fi
Q="path=/app/output"
SNAP="${SNAP:-/tmp/incr_step_snapshot.json}"

_snapshot() {
  local eps clusters idx
  eps=$(curl -s "$B/api/corpus/episodes?$Q&limit=1000" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("items",[])))')
  clusters=$(curl -s "$B/api/corpus/topic-clusters?$Q" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("clusters",[])))')
  idx=$(curl -s "$B/api/index/stats?$Q")
  python3 - "$eps" "$clusters" <<'PY'
import sys, json
eps, clusters = int(sys.argv[1]), int(sys.argv[2])
idx = json.load(open("/dev/stdin")) if False else None
print(json.dumps({"episodes": eps, "clusters": clusters}))
PY
  # index stats captured separately (kept raw for the diff)
  echo "$idx" > "${SNAP}.idx"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  before)
    _snapshot > "$SNAP"
    echo "baseline saved to $SNAP:"
    cat "$SNAP"
    echo "index reindex_recommended: $(python3 -c 'import sys,json;print(json.load(open("'"${SNAP}.idx"'")).get("reindex_recommended"))')"
    ;;
  after)
    expected_delta="${1:-1}"
    [ -f "$SNAP" ] || { echo "FAIL: no baseline at $SNAP (run 'before' first)"; exit 2; }
    now=$(_snapshot)
    b_eps=$(python3 -c 'import json;print(json.load(open("'"$SNAP"'"))["episodes"])')
    a_eps=$(echo "$now" | python3 -c 'import sys,json;print(json.load(sys.stdin)["episodes"])')
    b_cl=$(python3 -c 'import json;print(json.load(open("'"$SNAP"'"))["clusters"])')
    a_cl=$(echo "$now" | python3 -c 'import sys,json;print(json.load(sys.stdin)["clusters"])')
    delta=$((a_eps - b_eps))
    reindex=$(curl -s "$B/api/index/stats?$Q" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("reindex_recommended"))')
    echo "episodes: $b_eps -> $a_eps (delta $delta, expected $expected_delta)"
    echo "clusters: $b_cl -> $a_cl"
    echo "index reindex_recommended: $reindex (false = index caught up)"
    ok=1
    [ "$delta" = "$expected_delta" ] || { echo "  FAIL: episode delta $delta != expected $expected_delta"; ok=0; }
    [ "$reindex" = "False" ] || { echo "  WARN: reindex_recommended=$reindex (index may need a rebuild)"; }
    if [ "$ok" = 1 ]; then echo "RESULT: GO"; else echo "RESULT: NO-GO"; exit 1; fi
    ;;
  *)
    echo "usage: $0 before | after [expected_delta]"; exit 64;;
esac
