#!/usr/bin/env bash
# Rebuild the app validation corpus the way it is committed, and show or apply the difference:
# `make regen-fixtures` (dry run) / `make regen-fixtures WRITE=1`.
#
# The README's "Regenerate" procedure, as one command: build into a temp copy, realign the GI
# (and check a second realign changes nothing), run the two corpus enrichers on the realigned GI,
# then compare with tests/fixtures/app-validation-corpus/v3 file by file.
#
#   (default)  report added / changed / removed files; nothing is written
#   --write    copy added and changed files into the fixture. Removed files are LISTED, never
#              deleted: a file the builder no longer writes may be one nobody meant to drop.
#
# After an ad-cut change, re-derive the captured renders first (README): this does not.
set -euo pipefail
unset NODE_OPTIONS

cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
FIX=tests/fixtures/app-validation-corpus/v3
write=0
case "${1:-}" in --write) write=1 ;; "") ;; *) echo "usage: $0 [--write]"; exit 2 ;; esac

S=$(mktemp -d)
trap '[ -n "${KEEP_TMP:-}" ] && echo "kept $S" || rm -rf "$S"' EXIT
cp -R "$FIX" "$S/v3"
rm -rf "$S/v3/search/lance_index" "$S/v3/search/episode_fingerprints.json"

echo "--> build"
HF_HUB_OFFLINE=1 "$PY" scripts/build_app_validation_corpus.py --output "$S" >"$S/build.log" 2>&1 \
  || { tail -30 "$S/build.log"; echo "FAIL: build"; exit 1; }
echo "--> realign"
"$PY" scripts/tools/realign_fixture_gi.py --corpus "$S/v3" --apply >"$S/realign.log" 2>&1 \
  || { tail -30 "$S/realign.log"; echo "FAIL: realign"; exit 1; }
second=$("$PY" scripts/tools/realign_fixture_gi.py --corpus "$S/v3" 2>&1 | tail -5)
printf '    %s\n' "${second//$'\n'/$'\n'    }"
echo "--> corpus enrichers"
HF_HUB_OFFLINE=1 PYTHONPATH=src "$PY" -m podcast_scraper.cli enrich --output-dir "$S/v3" \
  --only grounding_rate,guest_coappearance --corpus-only --force >"$S/enrich.log" 2>&1 \
  || { tail -30 "$S/enrich.log"; echo "FAIL: enrich"; exit 1; }

# What the fixture deliberately does not commit (README: "The search index is built").
skip='^search/(lance_index/|episode_fingerprints\.json$)'
list() { (cd "$1" && find . -type f | sed 's#^\./##' | grep -vE "$skip" | sort); }
list "$S/v3" >"$S/new.txt"
list "$FIX" >"$S/old.txt"
added=$(comm -13 "$S/old.txt" "$S/new.txt")
removed=$(comm -23 "$S/old.txt" "$S/new.txt")
# A changed file is one whose CONTENT changed: an enrichment run stamps when it ran (computed_at,
# started_at, finished_at, ts, run_id, duration_ms, last_run_at, last_run_id) and appends to the
# run log, and those alone are not a change, or every regeneration would rewrite the same six
# files (measured 2026-10-17 on an up-to-date fixture).
changed=$(comm -12 "$S/old.txt" "$S/new.txt" | "$PY" -c '
import json, sys
VOLATILE = {"computed_at", "started_at", "finished_at", "ts", "run_id", "duration_ms",
            "last_run_at", "last_run_id"}
LOGS = {"enrichments/run.jsonl"}  # append-only: grows on every run
def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x
def load(path):
    raw = open(path, "rb").read()
    try:
        if path.endswith(".jsonl"):
            return [strip(json.loads(line)) for line in raw.decode().splitlines() if line.strip()]
        if path.endswith(".json"):
            return strip(json.loads(raw))
    except ValueError:
        pass
    return raw
new, old = sys.argv[1], sys.argv[2]
rels = sys.stdin.read().split()
out = [r for r in rels if r not in LOGS and open(f"{new}/{r}", "rb").read() != open(f"{old}/{r}", "rb").read()
       and load(f"{new}/{r}") != load(f"{old}/{r}")]
# The run log changes with every run; it is only part of a change that changed something else.
if out:
    out += [r for r in rels if r in LOGS and open(f"{new}/{r}", "rb").read() != open(f"{old}/{r}", "rb").read()]
print("\n".join(out))
' "$S/v3" "$FIX")

count() { if [ -z "$1" ]; then echo 0; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi; }
echo
echo "added $(count "$added"), changed $(count "$changed"), removed $(count "$removed")"
[ -z "$added" ] || printf '%s\n' "$added" | sed 's/^/  + /'
[ -z "$changed" ] || printf '%s\n' "$changed" | sed 's/^/  ~ /'
[ -z "$removed" ] || printf '%s\n' "$removed" | sed 's/^/  - (not deleted) /'

if [ "$write" = 1 ]; then
  for f in $added $changed; do
    mkdir -p "$(dirname "$FIX/$f")"
    cp "$S/v3/$f" "$FIX/$f"
  done
  echo "WROTE $(( $(count "$added") + $(count "$changed") )) file(s) into $FIX"
else
  echo "DRY RUN: nothing written (WRITE=1 to apply)"
fi
