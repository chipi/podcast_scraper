#!/usr/bin/env bash
# PA5: build a bake-off/finale corpus copy from the FROZEN v2.4 control (prod-v2.4-relabel-fixed)
# without the two footguns that bit the 100-ep run:
#   1. `cp -R run_dir/ dest/` (trailing slash) FLATTENS the run dir on BSD cp -> the pipeline can't
#      match episodes under run_*/metadata/ and produces empty output. We strip the trailing slash.
#   2. Copying ALL run_ dirs per feed (source + prior relabel) mixes 2.4 gi.json with fresh output.
#      We copy only the ONE canonical (non-empty-transcripts) run per feed.
# Also prints a verification block and asserts the SOURCE (relabel-fixed) is never written.
#
# Usage: scripts/eval/build_finale_corpus.sh <dest-name> [eps-per-feed|all]
#   dest-name    -> .test_outputs/manual/<dest-name>
#   eps-per-feed -> integer N (keep episodes 0001..000N per feed) or "all" (default: all)
set -euo pipefail

DEST_NAME="${1:?usage: build_finale_corpus.sh <dest-name> [eps-per-feed|all]}"
EPS="${2:-all}"
ROOT=".test_outputs/manual"
SRC="$ROOT/prod-v2.4-relabel-fixed/feeds"
DST="$ROOT/$DEST_NAME/feeds"

[ -d "$SRC" ] || { echo "FATAL: frozen control not found at $SRC"; exit 1; }
[ -e "$ROOT/$DEST_NAME" ] && { echo "FATAL: $ROOT/$DEST_NAME exists — remove it first"; exit 1; }

SRC_SNAPSHOT="$(find "$SRC" -type f | wc -l | tr -d ' ')"
mkdir -p "$DST"
total=0
for feed in "$SRC"/*/; do
  fn="$(basename "$feed")"
  best=""; bestn=0
  for r in "$feed"run_*/; do
    n="$(ls "$r/transcripts/" 2>/dev/null | grep -cE '\.adfree\.txt$' || true)"
    [ "${n:-0}" -gt "$bestn" ] && { bestn="$n"; best="${r%/}"; }   # strip trailing slash (footgun #1)
  done
  [ -n "$best" ] || { echo "WARN: no transcripts for $fn, skipping"; continue; }
  mkdir -p "$DST/$fn"
  cp -R "$best" "$DST/$fn/"                                          # creates $DST/$fn/run_<id>/ (nested)
  rundir="$DST/$fn/$(basename "$best")"
  if [ "$EPS" != "all" ]; then
    keep="$(printf '%04d' "$EPS")"
    for sub in transcripts metadata; do
      find "$rundir/$sub" -type f 2>/dev/null | while read -r f; do
        idx="$(basename "$f" | grep -oE '^[0-9]{4}' || echo 9999)"
        [ "$idx" -gt "$EPS" ] && rm -f "$f"
      done
    done
  fi
  kept="$(find "$rundir/transcripts" -name '*.adfree.txt' 2>/dev/null | wc -l | tr -d ' ')"
  total=$((total + kept))
  printf "  %-40s %s eps (from %s)\n" "$fn" "$kept" "$(basename "$best")"
done

echo "----------------------------------------------------------------"
echo "  corpus: $total episodes across $(ls "$DST" | wc -l | tr -d ' ') feeds -> $ROOT/$DEST_NAME"
echo "  layout check (must be feeds/<feed>/run_<id>/{transcripts,metadata}):"
find "$DST" -maxdepth 2 -type d -name 'run_*' | head -1 | sed "s#$DST#    feeds#"
SRC_AFTER="$(find "$SRC" -type f | wc -l | tr -d ' ')"
if [ "$SRC_SNAPSHOT" = "$SRC_AFTER" ]; then
  echo "  SOURCE relabel-fixed UNTOUCHED ($SRC_AFTER files, unchanged) ✓"
else
  echo "  FATAL: relabel-fixed file count changed ($SRC_SNAPSHOT -> $SRC_AFTER) — SOURCE WAS MODIFIED"; exit 1
fi
